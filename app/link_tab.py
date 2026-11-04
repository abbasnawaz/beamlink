from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import streamlit as st
from map_plots import build_scene_map_figure
from plots import (
    METHOD_LABELS,
    build_beam_pattern_figure,
    build_channel_response_figure,
    build_constellation_figure,
    build_pdp_figure,
)

from beamlink.array import beam_pattern
from beamlink.beamforming import dft_codebook
from beamlink.channel import subcarrier_frequencies_hz
from beamlink.config import (
    APP_DEFAULT_POSITION_XY_M,
    APP_LINK_SLOTS,
    APP_RAY_COUNT,
    BeamMethod,
    CsiMode,
    Modulation,
    SimulationMode,
    SystemConfig,
)
from beamlink.data import (
    ChannelPaths,
    Dataset,
    SceneMeta,
    n_probe_beams,
    nearest_position_idx,
    position_paths,
)
from beamlink.link import LinkResult, run_link
from beamlink.linkbudget import linear_to_db
from beamlink.modulation import constellation
from beamlink.predictor import BeamPredictor

PATTERN_POINTS = 361
MODULATION_LABELS = {Modulation.QPSK: "QPSK", Modulation.QAM16: "16-QAM", Modulation.QAM64: "64-QAM"}


@dataclass(frozen=True, slots=True)
class AppResources:
    config: SystemConfig
    dataset: Dataset
    meta: SceneMeta
    scene_image: bytes
    predictor: BeamPredictor | None
    ai_ranking: Callable[[int], npt.NDArray[np.int64] | None]
    coverage: Callable[[int, float, int], dict[str, npt.NDArray[np.float32]]]
    model_card: Callable[[int], dict[str, float]]


@dataclass(frozen=True, slots=True)
class Controls:
    config: SystemConfig
    tx_power_dbm: float | None
    snr_db: float | None
    modulation: Modulation
    method: BeamMethod
    top_k: int
    csi_mode: CsiMode
    simulation_mode: SimulationMode
    seed: int


def selected_position(resources: AppResources) -> int:
    if "position_idx" not in st.session_state:
        st.session_state["position_idx"] = nearest_position_idx(
            resources.dataset.positions_m, *APP_DEFAULT_POSITION_XY_M
        )
    return int(st.session_state["position_idx"])


def ai_available(resources: AppResources, n_antennas: int) -> bool:
    return resources.predictor is not None and resources.predictor.norm.n_antennas == n_antennas


def available_methods(resources: AppResources, n_antennas: int) -> list[BeamMethod]:
    return [
        method
        for method in BeamMethod
        if method is not BeamMethod.AI_TOP_K or ai_available(resources, n_antennas)
    ]


def simulate(
    resources: AppResources, paths: ChannelPaths, controls: Controls, method: BeamMethod, position_idx: int
) -> LinkResult:
    candidates, probes = None, 0
    if method is BeamMethod.AI_TOP_K:
        candidates = resources.ai_ranking(controls.top_k)[position_idx]
        probes = n_probe_beams(resources.predictor.norm.feature_set)
    return run_link(
        paths,
        controls.config,
        method=method,
        modulation=controls.modulation,
        csi_mode=controls.csi_mode,
        simulation_mode=controls.simulation_mode,
        snr_db=controls.snr_db,
        tx_power_dbm=controls.tx_power_dbm,
        seed=controls.seed,
        n_slots=APP_LINK_SLOTS,
        candidate_beams=candidates,
        n_probe_beams=probes,
    )


def _display_weights(result: LinkResult) -> npt.NDArray[np.complex64]:
    weights = result.beam_weights
    return weights if weights.ndim == 1 else weights[:, weights.shape[1] // 2]


def _pattern(config: SystemConfig, weights: npt.NDArray[np.complex64]) -> tuple[npt.NDArray, npt.NDArray]:
    relative = np.linspace(-np.pi / 2, np.pi / 2, PATTERN_POINTS, dtype=np.float32)
    azimuth = (config.bs_boresight_azimuth + relative).astype(np.float32)
    pattern = beam_pattern(
        weights,
        azimuth,
        zenith=np.pi / 2,
        spacing_wavelengths=config.antenna_spacing_wavelengths,
        boresight_azimuth=config.bs_boresight_azimuth,
    )
    return azimuth, pattern


def _strongest_rays(paths: ChannelPaths) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    power = np.where(paths.path_valid, np.abs(paths.path_gain) ** 2, 0.0)
    order = np.argsort(-power)[:APP_RAY_COUNT]
    order = order[power[order] > 0]
    return paths.path_azimuth[order], linear_to_db(power[order])


def _map_section(
    resources: AppResources, controls: Controls, result: LinkResult, paths: ChannelPaths, position_idx: int
) -> None:
    dataset, meta = resources.dataset, resources.meta
    azimuth, pattern = _pattern(controls.config, _display_weights(result))
    ray_azimuth, ray_power_db = _strongest_rays(paths)
    figure = build_scene_map_figure(
        image_png=resources.scene_image,
        image_extent_m=meta.image_extent_m,
        grid_extent_m=meta.grid_extent_m,
        positions_m=dataset.positions_m,
        is_los=dataset.is_los,
        bs_position_m=meta.bs_position_m,
        selected_idx=position_idx,
        lobe_azimuth=azimuth,
        lobe_gain=pattern / np.max(pattern),
        ray_azimuth=ray_azimuth,
        ray_power_db=ray_power_db,
        method=result.method,
    )
    event = st.plotly_chart(figure, on_select="rerun", selection_mode="points", width="stretch")
    points = event.selection.points if event and event.selection else []
    clicked = [point["customdata"] for point in points if point.get("curve_number") == 0]
    if clicked and int(np.ravel(clicked[0])[0]) != position_idx:
        st.session_state["position_idx"] = int(np.ravel(clicked[0])[0])
        st.rerun()
    with st.expander("Pick a position by coordinates"), st.form("coordinates"):
        location = dataset.positions_m[position_idx]
        x_m = st.slider("x (m)", *meta.grid_extent_m[:2], float(location[0]), step=meta.grid_spacing_m)
        y_m = st.slider("y (m)", *meta.grid_extent_m[2:], float(location[1]), step=meta.grid_spacing_m)
        if st.form_submit_button("Snap to nearest grid point"):
            st.session_state["position_idx"] = nearest_position_idx(dataset.positions_m, x_m, y_m)
            st.rerun()


def _metric_cards(result: LinkResult) -> None:
    columns = st.columns(6)
    columns[0].metric("Received SNR", f"{result.snr_db:.1f} dB")
    columns[1].metric("Beamforming gain", f"{result.beamforming_gain_db:.2f} dB")
    columns[2].metric("Bit error rate", f"{result.ber:.2e}")
    columns[3].metric("Spectral efficiency", f"{result.spectral_efficiency:.2f} bit/s/Hz")
    columns[4].metric("Beams tested", f"{result.beams_tested}")
    columns[5].metric("Power loss vs sweep", f"{result.power_loss_db:+.2f} dB")


def _pattern_figures(controls: Controls, result: LinkResult) -> None:
    config = controls.config
    left, right = st.columns(2)
    _, pattern = _pattern(config, _display_weights(result))
    relative_deg = np.rad2deg(np.linspace(-np.pi / 2, np.pi / 2, PATTERN_POINTS))
    codebook_db = None
    if result.method in (BeamMethod.SWEEP, BeamMethod.AI_TOP_K):
        codebook = dft_codebook(
            n_antennas=config.n_antennas,
            oversampling=config.codebook_oversampling,
            spacing_wavelengths=config.antenna_spacing_wavelengths,
            sector=config.codebook_sector,
        )
        codebook_db = linear_to_db(_pattern(config, codebook)[1])
    left.plotly_chart(
        build_beam_pattern_figure(relative_deg, linear_to_db(pattern), codebook_db, result.method)
    )
    ideal = constellation(controls.modulation)
    right.plotly_chart(
        build_constellation_figure(
            result.symbols_before_equalization, result.symbols_after_equalization, ideal
        )
    )


def _channel_figures(controls: Controls, result: LinkResult) -> None:
    config = controls.config
    frequency_megahertz = (
        subcarrier_frequencies_hz(
            n_subcarriers=config.n_subcarriers, subcarrier_spacing_hz=config.subcarrier_spacing_hz
        )
        / 1e6
    )
    unit_label = "normalised channel" if controls.snr_db is not None else "includes path loss"
    left, right = st.columns(2)
    visible = result.pdp > 0
    left.plotly_chart(build_pdp_figure(result.pdp_delay_s[visible] * 1e9, linear_to_db(result.pdp[visible])))
    right.plotly_chart(
        build_channel_response_figure(
            frequency_megahertz,
            linear_to_db(np.abs(result.cfr_reference_antenna) ** 2),
            linear_to_db(np.abs(result.h_eff_true) ** 2),
            linear_to_db(np.abs(result.h_eff_estimated) ** 2),
            result.method,
            unit_label,
        )
    )


def _comparison_table(
    resources: AppResources, controls: Controls, paths: ChannelPaths, position_idx: int
) -> None:
    rows = []
    for method in available_methods(resources, controls.config.n_antennas):
        result = simulate(resources, paths, controls, method, position_idx)
        rows.append(
            {
                "Method": METHOD_LABELS[method],
                "SNR (dB)": round(result.snr_db, 2),
                "BF gain (dB)": round(result.beamforming_gain_db, 2),
                "BER": f"{result.ber:.2e}",
                "Spectral eff. (bit/s/Hz)": round(result.spectral_efficiency, 2),
                "Beams tested": result.beams_tested,
                "Loss vs sweep (dB)": round(result.power_loss_db, 2),
            }
        )
    st.subheader("All methods on the same channel and noise")
    st.dataframe(rows, hide_index=True, width="stretch")


def render_link_tab(resources: AppResources, controls: Controls) -> None:
    position_idx = selected_position(resources)
    dataset = resources.dataset
    paths = position_paths(dataset.paths, position_idx)
    result = simulate(resources, paths, controls, controls.method, position_idx)
    region = "LOS" if dataset.is_los[position_idx] else "NLOS"
    location = dataset.positions_m[position_idx]
    st.markdown(f"**UE #{position_idx}** at ({location[0]:.1f}, {location[1]:.1f}) m, {region}")
    _map_section(resources, controls, result, paths, position_idx)
    _metric_cards(result)
    _pattern_figures(controls, result)
    _channel_figures(controls, result)
    _comparison_table(resources, controls, paths, position_idx)
