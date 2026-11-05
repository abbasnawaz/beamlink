from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st
from link_tab import MODULATION_LABELS, AppResources, Controls, available_methods, selected_position, simulate
from map_plots import build_coverage_figure, categorical_colorscale
from plots import CATEGORICAL, build_ber_figure

from beamlink.config import (
    APP_BER_SLOTS,
    APP_BER_SNR_DB,
    MIN_SERVICE_SNR_DB,
    SPLIT_TEST,
    BeamMethod,
    CsiMode,
    Modulation,
    SimulationMode,
)
from beamlink.data import labels_match_config, position_paths
from beamlink.linkbudget import db_to_linear
from beamlink.metrics import awgn_ber_theory

COVERAGE_VIEWS = (
    "Received SNR, sweep",
    "Received SNR, AI top-K",
    "AI power loss vs sweep",
    "Serving sector, sweep",
    "LOS / NLOS",
)
SNR_VIEWS = COVERAGE_VIEWS[:2]
NO_SERVICE_LABEL = f"No service (SNR < {MIN_SERVICE_SNR_DB:.0f} dB)"
REPOSITORY_URL = "https://github.com/abbasnawaz/beamlink"
SCOPE_AND_LIMITATIONS = (
    "- Static channel: the user does not move during a slot, and there is no Doppler.\n"
    "- A single user per simulation, so there is no multi-user interference.\n"
    "- One ray-traced scene (Sionna RT's simple street canyon at 28 GHz), so the predictor is trained "
    "for this street only.\n\n"
    f"More in the README: [Why four sectors]({REPOSITORY_URL}#why-four-sectors) and "
    f"[Limitations and future work]({REPOSITORY_URL}#limitations-and-future-work)."
)
PIPELINE_DOT = """
digraph pipeline {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#f4f3ef", color="#52514e", fontname="Helvetica"];
  rt [label="Sionna RT\\nstreet canyon, 28 GHz"]; paths [label="Ray paths\\n48 strongest per UE"];
  cfr [label="NumPy CFR\\nULA steering vectors"]; beam [label="Beam selection\\nsweep / AI top-K / MRT"];
  tx [label="QAM + pilots\\nIFFT + CP"]; channel [label="Channel + noise\\nfreq or time domain"];
  rx [label="FFT, LS estimate\\none-tap equalizer"]; ber [label="Demapper\\nBER, SNR, SE"];
  rt -> paths -> cfr -> beam -> tx -> channel -> rx -> ber;
}
"""


def _coverage_style(
    view: str, values: np.ndarray, n_categories: int
) -> tuple[str, object, tuple[float, float]]:
    if view in SNR_VIEWS:
        served = values[values >= MIN_SERVICE_SNR_DB]
        low, high = (
            np.percentile(served, (1, 99)) if served.size else (MIN_SERVICE_SNR_DB, MIN_SERVICE_SNR_DB + 1)
        )
        return "SNR (dB)", "Blues", (max(float(low), MIN_SERVICE_SNR_DB), float(high))
    if view == COVERAGE_VIEWS[2]:
        return "loss (dB)", "Oranges", (0.0, max(1.0, float(np.percentile(values, 99.9))))
    if view == COVERAGE_VIEWS[3]:
        return "sector", categorical_colorscale(n_categories), (-0.5, n_categories - 0.5)
    return "1 = LOS", [[0.0, CATEGORICAL[1]], [1.0, CATEGORICAL[0]]], (0.0, 1.0)


def render_coverage_tab(resources: AppResources, controls: Controls) -> None:
    n_antennas = controls.config.n_antennas
    labels = resources.dataset.labels
    if n_antennas not in labels.beam_power_db:
        st.info(
            f"Coverage maps are precomputed for N in {sorted(labels.beam_power_db)}. Change N in the sidebar."
        )
        return
    if (
        not labels_match_config(labels, controls.config)
        or labels.beam_power_db[n_antennas].shape[1] != controls.config.n_beams
    ):
        st.info("Coverage labels were built for a different codebook. Run offline/02_build_labels.py.")
        return
    tx_power_dbm = (
        controls.tx_power_dbm if controls.tx_power_dbm is not None else resources.config.tx_power_dbm
    )
    coverage = resources.coverage(n_antennas, tx_power_dbm, controls.top_k)
    view = st.radio("Map", [name for name in COVERAGE_VIEWS if name in coverage], horizontal=True)
    colorbar_title, colorscale, value_range = _coverage_style(view, coverage[view], controls.config.n_sectors)
    meta = resources.meta
    no_service = coverage[view] < MIN_SERVICE_SNR_DB if view in SNR_VIEWS else None
    figure = build_coverage_figure(
        image_png=resources.scene_image,
        image_extent_m=meta.image_extent_m,
        grid_extent_m=meta.grid_extent_m,
        positions_m=resources.dataset.positions_m,
        values=coverage[view],
        bs_position_m=meta.bs_position_m,
        selected_idx=selected_position(resources),
        title=f"{view} (N = {n_antennas}, tx power {tx_power_dbm:.0f} dBm, K = {controls.top_k})",
        colorbar_title=colorbar_title,
        colorscale=colorscale,
        value_range=value_range,
        no_service=no_service,
        no_service_label=NO_SERVICE_LABEL,
    )
    st.plotly_chart(figure, width="stretch")
    snr_view = view if view in SNR_VIEWS else COVERAGE_VIEWS[0]
    unserved_percent = 100.0 * float(np.mean(coverage[snr_view] < MIN_SERVICE_SNR_DB))
    st.caption(
        f"{unserved_percent:.1f}% of positions have no service (SNR < {MIN_SERVICE_SNR_DB:.0f} dB) "
        f"at {tx_power_dbm:.0f} dBm with {snr_view.split(', ')[1]} beam selection."
    )


@st.cache_data(show_spinner="Running Monte-Carlo BER...")
def ber_curves(
    _resources: AppResources,
    position_idx: int,
    n_antennas: int,
    modulation: str,
    csi_mode: str,
    simulation_mode: str,
    prefix_length: int,
    top_k: int,
    seed: int,
) -> dict[str, list[float]]:
    config = replace(_resources.config, n_antennas=n_antennas, cyclic_prefix_length=prefix_length)
    paths = position_paths(_resources.dataset.paths, position_idx)
    curves: dict[str, list[float]] = {}
    for method in available_methods(_resources, n_antennas):
        controls = Controls(
            config,
            None,
            None,
            Modulation(modulation),
            method,
            top_k,
            CsiMode(csi_mode),
            SimulationMode(simulation_mode),
            seed,
        )
        curves[method.value] = [
            simulate(_resources, paths, replace(controls, snr_db=float(snr_db)), method, position_idx).ber
            for snr_db in APP_BER_SNR_DB
        ]
    return curves


def render_ber_tab(resources: AppResources, controls: Controls) -> None:
    position_idx = selected_position(resources)
    curves = ber_curves(
        resources,
        position_idx,
        controls.config.n_antennas,
        controls.modulation.value,
        controls.csi_mode.value,
        controls.simulation_mode.value,
        controls.config.cyclic_prefix_length,
        controls.top_k,
        controls.seed,
    )
    snr_db = np.asarray(APP_BER_SNR_DB, dtype=np.float32)
    theory_snr_db = np.linspace(snr_db[0], snr_db[-1], 200, dtype=np.float32)
    figure = build_ber_figure(
        snr_db,
        {BeamMethod(name): np.asarray(values, dtype=np.float32) for name, values in curves.items()},
        theory_snr_db,
        awgn_ber_theory(db_to_linear(theory_snr_db), modulation=controls.modulation),
        f"BER at UE #{position_idx}, {MODULATION_LABELS[controls.modulation]}, "
        f"N = {controls.config.n_antennas}, {APP_BER_SLOTS} slot(s) per point",
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Fixed-SNR mode: the channel is scaled so one isotropic antenna receives unit power, so the gap "
        "between the curves and the AWGN line is the beamforming gain (element plus array) minus the cost "
        "of frequency-selective fading."
    )


def render_about_tab(resources: AppResources) -> None:
    dataset, meta, predictor, config = (
        resources.dataset,
        resources.meta,
        resources.predictor,
        resources.config,
    )
    boresights_deg = ", ".join(f"{np.rad2deg(azimuth):.0f}" for azimuth in config.sector_boresight_azimuths)
    st.subheader("Pipeline")
    st.graphviz_chart(PIPELINE_DOT)
    st.subheader("Dataset")
    st.markdown(
        f"- Scene: `{meta.scene_name}` at {meta.carrier_frequency_hz / 1e9:.0f} GHz, ray traced with "
        f"Sionna RT {meta.library_versions.get('sionna-rt', '?')}\n"
        f"- {dataset.positions_m.shape[0]} UE positions on a {meta.grid_spacing_m} m grid, "
        f"{int(dataset.is_los.sum())} with line of sight\n"
        f"- BS at ({meta.bs_position_m[0]:.1f}, {meta.bs_position_m[1]:.1f}, {meta.bs_position_m[2]:.1f}) m, "
        f"{config.n_sectors} sectors facing {boresights_deg} deg azimuth, "
        f"{np.rad2deg(config.sector_downtilt):.0f} deg downtilt, {config.n_antennas} elements per sector "
        f"with a {config.element_peak_gain_dbi:.0f} dBi TR 38.901 pattern\n"
        "- Beamforming gain is measured against one isotropic antenna at the BS, so it includes the "
        "element gain as well as the array gain"
    )
    if predictor is not None:
        top_k = resources.config.predictor_top_k
        n_test_positions = int(np.count_nonzero(dataset.labels.split == SPLIT_TEST))
        st.subheader("Model card")
        st.markdown(
            f"MLP beam predictor, features `{predictor.norm.feature_set}` "
            f"({', '.join(predictor.norm.feature_names)}), N = {predictor.norm.n_antennas}, "
            f"{predictor.norm.n_beams} beams. Evaluated on {n_test_positions} positions in spatially "
            "held-out test blocks."
        )
        card = resources.model_card(top_k)
        table = pd.DataFrame({"Metric": list(card), "Value": [f"{value:.2f}" for value in card.values()]})
        st.dataframe(table, hide_index=True)
        st.caption(
            "On an oversampled codebook, power loss is the metric that matters: a neighbouring beam is "
            "usually almost as good as the best one, so a top-1 miss often costs very little."
        )
    st.subheader("Scope and limitations")
    st.markdown(SCOPE_AND_LIMITATIONS)
