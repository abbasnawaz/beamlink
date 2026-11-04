import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from evaluate_link import synthetic_channel
from evaluate_style import CATEGORICAL, METHOD_COLORS, METHOD_LABELS, NEUTRAL, save_figure, style_axes
from matplotlib.axes import Axes

from beamlink.array import beam_pattern, steered_beam_weights, ula_steering_vectors
from beamlink.beamforming import (
    beamforming_gain,
    codebook_beam_azimuths,
    dft_codebook,
    effective_channel,
    spread_over_subcarriers,
    strongest_beam_idx,
    subcarrier_mrt_weights,
    sweep_beams,
    wideband_mrt_weights,
    wideband_power,
)
from beamlink.channel import paths_to_cfr
from beamlink.config import (
    BeamMethod,
    SystemConfig,
)
from beamlink.data import ChannelPaths, position_paths
from beamlink.linkbudget import linear_to_db
from beamlink.metrics import power_loss_db

logger = logging.getLogger("beamlink.offline")

PATTERN_FLOOR_DB = -25.0
PATTERN_GRID = np.linspace(-np.pi / 2, np.pi / 2, 1801, dtype=np.float32)
ANTENNA_COUNTS = (4, 8, 16, 32)
GAIN_STUDY_POSITIONS = 400
LOS_K_FACTOR_DB = 10.0
OVERSAMPLED = 4


def _pattern_db(beam_weights: npt.NDArray[np.complex64], config: SystemConfig) -> npt.NDArray[np.float32]:
    pattern = beam_pattern(
        beam_weights,
        PATTERN_GRID,
        zenith=np.pi / 2,
        spacing_wavelengths=config.antenna_spacing_wavelengths,
        boresight_azimuth=0.0,
    )
    return np.maximum(linear_to_db(pattern), PATTERN_FLOOR_DB)


def _half_polar(panel: Axes, max_db: float) -> None:
    panel.set_theta_zero_location("N")
    panel.set_thetamin(-90)
    panel.set_thetamax(90)
    panel.set_rlim(PATTERN_FLOOR_DB, max_db)
    panel.set_rticks(np.arange(-20.0, max_db, 10.0))
    panel.set_rlabel_position(-90)
    panel.tick_params(labelsize=8)


def _codebook(config: SystemConfig, n_antennas: int, oversampling: int) -> npt.NDArray[np.complex64]:
    return dft_codebook(
        n_antennas=n_antennas,
        oversampling=oversampling,
        spacing_wavelengths=config.antenna_spacing_wavelengths,
        sector=config.codebook_sector,
    )


def _cfr(paths: ChannelPaths, config: SystemConfig, n_antennas: int) -> npt.NDArray[np.complex64]:
    steering = ula_steering_vectors(
        paths.path_azimuth,
        paths.path_zenith,
        n_antennas=n_antennas,
        spacing_wavelengths=config.antenna_spacing_wavelengths,
        boresight_azimuth=0.0,
    )
    return paths_to_cfr(
        paths.path_gain,
        paths.path_delay_s,
        steering,
        n_subcarriers=config.n_subcarriers,
        subcarrier_spacing_hz=config.subcarrier_spacing_hz,
    )


def _beam_weights(
    cfr: npt.NDArray[np.complex64], codebook: npt.NDArray[np.complex64], method: BeamMethod
) -> npt.NDArray[np.complex64]:
    n_subcarriers = cfr.shape[-1]
    if method is BeamMethod.MRT_SUBCARRIER:
        return subcarrier_mrt_weights(cfr)
    if method is BeamMethod.MRT_WIDEBAND:
        return spread_over_subcarriers(wideband_mrt_weights(cfr), n_subcarriers=n_subcarriers)
    best = np.argmax(sweep_beams(cfr, codebook), axis=-1)
    return spread_over_subcarriers(codebook[best], n_subcarriers=n_subcarriers)


def build_beam_pattern_figure(config: SystemConfig, out_path: Path) -> None:
    figure = plt.figure(figsize=(17, 4.4))
    n_antennas = 8
    panel = figure.add_subplot(1, 3, 1, projection="polar")
    codebook = _codebook(config, n_antennas, 1)
    beam_azimuths = codebook_beam_azimuths(
        n_antennas=n_antennas,
        oversampling=1,
        spacing_wavelengths=config.antenna_spacing_wavelengths,
        sector=config.codebook_sector,
        boresight_azimuth=0.0,
    )
    for beam_idx, (pattern_db, azimuth) in enumerate(
        zip(_pattern_db(codebook, config), beam_azimuths, strict=True)
    ):
        panel.plot(
            PATTERN_GRID,
            pattern_db,
            color=CATEGORICAL[beam_idx],
            lw=1.8,
            label=f"beam {beam_idx}: {np.rad2deg(azimuth):+.1f} deg",
        )
    _half_polar(panel, float(linear_to_db(n_antennas)) + 1)
    panel.set_title("(a) N = 8 orthonormal DFT codebook\ngain (dBi, relative to one antenna)", fontsize=11)
    panel.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, frameon=False)

    panel = figure.add_subplot(1, 3, 2, projection="polar")
    for color, count in zip(CATEGORICAL, ANTENNA_COUNTS, strict=False):
        weights = steered_beam_weights(
            0.0,
            np.pi / 2,
            n_antennas=count,
            spacing_wavelengths=config.antenna_spacing_wavelengths,
            boresight_azimuth=0.0,
        )
        panel.plot(
            PATTERN_GRID,
            _pattern_db(weights, config),
            color=color,
            lw=1.8,
            label=f"N = {count}: peak {float(linear_to_db(count)):.1f} dBi",
        )
    _half_polar(panel, float(linear_to_db(max(ANTENNA_COUNTS))) + 1)
    panel.set_title("(b) Broadside beam: more antennas give\nmore gain and a narrower beam", fontsize=11)
    panel.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, fontsize=8, frameon=False)

    panel = figure.add_subplot(1, 3, 3, projection="polar")
    n_antennas = 16
    two_paths = ChannelPaths(
        path_gain=np.array([1.0, 0.8 * np.exp(1j)], dtype=np.complex64),
        path_delay_s=np.array([0.0, 40e-9], dtype=np.float32),
        path_azimuth=np.deg2rad([-25.0, 35.0]).astype(np.float32),
        path_zenith=np.full(2, np.pi / 2, dtype=np.float32),
        path_valid=np.ones(2, dtype=bool),
    )
    cfr = _cfr(two_paths, config, n_antennas)
    codebook = _codebook(config, n_antennas, 1)
    sweep_idx = strongest_beam_idx(sweep_beams(cfr, codebook), candidate_idx=np.arange(n_antennas))
    panel.plot(
        PATTERN_GRID,
        _pattern_db(codebook[sweep_idx], config),
        color=METHOD_COLORS[BeamMethod.SWEEP],
        lw=2,
        label=f"{METHOD_LABELS[BeamMethod.SWEEP]} (beam {sweep_idx})",
    )
    panel.plot(
        PATTERN_GRID,
        _pattern_db(wideband_mrt_weights(cfr), config),
        color=METHOD_COLORS[BeamMethod.MRT_WIDEBAND],
        lw=2,
        label=METHOD_LABELS[BeamMethod.MRT_WIDEBAND],
    )
    path_length_db = float(linear_to_db(n_antennas)) + linear_to_db(np.abs(two_paths.path_gain) ** 2)
    ray_azimuths = np.stack([two_paths.path_azimuth, two_paths.path_azimuth])
    ray_radii = np.stack([np.full_like(path_length_db, PATTERN_FLOOR_DB), path_length_db])
    ray_labels = ["path directions"] + ["_nolegend_"] * (two_paths.path_gain.size - 1)
    panel.plot(ray_azimuths, ray_radii, color=NEUTRAL, lw=2.5, ls=":", label=ray_labels)
    _half_polar(panel, float(linear_to_db(n_antennas)) + 1)
    panel.set_title(
        f"(c) N = {n_antennas}, two paths at -25 and +35 deg:\none DFT beam vs a two-lobe MRT beam",
        fontsize=11,
    )
    panel.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, fontsize=8, frameon=False)
    figure.text(
        0.5,
        -0.02,
        "Boresight is up. Positive azimuth is to the left (counter-clockwise seen from above, Sionna frame).",
        ha="center",
        fontsize=9,
        color=NEUTRAL,
    )
    figure.subplots_adjust(wspace=0.15, left=0.02, right=0.98, top=0.86, bottom=0.0)
    save_figure(figure, out_path)


def _gain_study(
    config: SystemConfig, rng: np.random.Generator, los_k_factor_db: float | None
) -> tuple[dict[str, list[float]], dict[int, npt.NDArray[np.float32]]]:
    gains: dict[str, list[float]] = {
        "sweep": [],
        "sweep_oversampled": [],
        "mrt_wideband": [],
        "mrt_subcarrier": [],
    }
    losses: dict[int, npt.NDArray[np.float32]] = {}
    paths = synthetic_channel(rng, GAIN_STUDY_POSITIONS, los_k_factor_db=los_k_factor_db)
    for n_antennas in ANTENNA_COUNTS:
        cfr = _cfr(paths, config, n_antennas)
        codebook = _codebook(config, n_antennas, 1)
        oversampled_power = np.max(sweep_beams(cfr, _codebook(config, n_antennas, OVERSAMPLED)), axis=-1)
        powers = {
            "sweep": wideband_power(effective_channel(cfr, _beam_weights(cfr, codebook, BeamMethod.SWEEP))),
            "sweep_oversampled": oversampled_power,
            "mrt_wideband": wideband_power(
                effective_channel(cfr, _beam_weights(cfr, codebook, BeamMethod.MRT_WIDEBAND))
            ),
            "mrt_subcarrier": wideband_power(
                effective_channel(cfr, _beam_weights(cfr, codebook, BeamMethod.MRT_SUBCARRIER))
            ),
        }
        for name, power in powers.items():
            gains[name].append(float(linear_to_db(np.mean(beamforming_gain(power, cfr)))))
        losses[n_antennas] = power_loss_db(powers["mrt_wideband"], powers["sweep"])
    return gains, losses


def build_gain_figure(config: SystemConfig, rng: np.random.Generator, out_path: Path) -> None:
    scenarios = {
        f"LOS-like (K = {LOS_K_FACTOR_DB:.0f} dB)": _gain_study(config, rng, LOS_K_FACTOR_DB),
        "NLOS (rich scattering)": _gain_study(config, rng, None),
    }
    styles = {
        "sweep": {
            "color": METHOD_COLORS[BeamMethod.SWEEP],
            "marker": "o",
            "ls": "-",
            "label": "DFT sweep, B = N",
        },
        "sweep_oversampled": {
            "color": METHOD_COLORS[BeamMethod.SWEEP],
            "marker": "o",
            "ls": "--",
            "mfc": "none",
            "label": f"DFT sweep, B = {OVERSAMPLED}N",
        },
        "mrt_wideband": {
            "color": METHOD_COLORS[BeamMethod.MRT_WIDEBAND],
            "marker": "s",
            "ls": "-",
            "label": METHOD_LABELS[BeamMethod.MRT_WIDEBAND],
        },
        "mrt_subcarrier": {
            "color": METHOD_COLORS[BeamMethod.MRT_SUBCARRIER],
            "marker": "^",
            "ls": "-",
            "label": METHOD_LABELS[BeamMethod.MRT_SUBCARRIER],
        },
    }
    figure, panels = plt.subplots(2, 2, figsize=(13, 9.5))
    for panel, (title, (gains, _)) in zip(panels[0], scenarios.items(), strict=True):
        panel.plot(
            ANTENNA_COUNTS,
            linear_to_db(np.asarray(ANTENNA_COUNTS)),
            color=NEUTRAL,
            lw=1.5,
            ls=":",
            label="Array gain 10 log10 N (per-subcarrier MRT reaches it exactly)",
        )
        for name, style in styles.items():
            panel.plot(ANTENNA_COUNTS, gains[name], lw=2, ms=8, **style)
            logger.info(
                f"gain scenario={title!r} method={name} gain_db={[round(value, 2) for value in gains[name]]}"
            )
        panel.set_xscale("log", base=2)
        panel.set_xticks(ANTENNA_COUNTS, [str(count) for count in ANTENNA_COUNTS])
        panel.set_xlabel("BS antennas N")
        panel.set_ylabel("Mean beamforming gain (dB)")
        panel.set_title(f"{title}, {GAIN_STUDY_POSITIONS} channels per N")
        style_axes(panel)
    panels[0, 0].legend(frameon=False, fontsize=9)

    n_antennas = 16
    cfr = _cfr(position_paths(synthetic_channel(rng, 1, los_k_factor_db=None), 0), config, n_antennas)
    codebook = _codebook(config, n_antennas, 1)
    frequency_megahertz = (
        (np.arange(config.n_subcarriers) - config.n_subcarriers // 2) * config.subcarrier_spacing_hz / 1e6
    )
    panels[1, 0].plot(
        frequency_megahertz,
        linear_to_db(np.abs(cfr[0]) ** 2),
        color=NEUTRAL,
        lw=1.5,
        label="Antenna 0 alone (no beamforming)",
    )
    for method in (BeamMethod.SWEEP, BeamMethod.MRT_WIDEBAND, BeamMethod.MRT_SUBCARRIER):
        h_eff = effective_channel(cfr, _beam_weights(cfr, codebook, method))
        panels[1, 0].plot(
            frequency_megahertz,
            linear_to_db(np.abs(h_eff) ** 2),
            color=METHOD_COLORS[method],
            lw=2,
            label=f"{METHOD_LABELS[method]} h_eff",
        )
    panels[1, 0].set_xlabel("Baseband frequency (MHz)")
    panels[1, 0].set_ylabel("|channel| (dB)")
    panels[1, 0].set_title(f"NLOS channel, N = {n_antennas}: beamforming lifts and flattens |H|")
    panels[1, 0].legend(frameon=False, fontsize=9)
    style_axes(panels[1, 0])

    dashes = dict(zip(scenarios, ("-", "--"), strict=True))
    for title, (_, losses) in scenarios.items():
        for color, count in zip(CATEGORICAL[4::2], (8, 32), strict=True):
            sorted_loss = np.sort(losses[count])
            cumulative_fraction = np.arange(1, sorted_loss.size + 1) / sorted_loss.size
            panels[1, 1].plot(
                sorted_loss,
                cumulative_fraction,
                color=color,
                ls=dashes[title],
                lw=2,
                label=f"N = {count}, {title}",
            )
            logger.info(
                f"sweep loss scenario={title!r} n_antennas={count} mean_db={sorted_loss.mean():.2f} "
                f"p95_db={np.percentile(sorted_loss, 95):.2f}"
            )
    panels[1, 1].set_xlabel("Sweep power loss vs wideband MRT (dB)")
    panels[1, 1].set_ylabel("CDF over channels")
    panels[1, 1].set_title("Price of a fixed codebook (B = N)")
    panels[1, 1].legend(frameon=False, fontsize=9, loc="lower right")
    style_axes(panels[1, 1])
    figure.tight_layout()
    save_figure(figure, out_path)
