from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from evaluate_style import CATEGORICAL, NEUTRAL, positive, save_figure, style_axes
from matplotlib.axes import Axes

from beamlink.config import (
    SYNTHETIC_AZIMUTH_SPAN,
    SYNTHETIC_DELAY_DECAY_S,
    SYNTHETIC_MAX_DELAY_S,
    SYNTHETIC_N_PATHS,
    SYNTHETIC_ZENITH_SPREAD,
    BeamMethod,
    CsiMode,
    Modulation,
    SimulationMode,
    SystemConfig,
)
from beamlink.data import ChannelPaths, position_paths, synthetic_paths
from beamlink.link import LinkResult, run_link
from beamlink.linkbudget import db_to_linear, linear_to_db, noise_variance_for_snr
from beamlink.metrics import awgn_ber_theory, fading_ber_theory
from beamlink.modulation import constellation

MODULATION_LABELS = {Modulation.QPSK: "QPSK", Modulation.QAM16: "16-QAM", Modulation.QAM64: "64-QAM"}
BER_SNR_DB = np.arange(0.0, 32.0, 2.0)
BER_SLOTS = 8
BER_FLOOR = 1e-5
DIAGNOSTIC_SNR_DB = 20.0
CP_SNR_DB = np.arange(0.0, 42.0, 3.0)
CP_SLOTS = 16
SHORT_PREFIX_LENGTH = 4
LONG_PREFIX_LENGTH = 21
LONG_DELAY_SPREAD_S = 500e-9
LONG_DELAY_DECAY_S = 250e-9


def synthetic_channel(
    rng: np.random.Generator,
    n_positions: int,
    *,
    los_k_factor_db: float | None,
    max_delay_s: float = SYNTHETIC_MAX_DELAY_S,
    delay_decay_s: float = SYNTHETIC_DELAY_DECAY_S,
) -> ChannelPaths:
    return synthetic_paths(
        rng=rng,
        n_positions=n_positions,
        n_paths=SYNTHETIC_N_PATHS,
        max_delay_s=max_delay_s,
        delay_decay_s=delay_decay_s,
        azimuth_span=SYNTHETIC_AZIMUTH_SPAN,
        zenith_spread=SYNTHETIC_ZENITH_SPREAD,
        los_power_ratio=None if los_k_factor_db is None else float(db_to_linear(los_k_factor_db)),
    )


def awgn_paths() -> ChannelPaths:
    return ChannelPaths(
        path_gain=np.ones(1, dtype=np.complex64),
        path_delay_s=np.zeros(1, dtype=np.float32),
        path_azimuth=np.zeros(1, dtype=np.float32),
        path_zenith=np.full(1, np.pi / 2, dtype=np.float32),
        path_valid=np.ones(1, dtype=bool),
    )


def _link(
    paths: ChannelPaths,
    config: SystemConfig,
    *,
    modulation: Modulation,
    csi_mode: CsiMode,
    simulation_mode: SimulationMode,
    snr_db: float,
    seed: int,
    n_slots: int,
) -> LinkResult:
    return run_link(
        paths,
        config,
        method=BeamMethod.SWEEP,
        modulation=modulation,
        csi_mode=csi_mode,
        simulation_mode=simulation_mode,
        snr_db=snr_db,
        seed=seed,
        n_slots=n_slots,
    )


def build_ber_figure(config: SystemConfig, rng: np.random.Generator, seed: int, out_path: Path) -> None:
    siso = replace(config, n_antennas=1)
    multipath = position_paths(synthetic_channel(rng, 1, los_k_factor_db=None), 0)
    fine_snr_db = np.linspace(BER_SNR_DB[0], BER_SNR_DB[-1], 200)
    figure, panels = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for panel, modulation in zip(panels, Modulation, strict=True):
        curves: dict[str, list[float]] = {"awgn": [], "perfect": [], "ls": [], "theory": []}
        for snr_db in BER_SNR_DB:
            common = {"modulation": modulation, "simulation_mode": SimulationMode.FREQUENCY, "snr_db": snr_db}
            common |= {"seed": seed, "n_slots": BER_SLOTS}
            curves["awgn"].append(_link(awgn_paths(), siso, csi_mode=CsiMode.PERFECT, **common).ber)
            perfect = _link(multipath, siso, csi_mode=CsiMode.PERFECT, **common)
            curves["perfect"].append(perfect.ber)
            curves["ls"].append(_link(multipath, siso, csi_mode=CsiMode.LS, **common).ber)
            noise_variance = noise_variance_for_snr(float(snr_db))
            curves["theory"].append(
                fading_ber_theory(perfect.h_eff_true, noise_variance=noise_variance, modulation=modulation)
            )
        awgn_theory = awgn_ber_theory(db_to_linear(fine_snr_db), modulation=modulation)
        panel.semilogy(fine_snr_db, awgn_theory, color=NEUTRAL, lw=2, label="AWGN theory")
        panel.semilogy(
            BER_SNR_DB,
            positive(curves["awgn"]),
            "o",
            color=CATEGORICAL[0],
            ms=8,
            mfc="none",
            mew=2,
            label="AWGN sim",
        )
        panel.semilogy(
            BER_SNR_DB, curves["theory"], "--", color=CATEGORICAL[1], lw=2, label="Multipath semi-analytic"
        )
        panel.semilogy(
            BER_SNR_DB,
            positive(curves["perfect"]),
            "s",
            color=CATEGORICAL[1],
            ms=8,
            mfc="none",
            mew=2,
            label="Multipath sim, perfect CSI",
        )
        panel.semilogy(
            BER_SNR_DB,
            positive(curves["ls"]),
            "^-",
            color=CATEGORICAL[2],
            ms=7,
            lw=2,
            label="Multipath sim, LS estimate",
        )
        panel.set_title(MODULATION_LABELS[modulation])
        panel.set_xlabel("SNR per resource element, Es/N0 (dB)")
        panel.set_ylim(BER_FLOOR, 0.6)
        style_axes(panel)
    panels[0].set_ylabel("Bit error rate")
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    figure.suptitle(
        f"OFDM BER vs SNR (K={config.n_subcarriers}, CP={config.cyclic_prefix_length}, one synthetic "
        f"{SYNTHETIC_N_PATHS}-path channel at unit average power, {config.equalizer.upper()} equalizer)"
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(figure, out_path)


def _cp_panel(panel: Axes, config: SystemConfig, rng: np.random.Generator, seed: int) -> None:
    siso = replace(config, n_antennas=1)
    paths = position_paths(
        synthetic_channel(
            rng, 1, los_k_factor_db=None, max_delay_s=LONG_DELAY_SPREAD_S, delay_decay_s=LONG_DELAY_DECAY_S
        ),
        0,
    )
    colors = {LONG_PREFIX_LENGTH: CATEGORICAL[0], SHORT_PREFIX_LENGTH: CATEGORICAL[1]}
    h_eff = None
    for prefix_length, color in colors.items():
        results = [
            _link(
                paths,
                replace(siso, cyclic_prefix_length=prefix_length),
                modulation=Modulation.QAM16,
                csi_mode=CsiMode.PERFECT,
                simulation_mode=SimulationMode.TIME,
                snr_db=float(snr_db),
                seed=seed,
                n_slots=CP_SLOTS,
            )
            for snr_db in CP_SNR_DB
        ]
        h_eff = results[0].h_eff_true
        panel.semilogy(
            CP_SNR_DB,
            positive([result.ber for result in results]),
            "o-",
            color=color,
            lw=2,
            ms=7,
            label=f"CP = {prefix_length} samples",
        )
    theory = [
        fading_ber_theory(
            h_eff, noise_variance=noise_variance_for_snr(float(snr_db)), modulation=Modulation.QAM16
        )
        for snr_db in CP_SNR_DB
    ]
    panel.semilogy(CP_SNR_DB, theory, "--", color=NEUTRAL, lw=1.5, label="Semi-analytic (no ISI)")
    panel.set_xlabel("SNR per resource element, Es/N0 (dB)")
    panel.set_ylabel("Bit error rate")
    panel.set_title("Cyclic prefix vs channel length, time-domain sim, 16-QAM")
    panel.set_ylim(BER_FLOOR, 0.6)
    panel.legend(frameon=False, fontsize=9)
    style_axes(panel)


def build_diagnostics_figure(
    config: SystemConfig, rng: np.random.Generator, seed: int, out_path: Path
) -> None:
    siso = replace(config, n_antennas=1)
    paths = position_paths(synthetic_channel(rng, 1, los_k_factor_db=None), 0)
    result = _link(
        paths,
        siso,
        modulation=Modulation.QAM16,
        csi_mode=CsiMode.LS,
        simulation_mode=SimulationMode.FREQUENCY,
        snr_db=DIAGNOSTIC_SNR_DB,
        seed=seed,
        n_slots=1,
    )
    figure, panels = plt.subplots(2, 2, figsize=(13, 9))

    pdp_db = linear_to_db(result.pdp)
    delay_nanoseconds = result.pdp_delay_s * 1e9
    panels[0, 0].vlines(delay_nanoseconds, pdp_db.min() - 5, pdp_db, color=CATEGORICAL[0], lw=2)
    panels[0, 0].plot(delay_nanoseconds, pdp_db, "o", color=CATEGORICAL[0], ms=8)
    prefix_nanoseconds = config.cyclic_prefix_length / config.sample_rate_hz * 1e9
    panels[0, 0].axvline(prefix_nanoseconds, color=NEUTRAL, ls="--", lw=1.5)
    panels[0, 0].text(prefix_nanoseconds, pdp_db.max(), "  CP length", color=NEUTRAL, va="top")
    panels[0, 0].set_xlabel("Delay (ns)")
    panels[0, 0].set_ylabel("Path power (dB)")
    panels[0, 0].set_title("Power delay profile")

    frequency_megahertz = (
        (np.arange(config.n_subcarriers) - config.n_subcarriers // 2) * config.subcarrier_spacing_hz / 1e6
    )
    panels[0, 1].plot(
        frequency_megahertz,
        linear_to_db(np.abs(result.h_eff_true) ** 2),
        color=NEUTRAL,
        lw=2,
        label="True |H|",
    )
    panels[0, 1].plot(
        frequency_megahertz,
        linear_to_db(np.abs(result.h_eff_estimated) ** 2),
        color=CATEGORICAL[2],
        lw=1.5,
        label=f"LS + linear interpolation ({DIAGNOSTIC_SNR_DB:.0f} dB)",
    )
    panels[0, 1].set_xlabel("Baseband frequency (MHz)")
    panels[0, 1].set_ylabel("|H| (dB)")
    panels[0, 1].set_title("Frequency-selective channel and its pilot estimate")
    panels[0, 1].legend(frameon=False)

    before, after = result.symbols_before_equalization, result.symbols_after_equalization
    panels[1, 0].plot(
        before.real, before.imag, ".", color=CATEGORICAL[1], ms=2, alpha=0.4, label="Received Y"
    )
    panels[1, 0].plot(
        after.real, after.imag, ".", color=CATEGORICAL[0], ms=2, alpha=0.5, label="After equalization"
    )
    ideal = constellation(Modulation.QAM16)
    panels[1, 0].plot(ideal.real, ideal.imag, "+", color="black", ms=10, mew=2)
    panels[1, 0].set_xlim(-2, 2)
    panels[1, 0].set_ylim(-2, 2)
    panels[1, 0].set_aspect("equal")
    panels[1, 0].set_xlabel("In-phase")
    panels[1, 0].set_ylabel("Quadrature")
    panels[1, 0].set_title(f"16-QAM at {DIAGNOSTIC_SNR_DB:.0f} dB, LS CSI (BER {result.ber:.2e}, + = ideal)")
    panels[1, 0].legend(frameon=False, markerscale=6, loc="upper right", fontsize=9)

    _cp_panel(panels[1, 1], config, rng, seed)
    for panel in panels.flat[:3]:
        style_axes(panel)
    figure.tight_layout()
    save_figure(figure, out_path)
