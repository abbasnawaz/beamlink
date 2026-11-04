import importlib
import logging
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from evaluate_style import (
    CATEGORICAL,
    METHOD_COLORS,
    METHOD_LABELS,
    NEUTRAL,
    map_panel,
    map_scatter,
    positive,
    save_figure,
    style_axes,
)
from matplotlib.axes import Axes

from beamlink.config import (
    SCENE_IMAGE_FILENAME,
    SPLIT_TEST,
    BeamMethod,
    CsiMode,
    FeatureSet,
    Modulation,
    SimulationMode,
    SystemConfig,
)
from beamlink.data import Dataset, SceneMeta, n_probe_beams, position_paths, predictor_features
from beamlink.link import run_link
from beamlink.linkbudget import db_to_linear, linear_to_db, noise_variance_for_tx_power
from beamlink.metrics import awgn_ber_theory, top_k_accuracy, top_k_power_loss_db
from beamlink.predictor import BeamPredictor, predict_top_k

training = importlib.import_module("03_train_beam_predictor")
logger = logging.getLogger("beamlink.offline")

MODEL_STYLES = {
    FeatureSet.POSITION: ("Model A (position)", "-"),
    FeatureSet.POSITION_WIDE_BEAMS: ("Model B (position + 4 wide beams)", "--"),
}
ANTENNA_COLORS = {8: CATEGORICAL[0], 16: CATEGORICAL[1]}
LOSS_CDF_K = 3
POSITION_BER_SNR_DB = np.arange(-10.0, 22.0, 4.0)
POSITION_BER_SLOTS = 6
SHORT_PREFIX_LENGTH = 2
BER_FLOOR = 1e-5
SPLIT_COLORS = ("#c3c2b7", CATEGORICAL[2], CATEGORICAL[1])
BER_METHODS = (BeamMethod.SWEEP, BeamMethod.AI_TOP_K, BeamMethod.MRT_WIDEBAND)
METHOD_MARKERS = {BeamMethod.SWEEP: "o-", BeamMethod.AI_TOP_K: "s--", BeamMethod.MRT_WIDEBAND: "^:"}
LOSS_SCALE_FLOOR_DB = 1.0


def _checkpoint_top_k(
    entry: dict[str, object], dataset: Dataset, feature_set: FeatureSet, n_candidates: int
) -> npt.NDArray[np.int64]:
    model = training.BeamClassifier(
        len(entry["feature_mean"]),
        int(entry["n_beams"]),
        int(entry["hidden_units"]),
        int(entry["hidden_layers"]),
    )
    model.load_state_dict(entry["state_dict"])
    model.eval()
    mean, std = entry["feature_mean"].numpy(), entry["feature_std"].numpy()
    normalized = ((predictor_features(dataset, feature_set) - mean) / std).astype(np.float32)
    return training.predicted_top_k(model, normalized, n_candidates)


def _accuracy_panels(panels: npt.NDArray[np.object_], dataset: Dataset, checkpoint: dict[str, dict]) -> None:
    test = dataset.labels.split == SPLIT_TEST
    for feature_set, (label, dash) in MODEL_STYLES.items():
        for n_antennas, color in ANTENNA_COLORS.items():
            entry = checkpoint[training.model_key(feature_set, n_antennas)]
            power_db = dataset.labels.beam_power_db[n_antennas][test]
            best = dataset.labels.best_beam[n_antennas][test].astype(np.int64)
            ranking = _checkpoint_top_k(entry, dataset, feature_set, power_db.shape[1])[test]
            k_values = np.arange(1, power_db.shape[1] + 1)
            measured = (k_values + n_probe_beams(feature_set)) / power_db.shape[1]
            accuracy = [top_k_accuracy(ranking[:, :n_candidates], best) for n_candidates in k_values]
            mean_loss = [
                float(top_k_power_loss_db(power_db, ranking[:, :n_candidates]).mean())
                for n_candidates in k_values
            ]
            name = f"{label}, N = {n_antennas}"
            panels[0].plot(measured, accuracy, dash, color=color, marker="o", ms=4, lw=2, label=name)
            panels[1].plot(measured, mean_loss, dash, color=color, marker="o", ms=4, lw=2, label=name)
            loss = np.sort(top_k_power_loss_db(power_db, ranking[:, :LOSS_CDF_K]))
            panels[2].plot(loss, np.arange(1, loss.size + 1) / loss.size, dash, color=color, lw=2, label=name)
    panels[0].set_ylabel("Top-K accuracy (best beam among the K tested)")
    panels[1].set_ylabel("Mean power loss vs exhaustive sweep (dB)")
    for panel in panels[:2]:
        panel.set_xlabel("Beams measured / codebook size (K + wide-beam probes) / B")
        panel.axvline(1.0, color=NEUTRAL, ls=":", lw=1.5)
    panels[2].set_xlabel(f"Power loss with K = {LOSS_CDF_K} (dB)")
    panels[2].set_ylabel("Fraction of test positions")
    panels[2].set_xlim(-0.2, 6.0)


def build_prediction_figure(dataset: Dataset, checkpoint_path: Path, out_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    figure, panels = plt.subplots(1, 3, figsize=(19, 5.4))
    _accuracy_panels(panels, dataset, checkpoint)
    titles = (
        "Accuracy vs measurement overhead",
        "Mean power loss vs overhead",
        "Power-loss CDF, test blocks",
    )
    for panel, title in zip(panels, titles, strict=True):
        panel.set_title(title)
        style_axes(panel)
    panels[2].legend(frameon=False, fontsize=9, loc="lower right")
    figure.tight_layout()
    save_figure(figure, out_path)


def ai_candidates(dataset: Dataset, predictor: BeamPredictor, n_candidates: int) -> npt.NDArray[np.int64]:
    return predict_top_k(
        predictor, predictor_features(dataset, predictor.norm.feature_set), n_candidates=n_candidates
    )


def build_ai_coverage_figure(
    dataset: Dataset,
    meta: SceneMeta,
    config: SystemConfig,
    predictor: BeamPredictor,
    data_dir: Path,
    out_path: Path,
) -> None:
    image = plt.imread(data_dir / SCENE_IMAGE_FILENAME)
    n_antennas = predictor.norm.n_antennas
    power_db = dataset.labels.beam_power_db[n_antennas]
    candidates = ai_candidates(dataset, predictor, config.predictor_top_k)
    loss_db = top_k_power_loss_db(power_db, candidates)
    noise_db = float(
        linear_to_db(
            noise_variance_for_tx_power(
                tx_power_dbm=config.tx_power_dbm,
                bandwidth_hz=config.bandwidth_hz,
                noise_figure_db=config.noise_figure_db,
            )
        )
    )
    ai_snr_db = np.max(np.take_along_axis(power_db, candidates, axis=-1), axis=-1) - noise_db
    figure, panels = plt.subplots(1, 3, figsize=(20, 5.2))
    map_panel(panels[0], meta, image, "Spatial split: grey train, aqua validation, orange test")
    map_scatter(panels[0], dataset, np.array(SPLIT_COLORS)[dataset.labels.split])
    n_candidates = config.predictor_top_k
    maps = (
        (
            panels[1],
            ai_snr_db,
            f"Received SNR with AI top-{n_candidates} at {config.tx_power_dbm:.0f} dBm (dB)",
            "Blues",
        ),
        (panels[2], loss_db, f"AI top-{n_candidates} power loss vs sweep (dB)", "Oranges"),
    )
    for panel, values, title, colormap in maps:
        map_panel(panel, meta, image, title)
        high = max(float(np.percentile(values, 99.9)), LOSS_SCALE_FLOOR_DB)
        low = float(np.percentile(values, 1)) if colormap == "Blues" else 0.0
        figure.colorbar(
            map_scatter(panel, dataset, values, cmap=colormap, vmin=low, vmax=high), ax=panel, shrink=0.8
        )
    test = dataset.labels.split == SPLIT_TEST
    logger.info(
        f"ai_coverage n_candidates={n_candidates} test_mean_loss_db={loss_db[test].mean():.2f} "
        f"test_p95_loss_db={np.percentile(loss_db[test], 95):.2f} all_mean_loss_db={loss_db.mean():.2f}"
    )
    figure.tight_layout()
    save_figure(figure, out_path)


def _median_test_position(dataset: Dataset, n_antennas: int, is_los: bool) -> int:
    mask = (dataset.labels.split == SPLIT_TEST) & (dataset.is_los == is_los)
    candidates = np.flatnonzero(mask)
    strength = dataset.labels.beam_power_db[n_antennas][candidates].max(axis=-1)
    return int(candidates[np.argsort(strength)[candidates.size // 2]])


def _ber_panel(
    panel: Axes,
    dataset: Dataset,
    config: SystemConfig,
    position_idx: int,
    candidates: npt.NDArray[np.int64],
    modulation: Modulation,
    seed: int,
) -> None:
    paths = position_paths(dataset.paths, position_idx)
    for method in BER_METHODS:
        ber = [
            run_link(
                paths,
                config,
                method=method,
                modulation=modulation,
                csi_mode=CsiMode.LS,
                simulation_mode=SimulationMode.FREQUENCY,
                snr_db=float(snr_db),
                seed=seed,
                n_slots=POSITION_BER_SLOTS,
                candidate_beams=candidates if method is BeamMethod.AI_TOP_K else None,
            ).ber
            for snr_db in POSITION_BER_SNR_DB
        ]
        panel.semilogy(
            POSITION_BER_SNR_DB,
            positive(ber),
            METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            lw=2,
            ms=7,
            mfc="none",
            label=METHOD_LABELS[method],
        )
    fine = np.linspace(POSITION_BER_SNR_DB[0], POSITION_BER_SNR_DB[-1], 200)
    panel.semilogy(
        fine,
        awgn_ber_theory(db_to_linear(fine), modulation=modulation),
        color=NEUTRAL,
        lw=1.5,
        ls="--",
        label="AWGN theory (no beamforming gain)",
    )
    panel.set_ylim(BER_FLOOR, 0.6)
    panel.set_xlabel("Per-antenna SNR before beamforming (dB)")
    panel.set_ylabel("Bit error rate")
    style_axes(panel)


def _prefix_panel(panel: Axes, dataset: Dataset, config: SystemConfig, position_idx: int, seed: int) -> None:
    paths = position_paths(dataset.paths, position_idx)
    for prefix_length, color in (
        (config.cyclic_prefix_length, CATEGORICAL[0]),
        (SHORT_PREFIX_LENGTH, CATEGORICAL[1]),
    ):
        ber = [
            run_link(
                paths,
                replace(config, cyclic_prefix_length=prefix_length),
                method=BeamMethod.SWEEP,
                modulation=Modulation.QAM16,
                csi_mode=CsiMode.PERFECT,
                simulation_mode=SimulationMode.TIME,
                snr_db=float(snr_db),
                seed=seed,
                n_slots=POSITION_BER_SLOTS,
            ).ber
            for snr_db in POSITION_BER_SNR_DB
        ]
        panel.semilogy(
            POSITION_BER_SNR_DB,
            positive(ber),
            "o-",
            color=color,
            lw=2,
            ms=5,
            label=f"CP = {prefix_length} samples",
        )
    panel.set_ylim(BER_FLOOR, 0.6)
    panel.set_xlabel("Per-antenna SNR before beamforming (dB)")
    panel.set_ylabel("Bit error rate")
    panel.set_title("16-QAM, time-domain sim, sweep beam")
    panel.legend(frameon=False, fontsize=8)
    style_axes(panel)


def build_position_ber_figure(
    dataset: Dataset, config: SystemConfig, predictor: BeamPredictor, seed: int, out_path: Path
) -> None:
    config = replace(config, n_antennas=predictor.norm.n_antennas)
    candidates = ai_candidates(dataset, predictor, config.predictor_top_k)
    figure, panels = plt.subplots(2, 4, figsize=(22, 9))
    for row, (region, is_los) in enumerate((("LOS", True), ("NLOS", False))):
        position_idx = _median_test_position(dataset, config.n_antennas, is_los)
        location = dataset.positions_m[position_idx]
        for column, modulation in enumerate(Modulation):
            panel = panels[row, column]
            _ber_panel(panel, dataset, config, position_idx, candidates[position_idx], modulation, seed)
            panel.set_title(
                f"{region} #{position_idx} ({location[0]:.0f}, {location[1]:.0f}) m, {modulation.upper()}"
            )
        _prefix_panel(panels[row, 3], dataset, config, position_idx, seed)
    panels[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    figure.tight_layout()
    save_figure(figure, out_path)
