from pathlib import Path

import numpy as np
import pytest

from beamlink.config import N_REFERENCE_SAMPLES, PREDICTOR_FILENAME
from beamlink.exceptions import ShapeError
from beamlink.metrics import adjacent_beam_accuracy, sector_accuracy, top_k_accuracy, top_k_power_loss_db
from beamlink.predictor import load_predictor, predict_logits, predict_top_k

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REFERENCE_RELATIVE_TOLERANCE = 1e-5
TOP_K = 3
BEAMS_PER_SECTOR = 16


def _predictor_available() -> bool:
    return (DATA_DIR / PREDICTOR_FILENAME).is_file()


def test_top_k_accuracy_counts_hits_inside_the_candidate_set() -> None:
    candidates = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0]])
    best = np.array([2, 0, 6])
    assert top_k_accuracy(candidates, best) == pytest.approx(2 / 3)


def test_sector_accuracy_compares_sectors_not_beams() -> None:
    best = np.array([3, 3, 20, 47, 63])
    predicted = np.array([3, 15, 16, 31, 47])
    assert sector_accuracy(predicted, best, n_beams_per_sector=BEAMS_PER_SECTOR) == pytest.approx(3 / 5)


def test_adjacent_beam_accuracy_accepts_only_neighbours_in_the_same_sector() -> None:
    best = np.array([5, 5, 5, 5, 16, 15])
    predicted = np.array([5, 4, 6, 7, 15, 16])
    assert adjacent_beam_accuracy(predicted, best, n_beams_per_sector=BEAMS_PER_SECTOR) == pytest.approx(
        3 / 6
    )


def test_beam_accuracies_are_ordered_exact_adjacent_sector() -> None:
    rng = np.random.default_rng(0)
    best = rng.integers(0, 4 * BEAMS_PER_SECTOR, 500)
    predicted = np.clip(best + rng.integers(-3, 4, 500), 0, 4 * BEAMS_PER_SECTOR - 1)
    exact = top_k_accuracy(predicted[:, np.newaxis], best)
    adjacent = adjacent_beam_accuracy(predicted, best, n_beams_per_sector=BEAMS_PER_SECTOR)
    sector = sector_accuracy(predicted, best, n_beams_per_sector=BEAMS_PER_SECTOR)
    assert exact <= adjacent <= sector
    assert exact < adjacent < sector


def test_top_k_power_loss_is_zero_when_the_best_beam_is_a_candidate() -> None:
    beam_power_db = np.array([[-90.0, -80.0, -85.0, -95.0], [-70.0, -71.0, -90.0, -60.0]], dtype=np.float32)
    candidates = np.array([[1, 3], [0, 1]])
    loss_db = top_k_power_loss_db(beam_power_db, candidates)
    assert loss_db[0] == pytest.approx(0.0)
    assert loss_db[1] == pytest.approx(10.0)


def test_onnx_predictor_matches_stored_torch_references() -> None:
    if not _predictor_available():
        pytest.skip("data/beam_predictor.onnx has not been exported")
    predictor = load_predictor(DATA_DIR)
    features = np.asarray(predictor.norm.reference_features, dtype=np.float32)
    expected = np.asarray(predictor.norm.reference_logits, dtype=np.float32)
    assert features.shape == (N_REFERENCE_SAMPLES, len(predictor.norm.feature_names))
    logits = predict_logits(predictor, features)
    assert logits.shape == (N_REFERENCE_SAMPLES, predictor.norm.n_beams)
    assert np.max(np.abs(logits - expected)) <= REFERENCE_RELATIVE_TOLERANCE * np.max(np.abs(expected))


def test_top_k_beams_are_sorted_by_logit() -> None:
    if not _predictor_available():
        pytest.skip("data/beam_predictor.onnx has not been exported")
    predictor = load_predictor(DATA_DIR)
    features = np.asarray(predictor.norm.reference_features, dtype=np.float32)
    logits = predict_logits(predictor, features)
    top = predict_top_k(predictor, features, n_candidates=TOP_K)
    assert top.shape == (N_REFERENCE_SAMPLES, TOP_K)
    ranked = np.take_along_axis(logits, top, axis=-1)
    assert np.all(np.diff(ranked, axis=-1) <= 0)
    assert np.all(ranked[:, -1] >= np.sort(logits, axis=-1)[:, -TOP_K])


def test_wrong_feature_count_is_rejected() -> None:
    if not _predictor_available():
        pytest.skip("data/beam_predictor.onnx has not been exported")
    predictor = load_predictor(DATA_DIR)
    wrong = np.zeros((2, len(predictor.norm.feature_names) + 1), dtype=np.float32)
    with pytest.raises(ShapeError):
        predict_logits(predictor, wrong)
