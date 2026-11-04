import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime

from beamlink.config import (
    NORM_FILENAME,
    NORM_SCHEMA_VERSION,
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    PREDICTOR_FILENAME,
    FeatureSet,
)
from beamlink.exceptions import ArtifactError, ShapeError

logger = logging.getLogger("beamlink")


@dataclass(frozen=True, slots=True)
class PredictorNorm:
    schema_version: int
    feature_set: FeatureSet
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    n_antennas: int
    n_beams: int
    reference_features: tuple[tuple[float, ...], ...]
    reference_logits: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class BeamPredictor:
    session: onnxruntime.InferenceSession
    norm: PredictorNorm


def norm_document(norm: PredictorNorm) -> dict[str, object]:
    return dataclasses.asdict(norm)


def norm_from_document(document: dict[str, object], path: Path) -> PredictorNorm:
    expected = {field.name for field in dataclasses.fields(PredictorNorm)}
    if set(document) != expected:
        missing, unknown = sorted(expected - set(document)), sorted(set(document) - expected)
        raise ArtifactError(str(path), f"missing keys {missing}, unknown keys {unknown}")
    if document["schema_version"] != NORM_SCHEMA_VERSION:
        raise ArtifactError(
            str(path), f"schema_version {document['schema_version']}, expected {NORM_SCHEMA_VERSION}"
        )
    norm = PredictorNorm(
        schema_version=int(document["schema_version"]),
        feature_set=FeatureSet(document["feature_set"]),
        feature_names=tuple(str(name) for name in document["feature_names"]),
        feature_mean=tuple(float(value) for value in document["feature_mean"]),
        feature_std=tuple(float(value) for value in document["feature_std"]),
        n_antennas=int(document["n_antennas"]),
        n_beams=int(document["n_beams"]),
        reference_features=tuple(
            tuple(float(value) for value in row) for row in document["reference_features"]
        ),
        reference_logits=tuple(tuple(float(value) for value in row) for row in document["reference_logits"]),
    )
    n_features = len(norm.feature_names)
    if len(norm.feature_mean) != n_features or len(norm.feature_std) != n_features:
        raise ArtifactError(str(path), "feature_mean and feature_std must match feature_names")
    return norm


def load_norm(data_dir: Path) -> PredictorNorm:
    path = data_dir / NORM_FILENAME
    if not path.is_file():
        raise ArtifactError(str(path), "file not found")
    return norm_from_document(json.loads(path.read_text(encoding="utf-8")), path)


def load_predictor(data_dir: Path) -> BeamPredictor:
    model_path = data_dir / PREDICTOR_FILENAME
    if not model_path.is_file():
        raise ArtifactError(str(model_path), "file not found")
    norm = load_norm(data_dir)
    session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = [node.name for node in session.get_inputs()]
    output_names = [node.name for node in session.get_outputs()]
    if input_names != [ONNX_INPUT_NAME] or output_names != [ONNX_OUTPUT_NAME]:
        raise ArtifactError(
            str(model_path), f"expected input {ONNX_INPUT_NAME!r} and output {ONNX_OUTPUT_NAME!r}"
        )
    size_kb = model_path.stat().st_size / 1024
    logger.info(
        f"loaded {model_path} feature_set={norm.feature_set} n_antennas={norm.n_antennas} "
        f"size_kb={size_kb:.0f}"
    )
    return BeamPredictor(session=session, norm=norm)


def normalize_features(features: npt.NDArray[np.float32], norm: PredictorNorm) -> npt.NDArray[np.float32]:
    mean = np.asarray(norm.feature_mean, dtype=np.float32)
    std = np.asarray(norm.feature_std, dtype=np.float32)
    return ((features - mean) / std).astype(np.float32)


def predict_logits(predictor: BeamPredictor, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    n_features = len(predictor.norm.feature_names)
    if features.ndim != 2 or features.shape[1] != n_features:
        raise ShapeError("features", (-1, n_features), features.shape)
    normalized = normalize_features(features.astype(np.float32), predictor.norm)
    (logits,) = predictor.session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: normalized})
    return np.asarray(logits, dtype=np.float32)


def predict_top_k(
    predictor: BeamPredictor, features: npt.NDArray[np.float32], *, n_candidates: int
) -> npt.NDArray[np.int64]:
    logits = predict_logits(predictor, features)
    return np.argsort(-logits, axis=-1, kind="stable")[:, :n_candidates].astype(np.int64)
