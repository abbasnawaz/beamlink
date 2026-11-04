import argparse
import dataclasses
import importlib
import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime
import torch

from beamlink.config import (
    CHECKPOINT_FILENAME,
    N_REFERENCE_SAMPLES,
    NORM_FILENAME,
    ONNX_INPUT_NAME,
    ONNX_OPSET,
    ONNX_OUTPUT_NAME,
    ONNX_PARITY_RELATIVE_TOLERANCE,
    ONNX_PARITY_SAMPLES,
    PREDICTOR_FILENAME,
    SPLIT_TEST,
    load_config,
)
from beamlink.data import load_dataset, predictor_features
from beamlink.predictor import load_norm, norm_document, normalize_features

training = importlib.import_module("03_train_beam_predictor")
logger = logging.getLogger("beamlink.offline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def load_classifier(checkpoint: dict[str, dict[str, object]], key: str) -> torch.nn.Module:
    entry = checkpoint[key]
    model = training.BeamClassifier(
        len(entry["feature_mean"]),
        int(entry["n_beams"]),
        int(entry["hidden_units"]),
        int(entry["hidden_layers"]),
    )
    model.load_state_dict(entry["state_dict"])
    model.eval()
    return model


def export_classifier(model: torch.nn.Module, example: npt.NDArray[np.float32], path: Path) -> None:
    batch = torch.export.Dim("batch")
    torch.onnx.export(
        model,
        (torch.from_numpy(example),),
        str(path),
        input_names=[ONNX_INPUT_NAME],
        output_names=[ONNX_OUTPUT_NAME],
        opset_version=ONNX_OPSET,
        dynamic_shapes={"features": {0: batch}},
        external_data=False,
        dynamo=True,
    )


def top_k(logits: npt.NDArray[np.float32], n_candidates: int) -> npt.NDArray[np.int64]:
    return np.argsort(-logits, axis=-1, kind="stable")[:, :n_candidates]


def torch_logits(model: torch.nn.Module, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    with torch.no_grad():
        return model(torch.from_numpy(features)).numpy()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    dataset = load_dataset(args.data_dir)
    norm = load_norm(args.data_dir)
    checkpoint = torch.load(args.data_dir / CHECKPOINT_FILENAME, weights_only=True)
    model = load_classifier(checkpoint, training.model_key(norm.feature_set, norm.n_antennas))
    raw_features = predictor_features(dataset, norm.feature_set)
    normalized = normalize_features(raw_features, norm)
    parity_rows = rng.choice(normalized.shape[0], ONNX_PARITY_SAMPLES, replace=False)
    model_path = args.data_dir / PREDICTOR_FILENAME
    export_classifier(model, normalized[parity_rows[:2]], model_path)
    session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (onnx_logits,) = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: normalized[parity_rows]})
    expected_logits = torch_logits(model, normalized[parity_rows])
    relative_difference = float(
        np.max(np.abs(onnx_logits - expected_logits)) / np.max(np.abs(expected_logits))
    )
    same_top_k = bool(
        np.array_equal(
            top_k(onnx_logits, config.predictor_top_k), top_k(expected_logits, config.predictor_top_k)
        )
    )
    test_rows = np.flatnonzero(dataset.labels.split == SPLIT_TEST)[:N_REFERENCE_SAMPLES]
    reference_logits = torch_logits(model, normalized[test_rows])
    updated = dataclasses.replace(
        norm,
        reference_features=tuple(tuple(float(value) for value in row) for row in raw_features[test_rows]),
        reference_logits=tuple(tuple(float(value) for value in row) for row in reference_logits),
    )
    (args.data_dir / NORM_FILENAME).write_text(json.dumps(norm_document(updated), indent=2), encoding="utf-8")
    logger.info(
        f"wrote {model_path} size_kb={model_path.stat().st_size / 1024:.0f} model="
        f"{training.model_key(norm.feature_set, norm.n_antennas)} opset={ONNX_OPSET} "
        f"parity_samples={ONNX_PARITY_SAMPLES} max_relative_difference={relative_difference:.2e} "
        f"identical_top{config.predictor_top_k}={same_top_k}"
    )
    if relative_difference > ONNX_PARITY_RELATIVE_TOLERANCE or not same_top_k:
        logger.error(f"ONNX and Torch disagree (relative tolerance {ONNX_PARITY_RELATIVE_TOLERANCE:.0e})")
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
