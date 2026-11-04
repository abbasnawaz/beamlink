import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from beamlink.config import (
    CHECKPOINT_FILENAME,
    NORM_FILENAME,
    NORM_SCHEMA_VERSION,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    FeatureSet,
    SystemConfig,
    load_config,
)
from beamlink.data import Dataset, Labels, feature_names, load_dataset, n_probe_beams, predictor_features
from beamlink.metrics import top_k_accuracy, top_k_power_loss_db
from beamlink.predictor import PredictorNorm, norm_document

logger = logging.getLogger("beamlink.offline")

REPORTED_TOP_K = (1, 3)


class BeamClassifier(nn.Module):
    def __init__(self, n_features: int, n_beams: int, hidden_units: int, hidden_layers: int) -> None:
        super().__init__()
        widths = [n_features] + [hidden_units] * hidden_layers
        layers: list[nn.Module] = []
        for width_in, width_out in zip(widths[:-1], widths[1:], strict=True):
            layers += [nn.Linear(width_in, width_out), nn.ReLU()]
        layers.append(nn.Linear(widths[-1], n_beams))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True, slots=True)
class TrainedModel:
    key: str
    feature_set: FeatureSet
    n_antennas: int
    n_beams: int
    feature_mean: npt.NDArray[np.float32]
    feature_std: npt.NDArray[np.float32]
    state_dict: dict[str, torch.Tensor]
    best_epoch: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def model_key(feature_set: FeatureSet, n_antennas: int) -> str:
    return f"{feature_set}_n{n_antennas}"


def build_classifier(n_features: int, n_beams: int, config: SystemConfig) -> BeamClassifier:
    return BeamClassifier(n_features, n_beams, config.predictor_hidden_units, config.predictor_hidden_layers)


def predicted_top_k(
    model: BeamClassifier, features: npt.NDArray[np.float32], n_candidates: int
) -> npt.NDArray[np.int64]:
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
    return torch.topk(logits, n_candidates, dim=-1).indices.numpy().astype(np.int64)


def soft_targets(beam_power_db: npt.NDArray[np.float32], temperature_db: float) -> torch.Tensor:
    power = torch.from_numpy(beam_power_db)
    if temperature_db <= 0.0:
        return nn.functional.one_hot(power.argmax(dim=-1), power.shape[-1]).float()
    return torch.softmax(power / temperature_db, dim=-1)


def _train_epoch(
    model: BeamClassifier,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: torch.Tensor,
    config: SystemConfig,
    generator: torch.Generator,
) -> float:
    model.train()
    order = torch.randperm(features.shape[0], generator=generator)
    total_loss = 0.0
    for start in range(0, features.shape[0], config.predictor_batch_size):
        batch = order[start : start + config.predictor_batch_size]
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(features[batch]), targets[batch])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.numel()
    return total_loss / features.shape[0]


def train_classifier(
    features: npt.NDArray[np.float32],
    labels: Labels,
    n_antennas: int,
    config: SystemConfig,
    seed: int,
) -> tuple[BeamClassifier, int]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    beam_power_db = labels.beam_power_db[n_antennas]
    train, validation = labels.split == SPLIT_TRAIN, labels.split == SPLIT_VALIDATION
    model = build_classifier(features.shape[1], beam_power_db.shape[1], config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.predictor_learning_rate)
    train_features = torch.from_numpy(features[train])
    train_targets = soft_targets(beam_power_db[train], config.soft_label_temperature_db)
    best_accuracy, best_epoch, best_state = -1.0, 0, {}
    for epoch in range(1, config.predictor_epochs + 1):
        train_loss = _train_epoch(model, optimizer, train_features, train_targets, config, generator)
        model.eval()
        validation_top = predicted_top_k(model, features[validation], max(REPORTED_TOP_K))
        accuracy = top_k_accuracy(validation_top, labels.best_beam[n_antennas][validation].astype(np.int64))
        logger.info(
            f"n_antennas={n_antennas} epoch={epoch} train_loss={train_loss:.4f} "
            f"validation_top3_accuracy={accuracy:.3f}"
        )
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def log_test_metrics(
    model: BeamClassifier,
    features: npt.NDArray[np.float32],
    labels: Labels,
    feature_set: FeatureSet,
    n_antennas: int,
) -> None:
    test = labels.split == SPLIT_TEST
    beam_power_db = labels.beam_power_db[n_antennas][test]
    best = labels.best_beam[n_antennas][test].astype(np.int64)
    for n_candidates in REPORTED_TOP_K:
        candidates = predicted_top_k(model, features[test], n_candidates)
        loss_db = top_k_power_loss_db(beam_power_db, candidates)
        accuracy = top_k_accuracy(candidates, best)
        logger.info(
            f"test model={model_key(feature_set, n_antennas)} n_candidates={n_candidates} "
            f"accuracy={accuracy:.3f} "
            f"mean_power_loss_db={loss_db.mean():.2f} p95_power_loss_db={np.percentile(loss_db, 95):.2f} "
            f"beams_tested={n_candidates + n_probe_beams(feature_set)}/{beam_power_db.shape[1]}"
        )


def train_all(dataset: Dataset, config: SystemConfig, seed: int) -> list[TrainedModel]:
    trained = []
    train = dataset.labels.split == SPLIT_TRAIN
    for feature_set in FeatureSet:
        raw = predictor_features(dataset, feature_set)
        mean, std = raw[train].mean(axis=0), raw[train].std(axis=0)
        normalized = ((raw - mean) / std).astype(np.float32)
        for n_antennas in config.label_antenna_counts:
            key = model_key(feature_set, n_antennas)
            model, best_epoch = train_classifier(normalized, dataset.labels, n_antennas, config, seed)
            log_test_metrics(model, normalized, dataset.labels, feature_set, n_antennas)
            n_beams = dataset.labels.beam_power_db[n_antennas].shape[1]
            trained.append(
                TrainedModel(key, feature_set, n_antennas, n_beams, mean, std, model.state_dict(), best_epoch)
            )
    return trained


def checkpoint_document(trained: list[TrainedModel], config: SystemConfig) -> dict[str, dict[str, object]]:
    return {
        model.key: {
            "feature_set": str(model.feature_set),
            "n_antennas": model.n_antennas,
            "n_beams": model.n_beams,
            "hidden_units": config.predictor_hidden_units,
            "hidden_layers": config.predictor_hidden_layers,
            "feature_mean": torch.from_numpy(model.feature_mean),
            "feature_std": torch.from_numpy(model.feature_std),
            "best_epoch": model.best_epoch,
            "state_dict": model.state_dict,
        }
        for model in trained
    }


def exported_norm(trained: list[TrainedModel], config: SystemConfig) -> PredictorNorm:
    chosen = next(
        model for model in trained if model.key == model_key(config.predictor_feature_set, config.n_antennas)
    )
    return PredictorNorm(
        schema_version=NORM_SCHEMA_VERSION,
        feature_set=chosen.feature_set,
        feature_names=feature_names(chosen.feature_set),
        feature_mean=tuple(float(value) for value in chosen.feature_mean),
        feature_std=tuple(float(value) for value in chosen.feature_std),
        n_antennas=chosen.n_antennas,
        n_beams=chosen.n_beams,
        reference_features=(),
        reference_logits=(),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    dataset = load_dataset(args.data_dir)
    trained = train_all(dataset, config, args.seed)
    checkpoint_path = args.data_dir / CHECKPOINT_FILENAME
    torch.save(checkpoint_document(trained, config), checkpoint_path)
    norm_path = args.data_dir / NORM_FILENAME
    norm_path.write_text(
        json.dumps(norm_document(exported_norm(trained, config)), indent=2), encoding="utf-8"
    )
    for model in trained:
        logger.info(f"model={model.key} best_epoch={model.best_epoch}")
    logger.info(
        f"wrote {checkpoint_path} size_kb={checkpoint_path.stat().st_size / 1024:.0f} and {norm_path}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
