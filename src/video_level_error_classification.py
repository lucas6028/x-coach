from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - imported at runtime on user machines
    raise SystemExit(
        "Video-level error classification requires `torch`.\n"
        "Install it with something like:\n"
        "  pip install torch\n"
        "Then rerun this command."
    ) from exc


TASKS = ("knees_forward", "knees_inward", "shallow_depth")


def load_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")
    return [str(item) for item in data]


def load_json_mapping(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}.")
    return {str(key): value for key, value in data.items()}


def label_from_error_intervals(intervals: object) -> int:
    if not isinstance(intervals, list):
        return 0
    return int(len(intervals) > 0)


def aggregate_frame_labels_to_video(frame_labels: dict[str, object]) -> dict[str, int]:
    video_labels: dict[str, int] = {}
    for frame_id, value in frame_labels.items():
        video_id = frame_id.rsplit("_", 1)[0]
        label = int(value) if isinstance(value, (bool, int, float)) else 0
        video_labels[video_id] = max(video_labels.get(video_id, 0), label)
    return video_labels


def load_split_video_ids(labeled_root: Path) -> dict[str, list[str]]:
    split_dir = labeled_root / "Splits"
    return {
        "train": load_json_list(split_dir / "train_keys.json"),
        "val": load_json_list(split_dir / "val_keys.json"),
        "test": load_json_list(split_dir / "test_keys.json"),
    }


def load_task_labels(labeled_root: Path) -> dict[str, dict[str, int]]:
    labels_dir = labeled_root / "Labels"
    shallow_dir = labeled_root / "Shallow_Squat_Error_Dataset"

    knees_forward = {
        video_id: label_from_error_intervals(intervals)
        for video_id, intervals in load_json_mapping(labels_dir / "error_knees_forward.json").items()
    }
    knees_inward = {
        video_id: label_from_error_intervals(intervals)
        for video_id, intervals in load_json_mapping(labels_dir / "error_knees_inward.json").items()
    }
    shallow_depth = aggregate_frame_labels_to_video(
        load_json_mapping(shallow_dir / "labels_shallow_depth.json")
    )

    return {
        "knees_forward": knees_forward,
        "knees_inward": knees_inward,
        "shallow_depth": shallow_depth,
    }


def summarize_task_labels(
    split_video_ids: dict[str, list[str]],
    task_labels: dict[str, dict[str, int]],
) -> dict[str, dict[str, dict[str, int]]]:
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for task_name, labels in task_labels.items():
        summary[task_name] = {}
        for split_name, video_ids in split_video_ids.items():
            available = [labels[video_id] for video_id in video_ids if video_id in labels]
            positives = int(sum(available))
            total = len(available)
            summary[task_name][split_name] = {
                "available_videos": total,
                "positives": positives,
                "negatives": total - positives,
            }
    return summary


def index_feature_paths(feature_root: Path) -> dict[str, Path]:
    feature_paths: dict[str, Path] = {}
    for path in feature_root.rglob("*.npz"):
        feature_paths[path.stem] = path
    return feature_paths


@dataclass(frozen=True)
class Sample:
    video_id: str
    feature_path: Path
    label: int


def build_samples(
    video_ids: list[str],
    labels: dict[str, int],
    feature_index: dict[str, Path],
) -> tuple[list[Sample], list[str], list[str]]:
    samples: list[Sample] = []
    missing_labels: list[str] = []
    missing_features: list[str] = []

    for video_id in video_ids:
        if video_id not in labels:
            missing_labels.append(video_id)
            continue
        feature_path = feature_index.get(video_id)
        if feature_path is None:
            missing_features.append(video_id)
            continue
        samples.append(Sample(video_id=video_id, feature_path=feature_path, label=int(labels[video_id])))

    return samples, missing_labels, missing_features


class FeatureDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with np.load(sample.feature_path, allow_pickle=False) as data:
            feature = data["video_feature"].astype(np.float32)
        return torch.from_numpy(feature), torch.tensor(sample.label, dtype=torch.float32)


class VideoFeatureClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def make_loader(samples: list[Sample], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(FeatureDataset(samples), batch_size=batch_size, shuffle=shuffle)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    if probabilities.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "threshold": float(threshold),
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
        }

    preds = (probabilities >= threshold).astype(np.int32)
    labels = labels.astype(np.int32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    accuracy = float((preds == labels).mean())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def find_best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    if probabilities.size == 0:
        return 0.5, compute_binary_metrics(probabilities, labels, threshold=0.5)

    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
                probabilities.astype(np.float32),
            ]
        )
    )

    best_threshold = 0.5
    best_metrics = compute_binary_metrics(probabilities, labels, threshold=0.5)

    for threshold in candidate_thresholds.tolist():
        metrics = compute_binary_metrics(probabilities, labels, threshold=float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def collect_predictions(
    model: nn.Module,
    samples: list[Sample],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray | list[str]]:
    loader = make_loader(samples, batch_size=batch_size, shuffle=False)
    model.eval()

    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    video_ids: list[str] = []

    offset = 0
    with torch.no_grad():
        for features, labels in loader:
            batch_size_now = int(features.shape[0])
            batch_samples = samples[offset : offset + batch_size_now]
            offset += batch_size_now

            logits = model(features.to(device)).detach().cpu().numpy()
            logits_list.append(logits)
            labels_list.append(labels.numpy())
            video_ids.extend(sample.video_id for sample in batch_samples)

    if logits_list:
        logits = np.concatenate(logits_list)
        labels = np.concatenate(labels_list)
        probabilities = sigmoid(logits)
    else:
        logits = np.asarray([], dtype=np.float32)
        labels = np.asarray([], dtype=np.float32)
        probabilities = np.asarray([], dtype=np.float32)

    return {
        "video_ids": video_ids,
        "logits": logits,
        "probabilities": probabilities,
        "labels": labels,
    }


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    dropout: float = 0.2
    weight_decay: float = 1e-2
    seed: int = 42
    device: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_binary_classifier(
    task_name: str,
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    config: TrainConfig,
) -> dict[str, object]:
    if not train_samples:
        raise ValueError(f"No training samples are available for task `{task_name}`.")

    set_seed(config.seed)
    device = resolve_device(config.device)

    with np.load(train_samples[0].feature_path, allow_pickle=False) as data:
        feature_dim = int(data["video_feature"].shape[0])

    model = VideoFeatureClassifier(
        feature_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    train_loader = make_loader(train_samples, batch_size=config.batch_size, shuffle=True)

    positives = sum(sample.label for sample in train_samples)
    negatives = len(train_samples) - positives
    pos_weight_value = negatives / max(positives, 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val_f1 = -math.inf
    best_state: dict[str, object] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(features.shape[0])

        train_predictions = collect_predictions(model, train_samples, config.batch_size, device)
        val_predictions = collect_predictions(model, val_samples, config.batch_size, device)

        train_metrics = compute_binary_metrics(
            train_predictions["probabilities"],
            train_predictions["labels"],
            threshold=0.5,
        )
        val_threshold, val_metrics = find_best_threshold(
            val_predictions["probabilities"],
            val_predictions["labels"],
        )

        epoch_summary = {
            "epoch": float(epoch),
            "loss": running_loss / max(len(train_samples), 1),
            "train_f1": float(train_metrics["f1"]),
            "val_f1": float(val_metrics["f1"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_threshold": float(val_threshold),
        }
        history.append(epoch_summary)

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_dim": feature_dim,
                "hidden_dim": config.hidden_dim,
                "dropout": config.dropout,
                "task_name": task_name,
                "val_threshold": float(val_threshold),
                "history": history.copy(),
            }

    if best_state is None:
        raise RuntimeError(f"Failed to train task `{task_name}`.")

    model.load_state_dict(best_state["model_state_dict"])

    train_predictions = collect_predictions(model, train_samples, config.batch_size, device)
    val_predictions = collect_predictions(model, val_samples, config.batch_size, device)
    test_predictions = collect_predictions(model, test_samples, config.batch_size, device)

    best_threshold = float(best_state["val_threshold"])
    final_train_metrics = compute_binary_metrics(
        train_predictions["probabilities"],
        train_predictions["labels"],
        threshold=best_threshold,
    )
    final_val_metrics = compute_binary_metrics(
        val_predictions["probabilities"],
        val_predictions["labels"],
        threshold=best_threshold,
    )
    final_test_metrics = compute_binary_metrics(
        test_predictions["probabilities"],
        test_predictions["labels"],
        threshold=best_threshold,
    )

    return {
        "task_name": task_name,
        "feature_dim": feature_dim,
        "device": str(device),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "positives": positives,
        "negatives": negatives,
        "pos_weight": float(pos_weight_value),
        "history": history,
        "best_state": best_state,
        "threshold": best_threshold,
        "train_metrics": final_train_metrics,
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "train_predictions": train_predictions,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
    }

