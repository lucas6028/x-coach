from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_MODES = ("combined", "knees_forward", "knees_inward")
THRESHOLD_KINDS = (
    "always_positive",
    "always_negative",
    "fixed_0_5",
    "f1_selected_threshold",
    "selected_threshold",
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - imported at runtime on user machines
    raise SystemExit(
        "VideoMAE classifier training requires `torch`.\n"
        "Install it with something like:\n"
        "  pip install torch\n"
        "Then rerun this command."
    ) from exc


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


def build_labels(
    video_ids: list[str],
    forward_errors: dict[str, object],
    inward_errors: dict[str, object],
    label_mode: str = "combined",
) -> dict[str, int]:
    labels: dict[str, int] = {}
    for video_id in video_ids:
        forward_label = label_from_error_intervals(forward_errors.get(video_id))
        inward_label = label_from_error_intervals(inward_errors.get(video_id))
        if label_mode == "combined":
            labels[video_id] = int(forward_label or inward_label)
        elif label_mode == "knees_forward":
            labels[video_id] = int(forward_label)
        elif label_mode == "knees_inward":
            labels[video_id] = int(inward_label)
        else:
            raise ValueError(f"Unsupported label mode: {label_mode}")
    return labels


@dataclass(frozen=True)
class Sample:
    video_id: str
    feature_path: Path
    label: int


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


def build_samples(feature_dir: Path, video_ids: list[str], labels: dict[str, int]) -> list[Sample]:
    samples: list[Sample] = []
    missing: list[str] = []
    for video_id in video_ids:
        candidates = list(feature_dir.rglob(f"{video_id}.npz"))
        if not candidates:
            missing.append(video_id)
            continue
        samples.append(Sample(video_id=video_id, feature_path=candidates[0], label=labels[video_id]))

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        print(f"Warning: {len(missing)} videos are missing features: {preview}{suffix}")

    return samples


def make_loader(
    samples: list[Sample],
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(FeatureDataset(samples), batch_size=batch_size, shuffle=shuffle, generator=generator)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    if probabilities.size == 0:
        return {
            "threshold": float(threshold),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
            "specificity": 0.0,
            "false_positive_rate": 0.0,
            "balanced_accuracy": 0.0,
            "negative_precision": 0.0,
            "negative_recall": 0.0,
            "negative_f1": 0.0,
            "macro_f1": 0.0,
            "youden_j": 0.0,
        }

    preds = (probabilities >= threshold).astype(np.int32)
    labels = labels.astype(np.int32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = float((preds == labels).mean())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    balanced_accuracy = (recall + specificity) / 2
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = specificity
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    macro_f1 = (f1 + negative_f1) / 2
    youden_j = recall + specificity - 1

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "balanced_accuracy": float(balanced_accuracy),
        "negative_precision": float(negative_precision),
        "negative_recall": float(negative_recall),
        "negative_f1": float(negative_f1),
        "macro_f1": float(macro_f1),
        "youden_j": float(youden_j),
    }


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(labels.numpy())

    if not logits_list:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    return sigmoid(np.concatenate(logits_list)), np.concatenate(labels_list)


def collect_sample_predictions(
    model: nn.Module,
    samples: list[Sample],
    batch_size: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    loader = make_loader(samples, batch_size=batch_size, shuffle=False)
    probabilities, labels = collect_predictions(model, loader, device)
    video_ids = [sample.video_id for sample in samples]
    return video_ids, probabilities, labels


def find_best_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    objective: str,
) -> tuple[float, dict[str, float]]:
    if probabilities.size == 0:
        return 0.5, compute_metrics(probabilities, labels, threshold=0.5)

    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
                probabilities.astype(np.float32),
            ]
        )
    )

    best_threshold = 0.5
    best_metrics = compute_metrics(probabilities, labels, threshold=0.5)
    for threshold in candidate_thresholds.tolist():
        metrics = compute_metrics(probabilities, labels, threshold=float(threshold))
        if metrics[objective] > best_metrics[objective]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def baseline_metrics(labels: np.ndarray, positive: bool) -> dict[str, float]:
    probability = 1.0 if positive else 0.0
    probabilities = np.full(labels.shape, probability, dtype=np.float32)
    threshold = 0.5
    return compute_metrics(probabilities, labels, threshold=threshold)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    probabilities, labels = collect_predictions(model, loader, device)
    return compute_metrics(probabilities, labels, threshold=threshold)


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name} metrics | threshold={metrics['threshold']:.3f} "
        f"accuracy={metrics['accuracy']:.3f} precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} specificity={metrics['specificity']:.3f} "
        f"false_positive_rate={metrics['false_positive_rate']:.3f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f} "
        f"f1={metrics['f1']:.3f} "
        f"tp={metrics['tp']:.0f} fp={metrics['fp']:.0f} "
        f"tn={metrics['tn']:.0f} fn={metrics['fn']:.0f}"
    )


def print_threshold_report(
    split_name: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    selected_threshold: float,
    f1_threshold: float,
) -> None:
    metrics_by_kind = build_threshold_metrics(
        probabilities,
        labels,
        selected_threshold=selected_threshold,
        f1_threshold=f1_threshold,
    )
    print(format_metrics(f"{split_name} always-positive baseline", metrics_by_kind["always_positive"]))
    print(format_metrics(f"{split_name} always-negative baseline", metrics_by_kind["always_negative"]))
    print(format_metrics(f"{split_name} fixed-0.5", metrics_by_kind["fixed_0_5"]))
    print(format_metrics(f"{split_name} f1-selected-threshold", metrics_by_kind["f1_selected_threshold"]))
    selected_metrics = metrics_by_kind["selected_threshold"]
    print(format_metrics(f"{split_name} selected-threshold", selected_metrics))
    if selected_metrics["specificity"] < 0.20:
        print(
            f"Warning: {split_name} selected-threshold specificity is "
            f"{selected_metrics['specificity']:.3f}; the classifier is still mostly predicting positive."
        )


def build_threshold_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    selected_threshold: float,
    f1_threshold: float,
) -> dict[str, dict[str, float]]:
    return {
        "always_positive": baseline_metrics(labels, positive=True),
        "always_negative": baseline_metrics(labels, positive=False),
        "fixed_0_5": compute_metrics(probabilities, labels, threshold=0.5),
        "f1_selected_threshold": compute_metrics(probabilities, labels, threshold=f1_threshold),
        "selected_threshold": compute_metrics(probabilities, labels, threshold=selected_threshold),
    }


def write_predictions_csv(
    path: Path,
    rows: list[tuple[str, list[str], np.ndarray, np.ndarray]],
    selected_threshold: float,
    label_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label_mode",
                "split",
                "video_id",
                "label",
                "probability",
                "fixed_0_5_prediction",
                "selected_threshold_prediction",
                "selected_threshold",
            ],
        )
        writer.writeheader()
        for split_name, video_ids, probabilities, labels in rows:
            for video_id, probability, label in zip(video_ids, probabilities, labels):
                writer.writerow(
                    {
                        "label_mode": label_mode,
                        "split": split_name,
                        "video_id": video_id,
                        "label": int(label),
                        "probability": f"{float(probability):.8f}",
                        "fixed_0_5_prediction": int(float(probability) >= 0.5),
                        "selected_threshold_prediction": int(float(probability) >= selected_threshold),
                        "selected_threshold": f"{selected_threshold:.8f}",
                    }
                )


def write_summary_json(
    path: Path,
    config: dict[str, object],
    class_balance: dict[str, dict[str, int]],
    best_state: dict[str, object],
    split_metrics: dict[str, dict[str, dict[str, float]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "class_balance": class_balance,
        "best_checkpoint": {
            "epoch": best_state["epoch"],
            "threshold": best_state["threshold"],
            "threshold_objective": best_state["threshold_objective"],
            "best_val_metrics": best_state["best_val_metrics"],
        },
        "metrics": split_metrics,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def label_counts(samples: list[Sample]) -> tuple[int, int]:
    positives = sum(sample.label for sample in samples)
    negatives = len(samples) - positives
    return positives, negatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight classifier on cached VideoMAE features.")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "videomae_features" / "train",
        help="Directory containing .npz feature bundles.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "videos",
        help="Root directory containing videos, used only for metadata/debugging.",
    )
    parser.add_argument(
        "--train-keys",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Splits" / "train_keys.json",
    )
    parser.add_argument(
        "--val-keys",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Splits" / "val_keys.json",
    )
    parser.add_argument(
        "--test-keys",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Splits" / "test_keys.json",
    )
    parser.add_argument(
        "--forward-labels",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Labels" / "error_knees_forward.json",
    )
    parser.add_argument(
        "--inward-labels",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Labels" / "error_knees_inward.json",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="combined")
    parser.add_argument("--output-path", type=Path, default=REPO_ROOT / "data" / "Squat" / "videomae_classifier.pt")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    parser.add_argument(
        "--threshold-objective",
        choices=("f1", "balanced_accuracy", "macro_f1", "youden_j"),
        default="balanced_accuracy",
        help="Validation metric used to select the decision threshold and best checkpoint.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids = load_json_list(args.train_keys)
    val_ids = load_json_list(args.val_keys)
    test_ids = load_json_list(args.test_keys)
    labels = build_labels(
        train_ids + val_ids + test_ids,
        load_json_mapping(args.forward_labels),
        load_json_mapping(args.inward_labels),
        label_mode=args.label_mode,
    )

    train_samples = build_samples(args.feature_dir, train_ids, labels)
    val_samples = build_samples(args.feature_dir, val_ids, labels)
    test_samples = build_samples(args.feature_dir, test_ids, labels)

    if not train_samples:
        raise SystemExit("No training feature files were found. Run feature extraction first.")

    with np.load(train_samples[0].feature_path, allow_pickle=False) as data:
        feature_dim = int(data["video_feature"].shape[0])

    model = VideoFeatureClassifier(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = make_loader(train_samples, batch_size=args.batch_size, shuffle=True, generator=train_generator)
    train_eval_loader = make_loader(train_samples, batch_size=args.batch_size, shuffle=False)
    val_loader = make_loader(val_samples, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_samples, batch_size=args.batch_size, shuffle=False)

    train_positives, train_negatives = label_counts(train_samples)
    val_positives, val_negatives = label_counts(val_samples)
    test_positives, test_negatives = label_counts(test_samples)
    pos_weight = torch.tensor([train_negatives / max(train_positives, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_score = -1.0
    epochs_without_improvement = 0

    print(f"Training on {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test samples.")
    print(f"Feature dim: {feature_dim}, seed: {args.seed}")
    print(
        "Class balance | "
        f"train positives={train_positives} negatives={train_negatives} | "
        f"val positives={val_positives} negatives={val_negatives} | "
        f"test positives={test_positives} negatives={test_negatives}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for features, labels_batch in train_loader:
            features = features.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * features.size(0)

        train_metrics = evaluate(model, train_eval_loader, device, threshold=0.5)
        val_probabilities, val_labels = collect_predictions(model, val_loader, device)
        val_threshold, val_metrics = find_best_threshold(
            val_probabilities,
            val_labels,
            objective=args.threshold_objective,
        )
        avg_loss = running_loss / max(len(train_samples), 1)
        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.4f} "
            f"| train_f1={train_metrics['f1']:.3f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.3f} val_f1={val_metrics['f1']:.3f} "
            f"val_specificity={val_metrics['specificity']:.3f} val_threshold={val_threshold:.3f} "
            f"early_stop_wait={epochs_without_improvement}"
        )

        if val_metrics[args.threshold_objective] > best_val_score:
            best_val_score = val_metrics[args.threshold_objective]
            epochs_without_improvement = 0
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "feature_dim": feature_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "threshold": float(val_threshold),
                "threshold_objective": args.threshold_objective,
                "label_mode": args.label_mode,
                "seed": args.seed,
                "epoch": epoch,
                "best_val_metrics": val_metrics,
                "labels_source": {
                    "forward_labels": str(args.forward_labels),
                    "inward_labels": str(args.inward_labels),
                },
            }
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch:02d}; "
                    f"best_{args.threshold_objective}={best_val_score:.3f} "
                    f"at epoch {best_state['epoch'] if best_state else 'unknown'}."
                )
                break

    if best_state is None:
        raise SystemExit("No best checkpoint was selected. Use --epochs greater than 0.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_path)
    print(f"Saved best checkpoint to {args.output_path}")

    model.load_state_dict(best_state["model_state_dict"])
    best_threshold = float(best_state["threshold"])

    train_video_ids, train_probabilities, train_labels = collect_sample_predictions(
        model,
        train_samples,
        args.batch_size,
        device,
    )
    val_video_ids, val_probabilities, val_labels = collect_sample_predictions(
        model,
        val_samples,
        args.batch_size,
        device,
    )
    f1_threshold, _ = find_best_threshold(val_probabilities, val_labels, objective="f1")

    split_metrics: dict[str, dict[str, dict[str, float]]] = {
        "train": build_threshold_metrics(
            train_probabilities,
            train_labels,
            selected_threshold=best_threshold,
            f1_threshold=f1_threshold,
        ),
        "val": build_threshold_metrics(
            val_probabilities,
            val_labels,
            selected_threshold=best_threshold,
            f1_threshold=f1_threshold,
        ),
    }

    print(f"Selected threshold objective: {args.threshold_objective}")
    print(f"Validation F1 comparison threshold: {f1_threshold:.3f}")
    print_threshold_report(
        "Train",
        train_probabilities,
        train_labels,
        selected_threshold=best_threshold,
        f1_threshold=f1_threshold,
    )
    print_threshold_report(
        "Val",
        val_probabilities,
        val_labels,
        selected_threshold=best_threshold,
        f1_threshold=f1_threshold,
    )

    if test_samples:
        test_video_ids, test_probabilities, test_labels = collect_sample_predictions(
            model,
            test_samples,
            args.batch_size,
            device,
        )
        print_threshold_report(
            "Test",
            test_probabilities,
            test_labels,
            selected_threshold=best_threshold,
            f1_threshold=f1_threshold,
        )
        split_metrics["test"] = build_threshold_metrics(
            test_probabilities,
            test_labels,
            selected_threshold=best_threshold,
            f1_threshold=f1_threshold,
        )
    else:
        test_video_ids = []
        test_probabilities = np.asarray([], dtype=np.float32)
        test_labels = np.asarray([], dtype=np.float32)
        print("No test samples were available for evaluation.")
        split_metrics["test"] = build_threshold_metrics(
            test_probabilities,
            test_labels,
            selected_threshold=best_threshold,
            f1_threshold=f1_threshold,
        )

    if args.predictions_output is not None:
        write_predictions_csv(
            args.predictions_output,
            rows=[
                ("train", train_video_ids, train_probabilities, train_labels),
                ("val", val_video_ids, val_probabilities, val_labels),
                ("test", test_video_ids, test_probabilities, test_labels),
            ],
            selected_threshold=best_threshold,
            label_mode=args.label_mode,
        )
        print(f"Saved predictions CSV to {args.predictions_output}")

    if args.summary_output is not None:
        write_summary_json(
            args.summary_output,
            config={
                "label_mode": args.label_mode,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "early_stopping_patience": args.early_stopping_patience,
                "threshold_objective": args.threshold_objective,
                "feature_dir": str(args.feature_dir),
                "train_keys": str(args.train_keys),
                "val_keys": str(args.val_keys),
                "test_keys": str(args.test_keys),
                "forward_labels": str(args.forward_labels),
                "inward_labels": str(args.inward_labels),
                "output_path": str(args.output_path),
                "predictions_output": str(args.predictions_output) if args.predictions_output else None,
            },
            class_balance={
                "train": {"positives": train_positives, "negatives": train_negatives},
                "val": {"positives": val_positives, "negatives": val_negatives},
                "test": {"positives": test_positives, "negatives": test_negatives},
            },
            best_state=best_state,
            split_metrics=split_metrics,
        )
        print(f"Saved summary JSON to {args.summary_output}")


if __name__ == "__main__":
    main()
