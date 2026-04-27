from __future__ import annotations

import argparse
import json
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
) -> dict[str, int]:
    labels: dict[str, int] = {}
    for video_id in video_ids:
        labels[video_id] = int(
            label_from_error_intervals(forward_errors.get(video_id))
            or label_from_error_intervals(inward_errors.get(video_id))
        )
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
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
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


def make_loader(samples: list[Sample], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(FeatureDataset(samples), batch_size=batch_size, shuffle=shuffle)


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.6).astype(np.int32)
    labels = labels.astype(np.int32)
    accuracy = float((preds == labels).mean()) if len(labels) else 0.0

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
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
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return compute_metrics(np.concatenate(logits_list), np.concatenate(labels_list))


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
    parser.add_argument("--output-path", type=Path, default=REPO_ROOT / "data" / "Squat" / "videomae_classifier.pt")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids = load_json_list(args.train_keys)
    val_ids = load_json_list(args.val_keys)
    test_ids = load_json_list(args.test_keys)
    labels = build_labels(train_ids + val_ids + test_ids, load_json_mapping(args.forward_labels), load_json_mapping(args.inward_labels))

    train_samples = build_samples(args.feature_dir, train_ids, labels)
    val_samples = build_samples(args.feature_dir, val_ids, labels)
    test_samples = build_samples(args.feature_dir, test_ids, labels)

    if not train_samples:
        raise SystemExit("No training feature files were found. Run feature extraction first.")

    with np.load(train_samples[0].feature_path, allow_pickle=False) as data:
        feature_dim = int(data["video_feature"].shape[0])

    model = VideoFeatureClassifier(feature_dim=feature_dim, hidden_dim=args.hidden_dim).to(device)

    train_loader = make_loader(train_samples, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_samples, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_samples, batch_size=args.batch_size, shuffle=False)

    positives = sum(sample.label for sample in train_samples)
    negatives = len(train_samples) - positives
    pos_weight = torch.tensor([negatives / max(positives, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_state = None
    best_val_f1 = -1.0

    print(f"Training on {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test samples.")
    print(f"Feature dim: {feature_dim}, positives: {positives}, negatives: {negatives}")

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

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(len(train_samples), 1)
        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.4f} "
            f"| train_f1={train_metrics['f1']:.3f} val_f1={val_metrics['f1']:.3f} val_acc={val_metrics['accuracy']:.3f}"
        )

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_dim": feature_dim,
                "hidden_dim": args.hidden_dim,
                "labels_source": {
                    "forward_labels": str(args.forward_labels),
                    "inward_labels": str(args.inward_labels),
                },
            }

    if best_state is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, args.output_path)
        print(f"Saved best checkpoint to {args.output_path}")

    if test_samples:
        test_metrics = evaluate(model, test_loader, device)
        print(
            f"Test metrics | accuracy={test_metrics['accuracy']:.3f} "
            f"precision={test_metrics['precision']:.3f} recall={test_metrics['recall']:.3f} f1={test_metrics['f1']:.3f}"
        )
    else:
        print("No test samples were available for evaluation.")


if __name__ == "__main__":
    main()
