"""Bound how much repeated people or scenes could inflate a subject-blind split.

    .venv\\Scripts\\python.exe scripts/video/find_near_duplicates.py

Fitness-AQA ships no participant mapping, so leave-one-subject-out is impossible and
every number on it is scored subject-blind. That is only a problem if the corpus
actually repeats people or scenes, which is measurable without a mapping: two clips of
the same athlete look alike in a person-only representation, and two clips in the same
gym look alike in a scene-only one. Both already exist as the Stage B control arms.

Run it on BOTH views. ``person_crop`` (scene removed) catches the same athlete filmed
somewhere new, which a scene view cannot see; ``background_only`` (athlete removed)
catches the same room with a different lifter, which a person view cannot see.

The reported ceiling is deliberately generous: it assumes every video with a near-twin
is classified perfectly for free while the rest sit near 0.6, i.e. ``share x 0.4``. The
real effect is smaller, because a twin in the same split leaks nothing and a twin with
the opposite label helps nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.squat_dataset import SQUAT_LABELED_ROOT

DEFAULT_THRESHOLDS = (0.999, 0.99, 0.95)
#: Assumed accuracy a non-leaked video would otherwise get, for the inflation ceiling.
BASELINE_ACCURACY = 0.6


def load_features(feature_root: Path) -> tuple[list[str], list[str], np.ndarray]:
    paths = sorted(feature_root.rglob("*.npz"))
    if not paths:
        raise SystemExit(f"No feature bundles under {feature_root}.")
    ids: list[str] = []
    splits: list[str] = []
    rows: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            rows.append(np.asarray(data["video_feature"], dtype=np.float64))
        ids.append(path.stem)
        splits.append(path.parent.name)
    matrix = np.stack(rows)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return ids, splits, matrix


def load_labels(labels_dir: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for name in ("error_knees_forward.json", "error_knees_inward.json"):
        path = labels_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for key, value in json.load(f).items():
                labels[str(key)] = labels.get(str(key), 0) or int(bool(value))
    return labels


def analyse(name: str, feature_root: Path, labels: dict[str, int], thresholds=DEFAULT_THRESHOLDS) -> dict:
    ids, splits, matrix = load_features(feature_root)
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)
    nearest = similarity.max(axis=1)
    nearest_index = similarity.argmax(axis=1)

    y = np.asarray([labels.get(video_id, 0) for video_id in ids])
    positive = y.mean()
    chance_agreement = positive**2 + (1 - positive) ** 2

    report = {
        "name": name,
        "n_videos": len(ids),
        "percentiles": {f"p{q}": float(np.percentile(nearest, q)) for q in (50, 90, 99, 99.9)},
        "max": float(nearest.max()),
        "chance_label_agreement": float(chance_agreement),
        "thresholds": {},
    }
    for threshold in thresholds:
        mask = nearest >= threshold
        count = int(mask.sum())
        if not count:
            report["thresholds"][str(threshold)] = {"n": 0}
            continue
        cross = sum(1 for i in np.flatnonzero(mask) if splits[i] != splits[nearest_index[i]])
        report["thresholds"][str(threshold)] = {
            "n": count,
            "share": float(mask.mean()),
            "cross_split": cross,
            "label_agreement": float((y[mask] == y[nearest_index[mask]]).mean()),
            "inflation_ceiling": float(mask.mean() * (1.0 - BASELINE_ACCURACY)),
        }

    report["pairs"] = [
        {
            "a": ids[i],
            "b": ids[int(nearest_index[i])],
            "split_a": splits[i],
            "split_b": splits[int(nearest_index[i])],
            "cosine": float(nearest[i]),
            "label_a": int(y[i]),
            "label_b": int(y[int(nearest_index[i])]),
        }
        for i in np.flatnonzero(nearest >= max(thresholds))
        if int(nearest_index[i]) > i
    ]
    return report


def format_report(report: dict) -> str:
    lines = [f"=== {report['name']} ({report['n_videos']} videos) ==="]
    percentiles = "  ".join(f"{k}={v:.4f}" for k, v in report["percentiles"].items())
    lines.append(f"  nearest-neighbour cosine: {percentiles}  max={report['max']:.4f}")
    for threshold, stats in report["thresholds"].items():
        if not stats.get("n"):
            lines.append(f"  cos>={threshold}: none")
            continue
        lines.append(
            f"  cos>={threshold}: {stats['n']:>4} videos ({stats['share']:>5.1%}), "
            f"{stats['cross_split']:>3} cross-split, label agreement {stats['label_agreement']:.3f} "
            f"vs chance {report['chance_label_agreement']:.3f}, inflation ceiling <={stats['inflation_ceiling']:.3f}"
        )
    if report["pairs"]:
        lines.append("  strictest-threshold pairs:")
        for pair in report["pairs"]:
            flag = "  CROSS-SPLIT" if pair["split_a"] != pair["split_b"] else ""
            lines.append(
                f"    {pair['a']} [{pair['split_a']}] ~ {pair['b']} [{pair['split_b']}] "
                f"cos={pair['cosine']:.5f} labels {pair['label_a']}/{pair['label_b']}{flag}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bound near-duplicate leakage in a subject-blind corpus.")
    parser.add_argument(
        "--person-features",
        type=Path,
        default=SQUAT_LABELED_ROOT / "person_crop" / "videomae_mean_pool_fc_norm_mean",
    )
    parser.add_argument(
        "--scene-features",
        type=Path,
        default=SQUAT_LABELED_ROOT / "background_only" / "videomae_mean_pool_fc_norm_mean",
    )
    parser.add_argument("--labels-dir", type=Path, default=SQUAT_LABELED_ROOT / "Labels")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    labels = load_labels(args.labels_dir)
    reports = [
        analyse("person_crop -- the ATHLETE, scene removed", args.person_features, labels),
        analyse("background_only -- the SCENE, athlete removed", args.scene_features, labels),
    ]
    for report in reports:
        print(format_report(report))
        print()

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
