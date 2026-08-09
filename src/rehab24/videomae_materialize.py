"""Derive LOSO-ready VideoMAE feature dirs from the raw per-clip bundles.

Extraction is the expensive GPU step and stores per-clip stacks for both token
pooling modes. This step is pure numpy over ~52 MB and takes seconds, so every
(token pooling x clip aggregation) combination can be produced and evaluated
without re-running the GPU.

Each output dir stores its repetition vector under ``video_feature``, which is the
key ``videomae_video_classifier.build_samples`` already reads -- so
``loso_cross_validation`` and ``paired_loso`` consume these unmodified, via
``--feature-dir`` / ``--baseline-dir`` / ``--candidate-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT
from src.video.videomae_pooling import (
    CLIP_AGGREGATIONS,
    TOKEN_POOLING_MODES,
    aggregate_clips,
    feature_dir_name,
)

#: Metadata copied verbatim from the raw bundle into every materialized bundle, so
#: the derived dirs stay self-describing for stratified analysis.
CARRIED_KEYS = ("sample_id", "video_id", "exercise_id", "person_id", "camera", "correctness")


def read_provenance(data: np.lib.npyio.NpzFile) -> dict[str, str]:
    return {
        key[len("provenance_") :]: str(data[key])
        for key in data.files
        if key.startswith("provenance_")
    }


def materialize_bundle(
    raw_path: Path,
    output_root: Path,
    split: str,
    token_pooling: str,
    aggregation: str,
) -> Path:
    """Write one aggregated feature bundle for a single (pooling, aggregation) pair."""
    with np.load(raw_path, allow_pickle=False) as data:
        clip_features = data[f"clip_features_{token_pooling}"]
        payload = {key: data[key] for key in CARRIED_KEYS if key in data.files}
        provenance = read_provenance(data)
        clip_starts = data["clip_starts"] if "clip_starts" in data.files else None

    provenance["token_pooling"] = token_pooling
    provenance["clip_aggregation"] = aggregation

    output_path = output_root / split / raw_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        video_feature=aggregate_clips(clip_features, aggregation),
        clip_features=clip_features.astype(np.float32, copy=False),
        **({"clip_starts": clip_starts} if clip_starts is not None else {}),
        **payload,
        **{f"provenance_{key}": np.asarray(value) for key, value in provenance.items()},
    )
    return output_path


def materialize_all(
    raw_dir: Path,
    output_parent: Path,
    token_poolings: tuple[str, ...] = TOKEN_POOLING_MODES,
    aggregations: tuple[str, ...] = CLIP_AGGREGATIONS,
) -> dict[str, int]:
    """Materialize every requested combination; return per-dir written counts."""
    raw_paths = sorted(raw_dir.rglob("*.npz"))
    if not raw_paths:
        raise SystemExit(f"No raw feature bundles found under {raw_dir}. Run videomae_features first.")

    counts: dict[str, int] = {}
    for token_pooling in token_poolings:
        for aggregation in aggregations:
            name = feature_dir_name(token_pooling, aggregation)
            output_root = output_parent / name
            for raw_path in raw_paths:
                materialize_bundle(
                    raw_path=raw_path,
                    output_root=output_root,
                    split=raw_path.parent.name,
                    token_pooling=token_pooling,
                    aggregation=aggregation,
                )
            counts[name] = len(raw_paths)
            print(f"{name:<40} {len(raw_paths)} bundles -> {output_root}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive LOSO-ready VideoMAE feature dirs from raw per-clip bundles.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "videomae_raw")
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument(
        "--token-pooling",
        nargs="+",
        choices=TOKEN_POOLING_MODES,
        default=list(TOKEN_POOLING_MODES),
    )
    parser.add_argument("--aggregation", nargs="+", choices=CLIP_AGGREGATIONS, default=list(CLIP_AGGREGATIONS))
    args = parser.parse_args()

    counts = materialize_all(
        raw_dir=args.raw_dir,
        output_parent=args.output_parent,
        token_poolings=tuple(args.token_pooling),
        aggregations=tuple(args.aggregation),
    )
    print(f"\nMaterialized {len(counts)} feature dirs from {args.raw_dir}")


if __name__ == "__main__":
    main()
