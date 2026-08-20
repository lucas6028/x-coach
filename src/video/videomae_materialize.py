"""Derive classifier-ready VideoMAE feature dirs from raw per-clip bundles.

Extraction is the expensive step and stores per-clip stacks for both token pooling
modes. This step is pure numpy over a few tens of MB, so every (token pooling x clip
aggregation) combination can be produced and evaluated without re-running the model.

Each output dir stores its clip-aggregated vector under ``video_feature``, which is
the key ``videomae_video_classifier.build_samples`` already reads -- so every
downstream driver consumes these unmodified.

Dataset-agnostic on purpose: REHAB24-6 repetitions and Fitness-AQA videos both land
here as ``<raw_dir>/<split>/<sample_id>.npz``, and the per-dataset metadata columns
ride along through ``carried_keys``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.video.videomae_pooling import (
    CLIP_AGGREGATIONS,
    TOKEN_POOLING_MODES,
    aggregate_clips,
    feature_dir_name,
)


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
    carried_keys: tuple[str, ...],
) -> Path:
    """Write one aggregated feature bundle for a single (pooling, aggregation) pair."""
    with np.load(raw_path, allow_pickle=False) as data:
        clip_features = data[f"clip_features_{token_pooling}"]
        payload = {key: data[key] for key in carried_keys if key in data.files}
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
    carried_keys: tuple[str, ...],
    token_poolings: tuple[str, ...] = TOKEN_POOLING_MODES,
    aggregations: tuple[str, ...] = CLIP_AGGREGATIONS,
) -> dict[str, int]:
    """Materialize every requested combination; return per-dir written counts."""
    raw_paths = sorted(raw_dir.rglob("*.npz"))
    if not raw_paths:
        raise SystemExit(f"No raw feature bundles found under {raw_dir}. Run the extractor first.")

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
                    carried_keys=carried_keys,
                )
            counts[name] = len(raw_paths)
            print(f"{name:<40} {len(raw_paths)} bundles -> {output_root}")
    return counts
