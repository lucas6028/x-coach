"""REHAB24-6 CLI over the shared VideoMAE materialize step.

The arithmetic lives in ``src.video.videomae_materialize`` because Fitness-AQA runs
the identical step; only the default paths and the carried metadata columns are
REHAB24-specific.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT
from src.video.videomae_materialize import (  # noqa: F401 - re-exported for callers/tests
    materialize_bundle as _materialize_bundle,
    read_provenance,
)
from src.video.videomae_materialize import materialize_all as _materialize_all
from src.video.videomae_pooling import CLIP_AGGREGATIONS, TOKEN_POOLING_MODES

#: Metadata copied verbatim from the raw bundle into every materialized bundle, so
#: the derived dirs stay self-describing for stratified analysis.
CARRIED_KEYS = ("sample_id", "video_id", "exercise_id", "person_id", "camera", "correctness")


def materialize_bundle(
    raw_path: Path,
    output_root: Path,
    split: str,
    token_pooling: str,
    aggregation: str,
    carried_keys: tuple[str, ...] = CARRIED_KEYS,
) -> Path:
    return _materialize_bundle(
        raw_path=raw_path,
        output_root=output_root,
        split=split,
        token_pooling=token_pooling,
        aggregation=aggregation,
        carried_keys=carried_keys,
    )


def materialize_all(
    raw_dir: Path,
    output_parent: Path,
    token_poolings: tuple[str, ...] = TOKEN_POOLING_MODES,
    aggregations: tuple[str, ...] = CLIP_AGGREGATIONS,
    carried_keys: tuple[str, ...] = CARRIED_KEYS,
) -> dict[str, int]:
    return _materialize_all(
        raw_dir=raw_dir,
        output_parent=output_parent,
        carried_keys=carried_keys,
        token_poolings=token_poolings,
        aggregations=aggregations,
    )


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
