from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight REHAB24-6 metadata for Colab experiments.")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "colab_package")
    parser.add_argument("--include-skeleton-features", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(args.processed_root / "manifest.csv", args.output_dir / "manifest.csv")
    copy_if_exists(args.processed_root / "labels" / "correctness.json", args.output_dir / "labels" / "correctness.json")
    for split_name in ("train", "val", "test"):
        copy_if_exists(
            args.processed_root / "splits" / f"{split_name}_keys.json",
            args.output_dir / "splits" / f"{split_name}_keys.json",
        )

    if args.include_skeleton_features:
        source = args.processed_root / "skeleton_features"
        if not source.exists():
            raise FileNotFoundError(source)
        destination = args.output_dir / "skeleton_features"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    readme = args.output_dir / "README.md"
    readme.write_text(
        "# REHAB24-6 Colab Package\n\n"
        "Upload this folder together with the raw `REHAB24-6` dataset folder, or mount Drive so the relative paths in "
        "`manifest.csv` resolve under your chosen data root.\n\n"
        "Typical commands after cloning the repo in Colab:\n\n"
        "```bash\n"
        "python scripts/rehab24/extract_skeleton_features.py --data-root /content/REHAB24-6 --manifest manifest.csv \\\n"
        "  --output-dir skeleton_features\n"
        "python scripts/rehab24/extract_videomae_features.py --data-root /content/REHAB24-6 --manifest manifest.csv \\\n"
        "  --output-dir videomae_features --device cuda\n"
        "python scripts/rehab24/fuse_features.py --manifest manifest.csv --first-feature-dir skeleton_features \\\n"
        "  --second-feature-dir videomae_features --output-dir fused_features\n"
        "python scripts/rehab24/train_correctness_classifier.py --feature-dir fused_features --manifest manifest.csv \\\n"
        "  --train-keys splits/train_keys.json --val-keys splits/val_keys.json --test-keys splits/test_keys.json \\\n"
        "  --labels labels/correctness.json --device cuda\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"Exported REHAB24-6 Colab package to {args.output_dir}")


if __name__ == "__main__":
    main()
