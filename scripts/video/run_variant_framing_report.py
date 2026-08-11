"""Report F3 and centre-crop survival per B1 arm, from a person_crop manifest.

    .venv\\Scripts\\python.exe scripts/video/run_variant_framing_report.py

Geometry only -- no video decoding, no model. See ``src/video/variant_framing.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.squat_dataset import SQUAT_LABELED_ROOT
from src.video.variant_framing import FRAMING_VARIANTS, SUBSETS, summarize_manifest


def format_report(summary: dict) -> str:
    lines = [
        f"subset {summary['subset']}: {summary['n_selected']} of {summary['n_rows']} videos "
        f"({summary['n_boxless']} with no visible person, excluded)",
        "",
        f"{'arm':<24}{'body area of 224^2':>22}{'box kept':>10}"
        f"{'top lost':>10}{'foot lost':>11}{'truncated':>11}{'unchanged':>11}",
    ]
    for variant, arm in summary["arms"].items():
        area = arm["body_area_fraction"]
        lines.append(
            f"{variant:<24}"
            + f"{area['p10']:.3f} / {area['median']:.3f} / {area['p90']:.3f}".rjust(22)
            + f"{arm['box_survival']['median']:.3f}".rjust(10)
            + f"{arm['top_loss']['p90']:.1%}".rjust(10)
            + f"{arm['bottom_loss']['p90']:.1%}".rjust(11)
            + f"{arm['fraction_truncated']:.1%}".rjust(11)
            + f"{arm['n_identical_to_source']:>11}"
        )
    lines.append("")
    lines.append(
        "body area p10/median/p90; box kept median; top and foot loss are the p90 "
        "fraction of the athlete's HEIGHT cut off that end."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-arm framing report for the B1 matrix.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SQUAT_LABELED_ROOT / "videos_person_crop" / "manifest.json",
        help="A person_crop manifest: both crop arms take its expanded box, and tracking "
        "one box through every arm is what makes the arms comparable.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("variant") != "person_crop":
        raise SystemExit(
            f"{args.manifest} is a {manifest.get('variant')!r} manifest. Only the person_crop "
            "manifest carries the expanded box the crop arms actually use."
        )

    summaries = {subset: summarize_manifest(manifest["rows"], FRAMING_VARIANTS, subset) for subset in SUBSETS}
    for subset in SUBSETS:
        print(format_report(summaries[subset]))
        print()

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
