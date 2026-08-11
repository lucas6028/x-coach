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

from src.video.squat_dataset import SPLIT_NAMES, SQUAT_LABELED_ROOT, load_json_list
from src.video.variant_framing import (
    FRAMING_VARIANTS,
    SUBSETS,
    Box,
    select_rows,
    split_counts,
    summarize_manifest,
    truncation_cause,
)

#: Stage B's paired bootstrap gave a half-width of about 0.052 on 244 test videos.
#: A bootstrap CI narrows as 1/sqrt(n), so this extrapolates the width any smaller
#: subset can deliver. It is an extrapolation, labelled as one -- not a measurement.
STAGE_B_HALF_WIDTH = 0.052
STAGE_B_TEST_N = 244


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


def format_power(rows: list[dict], split_map: dict[str, str]) -> str:
    lines = [
        "power: a contrast is only as strong as its TEST videos",
        "",
        f"{'subset':<30}{'corpus':>8}{'test':>7}{'extrapolated CI half-width':>29}",
    ]
    for subset in SUBSETS:
        counts = split_counts(rows, split_map, subset)
        test_n = counts.get("test", 0)
        width = STAGE_B_HALF_WIDTH * (STAGE_B_TEST_N / test_n) ** 0.5 if test_n else float("inf")
        lines.append(f"{subset:<30}{sum(counts.values()):>8}{test_n:>7}{width:>28.3f}")

    causes: dict[str, int] = {}
    for row in select_rows(rows, "all"):
        frame_w, frame_h = (int(value) for value in row["frame_size"])
        cause = truncation_cause(frame_w, frame_h, Box(*row["box"]))
        causes[cause] = causes.get(cause, 0) + 1
    truncated = causes.get("scale", 0) + causes.get("framing", 0)
    lines.append("")
    lines.append(
        f"why full_frame truncates ({truncated} videos): "
        f"scale {causes.get('scale', 0)} ({causes.get('scale', 0) / truncated:.1%}), "
        f"framing {causes.get('framing', 0)} ({causes.get('framing', 0) / truncated:.1%})"
    )
    lines.append(
        "  scale  = box taller than the centre-crop window; only zooming out helps, which costs F3"
    )
    lines.append(
        "  framing = box fits but sits off the frame's centre; re-centring costs no zoom at all"
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
    parser.add_argument("--split-dir", type=Path, default=SQUAT_LABELED_ROOT / "Splits")
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

    split_map = {
        video_id: split_name
        for split_name in SPLIT_NAMES
        for video_id in load_json_list(args.split_dir / f"{split_name}_keys.json")
    }
    print(format_power(manifest["rows"], split_map))
    print()

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
