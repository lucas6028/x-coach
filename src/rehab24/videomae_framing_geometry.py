"""What each REHAB24-6 framing arm actually shows the model -- before any accuracy.

Plan §6.1 makes this a gate, not a diagnostic: the framing report is produced and read
FIRST, so nobody can look at a delta and then decide which arm was "really" showing the
whole athlete. The arithmetic is exact -- ``variant_framing`` models the processor's
shortest-edge resize plus 224 centre crop on the frame size and the box -- so nothing is
decoded and no model is loaded.

Two REHAB24-6 specifics separate this from the Fitness-AQA report next door:

*There are no no-op videos.* Both cameras are non-square (cam17 1920x1080, cam18
1080x1920), so ``full_frame_letterbox`` transforms every single video. On Fitness-AQA
47.3% of videos were already square and the arm had nothing to restore there; here a
single bit-identical output would mean the transform did not run.

*The two cameras fail differently, and averaging them hides it.* cam17's box is shorter
than the crop window, so it can only lose the athlete left and right -- an area loss.
cam18's is taller, so it loses the ends: the feet, and squat depth is judged at the
feet. Reporting one marginal number over both cameras would dilute exactly the effect
the primary comparison is about, which is why everything here is stratified.

The report is also where the honest cost of the letterbox is written down. Padding
1920x1080 to 1920x1920 restores the cropped-off athlete AND shrinks them inside the
224x224 input; ``full_frame``'s centre crop zooms in on whatever it keeps. The two arms
scoring the same is therefore consistent with a completeness gain cancelling a
resolution loss, not only with "completeness does not matter". That reading has to be
available before the number is, so ``body_area_fraction`` is reported alongside
survival rather than left to a footnote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT, load_manifest
from src.rehab24.videomae_boxes import load_index
from src.video.variant_framing import frame_variant, summarize_manifest
from src.video.variant_geometry import Box

#: The arms this plan measures. ``person_crop`` and ``background_only`` are reported
#: even while only the letterbox is being extracted: their geometry costs nothing to
#: compute and pre-registering it stops a later arm from being scoped to flatter it.
REHAB24_FRAMING_VARIANTS = ("full_frame", "full_frame_letterbox", "person_crop", "background_only")

#: A skeleton one frame longer than its container is a decoder/header off-by-one and is
#: recorded rather than treated as a mismatch; anything larger means the two files
#: describe different footage and no box built from one can be trusted on the other.
FRAME_COUNT_TOLERANCE = 1


def framing_rows(manifest_rows: Sequence[dict[str, str]], box_index: dict) -> list[dict]:
    """One row per SAMPLE, carrying its video's fixed box.

    Sample-weighted rather than video-weighted because the LOSO unit is the repetition:
    a video with 30 repetitions has 30 times the influence on the primary endpoint that
    a video with one has, and a video-weighted percentile would misstate what the
    classifier sees.
    """
    videos = box_index.get("videos", {})
    rows: list[dict] = []
    for row in manifest_rows:
        entry = videos.get(row["video_path"])
        if entry is None:
            raise KeyError(f"{row['sample_id']}: no box index entry for {row['video_path']!r}.")
        rows.append(
            {
                "video_id": row["sample_id"],
                "camera": row["camera"],
                "exercise_id": row["exercise_id"],
                "frame_size": list(entry["frame_size"]),
                "box": list(entry["box"]),
            }
        )
    return rows


def frame_count_findings(box_index: dict, tolerance: int = FRAME_COUNT_TOLERANCE) -> dict:
    """Whether each camera's skeleton and its video describe the same footage."""
    over_tolerance: list[str] = []
    off_by_one = 0
    for path, entry in sorted(box_index.get("videos", {}).items()):
        delta = int(entry["skeleton_frames"]) - int(entry["video_frames"])
        if abs(delta) > tolerance:
            over_tolerance.append(f"{path}: video {entry['video_frames']} vs skeleton {entry['skeleton_frames']}")
        elif delta:
            off_by_one += 1
    return {
        "tolerance": tolerance,
        "n_within_tolerance_but_unequal": off_by_one,
        "over_tolerance": over_tolerance,
    }


def camera_frame_sizes(box_index: dict) -> dict[str, list[list[int]]]:
    sizes: dict[str, set[tuple[int, int]]] = {}
    for entry in box_index.get("videos", {}).values():
        sizes.setdefault(entry["camera"], set()).add(tuple(int(v) for v in entry["frame_size"]))
    return {camera: sorted([list(size) for size in values]) for camera, values in sorted(sizes.items())}


def box_inside_frame(rows: Sequence[dict]) -> list[str]:
    """Boxes that leave the frame -- ``background_only``'s mask must cover its box."""
    escaped = []
    for row in rows:
        width, height = (int(value) for value in row["frame_size"])
        x0, y0, x1, y1 = (int(value) for value in row["box"])
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 <= x0 or y1 <= y0:
            escaped.append(f"{row['video_id']}: box {row['box']} in {width}x{height}")
    return escaped


def build_report(
    manifest_rows: Sequence[dict[str, str]],
    box_index: dict,
    variants: tuple[str, ...] = REHAB24_FRAMING_VARIANTS,
) -> dict:
    rows = framing_rows(manifest_rows, box_index)
    cameras = sorted({row["camera"] for row in rows})

    report = {
        "n_samples": len(rows),
        "n_videos": len(box_index.get("videos", {})),
        "box_source": box_index.get("box_source"),
        "box_margin": box_index.get("margin"),
        "camera_frame_sizes": camera_frame_sizes(box_index),
        "frame_counts": frame_count_findings(box_index),
        "boxes_outside_frame": box_inside_frame(rows),
        "overall": summarize_manifest(list(rows), variants=variants),
        "by_camera": {
            camera: summarize_manifest([row for row in rows if row["camera"] == camera], variants=variants)
            for camera in cameras
        },
    }
    report["checks"] = gate_checks(report, variants)
    report["passed"] = all(report["checks"].values())
    return report


def gate_checks(report: dict, variants: tuple[str, ...]) -> dict[str, bool]:
    """Plan §6.1, expressed so a failure stops the run instead of being interpreted."""
    strata = [report["overall"], *report["by_camera"].values()]

    def arm(stratum: dict, variant: str) -> dict:
        return stratum["arms"][variant]

    checks = {
        "each_camera_has_one_frame_size": all(len(sizes) == 1 for sizes in report["camera_frame_sizes"].values()),
        "skeletons_match_their_videos": not report["frame_counts"]["over_tolerance"],
        "boxes_lie_inside_their_frames": not report["boxes_outside_frame"],
        "every_sample_has_a_box": report["overall"]["n_boxless"] == 0,
    }

    if "full_frame_letterbox" in variants:
        checks["letterbox_truncates_nothing"] = all(
            arm(stratum, "full_frame_letterbox")["n_truncated"] == 0 for stratum in strata
        )
        # Both REHAB24-6 cameras are non-square, so an untransformed video here means
        # the transform silently did not run -- the failure that already cost this
        # study half a control arm.
        checks["letterbox_transforms_every_sample"] = all(
            arm(stratum, "full_frame_letterbox")["n_identical_to_source"] == 0 for stratum in strata
        )
    if "person_crop" in variants:
        checks["person_crop_keeps_the_whole_box"] = all(
            arm(stratum, "person_crop")["n_truncated"] == 0 for stratum in strata
        )
    return checks


def print_report(report: dict) -> None:
    print(f"\n=== REHAB24-6 framing geometry ({report['n_samples']} samples, {report['n_videos']} videos) ===")
    print(f"  box source     : {report['box_source']} (margin {report['box_margin']})")
    print(f"  frame sizes    : {report['camera_frame_sizes']}")
    counts = report["frame_counts"]
    print(
        f"  frame counts   : {counts['n_within_tolerance_but_unequal']} videos off by <={counts['tolerance']} frame, "
        f"{len(counts['over_tolerance'])} beyond tolerance"
    )

    for name, stratum in [("overall", report["overall"]), *report["by_camera"].items()]:
        print(f"\n  --- {name} (n={stratum['n_selected']}) ---")
        print(f"  {'arm':<22} {'body area':>18} {'box survival':>14} {'top loss':>10} {'bot loss':>10} {'trunc':>8}")
        for variant, values in stratum["arms"].items():
            area = values["body_area_fraction"]
            survival = values["box_survival"]
            print(
                f"  {variant:<22} "
                f"{area['p10']:.3f}/{area['median']:.3f}/{area['p90']:.3f}".rjust(19)
                + f"{survival['median']:>14.3f}"
                + f"{values['top_loss']['median']:>10.3f}"
                + f"{values['bottom_loss']['median']:>10.3f}"
                + f"{values['fraction_truncated']:>8.1%}"
            )

    print("\n  gates:")
    for name, ok in report["checks"].items():
        print(f"    {'PASS' if ok else 'FAIL'} {name}")
    print(f"  => {'PASS' if report['passed'] else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="REHAB24-6 framing geometry gate (no decoding, no model).")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--box-index", type=Path, default=DEFAULT_PROCESSED_ROOT / "videomae_boxes.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_ROOT / "videomae_framing_geometry.json")
    parser.add_argument("--variants", nargs="+", default=list(REHAB24_FRAMING_VARIANTS))
    args = parser.parse_args()

    report = build_report(load_manifest(args.manifest), load_index(args.box_index), tuple(args.variants))
    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"\nSaved framing geometry report to {args.output}")

    if not report["passed"]:
        failed = [name for name, ok in report["checks"].items() if not ok]
        raise SystemExit(f"Framing geometry gate FAILED: {', '.join(failed)}. Do not extract features.")


if __name__ == "__main__":
    main()
