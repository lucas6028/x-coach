"""Build the Stage B shortcut-control videos (person-crop and background-only).

Reads the MediaPipe pose JSON produced by ``scripts/pose/run_pose_extraction.py``,
derives one box per video, and writes a parallel video tree per variant plus a
manifest recording every box and every fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.squat_dataset import SPLIT_NAMES, SQUAT_LABELED_ROOT, load_json_list
from src.video.squat_video_variants import VARIANTS, build_variant_video, verify_variant_video


def find_video_path(video_root: Path, video_id: str) -> Path | None:
    direct = video_root / f"{video_id}.mp4"
    if direct.exists():
        return direct
    matches = sorted(video_root.rglob(f"{video_id}.mp4"))
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build person-crop / background-only squat videos.")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--video-root", type=Path, default=SQUAT_LABELED_ROOT / "videos")
    parser.add_argument("--pose-json-dir", type=Path, default=SQUAT_LABELED_ROOT / "pose_json")
    parser.add_argument("--split-dir", type=Path, default=SQUAT_LABELED_ROOT / "Splits")
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to videos_<variant>/.")
    parser.add_argument("--manifest-output", type=Path, default=None, help="Defaults to <output-root>/manifest.json.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root or SQUAT_LABELED_ROOT / f"videos_{args.variant}"
    manifest_output = args.manifest_output or output_root / "manifest.json"

    work: list[tuple[str, str, Path, Path, Path]] = []
    missing: list[str] = []
    for split_name in args.splits:
        for video_id in load_json_list(args.split_dir / f"{split_name}_keys.json"):
            video_path = find_video_path(args.video_root, video_id)
            pose_path = args.pose_json_dir / split_name / f"{video_id}.json"
            if video_path is None or not pose_path.exists():
                missing.append(f"{split_name}/{video_id}")
                continue
            work.append((video_id, split_name, video_path, pose_path, output_root / f"{video_id}.mp4"))

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise SystemExit(
            f"{len(missing)} videos have no source video or no pose JSON: {preview}{suffix}\n"
            "Refusing to build a partial control set -- a variant arm that silently covers "
            "fewer videos than the main arm is not a paired comparison."
        )

    if args.limit is not None:
        work = work[: args.limit]

    def build(item: tuple[str, str, Path, Path, Path]) -> dict:
        video_id, split_name, video_path, pose_path, output_path = item
        if output_path.exists() and not args.overwrite:
            return {"video_id": video_id, "split": split_name, "variant": args.variant, "skipped": True}
        row = build_variant_video(
            video_path=video_path,
            pose_path=pose_path,
            output_path=output_path,
            variant=args.variant,
        )
        row["split"] = split_name
        return row

    print(f"Building {len(work)} {args.variant} videos under {output_root} with {args.jobs} job(s).")
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        for index, row in enumerate(pool.map(build, work), start=1):
            rows.append(row)
            if index % 100 == 0:
                print(f"  {index}/{len(work)} done")

    print("Verifying every output against its source frame count...")
    corrupt: list[tuple[str, int]] = []
    for video_id, split_name, _, pose_path, output_path in work:
        with pose_path.open("r", encoding="utf-8") as f:
            expected = int(json.load(f)["metadata"]["total_frames"])
        found = verify_variant_video(output_path, expected)
        if found is not None:
            corrupt.append((video_id, found))
            output_path.unlink(missing_ok=True)

    fallbacks = [row["video_id"] for row in rows if row.get("pose_detected") is False]
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_output.open("w", encoding="utf-8") as f:
        json.dump({"variant": args.variant, "n_videos": len(rows), "fallbacks": fallbacks, "rows": rows}, f, indent=2)

    print(f"Wrote {len(rows)} videos and {manifest_output}")
    if corrupt:
        preview = ", ".join(f"{video_id}({frames} frames)" for video_id, frames in corrupt[:10])
        raise SystemExit(
            f"{len(corrupt)} outputs did not match their source frame count and were deleted: "
            f"{preview}\n"
            "Re-run this command to rebuild them -- a variant off the source's frame "
            "grid samples different clips and silently breaks the pairing."
        )
    if fallbacks:
        print(
            f"WARNING: {len(fallbacks)} videos had no visible pose and were copied unmodified: "
            f"{', '.join(fallbacks[:10])}"
        )


if __name__ == "__main__":
    main()
