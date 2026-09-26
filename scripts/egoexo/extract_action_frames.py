"""Extract the frames of movements' judged actions from the split ``frames_open`` archive.

``data/EgoExo-Fitness/frames_open`` ships as 21 parts (``.aa`` .. ``.au``, ~66 GB) of ONE gzip
stream. While only a prefix is on disk, ``tarfile``'s ``"r|gz"`` (streaming) mode reads exactly
that far and then raises, which is why this script writes as it goes and, by default, treats the
terminal error as the end of the data rather than a failure. Once the whole archive is expected,
pass ``--require-complete``: then a stream error, a part after a gap, or any planned
(action, view) pair short of its window exits non-zero instead of quietly yielding fewer actions.

Deliberately, this carries the frame ranges of EVERY judged action of the requested movements and
writes whatever the stream reaches. Which records are recoverable is then DISCOVERED rather than
predicted from an assumed archive ordering -- the same cost, and it cannot be wrong.

``--movement`` repeats, so one pass over the archive serves several movements (each pass
decompresses the whole stream). With ONE movement the pair directories go straight under
``--out`` as before; with several, each movement gets ``--out/<slug>/`` so the per-movement
runners never see another movement's actions.

    .venv\\Scripts\\python.exe scripts/egoexo/extract_action_frames.py ^
      --movement "Jumping Jacks" --movement "High Knee" --out <dir> --require-complete
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.egoexo.frame_extraction import (  # noqa: E402
    ExtractionPlan,
    build_plan,
    completeness_problems,
    concatenated_parts,
    movement_slug,
    parse_member_path,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--movement", required=True, action="append",
        help='action_name, e.g. "High Knee"; repeat for several movements in one pass',
    )
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/EgoExo-Fitness/processed/manifest.csv",
    )
    ap.add_argument(
        "--frames-root",
        type=Path,
        default=REPO_ROOT / "data/EgoExo-Fitness/frames_open",
    )
    ap.add_argument(
        "--views",
        default="exo_l,exo_m,exo_r",
        help="comma-separated views to keep (default: the three exocentric cameras)",
    )
    ap.add_argument("--report", type=Path, default=None, help="write a JSON report here")
    ap.add_argument(
        "--require-complete", action="store_true",
        help="exit 1 unless the whole archive decoded and every planned pair is complete",
    )
    args = ap.parse_args()

    movements = list(dict.fromkeys(args.movement))
    with open(args.manifest, encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["action_name"] in movements]
    missing = [m for m in movements if not any(r["action_name"] == m for r in rows)]
    if missing:
        raise SystemExit(f"no actions named {missing!r} in {args.manifest}")

    per_movement_dirs = len(movements) > 1
    out_dir_of = {
        r["sample_id"]: (args.out / movement_slug(r["action_name"]) if per_movement_dirs else args.out)
        for r in rows
    }

    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    plan: ExtractionPlan = build_plan(rows, views)
    print(
        f"{len(rows)} judged actions of {movements}, {len(plan.by_record)} records, views={views}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    truncated_at: str | None = None
    parts = sorted(args.frames_root.glob("frames_open.tar.gz.*"))
    stream = concatenated_parts(parts)
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parsed = parse_member_path(member.name)
                if parsed is None:
                    continue
                record, view, frame_index = parsed
                targets = plan.lookup(record, view, frame_index)
                if not targets:
                    continue
                payload = tar.extractfile(member)
                if payload is None:
                    continue
                data = payload.read()
                for sample_id in targets:
                    directory = out_dir_of[sample_id] / f"{sample_id}__{view}"
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / f"{frame_index:06d}.jpg").write_bytes(data)
                    key = f"{sample_id}__{view}"
                    written[key] = written.get(key, 0) + 1
                    if written[key] % 200 == 0:
                        print(f"  {key}: {written[key]}", flush=True)
    except (tarfile.ReadError, EOFError, OSError) as exc:
        truncated_at = f"{type(exc).__name__}: {exc}"
        print(f"stream ended: {truncated_at}", flush=True)

    problems = completeness_problems(plan.expected, written, truncated_at, parts)
    report = {
        "movements": movements,
        "views": list(views),
        "actions_planned": len(rows),
        "pairs_planned": len(plan.expected),
        "pairs_written": len(written),
        "frames_written": sum(written.values()),
        "truncated_at": truncated_at,
        "parts_read": [p.name for p in parts],
        "problems": problems,
        "per_pair": dict(sorted(written.items())),
        "complete_pairs": sorted(
            key for key, count in written.items() if count >= plan.expected[key]
        ),
    }
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_pair", "complete_pairs")},
                     indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.require_complete and problems:
        raise SystemExit(f"--require-complete: {len(problems)} problem(s); see above")


if __name__ == "__main__":
    main()
