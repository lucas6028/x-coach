"""Extract the frames of a movement's judged actions from the split ``frames_open`` archive.

``data/EgoExo-Fitness/frames_open`` ships as 3 GiB parts of ONE gzip stream. Part ``.ac`` is
missing on this machine, so ``.aa``+``.ab`` is a contiguous *prefix* that decodes until it runs
out. Python's ``tarfile`` in ``"r|gz"`` (streaming) mode reads exactly that far and then raises,
which is why this script writes as it goes and treats the terminal error as the end of the data
rather than a failure.

Deliberately, this carries the frame ranges of EVERY judged action of the requested movement and
writes whatever the stream reaches. Which records are recoverable is then DISCOVERED rather than
predicted from an assumed archive ordering -- the same cost, and it cannot be wrong.

    .venv\\Scripts\\python.exe scripts/egoexo/extract_action_frames.py ^
      --movement "High Knee" --out <dir>
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
    concatenated_parts,
    parse_member_path,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--movement", required=True, help='action_name, e.g. "High Knee"')
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
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["action_name"] == args.movement]
    if not rows:
        raise SystemExit(f"no actions named {args.movement!r} in {args.manifest}")

    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    plan: ExtractionPlan = build_plan(rows, views)
    print(f"{len(rows)} judged actions, {len(plan.by_record)} records, views={views}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    truncated_at: str | None = None
    stream = concatenated_parts(sorted(args.frames_root.glob("frames_open.tar.gz.*")))
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
                    directory = args.out / f"{sample_id}__{view}"
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / f"{frame_index:06d}.jpg").write_bytes(data)
                    key = f"{sample_id}__{view}"
                    written[key] = written.get(key, 0) + 1
                    if written[key] % 200 == 0:
                        print(f"  {key}: {written[key]}", flush=True)
    except (tarfile.ReadError, EOFError, OSError) as exc:
        truncated_at = f"{type(exc).__name__}: {exc}"
        print(f"stream ended: {truncated_at}", flush=True)

    report = {
        "movement": args.movement,
        "views": list(views),
        "actions_planned": len(rows),
        "pairs_written": len(written),
        "frames_written": sum(written.values()),
        "truncated_at": truncated_at,
        "per_pair": dict(sorted(written.items())),
        "complete_pairs": sorted(
            key for key, count in written.items() if count >= plan.expected[key]
        ),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "per_pair"}, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
