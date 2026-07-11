"""CLI for Fit3D experiment 2 (view-dependence of 2D squat-rule readings).

Run from the repository root:

    python scripts/fit3d/run_view_dependence.py --action squat
    python scripts/fit3d/run_view_dependence.py --action squat --json data/Fit3D/derived/view_dependence_squat.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fit3d import view_dependence as vd  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default="squat", help="Fit3D action name (default: squat)")
    parser.add_argument("--split", default="train", help="dataset split (train has 3D GT)")
    parser.add_argument("--subjects", nargs="*", default=None, help="restrict to these subject ids")
    parser.add_argument("--json", type=Path, default=None, help="write the full result as JSON here")
    parser.add_argument("--csv", type=Path, default=None, help="write per-rep readings as CSV here")
    args = parser.parse_args()

    result = vd.run(action=args.action, split=args.split, subjs=args.subjects)
    print(vd.format_report(result))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")

    if args.csv:
        records, _ = vd.collect_records(args.action, args.split, args.subjects)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            cols = ["subject", "rep_index", "source", "metric", "value"]
            fh.write(",".join(cols) + "\n")
            for rec in records:
                for metric, val in rec.truth.items():
                    fh.write(f"{rec.subject},{rec.rep_index},truth,{metric},{val}\n")
                for cam, mvals in rec.views.items():
                    for metric, val in mvals.items():
                        fh.write(f"{rec.subject},{rec.rep_index},{cam},{metric},{val}\n")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
