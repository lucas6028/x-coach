"""Does MediaPipe complexity (Lite=0 / Full=1 / Heavy=2) change SQUAT rule-detector verdicts?

The shipped "Lite is fine" claim came from a downstream correctness classifier pooled over 6
rehab exercises (notes/rehab24_correctness_experiment_summary.md), NOT the squat fault verdicts.
This measures the thing that actually matters for SP1: for each squat clip, extract pose at each
tier and compare the SET OF DETECTED fault_ids. If Lite and Heavy agree on essentially every
clip, defaulting the client analysis extraction to Lite is defensible.

Run:  .venv\\Scripts\\python.exe scripts/pose/validate_complexity_verdicts.py --limit 40
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pose.pose_rule_detector import detect_pose_rules_from_json  # noqa: E402
from src.pose.process_videos import process_video  # noqa: E402

TIERS = {"lite": 0, "full": 1, "heavy": 2}
SQUAT_VIDEOS = REPO_ROOT / "data" / "Fitness-AQA" / "Squat" / "Labeled_Dataset" / "videos"


def verdict_set(pose_json: Path) -> frozenset[str]:
    result = detect_pose_rules_from_json(pose_json, include_retrieval=False, movement="Squat")
    return frozenset(d["fault_id"] for d in result.get("detections", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path, default=SQUAT_VIDEOS)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "notes" / "mediapipe_complexity_squat_verdicts.md")
    args = ap.parse_args()

    clips = sorted(p for p in args.videos_dir.glob("*.mp4"))[: args.limit]
    if not clips:
        print(f"No .mp4 under {args.videos_dir}", file=sys.stderr)
        return 1

    disagreements = 0
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for clip in clips:
            verdicts = {}
            for tier, cx in TIERS.items():
                out = tmp / f"{clip.stem}_{tier}.json"
                process_video(str(clip), str(out), None, cx)
                verdicts[tier] = verdict_set(out)
            agree = verdicts["lite"] == verdicts["heavy"]
            disagreements += 0 if agree else 1
            rows.append((clip.name, verdicts, agree))

    n = len(clips)
    agree_pct = 100.0 * (n - disagreements) / n
    lines = [
        "# MediaPipe complexity vs squat rule-detector verdicts",
        "",
        f"- clips: {n}",
        f"- Lite==Heavy verdict agreement: {agree_pct:.1f}% ({n - disagreements}/{n})",
        "",
        "| clip | lite | full | heavy | lite==heavy |",
        "|---|---|---|---|---|",
    ]
    for name, v, agree in rows:
        fmt = lambda s: ",".join(sorted(s)) or "—"
        lines.append(f"| {name} | {fmt(v['lite'])} | {fmt(v['full'])} | {fmt(v['heavy'])} | {'yes' if agree else 'NO'} |")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} — Lite==Heavy on {agree_pct:.1f}% of {n} clips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
