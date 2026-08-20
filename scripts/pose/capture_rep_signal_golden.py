"""Freeze real landmarks and the angle Python computes from them, for the TypeScript port to match.

RS-SP2 recomputes the squat rep signal in the browser, and the backend then TRUSTS the rep windows
that signal produces -- so a divergence between the two implementations would never surface on its
own (spec §2.3, §2.7). tests/fixtures/rep_segmentation_cases.json pins signal->windows; this file
pins landmarks->signal, which is the other half and the one nothing else covers.

Regenerate only when the Python formula deliberately changes:
    .venv\\Scripts\\python.exe scripts/pose/capture_rep_signal_golden.py <pose.json>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pose.pose_rule_detector import raw_frame_metrics  # noqa: E402

OUTPUT = REPO_ROOT / "tests" / "fixtures" / "rep_signal_golden.json"
# Enough frames to cover a full rep's range of angles without bloating the repo.
STRIDE = 3
MAX_FRAMES = 60


def main(source: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    fps = float((payload.get("metadata") or {}).get("fps", 30.0) or 30.0)
    cases = []
    for frame in payload.get("frames", [])[::STRIDE][:MAX_FRAMES]:
        metrics = raw_frame_metrics(frame, fps)
        angle = metrics.get("avg_knee_angle")
        cases.append({
            "landmarks": frame.get("landmarks"),
            # null encodes "no measurable angle" -- JSON has no NaN, and the TS side asserts
            # Number.isNaN for these rather than an equality that would silently pass on undefined.
            "avg_knee_angle": None if angle is None or not math.isfinite(angle) else float(angle),
        })
    OUTPUT.write_text(json.dumps({"source": source.name, "cases": cases}), encoding="utf-8")
    print(f"wrote {len(cases)} cases from {source.name} to {OUTPUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
