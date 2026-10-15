"""Freeze real landmarks and the rep signal Python computes from them, for the TypeScript port to match.

RS-SP2 recomputes each movement's rep signal in the browser, and the backend then TRUSTS the rep
windows that signal produces -- so a divergence between the two implementations would never surface
on its own (spec §2.3, §2.7). tests/fixtures/rep_segmentation_cases.json pins signal->windows; this
file pins landmarks->signal, which is the other half and the one nothing else covers.

Every value comes from the movement's REAL detector -- `compute_raw`, read at `rep_signal` -- not
from a restated formula, so the fixture pins the TS port to whatever the backend actually segments
on. The detector's rep knobs are recorded too: the browser passes none, so a movement whose knobs
leave the defaults must not be ported until the TS side learns to pass them.

Regenerate only when the Python formula deliberately changes, or a movement is ported:
    .venv\\Scripts\\python.exe scripts/pose/capture_rep_signal_golden.py <pose.json>
or, without the original clip, re-derive the values from the landmarks already frozen here:
    .venv\\Scripts\\python.exe scripts/pose/capture_rep_signal_golden.py tests/fixtures/rep_signal_golden.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pose.movements import registry  # noqa: E402

OUTPUT = REPO_ROOT / "tests" / "fixtures" / "rep_signal_golden.json"
# Enough frames to cover a full rep's range of angles without bloating the repo.
STRIDE = 3
MAX_FRAMES = 60
# The movements frontend/src/lib/repSignal.ts ports (its TS_REP_SIGNALS keys), by registry name.
# The TS golden test fails on a ported movement this list does not cover.
PORTED_MOVEMENTS = ("Squat", "Push-up", "Lunge", "Row", "Bicep Curl", "Sit-up")


def load_frames(payload: dict) -> tuple[str | None, float, list[dict]]:
    """`(source name or None, fps, frames)` from either a pose JSON or an existing golden file."""
    if "cases" in payload:
        # Already sampled: keep every frozen frame, and the source it was originally sampled from.
        frames = [
            {"frame_index": index, "landmarks": case.get("landmarks")}
            for index, case in enumerate(payload["cases"])
        ]
        return payload.get("source"), 30.0, frames
    fps = float((payload.get("metadata") or {}).get("fps", 30.0) or 30.0)
    return None, fps, payload.get("frames", [])[::STRIDE][:MAX_FRAMES]


def capture(frames: list[dict], fps: float) -> tuple[dict, list[dict]]:
    """`(movements, cases)` exactly as written to the golden file."""
    movements: dict[str, dict] = {}
    signals: list[dict[str, float | None]] = [{} for _ in frames]
    for name in PORTED_MOVEMENTS:
        detector = registry.get_detector(name)
        movements[name] = {
            "rep_signal": detector.rep_signal,
            "polarity": detector.rep_polarity,
            "rectify": detector.rep_rectify,
            "rep_start": detector.rep_start,
            "min_rep_seconds": detector.min_rep_seconds,
        }
        for row, item in zip(signals, detector.compute_raw(frames, fps)):
            value = item.get(detector.rep_signal)
            # null encodes "no measurable signal" -- JSON has no NaN, and the TS side asserts
            # Number.isNaN for these rather than an equality that would silently pass on undefined.
            row[name] = None if value is None or not math.isfinite(value) else float(value)
    cases = [
        {"landmarks": frame.get("landmarks"), "signals": row} for frame, row in zip(frames, signals)
    ]
    return movements, cases


def main(source: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    source_name, fps, frames = load_frames(payload)
    movements, cases = capture(frames, fps)
    OUTPUT.write_text(
        json.dumps({"source": source_name or source.name, "movements": movements, "cases": cases}),
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases x {len(movements)} movements from {source_name or source.name} to {OUTPUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
