"""Capture view-estimation verdicts over every real pose JSON in the repo.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/pose/capture_view_baseline.py

Writes tests/fixtures/view_baseline.json, the frozen corpus the
tests/test_view_regression_corpus.py gate diffs against. Regenerate ONLY when a
verdict change is intentional and reviewed -- this file is the record that a
refactor did not move production squat behavior.
"""
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pose.view_estimation import estimate_view_for_pose  # noqa: E402

CORPUS_ROOTS = (
    Path("data/runtime/pose_json"),
    Path("data/Fitness-AQA/Squat/Labeled_Dataset/pose_json"),
)
FIXTURE_PATH = Path("tests/fixtures/view_baseline.json")


def _round_or_none(value: float, ndigits: int) -> float | None:
    """Round a float, mapping non-finite values (NaN from absent evidence, e.g.
    unmeasurable torso width) to None so the fixture stays strict-JSON valid.

    Bare `NaN`/`Infinity` tokens are accepted by Python's json module on both
    read and write, but they are not valid RFC 8259 JSON. If any stricter tool
    ever touches this fixture, tests/test_view_regression_corpus.py's
    _load_baseline() would catch the resulting JSONDecodeError and silently
    degrade to an empty baseline, which skips all three corpus tests instead
    of failing them -- a safety net that disables itself with no signal.
    """
    return round(value, ndigits) if math.isfinite(value) else None


def capture(repo_root: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for root in CORPUS_ROOTS:
        for path in sorted((repo_root / root).rglob("*.json")):
            estimate = estimate_view_for_pose(path)
            key = path.relative_to(repo_root).as_posix()
            records[key] = {
                "view_type": estimate.view_type,
                "view_confidence": _round_or_none(estimate.view_confidence, 6),
                "side_score": _round_or_none(estimate.side_score, 6),
                "front_score": _round_or_none(estimate.front_score, 6),
                "rear_score": _round_or_none(estimate.rear_score, 6),
                "oblique_score": _round_or_none(estimate.oblique_score, 6),
                "torso_width_ratio_mean": _round_or_none(estimate.torso_width_ratio_mean, 6),
                "orientation_score_mean": _round_or_none(estimate.orientation_score_mean, 6),
                "valid_frame_ratio": _round_or_none(estimate.valid_frame_ratio, 6),
                "total_frames": estimate.total_frames,
            }
    return records


def main() -> None:
    records = capture(REPO_ROOT)
    out = REPO_ROOT / FIXTURE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    counts: dict[str, int] = {}
    for record in records.values():
        counts[record["view_type"]] = counts.get(record["view_type"], 0) + 1
    print(f"{len(records)} files -> {FIXTURE_PATH}")
    for key in sorted(counts):
        print(f"  {key:15s} {counts[key]}")


if __name__ == "__main__":
    main()
