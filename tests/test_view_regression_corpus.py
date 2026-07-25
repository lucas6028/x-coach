import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "view_baseline.json"


def _load_baseline() -> dict:
    if not FIXTURE.exists():
        return {}
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


BASELINE = _load_baseline()
AVAILABLE = [key for key in BASELINE if (REPO_ROOT / key).exists()]


class ViewRegressionCorpusTests(unittest.TestCase):
    """Pose corpora live under gitignored data/, so this gate is local-only: it
    skips in CI and on fresh clones, and bites on the machine that has the data.
    A verdict move here means production squat gating changed."""

    @unittest.skipUnless(AVAILABLE, "view baseline corpus not present (data/ is gitignored)")
    def test_view_verdicts_match_frozen_baseline(self) -> None:
        from src.pose.view_estimation import estimate_view_for_pose

        drifted = []
        for key in AVAILABLE:
            expected = BASELINE[key]
            actual = estimate_view_for_pose(REPO_ROOT / key)
            if actual.view_type != expected["view_type"]:
                drifted.append(
                    f"{key}: {expected['view_type']} -> {actual.view_type}"
                )
        self.assertEqual(drifted, [], f"view verdicts moved for {len(drifted)} file(s)")

    @unittest.skipUnless(AVAILABLE, "view baseline corpus not present (data/ is gitignored)")
    def test_confidence_does_not_drift_far(self) -> None:
        from src.pose.view_estimation import estimate_view_for_pose

        for key in AVAILABLE:
            expected = BASELINE[key]
            actual = estimate_view_for_pose(REPO_ROOT / key)
            self.assertLess(
                abs(actual.view_confidence - expected["view_confidence"]),
                0.10,
                f"{key}: confidence moved from {expected['view_confidence']} to {actual.view_confidence}",
            )
