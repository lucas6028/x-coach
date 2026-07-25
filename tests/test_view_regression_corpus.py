import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "view_baseline.json"


def _load_baseline() -> dict:
    """Load the frozen baseline, degrading to {} (all-skip) on any problem.

    A missing fixture is the expected fresh-clone/CI state. A malformed fixture
    (bad JSON, unreadable file -- e.g. a stray merge-conflict marker left in the
    542-line file that Tasks 2/3 will regenerate) must NOT raise at import time:
    that would abort collection for the entire test suite, not just this module.
    Both cases collapse to an empty baseline, which the skipUnless guards below
    turn into a clean skip for these tests only.
    """
    if not FIXTURE.exists():
        return {}
    try:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


BASELINE = _load_baseline()
AVAILABLE = [key for key in BASELINE if (REPO_ROOT / key).exists()]
SKIP_REASON = "view baseline corpus not present (data/ is gitignored, or fixture missing/malformed)"


class ViewRegressionCorpusTests(unittest.TestCase):
    """Pose corpora live under gitignored data/, so this gate is local-only: it
    skips in CI and on fresh clones, and bites on the machine that has the data.
    A verdict move here means production squat gating changed."""

    @unittest.skipUnless(AVAILABLE, SKIP_REASON)
    def test_corpus_coverage_matches_baseline(self) -> None:
        """A partial local corpus must fail loudly, not pass quietly.

        40 of the 45 baseline files live under data/runtime/pose_json, a
        transient upload cache -- partial presence is a realistic dev-machine
        state, not a hypothetical. If some baseline files are missing while
        others are present, AVAILABLE is a nonempty strict subset of BASELINE
        and the other two tests would silently check fewer files than the
        baseline covers, reporting a clean "2 passed" that hides the shrunken
        coverage. This test makes that shortfall an explicit failure.
        """
        missing = sorted(set(BASELINE) - set(AVAILABLE))
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" ... (+{len(missing) - 5} more)"
        self.assertEqual(
            len(AVAILABLE),
            len(BASELINE),
            f"Partial corpus: only {len(AVAILABLE)}/{len(BASELINE)} baseline files "
            f"present locally; missing {len(missing)}: {preview}{suffix}. Restore the "
            "full corpus, or this safety net is silently checking less than it claims.",
        )

    @unittest.skipUnless(AVAILABLE, SKIP_REASON)
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

    @unittest.skipUnless(AVAILABLE, SKIP_REASON)
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
