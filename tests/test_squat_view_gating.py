"""View gating for the two squat rules that carried none.

`docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` §3 states the
convention for every rule: "Confidence is scaled down when the required view is unavailable
(the coded squat detector multiplies by ~0.65)". Two squat rules did not follow it:

- `heel_rise` — spec observability "medium on **side / oblique** (heel-vs-toe height needs a
  lateral or oblique view; nearly invisible head-on)", but the rule hardcoded
  `observability="medium"` with an undiscounted confidence, so a head-on clip emitted a
  heel verdict as confidently as a sagittal one.
- `shallow_depth` — the spec enumerates side/front/front_oblique (high) and rear/rear_oblique
  (medium) and says nothing about `unknown`, which `estimate_view_for_pose` returns exactly
  when a clip FAILS its evidence floor (`valid_frame_ratio < 0.15`, or every view score
  < 0.20). The rule's `else` branch resolved that worst case to its BEST branch:
  observability "high" at full confidence. Its two siblings (`knees_inward`,
  `excessive_forward_lean`) already fold `unknown` into their medium/x0.65 branch.

These tests drive the real registry path (`run_detector`), not a re-typed threshold.
"""

from __future__ import annotations

import unittest

from src.pose.movements import registry
from src.pose.movements.base import run_detector
from src.pose.pose_rule_detector import VIEW_UNAVAILABLE_CONFIDENCE_SCALE
from tests.test_pose_rule_detector import frame, landmark

BOTTOM_RANGE = range(6, 14)


def heel_clip(heel_y_at_bottom: float, frame_count: int = 20) -> list[dict]:
    """A clip whose middle frames sink into `bottom` with the heels displaced.

    `heel_y_at_bottom` is an image-y coordinate (y grows DOWNWARD), against an ankle/toe
    line at 0.90. See `test_a_real_heel_rise_never_fires` for why the firing direction is
    the one it is.
    """
    frames = []
    for index in range(frame_count):
        at_bottom = index in BOTTOM_RANGE
        item = frame(hip_y=0.86 if at_bottom else 0.72, knee_y=0.68, frame_index=index)
        if at_bottom:
            item["landmarks"][29] = landmark(0.30, heel_y_at_bottom)
            item["landmarks"][30] = landmark(0.70, heel_y_at_bottom)
        frames.append(item)
    return frames


def shallow_clip(frame_count: int = 14) -> list[dict]:
    """Hips stay well above the knees for the whole clip -- an above-parallel squat."""
    return [frame(hip_y=0.45, knee_y=0.70, frame_index=index) for index in range(frame_count)]


def detect(frames: list[dict], view_type: str, view_confidence: float = 0.8):
    _, detections = run_detector(registry.get_detector("Squat"), frames, 30.0, view_type, view_confidence)
    return detections


def fault(detections, fault_id: str):
    matches = [d for d in detections if d.fault_id == fault_id]
    assert len(matches) == 1, f"expected exactly one {fault_id}, got {len(matches)}"
    return matches[0]


class HeelRiseViewGatingTests(unittest.TestCase):
    """Heel-vs-toe height is a sagittal cue; a head-on camera cannot resolve it."""

    def test_side_view_keeps_full_confidence(self) -> None:
        detection = fault(detect(heel_clip(0.95), "side"), "heel_rise")
        self.assertEqual(detection.observability, "medium")
        self.assertAlmostEqual(detection.confidence, detection.severity, places=6)

    def test_oblique_views_keep_full_confidence(self) -> None:
        """The spec grants the cue to "side / oblique", so both obliques stay undiscounted."""
        for view_type in ("front_oblique", "rear_oblique"):
            with self.subTest(view_type=view_type):
                detection = fault(detect(heel_clip(0.95), view_type), "heel_rise")
                self.assertEqual(detection.observability, "medium")
                self.assertAlmostEqual(detection.confidence, detection.severity, places=6)

    def test_head_on_views_are_discounted(self) -> None:
        for view_type in ("front", "rear"):
            with self.subTest(view_type=view_type):
                detection = fault(detect(heel_clip(0.95), view_type), "heel_rise")
                self.assertEqual(detection.observability, "low")
                # `build_detection` rounds severity and confidence independently to 4 dp, so
                # the two can disagree in the last digit; 1e-4 is that rounding, not slack.
                self.assertAlmostEqual(
                    detection.confidence,
                    detection.severity * VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
                    delta=1e-4,
                )

    def test_unknown_view_is_discounted(self) -> None:
        """`unknown` means view estimation failed its evidence floor -- the clip is poor, so
        this is the LAST case that should earn an undiscounted verdict."""
        detection = fault(detect(heel_clip(0.95), "unknown", view_confidence=0.0), "heel_rise")
        self.assertEqual(detection.observability, "low")
        self.assertAlmostEqual(
            detection.confidence, detection.severity * VIEW_UNAVAILABLE_CONFIDENCE_SCALE, delta=1e-4
        )

    def test_view_changes_confidence_but_never_severity(self) -> None:
        """Severity is a property of the movement, not the camera. Only the confidence in
        having SEEN it depends on the view."""
        severities = {
            view_type: fault(detect(heel_clip(0.95), view_type), "heel_rise").severity
            for view_type in ("side", "rear_oblique", "rear", "unknown")
        }
        self.assertEqual(len(set(round(value, 9) for value in severities.values())), 1, severities)
        self.assertGreater(min(severities.values()), 0.0)

    def test_a_discounted_heel_verdict_sorts_behind_observed_faults(self) -> None:
        """`run_detector` sorts on `(observability == "low", -severity, start_frame)`, so the
        downgrade also demotes the verdict below any fault seen from a view that works --
        which is the point of downgrading it rather than only scaling its number."""
        detections = detect(heel_clip(0.95), "rear")
        ids = [d.fault_id for d in detections]
        self.assertIn("heel_rise", ids)
        self.assertIn("knees_inward", ids)  # severity 1.0, observability "high" on rear
        self.assertLess(ids.index("knees_inward"), ids.index("heel_rise"))

    @unittest.expectedFailure
    def test_a_real_heel_rise_never_fires(self) -> None:
        """KNOWN DEFECT, pinned as an expected failure rather than papered over.

        The spec's own heuristic -- `heel_height_delta = heel_y - toe_y` (image-y), flag when
        `heel_height_delta - baseline > 0.015` -- is INVERTED against its own stated coordinate
        convention (§3: "y increasing downward"). A heel lifting off the floor moves UP the
        image, so `heel_y` DECREASES and the delta goes NEGATIVE; the rule can only fire when
        the heel drops BELOW the toe line, i.e. on a toe rise. `src/pose/geometry.py`'s
        `heel_height_delta` implements the spec faithfully, so code and spec agree -- and are
        both wrong. Fixing it means amending the spec text and flipping one sign, which is
        outside this change's scope; when that lands, this test starts passing and the run goes
        RED on the unexpected success, forcing the marker's removal. (Verified, not assumed:
        with the fixture temporarily set to the firing direction, `pytest` reported
        `FAILED ... test_a_real_heel_rise_never_fires` rather than a quiet XPASS.)
        """
        detections = detect(heel_clip(0.85), "side")  # heels lifted 0.05 above the toe line
        self.assertTrue(any(d.fault_id == "heel_rise" for d in detections))


class ShallowDepthUnknownViewTests(unittest.TestCase):
    def test_side_view_keeps_high_observability_and_full_confidence(self) -> None:
        detection = fault(detect(shallow_clip(), "side"), "shallow_depth")
        self.assertEqual(detection.observability, "high")
        self.assertAlmostEqual(detection.confidence, detection.severity, places=6)

    def test_rear_views_stay_medium_at_full_confidence(self) -> None:
        """Unchanged by this fix: the spec lists rear/rear_oblique as a MEDIUM view for depth
        (the hip crease is occluded), not an unavailable one, so it earns no discount."""
        for view_type in ("rear", "rear_oblique"):
            with self.subTest(view_type=view_type):
                detection = fault(detect(shallow_clip(), view_type), "shallow_depth")
                self.assertEqual(detection.observability, "medium")
                self.assertAlmostEqual(detection.confidence, detection.severity, places=6)

    def test_unknown_view_is_downgraded_and_discounted(self) -> None:
        detection = fault(detect(shallow_clip(), "unknown", view_confidence=0.0), "shallow_depth")
        self.assertEqual(detection.observability, "medium")
        self.assertAlmostEqual(
            detection.confidence, detection.severity * VIEW_UNAVAILABLE_CONFIDENCE_SCALE, delta=1e-4
        )

    def test_unknown_view_matches_how_its_siblings_treat_unknown(self) -> None:
        """`knees_inward` and `excessive_forward_lean` already resolve `unknown` to
        medium/x0.65. Depth is now consistent with them instead of resolving the worst
        available view evidence to its best branch."""
        detections = detect(shallow_clip(), "unknown", view_confidence=0.0)
        depth = fault(detections, "shallow_depth")
        inward = fault(detections, "knees_inward")
        self.assertEqual(depth.observability, inward.observability)
        self.assertAlmostEqual(
            depth.confidence / depth.severity, inward.confidence / inward.severity, places=6
        )

    def test_severity_is_unchanged_by_view(self) -> None:
        severities = {
            view_type: fault(detect(shallow_clip(), view_type), "shallow_depth").severity
            for view_type in ("side", "rear", "unknown")
        }
        self.assertEqual(len(set(round(value, 9) for value in severities.values())), 1, severities)
        self.assertGreater(min(severities.values()), 0.0)


if __name__ == "__main__":
    unittest.main()
