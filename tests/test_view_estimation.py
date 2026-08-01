"""Unit tests for camera-view estimation helpers (src/pose/view_estimation.py).

Covers landmark array parsing, visibility/geometry primitives, the clip/mean utilities,
per-frame signal extraction, the score_view decision regimes, and estimate_view_for_pose
run end-to-end over a synthetic pose JSON.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pose.view_estimation import (
    LANDMARK_COUNT,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    body_axis_extent,
    body_height,
    clip01,
    estimate_view_for_pose,
    finite_visibility,
    frame_view_signals,
    landmarks_to_array,
    load_json_list,
    load_pose_json,
    mean_finite,
    parse_split_names,
    score_view,
    signed_orientation,
    visible_point,
    xy_distance,
    z_asymmetry,
)


def _lm(overrides: dict[int, tuple[float, float, float, float]]) -> list[dict]:
    """Build LANDMARK_COUNT landmark dicts; overrides maps index -> (x, y, z, visibility)."""
    points = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(LANDMARK_COUNT)]
    for index, (x, y, z, vis) in overrides.items():
        points[index] = {"x": x, "y": y, "z": z, "visibility": vis}
    return points


def _landmark(x: float, y: float, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _upright_landmarks() -> list[dict]:
    """Standing subject: shoulders high, ankles low, body axis vertical (extent ~0.60)."""
    lm = [_landmark(0.5, 0.5) for _ in range(33)]
    lm[11], lm[12] = _landmark(0.46, 0.20), _landmark(0.54, 0.20)
    lm[23], lm[24] = _landmark(0.47, 0.50), _landmark(0.53, 0.50)
    lm[25], lm[26] = _landmark(0.47, 0.65), _landmark(0.53, 0.65)
    lm[27], lm[28] = _landmark(0.47, 0.80), _landmark(0.53, 0.80)
    return lm


def _horizontal_landmarks() -> list[dict]:
    """The same body rotated 90 degrees (push-up/plank): axis runs along image x."""
    lm = [_landmark(0.5, 0.5) for _ in range(33)]
    lm[11], lm[12] = _landmark(0.20, 0.46), _landmark(0.20, 0.54)
    lm[23], lm[24] = _landmark(0.50, 0.47), _landmark(0.50, 0.53)
    lm[25], lm[26] = _landmark(0.65, 0.47), _landmark(0.65, 0.53)
    lm[27], lm[28] = _landmark(0.80, 0.47), _landmark(0.80, 0.53)
    return lm


def _side_view_landmarks() -> list[dict]:
    """Sideways skeleton: left/right pairs coincide (near-zero torso width) -> narrow body."""
    return _lm(
        {
            0: (0.5, 0.12, 0.0, 1.0),  # nose, for body height span
            LEFT_SHOULDER: (0.5, 0.30, 0.0, 1.0),
            RIGHT_SHOULDER: (0.5, 0.30, 0.0, 1.0),
            LEFT_HIP: (0.5, 0.55, 0.0, 1.0),
            RIGHT_HIP: (0.5, 0.55, 0.0, 1.0),
            25: (0.5, 0.75, 0.0, 1.0),
            26: (0.5, 0.75, 0.0, 1.0),
            27: (0.5, 0.92, 0.0, 1.0),
            28: (0.5, 0.92, 0.0, 1.0),
        }
    )


class ParseSplitNamesTests(unittest.TestCase):
    def test_parses_comma_separated(self):
        self.assertEqual(parse_split_names("train,val"), ["train", "val"])

    def test_strips_blanks(self):
        self.assertEqual(parse_split_names("train, , test"), ["train", "test"])

    def test_rejects_unknown_split(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_split_names("train,bogus")


class LandmarksToArrayTests(unittest.TestCase):
    def test_full_landmarks_become_33x4(self):
        array = landmarks_to_array(_lm({}))
        self.assertEqual(array.shape, (LANDMARK_COUNT, 4))
        self.assertEqual(float(array[0, 3]), 1.0)

    def test_too_few_landmarks_returns_none(self):
        self.assertIsNone(landmarks_to_array([{"x": 0.0, "y": 0.0}] * 10))

    def test_non_list_returns_none(self):
        self.assertIsNone(landmarks_to_array("not a list"))

    def test_non_dict_entry_stays_nan(self):
        landmarks = _lm({})
        landmarks[5] = "garbage"
        array = landmarks_to_array(landmarks)
        self.assertTrue(np.isnan(array[5]).all())


class VisibilityGeometryTests(unittest.TestCase):
    def test_finite_visibility_filters_nan(self):
        points = landmarks_to_array(_lm({1: (0.5, 0.5, 0.0, float("nan"))}))
        result = finite_visibility(points, [0, 1])
        self.assertEqual(result.tolist(), [1.0])

    def test_finite_visibility_none_points(self):
        self.assertEqual(finite_visibility(None, [0, 1]).size, 0)

    def test_visible_point_returns_coords(self):
        points = landmarks_to_array(_lm({3: (0.2, 0.4, 0.1, 0.9)}))
        np.testing.assert_allclose(visible_point(points, 3), [0.2, 0.4, 0.1], atol=1e-6)

    def test_visible_point_below_threshold_is_none(self):
        points = landmarks_to_array(_lm({3: (0.2, 0.4, 0.1, 0.1)}))
        self.assertIsNone(visible_point(points, 3))

    def test_xy_distance_between_visible_points(self):
        points = landmarks_to_array(
            _lm({LEFT_SHOULDER: (0.4, 0.5, 0.0, 1.0), RIGHT_SHOULDER: (0.6, 0.5, 0.0, 1.0)})
        )
        self.assertAlmostEqual(xy_distance(points, LEFT_SHOULDER, RIGHT_SHOULDER), 0.2, places=6)

    def test_xy_distance_missing_point_is_nan(self):
        points = landmarks_to_array(_lm({LEFT_SHOULDER: (0.4, 0.5, 0.0, 0.0)}))
        self.assertTrue(np.isnan(xy_distance(points, LEFT_SHOULDER, RIGHT_SHOULDER)))

    def test_signed_orientation_sign(self):
        points = landmarks_to_array(
            _lm({LEFT_SHOULDER: (0.4, 0.5, 0.0, 1.0), RIGHT_SHOULDER: (0.6, 0.5, 0.0, 1.0)})
        )
        self.assertEqual(signed_orientation(points, LEFT_SHOULDER, RIGHT_SHOULDER), -1.0)

    def test_z_asymmetry_absolute_difference(self):
        points = landmarks_to_array(
            _lm({LEFT_HIP: (0.4, 0.5, 0.1, 1.0), RIGHT_HIP: (0.6, 0.5, 0.3, 1.0)})
        )
        self.assertAlmostEqual(z_asymmetry(points, LEFT_HIP, RIGHT_HIP), 0.2, places=6)

    def test_body_height_span(self):
        points = landmarks_to_array(_side_view_landmarks())
        self.assertAlmostEqual(body_height(points), 0.80, places=2)

    def test_body_height_none_points(self):
        self.assertTrue(np.isnan(body_height(None)))

    def test_body_height_too_few_points_is_nan(self):
        points = np.full((LANDMARK_COUNT, 4), np.nan, dtype=np.float32)
        for index in (0, LEFT_SHOULDER, LEFT_HIP):  # only 3 visible body landmarks
            points[index] = [0.5, 0.5, 0.0, 1.0]
        self.assertTrue(np.isnan(body_height(points)))

    def test_axis_extent_matches_y_extent_for_an_upright_body(self) -> None:
        # For a vertical body axis the projection reduces to the y-extent, so the
        # upright behaviour this classifier was tuned on must be preserved exactly.
        points = landmarks_to_array(_upright_landmarks())
        self.assertAlmostEqual(body_axis_extent(points), 0.60, delta=0.02)

    def test_axis_extent_recovers_body_length_when_horizontal(self) -> None:
        # Same body rotated 90 degrees: the y-extent collapses to the body's thickness,
        # but the axis-relative extent must still recover its full length.
        points = landmarks_to_array(_horizontal_landmarks())
        self.assertGreater(body_axis_extent(points), 0.45)


class ClipAndMeanTests(unittest.TestCase):
    def test_clip01_clamps_range(self):
        self.assertEqual(clip01(1.5), 1.0)
        self.assertEqual(clip01(-0.5), 0.0)
        self.assertEqual(clip01(0.3), 0.3)

    def test_clip01_non_finite_is_zero(self):
        self.assertEqual(clip01(float("nan")), 0.0)

    def test_mean_finite_ignores_nan(self):
        self.assertAlmostEqual(mean_finite([1.0, float("nan"), 3.0]), 2.0)

    def test_mean_finite_uses_default_when_empty(self):
        self.assertEqual(mean_finite([float("nan")], default=-1.0), -1.0)


class FrameViewSignalsTests(unittest.TestCase):
    def test_non_dict_frame_is_none(self):
        self.assertIsNone(frame_view_signals("nope"))

    def test_frame_without_landmarks_is_none(self):
        self.assertIsNone(frame_view_signals({}))

    def test_valid_frame_returns_signal_dict(self):
        signals = frame_view_signals({"landmarks": _side_view_landmarks()})
        self.assertIsNotNone(signals)
        self.assertEqual(
            set(signals.keys()),
            {"orientation_score", "torso_width_ratio", "face_visibility", "z_asymmetry"},
        )


class ScoreViewTests(unittest.TestCase):
    def test_low_valid_ratio_is_unknown(self):
        view_type, confidence, *_ = score_view(0.5, 0.5, 0.2, 0.0, valid_frame_ratio=0.05)
        self.assertEqual(view_type, "unknown")
        self.assertEqual(confidence, 0.0)

    def test_narrow_ambiguous_body_is_side(self):
        view_type, *_ = score_view(0.0, 0.0, 0.05, 0.0, valid_frame_ratio=0.8)
        self.assertEqual(view_type, "side")

    def test_broad_oriented_body_is_front_when_allowed(self):
        view_type, *_ = score_view(1.0, 0.5, 0.30, 0.0, valid_frame_ratio=0.8, allow_front=True)
        self.assertEqual(view_type, "front")

    def test_front_signal_defaults_to_rear_oblique(self):
        # Without allow_front, front-like orientation is reported as rear_oblique by design.
        view_type, *_ = score_view(1.0, 0.5, 0.30, 0.0, valid_frame_ratio=0.8, allow_front=False)
        self.assertEqual(view_type, "rear_oblique")

    def test_absent_width_evidence_does_not_score_as_side(self) -> None:
        # A clip where torso width is unmeasurable in every frame must NOT be called
        # "side": narrow_body_signal previously read the 0.0 default as maximally narrow
        # and returned side @ 0.9 with no width evidence at all.
        view_type, confidence, _front, _rear, side_score, _oblique = score_view(
            orientation_score=0.0,
            face_visibility=0.5,
            torso_width_ratio=float("nan"),
            z_asymmetry_value=0.0,
            valid_frame_ratio=1.0,
        )
        self.assertNotEqual(view_type, "side")
        self.assertLess(side_score, 0.62)

    def test_sagittal_horizontal_body_scores_side_not_oblique(self) -> None:
        # A sagittal push-up with a realistic residual left/right separation was
        # misclassified `rear_oblique` when body extent was measured vertically,
        # because the collapsed denominator inflated torso_width_ratio ~2.3x.
        lm = _horizontal_landmarks()
        for index in (12, 24, 26, 28):  # nudge the far side to a 0.04 residual gap
            lm[index] = _landmark(lm[index]["x"] + 0.04, lm[index]["y"])
        points = landmarks_to_array(lm)
        width = mean_finite(
            [xy_distance(points, 11, 12), xy_distance(points, 23, 24)], default=float("nan")
        )
        ratio = width / body_axis_extent(points)
        view_type, _confidence, _f, _r, _s, _o = score_view(
            orientation_score=0.0,
            face_visibility=0.5,
            torso_width_ratio=ratio,
            z_asymmetry_value=0.0,
            valid_frame_ratio=1.0,
        )
        self.assertEqual(view_type, "side")

    def test_sagittal_upright_body_still_scores_side(self) -> None:
        # Companion to the horizontal case above: for a genuine UPRIGHT sagittal view
        # (the case this classifier was tuned on, and the case most at risk if the
        # axis-relative change regressed anything) the verdict must still be `side`.
        # After Task 2 the 45-file regression corpus contains zero `side` verdicts
        # (its only sample was degenerate and was correctly reclassified), so the
        # corpus alone cannot prove this; this synthetic case fills that gap.
        lm = _upright_landmarks()
        points = landmarks_to_array(lm)
        width = mean_finite(
            [xy_distance(points, 11, 12), xy_distance(points, 23, 24)], default=float("nan")
        )
        ratio = width / body_axis_extent(points)
        view_type, _confidence, _f, _r, _s, _o = score_view(
            orientation_score=0.0,
            face_visibility=0.5,
            torso_width_ratio=ratio,
            z_asymmetry_value=0.0,
            valid_frame_ratio=1.0,
        )
        self.assertEqual(view_type, "side")


class EstimateViewForPoseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, payload: dict) -> Path:
        path = self.tmp / "pose.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_empty_frames_is_unknown(self):
        estimate = estimate_view_for_pose(self._write({"frames": []}))
        self.assertEqual(estimate.view_type, "unknown")
        self.assertEqual(estimate.total_frames, 0)
        self.assertEqual(estimate.valid_frame_count, 0)

    def test_side_view_sequence_is_classified_side(self):
        frames = [{"frame_index": i, "landmarks": _side_view_landmarks()} for i in range(6)]
        estimate = estimate_view_for_pose(self._write({"frames": frames}), video_id="vid42")
        self.assertEqual(estimate.view_type, "side")
        self.assertEqual(estimate.total_frames, 6)
        self.assertEqual(estimate.valid_frame_count, 6)
        self.assertEqual(estimate.video_id, "vid42")
        self.assertTrue(0.0 <= estimate.view_confidence <= 1.0)

    def test_degenerate_all_coincident_landmarks_is_not_side(self) -> None:
        # Mirrors data/runtime/pose_json/vid1.json: every landmark at the same point,
        # so shoulder/hip widths are 0 and body extent is 0 -> torso_width_ratio NaN in
        # every frame. This produced `side` @ conf 0.9 before the fix.
        landmarks = [{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 0.9} for _ in range(33)]
        payload = {"metadata": {}, "frames": [{"frame_index": 0, "landmarks": landmarks}]}
        estimate = estimate_view_for_pose(self._write(payload))
        self.assertNotEqual(estimate.view_type, "side")


class JsonLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, payload) -> Path:
        path = self.tmp / "data.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_load_json_list_coerces_to_str(self):
        self.assertEqual(load_json_list(self._write([1, "two", 3])), ["1", "two", "3"])

    def test_load_json_list_rejects_non_list(self):
        with self.assertRaises(ValueError):
            load_json_list(self._write({"a": 1}))

    def test_load_pose_json_returns_dict(self):
        self.assertEqual(load_pose_json(self._write({"frames": []})), {"frames": []})

    def test_load_pose_json_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            load_pose_json(self._write([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
