"""Unit tests for the pure-logic NLF worker package (src/nlf_worker/). No torch, PyAV,
pyrender, boto3 or supabase import anywhere here -- see the interfaces/loop tests, which use
plain fakes instead of the real runtime adapters (a later task).
"""
from __future__ import annotations

import gzip
import json
import math
import os
import pickle
import sys
import tempfile
import types
import dataclasses
import unittest
import weakref

import numpy as np

from src.nlf_worker import config, geometry, keyframes, loop, result, smpl_faces, timeline
from src.nlf_worker.geometry import (
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    NECK,
    PELVIS,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
)


def _pt24(**overrides: tuple[float, float, float]) -> np.ndarray:
    """A (24, 3) joint frame with only the given indices set; the rest NaN (unused)."""
    pts = np.full((24, 3), np.nan)
    for index, xyz in overrides.items():
        pts[int(index)] = xyz
    return pts


def _leg_frame_direct(pelvis, hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r) -> np.ndarray:
    """A realistic-shaped (24, 3) squat frame built from explicit joint positions (camera y
    pointing down)."""
    pts = np.full((24, 3), np.nan)
    pts[PELVIS] = pelvis
    pts[L_HIP] = hip_l
    pts[R_HIP] = hip_r
    pts[L_KNEE] = knee_l
    pts[R_KNEE] = knee_r
    pts[L_ANKLE] = ankle_l
    pts[R_ANKLE] = ankle_r
    pts[NECK] = pelvis + np.array([0.0, -600.0, 0.0])
    pts[L_SHOULDER] = pelvis + np.array([-150.0, -800.0, 0.0])
    pts[R_SHOULDER] = pelvis + np.array([150.0, -800.0, 0.0])
    return pts


# ---------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------
class KneeAngleTests(unittest.TestCase):
    def test_straight_leg_is_about_180(self):
        pts = _pt24(
            **{
                str(L_HIP): (0, 1, 0), str(L_KNEE): (0, 0, 0), str(L_ANKLE): (0, -1, 0),
                str(R_HIP): (0, 1, 0), str(R_KNEE): (0, 0, 0), str(R_ANKLE): (0, -1, 0),
            }
        )
        self.assertAlmostEqual(float(geometry.knee_angle(pts)), 180.0, places=3)

    def test_right_angle_leg_is_about_90(self):
        pts = _pt24(
            **{
                str(L_HIP): (0, 1, 0), str(L_KNEE): (0, 0, 0), str(L_ANKLE): (1, 0, 0),
                str(R_HIP): (0, 1, 0), str(R_KNEE): (0, 0, 0), str(R_ANKLE): (1, 0, 0),
            }
        )
        self.assertAlmostEqual(float(geometry.knee_angle(pts)), 90.0, places=3)

    def test_batch_with_undetected_row_is_nan(self):
        straight = _pt24(
            **{
                str(L_HIP): (0, 1, 0), str(L_KNEE): (0, 0, 0), str(L_ANKLE): (0, -1, 0),
                str(R_HIP): (0, 1, 0), str(R_KNEE): (0, 0, 0), str(R_ANKLE): (0, -1, 0),
            }
        )
        undetected = np.full((24, 3), np.nan)
        batch = np.stack([straight, undetected])
        out = geometry.knee_angle(batch)
        self.assertAlmostEqual(float(out[0]), 180.0, places=3)
        self.assertTrue(np.isnan(out[1]))


class UpAxisTests(unittest.TestCase):
    def test_upright_skeleton_gives_minus_y(self):
        # Standing frames (high knee angle) encode shoulder-ankle == height * (0, -1, 0).
        joints = np.zeros((5, 24, 3))
        knee = np.array([170.0, 170.0, 170.0, 90.0, 90.0])
        direction = np.array([0.0, -1.0, 0.0])
        for i in range(5):
            frame = np.zeros((24, 3))
            if knee[i] >= 170.0:
                frame[L_ANKLE] = (-100.0, 0.0, 0.0)
                frame[R_ANKLE] = (100.0, 0.0, 0.0)
                frame[L_SHOULDER] = np.array([-100.0, 0.0, 0.0]) + 1000.0 * direction
                frame[R_SHOULDER] = np.array([100.0, 0.0, 0.0]) + 1000.0 * direction
            joints[i] = frame
        up = geometry.estimate_up_axis(joints, knee)
        self.assertAlmostEqual(np.linalg.norm(up), 1.0, places=6)
        np.testing.assert_allclose(up, direction, atol=1e-6)

    def test_tilted_skeleton_gives_tilted_axis(self):
        theta = math.radians(15.0)
        direction = np.array([math.sin(theta), -math.cos(theta), 0.0])
        joints = np.zeros((5, 24, 3))
        knee = np.array([170.0, 170.0, 170.0, 90.0, 90.0])
        for i in range(5):
            frame = np.zeros((24, 3))
            if knee[i] >= 170.0:
                frame[L_ANKLE] = (-100.0, 0.0, 0.0)
                frame[R_ANKLE] = (100.0, 0.0, 0.0)
                frame[L_SHOULDER] = np.array([-100.0, 0.0, 0.0]) + 1000.0 * direction
                frame[R_SHOULDER] = np.array([100.0, 0.0, 0.0]) + 1000.0 * direction
            joints[i] = frame
        up = geometry.estimate_up_axis(joints, knee)
        np.testing.assert_allclose(up, direction, atol=1e-6)

    def test_raises_when_no_standing_frames(self):
        joints = np.full((3, 24, 3), np.nan)
        knee = np.full(3, np.nan)
        with self.assertRaises(ValueError):
            geometry.estimate_up_axis(joints, knee)


class BodyBasisTests(unittest.TestCase):
    def _assert_orthonormal_right_handed(self, up: np.ndarray) -> np.ndarray:
        basis = geometry.body_basis(up)
        self.assertEqual(basis.shape, (3, 3))
        for row in basis:
            self.assertAlmostEqual(np.linalg.norm(row), 1.0, places=6)
        right, up_row, back = basis
        self.assertAlmostEqual(float(np.dot(right, up_row)), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(up_row, back)), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(right, back)), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.det(basis)), 1.0, places=6)
        return basis

    def test_axis_aligned_up(self):
        basis = self._assert_orthonormal_right_handed(np.array([0.0, -1.0, 0.0]))
        np.testing.assert_allclose(basis[0], [1.0, 0.0, 0.0], atol=1e-6)  # right
        np.testing.assert_allclose(basis[2], [0.0, 0.0, -1.0], atol=1e-6)  # back

    def test_tilted_up(self):
        theta = math.radians(15.0)
        up = np.array([math.sin(theta), -math.cos(theta), 0.05])
        up = up / np.linalg.norm(up)
        self._assert_orthonormal_right_handed(up)


class ToBodyFrameTests(unittest.TestCase):
    def test_point_along_up_maps_to_positive_y(self):
        up = np.array([0.0, -1.0, 0.0])
        basis = geometry.body_basis(up)
        pelvis = np.array([10.0, 20.0, 1000.0])
        point = pelvis + np.array([0.0, -1000.0, 0.0])  # 1000 mm "up" from the pelvis
        out = geometry.to_body_frame(point, pelvis, basis)
        self.assertAlmostEqual(float(out[1]), 1.0, places=6)  # 1 metre
        self.assertAlmostEqual(float(out[0]), 0.0, places=6)


class OrthoPixelYTests(unittest.TestCase):
    def test_maps_center_and_edges(self):
        self.assertEqual(geometry.ortho_pixel_y(0.0, 0.0, 1.0, 384), 192)
        self.assertEqual(geometry.ortho_pixel_y(1.0, 0.0, 1.0, 384), 0)
        self.assertEqual(geometry.ortho_pixel_y(-1.0, 0.0, 1.0, 384), 384)


# ---------------------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------------------
class RotationKTests(unittest.TestCase):
    def test_known_angles(self):
        self.assertEqual(timeline.rotation_k(0), 0)
        self.assertEqual(timeline.rotation_k(90), 1)
        self.assertEqual(timeline.rotation_k(-90), 3)
        self.assertEqual(timeline.rotation_k(180), 2)
        self.assertEqual(timeline.rotation_k(270), 3)
        self.assertEqual(timeline.rotation_k(None), 0)


class SampleClockTests(unittest.TestCase):
    def test_keeps_first_frame_at_or_after_each_tick(self):
        clock = timeline.SampleClock(fps=10.0)
        times = [0.0, 0.05, 0.11, 0.15, 0.22, 0.29, 0.31, 0.42]
        kept = [clock.keep(t) for t in times]
        self.assertEqual(kept, [True, False, True, False, True, False, True, True])

    def test_rejects_non_positive_fps(self):
        with self.assertRaises(ValueError):
            timeline.SampleClock(fps=0)

    def test_catches_up_ticks_across_a_gap_instead_of_bursting(self):
        # A gap (no frames near t=0.1..0.5) followed by frames close together must still yield
        # about one keep per tick, not one keep per frame in the burst.
        clock = timeline.SampleClock(fps=10.0)
        times = [0.0, 0.55, 0.58, 0.61]
        kept = [clock.keep(t) for t in times]
        self.assertEqual(kept, [True, True, False, True])


class FrameToSecondsTests(unittest.TestCase):
    def test_converts_with_fps(self):
        self.assertAlmostEqual(timeline.frame_to_seconds(30, 30.0), 1.0)
        self.assertAlmostEqual(timeline.frame_to_seconds(15, 10.0), 1.5)

    def test_none_when_fps_missing_or_invalid(self):
        self.assertIsNone(timeline.frame_to_seconds(15, None))
        self.assertIsNone(timeline.frame_to_seconds(15, 0))
        self.assertIsNone(timeline.frame_to_seconds(15, -5.0))


# ---------------------------------------------------------------------------------------
# keyframes
# ---------------------------------------------------------------------------------------
class SelectKeyFramesTests(unittest.TestCase):
    # A distinct fps from the 10 fps NLF sample rate, so a test that accidentally indexed by
    # sample count instead of converting through pose_fps would fail loudly.
    POSE_FPS = 30.0

    def _seg(self, index, start_time, end_time):
        # rep_segments carry frame numbers on the wire; the worker converts them with
        # pose_fps, same as detections' peak_frame (MODULES TO WRITE item 3).
        return {
            "index": index,
            "start_frame": start_time * self.POSE_FPS,
            "end_frame": end_time * self.POSE_FPS,
        }

    def test_squat_bottom_ignores_crouch_outside_segments(self):
        sample_times = np.round(np.arange(0.0, 2.0, 0.1), 6)
        knee = np.full(sample_times.shape, 170.0)
        knee[0] = 10.0  # a deeper, set-up crouch OUTSIDE any rep segment
        knee[7] = 60.0  # t = 0.7, inside the segment below

        rep_segments = [self._seg(0, 0.5, 1.0)]
        out = keyframes.select_key_frames("Squat", sample_times, knee, rep_segments, [], self.POSE_FPS)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["kind"], "rep_bottom")
        self.assertEqual(out[0]["rep_index"], 0)
        self.assertAlmostEqual(out[0]["t_s"], 0.7, places=6)
        self.assertEqual(out[0]["fault_ids"], [])

    def test_non_squat_ignores_rep_segments(self):
        sample_times = np.round(np.arange(0.0, 2.0, 0.1), 6)
        knee = np.full(sample_times.shape, 170.0)
        knee[7] = 60.0
        rep_segments = [self._seg(0, 0.5, 1.0)]
        out = keyframes.select_key_frames("Push-up", sample_times, knee, rep_segments, [], self.POSE_FPS)
        self.assertEqual(out, [])

    def test_squat_is_the_fallback_for_a_missing_movement(self):
        # registry.get_detector falls back to Squat when the analysis has no movement column
        # (older rows); the worker must treat None/"" the same way.
        sample_times = np.round(np.arange(0.0, 2.0, 0.1), 6)
        knee = np.full(sample_times.shape, 170.0)
        knee[7] = 60.0
        rep_segments = [self._seg(0, 0.5, 1.0)]
        out = keyframes.select_key_frames(None, sample_times, knee, rep_segments, [], self.POSE_FPS)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["kind"], "rep_bottom")

    def test_segment_with_no_finite_knee_angle_is_skipped(self):
        sample_times = np.round(np.arange(0.0, 1.0, 0.1), 6)
        knee = np.full(sample_times.shape, np.nan)
        rep_segments = [self._seg(0, 0.0, 1.0)]
        out = keyframes.select_key_frames("squat", sample_times, knee, rep_segments, [], self.POSE_FPS)
        self.assertEqual(out, [])

    def test_rep_bottom_skipped_when_pose_fps_missing(self):
        sample_times = np.round(np.arange(0.0, 2.0, 0.1), 6)
        knee = np.full(sample_times.shape, 170.0)
        knee[7] = 60.0
        rep_segments = [self._seg(0, 0.5, 1.0)]
        out = keyframes.select_key_frames("Squat", sample_times, knee, rep_segments, [], None)
        self.assertEqual(out, [])

    def test_fault_peak_mapped_by_pose_fps(self):
        sample_times = np.round(np.arange(0.0, 2.0, 0.1), 6)
        knee = np.full(sample_times.shape, 170.0)
        detections = [{"fault_id": "f1", "peak_frame": 1.5 * self.POSE_FPS}]
        out = keyframes.select_key_frames("Squat", sample_times, knee, None, detections, self.POSE_FPS)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["kind"], "fault_peak")
        self.assertEqual(out[0]["fault_ids"], ["f1"])
        self.assertAlmostEqual(out[0]["t_s"], 1.5, places=6)

    def test_detection_dropped_when_pose_fps_missing(self):
        sample_times = np.round(np.arange(0.0, 2.0, 0.1), 6)
        knee = np.full(sample_times.shape, 170.0)
        detections = [{"fault_id": "f1", "peak_frame": 1.5 * self.POSE_FPS}]
        out = keyframes.select_key_frames("Squat", sample_times, knee, None, detections, None)
        self.assertEqual(out, [])

    def test_fault_peak_snaps_to_nearest_detected_sample(self):
        # peak_frame maps to t=0.5 (sample index 5), but that sample has no detection; the
        # nearest DETECTED sample is at t=0.3 (index 3), not the equidistant-looking t=0.6/0.7.
        sample_times = np.round(np.arange(0.0, 2.0, 0.1), 6)
        knee = np.full(sample_times.shape, 170.0)
        knee[4:8] = np.nan  # samples at t=0.4..0.7 are all undetected
        detections = [{"fault_id": "f1", "peak_frame": 0.5 * self.POSE_FPS}]  # t = 0.5
        out = keyframes.select_key_frames("Push-up", sample_times, knee, None, detections, self.POSE_FPS)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["t_s"], 0.3, places=6)
        self.assertTrue(np.isfinite(knee[sample_times.tolist().index(out[0]["t_s"])]))

    def test_merge_within_threshold_unions_fault_ids(self):
        sample_times = np.round(np.arange(0.0, 2.0, 0.1), 6)
        knee = np.full(sample_times.shape, 170.0)
        detections = [
            {"fault_id": "a", "peak_frame": 1.0 * self.POSE_FPS},  # t = 1.0
            {"fault_id": "b", "peak_frame": 1.2 * self.POSE_FPS},  # t = 1.2 (0.2s away, <= 0.3 merge_s)
        ]
        out = keyframes.select_key_frames("Squat", sample_times, knee, None, detections, self.POSE_FPS)
        self.assertEqual(len(out), 1)
        self.assertEqual(sorted(out[0]["fault_ids"]), ["a", "b"])
        self.assertAlmostEqual(out[0]["t_s"], 1.0, places=6)

    def test_merge_prefers_rep_bottom_kind_and_keeps_its_time(self):
        sample_times = np.round(np.arange(0.0, 2.0, 0.1), 6)
        knee = np.full(sample_times.shape, 170.0)
        knee[10] = 40.0  # t = 1.0, the bottom
        rep_segments = [self._seg(0, 0.8, 1.2)]
        detections = [{"fault_id": "f1", "peak_frame": 1.1 * self.POSE_FPS}]  # t=1.1, within 0.3s
        out = keyframes.select_key_frames(
            "Squat", sample_times, knee, rep_segments, detections, self.POSE_FPS
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["kind"], "rep_bottom")
        self.assertEqual(out[0]["fault_ids"], ["f1"])
        self.assertAlmostEqual(out[0]["t_s"], 1.0, places=6)

    def test_cap_keeps_earliest_when_all_peaks(self):
        sample_times = np.arange(0.0, 20.0, 1.0)
        knee = np.full(sample_times.shape, 170.0)
        detections = [{"fault_id": f"d{i}", "peak_frame": i * self.POSE_FPS} for i in range(15)]
        out = keyframes.select_key_frames(
            "Push-up", sample_times, knee, None, detections, self.POSE_FPS, cap=12
        )
        self.assertEqual(len(out), 12)
        self.assertEqual([kf["t_s"] for kf in out], [float(i) for i in range(12)])

    def test_cap_prefers_rep_bottoms_over_later_peaks(self):
        sample_times = np.arange(0.0, 103.0, 1.0)
        knee = np.full(sample_times.shape, 170.0)
        for i in range(3):
            knee[int(100 + i)] = 50.0 + i
        rep_segments = [self._seg(i, 100 + i - 0.4, 100 + i + 0.4) for i in range(3)]
        detections = [{"fault_id": f"d{i}", "peak_frame": i * self.POSE_FPS} for i in range(12)]  # t=0..11
        out = keyframes.select_key_frames(
            "Squat", sample_times, knee, rep_segments, detections, self.POSE_FPS, cap=12
        )
        self.assertEqual(len(out), 12)
        bottoms = [kf for kf in out if kf["kind"] == "rep_bottom"]
        peaks = [kf for kf in out if kf["kind"] == "fault_peak"]
        self.assertEqual(len(bottoms), 3)
        self.assertEqual(len(peaks), 9)
        self.assertEqual([kf["t_s"] for kf in peaks], [float(i) for i in range(9)])
        # chronological order preserved
        self.assertEqual([kf["t_s"] for kf in out], sorted(kf["t_s"] for kf in out))


# ---------------------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------------------
class ResultTests(unittest.TestCase):
    def test_prefix_and_strip_keys(self):
        prefix = result.result_prefix("uploads/owner/vid", 3)
        self.assertEqual(prefix, "uploads/owner/vid/nlf.v1.a3")
        self.assertEqual(result.strip_key(prefix, 0), "uploads/owner/vid/nlf.v1.a3/turntable/k00.webp")
        self.assertEqual(result.strip_key(prefix, 11), "uploads/owner/vid/nlf.v1.a3/turntable/k11.webp")

    def test_pose3d_nan_to_none_and_gzip_round_trip(self):
        joints = [np.array([[float("nan")] * 3] * 24), None]
        unc = [np.zeros(24), None]
        pose3d = result.build_pose3d(
            model_sha256="deadbeef",
            torch_version="2.5.1",
            sample_fps=10.0,
            sample_times_s=[0.0, 0.1],
            joints3d=joints,
            joint_uncertainties=unc,
            up_axis=np.array([0.0, -1.0, 0.0]),
        )
        self.assertIsNone(pose3d["joints3d"][0][0][0])
        self.assertIsNone(pose3d["joints3d"][1])
        self.assertIsNone(pose3d["joint_uncertainties"][1])

        encoded = result.encode_pose3d(pose3d)
        self.assertIsInstance(encoded, bytes)
        # round trip through real gzip/json, independent of the helper
        decoded = json.loads(gzip.decompress(encoded).decode("utf-8"))
        self.assertEqual(decoded, pose3d)
        self.assertEqual(result.decode_pose3d(encoded), pose3d)

    def test_build_meta_shape(self):
        prefix = "uploads/o/v/nlf.v1.a1"
        key_frames = [
            {"t_s": 0.5, "kind": "rep_bottom", "fault_ids": [], "rep_index": 0, "sample_index": 5},
            {"t_s": 1.1, "kind": "fault_peak", "fault_ids": ["f1"], "rep_index": None, "sample_index": 11},
        ]
        meta = result.build_meta(analysis_id="an-1", angles=24, tile_px=384, prefix=prefix, key_frames=key_frames)
        self.assertEqual(meta["schema_version"], 1)
        self.assertEqual(meta["analysis_id"], "an-1")
        self.assertEqual(meta["angles"], 24)
        self.assertEqual(meta["tile_px"], 384)
        self.assertEqual(meta["prefix"], prefix)
        self.assertEqual(len(meta["key_frames"]), 2)
        self.assertEqual(meta["key_frames"][0]["strip_key"], result.strip_key(prefix, 0))
        self.assertEqual(meta["key_frames"][1]["strip_key"], result.strip_key(prefix, 1))
        self.assertEqual(meta["key_frames"][1]["rep_index"], None)
        self.assertEqual(meta["key_frames"][1]["fault_ids"], ["f1"])
        self.assertNotIn("sample_index", meta["key_frames"][0])


# ---------------------------------------------------------------------------------------
# smpl_faces
# ---------------------------------------------------------------------------------------
class SmplFacesTests(unittest.TestCase):
    def test_loads_faces_ignoring_chumpy(self):
        fake_chumpy = types.ModuleType("chumpy")
        fake_chumpy_ch = types.ModuleType("chumpy.ch")

        class Ch:
            def __init__(self):
                self.dummy_state = "whatever chumpy would restore"

        Ch.__module__ = "chumpy.ch"
        Ch.__qualname__ = "Ch"
        fake_chumpy_ch.Ch = Ch
        sys.modules["chumpy"] = fake_chumpy
        sys.modules["chumpy.ch"] = fake_chumpy_ch
        try:
            faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 0]], dtype=np.int64)
            payload = {"f": faces, "chumpy_thing": Ch()}
            data = pickle.dumps(payload, protocol=2)
        finally:
            sys.modules.pop("chumpy", None)
            sys.modules.pop("chumpy.ch", None)

        # "chumpy" is now fully absent from sys.modules, simulating the CI/runtime environment.
        self.assertNotIn("chumpy", sys.modules)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            fh.write(data)
            path = fh.name
        try:
            out = smpl_faces.load_smpl_faces(path)
        finally:
            os.remove(path)

        self.assertEqual(out.dtype, np.int64)
        np.testing.assert_array_equal(out, faces)


# ---------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------
# Built at runtime, not written as a literal, so secret scanners (GitGuardian) don't read the
# config fixtures as leaked credentials.
_FAKE_CREDENTIAL = "-".join(("test", "only", "value"))

_REQUIRED_ENV_KEYS = (
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "NLF_WORKER_EMAIL",
    "NLF_WORKER_PASSWORD",
    "NLF_R2_ACCOUNT_ID",
    "NLF_R2_ACCESS_KEY_ID",
    "NLF_R2_SECRET_ACCESS_KEY",
    "NLF_R2_BUCKET",
)
_OPTIONAL_ENV_KEYS = (
    "NLF_MODEL_PATH",
    "NLF_MODEL_FP32_PATH",
    "NLF_SMPL_PKL",
    "NLF_POLL_SECONDS",
    "NLF_SAMPLE_FPS",
    "NLF_BATCH",
    "NLF_TMP_DIR",
)


class LoadWorkerEnvTests(unittest.TestCase):
    def setUp(self):
        # Optional keys too: a stray NLF_BATCH in the shell must not leak into the defaults tests.
        self._saved = {k: os.environ.pop(k, None) for k in _REQUIRED_ENV_KEYS + _OPTIONAL_ENV_KEYS}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _write(self, tmp_path, text):
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def test_parses_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "worker.env")
            self._write(
                path,
                "\n".join(
                    [
                        "# a comment",
                        "",
                        "SUPABASE_URL=https://example.supabase.co",
                        "SUPABASE_ANON_KEY = 'anon-key'",
                        'NLF_WORKER_EMAIL="worker@example.com"',
                        f"NLF_WORKER_PASSWORD={_FAKE_CREDENTIAL}",
                        "NLF_R2_ACCOUNT_ID=acct",
                        "NLF_R2_ACCESS_KEY_ID=akid",
                        f"NLF_R2_SECRET_ACCESS_KEY={_FAKE_CREDENTIAL}",
                        "NLF_R2_BUCKET=bucket",
                    ]
                ),
            )
            env = config.load_worker_env(path)
            self.assertEqual(env.SUPABASE_URL, "https://example.supabase.co")
            self.assertEqual(env.SUPABASE_ANON_KEY, "anon-key")
            self.assertEqual(env.NLF_WORKER_EMAIL, "worker@example.com")
            self.assertEqual(env.NLF_R2_BUCKET, "bucket")

    def test_env_var_overrides_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "worker.env")
            self._write(
                path,
                "\n".join(f"{k}=from-file-{k}" for k in _REQUIRED_ENV_KEYS),
            )
            os.environ["NLF_R2_BUCKET"] = "from-env"
            env = config.load_worker_env(path)
            self.assertEqual(env.NLF_R2_BUCKET, "from-env")
            self.assertEqual(env.SUPABASE_URL, "from-file-SUPABASE_URL")

    def test_missing_keys_raise_value_error_naming_them(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "worker.env")
            self._write(path, "SUPABASE_URL=https://example.supabase.co\n")
            with self.assertRaises(ValueError) as ctx:
                config.load_worker_env(path)
            msg = str(ctx.exception)
            self.assertIn("NLF_R2_BUCKET", msg)
            self.assertIn("SUPABASE_ANON_KEY", msg)

    def test_missing_file_with_no_env_raises(self):
        with self.assertRaises(ValueError):
            config.load_worker_env("does/not/exist.env")

    def _write_required(self, path, extra_lines=()):
        self._write(path, "\n".join([f"{k}=v-{k}" for k in _REQUIRED_ENV_KEYS] + list(extra_lines)))

    def test_optional_settings_default(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "worker.env")
            self._write_required(path)
            env = config.load_worker_env(path)
        self.assertEqual(env.NLF_MODEL_PATH, "data/models/nlf/nlf_l_multi.torchscript")
        self.assertEqual(env.NLF_MODEL_FP32_PATH, "data/models/nlf/nlf_l_multi_fp32.torchscript")
        self.assertEqual(
            env.NLF_SMPL_PKL,
            "data/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
        )
        self.assertEqual(env.NLF_POLL_SECONDS, 60.0)
        self.assertEqual(env.NLF_SAMPLE_FPS, 10.0)
        self.assertEqual(env.NLF_BATCH, 8)
        self.assertIsInstance(env.NLF_BATCH, int)
        self.assertEqual(env.NLF_TMP_DIR, "data/runtime/nlf_worker_tmp")

    def test_optional_settings_from_file_then_env(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "worker.env")
            self._write_required(
                path, ["NLF_BATCH=16", "NLF_SAMPLE_FPS=7.5", "NLF_TMP_DIR='D:/nlf tmp'", "NLF_POLL_SECONDS=30"]
            )
            os.environ["NLF_BATCH"] = "4"
            env = config.load_worker_env(path)
        self.assertEqual(env.NLF_BATCH, 4)  # the real environment wins over the file
        self.assertEqual(env.NLF_SAMPLE_FPS, 7.5)
        self.assertEqual(env.NLF_TMP_DIR, "D:/nlf tmp")
        self.assertEqual(env.NLF_POLL_SECONDS, 30.0)

    def test_invalid_numeric_setting_raises_naming_the_key(self):
        for line in ("NLF_BATCH=eight", "NLF_BATCH=2.5", "NLF_POLL_SECONDS=0", "NLF_SAMPLE_FPS=-10"):
            with self.subTest(line=line), tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "worker.env")
                self._write_required(path, [line])
                with self.assertRaises(ValueError) as ctx:
                    config.load_worker_env(path)
                self.assertIn(line.split("=")[0], str(ctx.exception))

    def test_settings_alone_need_no_credentials(self):
        settings = config.load_worker_settings("does/not/exist.env")
        self.assertEqual(settings.NLF_BATCH, 8)
        os.environ["NLF_SAMPLE_FPS"] = "5"
        self.assertEqual(config.load_worker_settings("does/not/exist.env").NLF_SAMPLE_FPS, 5.0)

    def test_repr_hides_secrets(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "worker.env")
            self._write_required(path)
            text = repr(config.load_worker_env(path))
        hidden = [f.name for f in dataclasses.fields(config.WorkerEnv) if not f.repr]
        self.assertEqual(len(hidden), 4)
        for name in hidden:
            self.assertNotIn(f"v-{name}", text)
        self.assertIn("v-SUPABASE_URL", text)


# ---------------------------------------------------------------------------------------
# loop, with fakes for every interface
# ---------------------------------------------------------------------------------------
class FakeDb:
    def __init__(self, jobs=None, complete_result=True):
        self.jobs = list(jobs or [])
        self.heartbeats = 0
        self.completed = []
        self.failed = []
        self.complete_result = complete_result

    def heartbeat(self):
        self.heartbeats += 1

    def claim(self):
        if not self.jobs:
            return None
        return self.jobs.pop(0)

    def complete(self, job_id, result_key, meta):
        self.completed.append((job_id, result_key, meta))
        return self.complete_result

    def fail(self, job_id, error):
        self.failed.append((job_id, error))


class FakeStore:
    def __init__(self):
        self.puts = {}
        self.deleted_prefixes = []

    def put(self, key, data, *, content_type):
        self.puts[key] = (data, content_type)

    def delete_prefix(self, prefix):
        self.deleted_prefixes.append(prefix)


class FakeFetcher:
    def __init__(self, content=b"fake-video-bytes"):
        self.content = content
        self.downloaded_keys = []
        self.last_path = None

    def download(self, key, dest_dir):
        self.downloaded_keys.append(key)
        path = os.path.join(dest_dir, "source.bin")
        with open(path, "wb") as fh:
            fh.write(self.content)
        self.last_path = path
        return path


class FakeDecoder:
    def __init__(self, samples):
        self.samples = list(samples)
        self.calls = []

    def iterate(self, path, sample_fps):
        self.calls.append((path, sample_fps))
        return list(self.samples)


class FakeModel:
    def __init__(self, results):
        self._results = list(results)
        self._i = 0
        self.batch_sizes = []

    def infer(self, frames):
        n = len(frames)
        self.batch_sizes.append(n)
        out = self._results[self._i : self._i + n]
        self._i += n
        return out


class RaisingModel:
    def infer(self, frames):
        raise RuntimeError("boom: the model blew up")


def _detected_result(theta_deg):
    frame = _leg_frame_direct(
        np.array([0.0, 0.0, 2000.0]),
        np.array([-100.0, 0.0, 2000.0]),
        np.array([100.0, 0.0, 2000.0]),
        np.array([-100.0, 400.0, 2000.0]),
        np.array([100.0, 400.0, 2000.0]),
        np.array([-100.0, 400.0, 2000.0])
        + 400.0 * np.array([math.sin(math.radians(theta_deg)), math.cos(math.radians(theta_deg)), 0.0]),
        np.array([100.0, 400.0, 2000.0])
        + 400.0 * np.array([math.sin(math.radians(theta_deg)), math.cos(math.radians(theta_deg)), 0.0]),
    )
    return {
        "joints3d": frame,
        "joint_uncertainties": np.zeros(24),
        "vertices3d": np.zeros((8, 3)),
    }


class RunOneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _job(self, **overrides):
        job = {
            "job_id": "job-1",
            "owner_id": "owner-1",
            "video_id": "vid-1",
            "attempts": 1,
            "storage_key": "uploads/owner-1/vid-1",
            "analysis_id": "an-1",
            "movement": "Squat",
            "pose_fps": 10.0,
            "rep_segments": [
                {"index": 0, "start_frame": 0, "end_frame": 8, "start_time": 0.0, "end_time": 0.8,
                 "analyzed": True, "partial": False}
            ],
            "detections": [{"fault_id": "f1", "peak_frame": 0}],
        }
        job.update(overrides)
        return job

    def _samples(self, thetas):
        # 9 samples at 0.1s apart -> 0.0 .. 0.8
        return [(round(i * 0.1, 6), np.zeros((2, 2, 3), dtype=np.uint8)) for i in range(len(thetas))]

    def test_idle_when_no_job(self):
        db = FakeDb(jobs=[])
        status = loop.run_one(
            db, FakeStore(), FakeFetcher(), FakeDecoder([]), FakeModel([]), object(),
            tmp_dir=self.tmp.name, model_sha256="sha", torch_version="2.5.1",
        )
        self.assertEqual(status, "idle")
        self.assertEqual(db.heartbeats, 1)
        self.assertEqual(db.completed, [])
        self.assertEqual(db.failed, [])

    def test_heartbeat_called_every_run(self):
        db = FakeDb(jobs=[])
        for expected in (1, 2, 3):
            loop.run_one(
                db, FakeStore(), FakeFetcher(), FakeDecoder([]), FakeModel([]), object(),
                tmp_dir=self.tmp.name, model_sha256="sha", torch_version="2.5.1",
            )
            self.assertEqual(db.heartbeats, expected)

    def test_done_uploads_strips_and_pose3d_and_completes(self):
        thetas = [5, 5, 5, 5, 80, 5, 5, 5, 5]  # bottom at index 4 (t=0.4)
        job = self._job(detections=[{"fault_id": "f1", "peak_frame": 0}])  # t = 0.0, far from bottom
        db = FakeDb(jobs=[job])
        store = FakeStore()
        fetcher = FakeFetcher()
        decoder = FakeDecoder(self._samples(thetas))
        model = FakeModel([_detected_result(t) for t in thetas])
        renderer_calls = []

        class Renderer:
            def render_strip(self, vertices_mm, joints_mm, basis, framing=None):
                renderer_calls.append((vertices_mm, joints_mm, basis, framing))
                return b"webp-bytes-%d" % len(renderer_calls)

        status = loop.run_one(
            db, store, fetcher, decoder, model, Renderer(),
            tmp_dir=self.tmp.name, model_sha256="sha256hex", torch_version="2.5.1",
        )

        self.assertEqual(status, "done")
        self.assertEqual(len(db.completed), 1)
        job_id, pose3d_key, meta = db.completed[0]
        self.assertEqual(job_id, "job-1")

        prefix = result.result_prefix(job["storage_key"], job["attempts"])
        self.assertEqual(pose3d_key, f"{prefix}/pose3d.json.gz")
        self.assertEqual(meta["prefix"], prefix)
        self.assertEqual(len(meta["key_frames"]), 2)
        self.assertEqual(len(renderer_calls), 2)

        expected_strip_keys = {result.strip_key(prefix, i) for i in range(2)}
        uploaded_keys = set(store.puts.keys())
        self.assertEqual(uploaded_keys, expected_strip_keys | {pose3d_key})
        for i, kf in enumerate(meta["key_frames"]):
            self.assertEqual(kf["strip_key"], result.strip_key(prefix, i))
            self.assertIn(kf["strip_key"], store.puts)
            self.assertEqual(store.puts[kf["strip_key"]][1], "image/webp")
        self.assertEqual(store.puts[pose3d_key][1], "application/json")

        # The renderer must get raw camera-coordinate mm (not pelvis-relative/rotated/metres):
        # it does that conversion itself (geometry.to_body_frame), same as phase0_6 does inline.
        for i, kf in enumerate(meta["key_frames"]):
            sample_index = None
            # recover which sampled frame this key frame came from via its time
            for idx, t in enumerate(round(x * 0.1, 6) for x in range(9)):
                if abs(t - kf["t_s"]) < 1e-6:
                    sample_index = idx
                    break
            self.assertIsNotNone(sample_index)
            expected_joints_mm = _detected_result(thetas[sample_index])["joints3d"]
            _, joints_mm, basis, _ = renderer_calls[i]
            np.testing.assert_allclose(joints_mm, expected_joints_mm)
            self.assertEqual(basis.shape, (3, 3))

        # One framing for the whole job: every strip gets the same view, taken from the union of
        # all key frames' body-frame meshes.
        framings = [call[3] for call in renderer_calls]
        self.assertIsNotNone(framings[0])
        self.assertTrue(all(f == framings[0] for f in framings))
        expected = geometry.turntable_framing(
            [geometry.to_body_frame(v, j[PELVIS], b) for v, j, b, _ in renderer_calls]
        )
        self.assertEqual(framings[0], expected)

        kinds = {kf["kind"] for kf in meta["key_frames"]}
        self.assertEqual(kinds, {"fault_peak", "rep_bottom"})

        pose3d_bytes, pose3d_ct = store.puts[pose3d_key]
        pose3d = result.decode_pose3d(pose3d_bytes)
        self.assertEqual(pose3d["schema_version"], 1)
        self.assertEqual(len(pose3d["sample_times_s"]), 9)
        self.assertEqual(len(pose3d["joints3d"]), 9)

        self.assertEqual(store.deleted_prefixes, [])
        self.assertFalse(os.path.exists(fetcher.last_path))

    def test_orphaned_when_complete_returns_false(self):
        thetas = [5, 5, 5, 5, 80, 5, 5, 5, 5]
        job = self._job()
        db = FakeDb(jobs=[job], complete_result=False)
        store = FakeStore()
        fetcher = FakeFetcher()
        decoder = FakeDecoder(self._samples(thetas))
        model = FakeModel([_detected_result(t) for t in thetas])

        class Renderer:
            def render_strip(self, vertices_mm, joints_mm, basis, framing=None):
                return b"strip"

        status = loop.run_one(
            db, store, fetcher, decoder, model, Renderer(),
            tmp_dir=self.tmp.name, model_sha256="sha", torch_version="2.5.1",
        )

        self.assertEqual(status, "orphaned")
        prefix = result.result_prefix(job["storage_key"], job["attempts"])
        self.assertEqual(store.deleted_prefixes, [prefix])
        self.assertEqual(db.failed, [])
        self.assertFalse(os.path.exists(fetcher.last_path))

    def test_failed_when_model_raises(self):
        thetas = [5, 5, 5]
        job = self._job(
            rep_segments=[{"index": 0, "start_frame": 0, "end_frame": 2,
                           "start_time": 0.0, "end_time": 0.2, "analyzed": True, "partial": False}],
            detections=[],
        )
        db = FakeDb(jobs=[job])
        store = FakeStore()
        fetcher = FakeFetcher()
        decoder = FakeDecoder(self._samples(thetas))
        model = RaisingModel()

        status = loop.run_one(
            db, store, fetcher, decoder, model, object(),
            tmp_dir=self.tmp.name, model_sha256="sha", torch_version="2.5.1",
        )

        self.assertEqual(status, "failed")
        prefix = result.result_prefix(job["storage_key"], job["attempts"])
        self.assertEqual(store.deleted_prefixes, [prefix])
        self.assertEqual(len(db.failed), 1)
        failed_job_id, error = db.failed[0]
        self.assertEqual(failed_job_id, "job-1")
        self.assertIn("boom", error)
        self.assertFalse(os.path.exists(fetcher.last_path))

    def test_failed_when_no_frame_detected(self):
        n = 5
        job = self._job(detections=[])
        db = FakeDb(jobs=[job])
        store = FakeStore()
        fetcher = FakeFetcher()
        decoder = FakeDecoder(
            [(round(i * 0.1, 6), np.zeros((2, 2, 3), dtype=np.uint8)) for i in range(n)]
        )
        model = FakeModel(
            [{"joints3d": None, "joint_uncertainties": None, "vertices3d": None} for _ in range(n)]
        )

        status = loop.run_one(
            db, store, fetcher, decoder, model, object(),
            tmp_dir=self.tmp.name, model_sha256="sha", torch_version="2.5.1",
        )

        self.assertEqual(status, "failed")
        prefix = result.result_prefix(job["storage_key"], job["attempts"])
        self.assertEqual(store.deleted_prefixes, [prefix])
        self.assertEqual(len(db.failed), 1)
        self.assertIn("detection", db.failed[0][1])
        self.assertFalse(os.path.exists(fetcher.last_path))


class LiveFrameDecoder:
    """A streaming decoder that, before producing each frame, counts how many of the frames it
    already yielded are still alive (held by anyone) -- the loop's frame footprint."""

    def __init__(self, n, fps=10.0):
        self.n = n
        self.fps = fps
        self.max_alive = 0
        self.yielded = 0

    def iterate(self, path, sample_fps):
        refs = []
        for i in range(self.n):
            alive = sum(r() is not None for r in refs)
            self.max_alive = max(self.max_alive, alive)
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            refs.append(weakref.ref(frame))
            self.yielded += 1
            yield round(i / self.fps, 6), frame
            del frame
        self.max_alive = max(self.max_alive, sum(r() is not None for r in refs))


class ListLengthModel:
    """Records the longest frame list it was ever handed; answers like a detected squat."""

    def __init__(self, thetas):
        self.thetas = list(thetas)
        self.i = 0
        self.max_len = 0
        self.calls = 0

    def infer(self, frames):
        self.calls += 1
        self.max_len = max(self.max_len, len(frames))
        out = [_detected_result(self.thetas[self.i + k]) for k in range(len(frames))]
        self.i += len(frames)
        return out


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _run(self, decoder, model, batch, job=None, renderer=None):
        class Renderer:
            def render_strip(self, vertices_mm, joints_mm, basis, framing=None):
                return b"strip"

        job = job or {
            "job_id": "job-1", "owner_id": "o", "video_id": "v", "attempts": 1,
            "storage_key": "uploads/o/v", "analysis_id": "an-1", "movement": "Squat",
            "pose_fps": 10.0,
            "rep_segments": [{"index": 0, "start_frame": 0, "end_frame": 24}],
            "detections": [],
        }
        db = FakeDb(jobs=[job])
        status = loop.run_one(
            db, FakeStore(), FakeFetcher(), decoder, model, renderer or Renderer(),
            batch=batch, tmp_dir=self.tmp.name, model_sha256="sha", torch_version="t",
        )
        return status, db

    def test_never_holds_more_than_one_batch_of_frames(self):
        n, batch = 25, 4
        thetas = [5] * 12 + [80] + [5] * 12
        decoder, model = LiveFrameDecoder(n), ListLengthModel(thetas)
        status, db = self._run(decoder, model, batch)
        self.assertEqual(status, "done", db.failed)
        self.assertEqual(decoder.yielded, n)
        self.assertEqual(model.calls, 7)  # 6 full batches of 4, then the remaining 1
        self.assertLessEqual(model.max_len, batch)
        self.assertLessEqual(decoder.max_alive, batch)
        # Every decoded frame was released by the end, and the bottom was still found.
        self.assertEqual(db.completed[0][2]["key_frames"][0]["t_s"], 1.2)

    def test_model_outputs_are_kept_as_detached_float32(self):
        base = np.arange(8 * 24 * 3, dtype=np.float64).reshape(8, 24, 3)
        seen = {}

        class ViewModel:
            def infer(self, frames):
                # Views into one big batch buffer, as a careless adapter might return.
                out = [
                    {"joints3d": base[k], "joint_uncertainties": base[k, :, 0], "vertices3d": base[k]}
                    for k in range(len(frames))
                ]
                return out

        kept = loop._decode_and_infer(
            FakeDecoder([(i / 10, np.zeros((2, 2, 3), np.uint8)) for i in range(3)]),
            ViewModel(), "unused", 10.0, 8,
        )
        times, joints, uncertainties, vertices = kept
        self.assertEqual(times, [0.0, 0.1, 0.2])
        for arr in joints + uncertainties + vertices:
            self.assertEqual(arr.dtype, np.float32)
            self.assertIsNone(arr.base)  # an owned copy, not a view keeping `base` alive
        np.testing.assert_array_equal(joints[1], base[1].astype(np.float32))

    def test_undetected_frames_stay_none(self):
        _, joints, uncertainties, vertices = loop._decode_and_infer(
            FakeDecoder([(0.0, np.zeros((2, 2, 3), np.uint8))]),
            FakeModel([{"joints3d": None, "joint_uncertainties": None, "vertices3d": None}]),
            "unused", 10.0, 8,
        )
        self.assertEqual((joints, uncertainties, vertices), ([None], [None], [None]))

    def test_model_returning_the_wrong_count_fails_the_job(self):
        class ShortModel:
            def infer(self, frames):
                return [_detected_result(5)] * (len(frames) - 1)

        decoder = FakeDecoder([(i / 10, np.zeros((2, 2, 3), np.uint8)) for i in range(4)])
        status, db = self._run(decoder, ShortModel(), 4)
        self.assertEqual(status, "failed")
        self.assertIn("3 results for 4 frames", db.failed[0][1])


class SharedFramingTests(unittest.TestCase):
    def test_union_of_key_frames_and_meshless_entries_ignored(self):
        basis = geometry.body_basis(np.array([0.0, -1.0, 0.0]))
        pelvis = np.array([0.0, 0.0, 2000.0])
        joints = np.zeros((24, 3))
        joints[PELVIS] = pelvis
        # Camera y is down: a tall standing body (1.8 m) and a short crouched one (1.0 m).
        tall = pelvis + np.array([[0.0, -900.0, 0.0], [0.0, 900.0, 0.0]])
        short = pelvis + np.array([[0.0, -600.0, 0.0], [0.0, 400.0, 0.0]])
        framing = loop.shared_framing([(tall, joints), (None, joints), (short, joints)], basis)
        self.assertAlmostEqual(framing["cy"], 0.0)
        self.assertAlmostEqual(framing["mag"], 0.9 * 1.12)
        self.assertEqual(framing, loop.shared_framing([(short, joints), (tall, joints)], basis))

    def test_none_when_nothing_to_frame(self):
        self.assertIsNone(loop.shared_framing([], np.eye(3)))
        self.assertIsNone(loop.shared_framing([(None, np.zeros((24, 3)))], np.eye(3)))


if __name__ == "__main__":
    unittest.main()
