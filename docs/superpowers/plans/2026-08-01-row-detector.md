# Row Rule Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a cited, unvalidated (Beta) rule detector for the bent-over barbell Row — four of the parent spec's five rules, with the fifth proven unimplementable and documented as a spec defect.

**Architecture:** One new module `src/pose/movements/row.py` following `src/pose/movements/lunge.py` exactly: threshold-free raw metrics → phase assignment → cited rule functions → an assembled `MovementDetector` registered by side-effect import. The shared `run_detector` in `src/pose/movements/base.py` does segmentation, smoothing, per-rep slicing and merging; nothing in this plan changes it.

**Tech Stack:** Python 3.12, numpy, `unittest.TestCase`. No new dependencies.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-08-01-row-detector-design.md`. **Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` §Row (lines 621–689). Citations are **copied verbatim from the parent spec at implementation time, never recalled from memory**.
- **Interpreter:** `.venv\Scripts\python.exe` from the repository root. Never bare `python`/`pip`, never `source .venv/bin/activate`.
- **Run everything from the repository root** (this worktree's root). Modules import as `from src.pose... import ...`.
- **Every threshold is labeled in-code as exactly one of two categories**, in the style of `src/pose/movements/pushup.py`: **`FROM THE SPEC`** or **`RULE-LEVEL CHOICE MADE HERE`**. Never blur them.
- **All four severity ramps are RULE-LEVEL.** The parent spec's Row section states no ramp for any fault. Convention taken from `pushup.rule_hip_sag`: ramp endpoint = 2.5× the fire threshold, documented as a display/ranking curve, not a cited quantity. The one exception is the elbow-angle ramp `100 → 140°`, taken verbatim from `pushup.rule_shallow_depth` so the two elbow ramps cannot drift.
- **No threshold tuning.** Cited numbers stay as the spec states them. Weak behavior is written up, never repaired by moving a number.
- `ROW_DETECTOR.validated` stays `False`. There is no labeled row data anywhere in this repository.
- **Metric layer contains no thresholds.** `row_compute_raw` / `row_assign_phases` emit scale-free per-frame quantities and phase labels only. The sole constant they may define is a division-by-zero guard.
- Test command: `.venv\Scripts\python.exe -m pytest tests/ -q` (always scoped to `tests/`). Coverage gate: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.
- Commit after every task. Commit message body explains **why**, in the style of the repository's recent history.

---

### Task 1: Raw metrics and phase assignment

**Files:**
- Create: `src/pose/movements/row.py`
- Test: `tests/test_row.py`

**Interfaces:**
- Consumes: `src.pose.geometry` (`landmarks_to_array`, `visible_point`, `angle_degrees`, `midpoint`, `distance`, `mean_visibility`), `src.pose.movements.base.CoreFrame`.
- Produces:
  - `ROW_METRIC_KEYS: tuple[str, ...]`
  - `row_compute_raw(frames: Sequence[object], fps: float) -> list[dict]`
  - `row_assign_phases(raw: list[dict]) -> list[str]`
  - Landmark constants `LEFT_ELBOW = 13`, `RIGHT_ELBOW = 14`, `LEFT_WRIST = 15`, `RIGHT_WRIST = 16`
  - Module-private `_derivative(values, fps)`, `_DEGENERATE_LENGTH = 1e-6`

- [ ] **Step 1: Write the failing metric tests**

Create `tests/test_row.py`:

```python
import math
import unittest

import numpy as np

from src.pose.movements.base import RuleContext, run_detector


def _lm(x: float, y: float, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _elbow_xy(
    shoulder: tuple[float, float],
    wrist: tuple[float, float],
    elbow_angle_deg: float,
    side_sign: float,
) -> tuple[float, float]:
    """Place an elbow so that angle(shoulder, elbow, wrist) EQUALS `elbow_angle_deg` exactly.

    Two equal-length segments of length r spanning a shoulder-wrist chord of length d subtend
    an elbow angle of 2*asin(d / (2r)), so the r that produces a requested angle is
    r = d / (2*sin(angle/2)). The elbow then sits on the chord's perpendicular bisector at
    height h = sqrt(r^2 - (d/2)^2). Controlling the ANGLE directly is the property the ROM
    rule's fixtures need: `max_elbow_angle` equals the requested number by construction, so a
    boundary fixture really does sit one step either side of the 100-degree threshold.
    """
    sx, sy = shoulder
    wx, wy = wrist
    dx, dy = wx - sx, wy - sy
    d = math.hypot(dx, dy)
    half = math.radians(elbow_angle_deg) / 2.0
    r = d / (2.0 * math.sin(half))
    h = math.sqrt(max(r * r - (d / 2.0) ** 2, 0.0))
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    return (sx + dx / 2.0 + side_sign * h * px, sy + dy / 2.0 + side_sign * h * py)


def row_frame(
    trunk_angle_deg: float = 20.0,
    wrist_hip_dist: float = 0.08,
    elbow_angle_deg: float = 70.0,
    elbow_dy: float = 0.0,
    shoulder_tilt: float = 0.0,
    wrist_shift: float = 0.0,
    frame_index: int = 0,
    visibility: float = 0.95,
) -> dict:
    """One bent-over row frame, image y growing DOWNWARD, viewed obliquely.

    Knobs, each controlling exactly one metric BY CONSTRUCTION:
      trunk_angle_deg -- angle of shoulder_mid -> hip_mid from horizontal. 0 = perfectly
                         hinged (torso horizontal), 90 = upright. Equals
                         `trunk_angle_from_horizontal_deg`.
      wrist_hip_dist  -- distance from each wrist to its same-side hip, in image units.
                         Equals `mean_wrist_hip_dist` when `wrist_shift` is 0.
      elbow_angle_deg -- angle(shoulder, elbow, wrist) per side; equals both
                         `min_elbow_angle` and `max_elbow_angle` when `elbow_dy` is 0.
      elbow_dy        -- extra downward displacement applied to the LEFT elbow only. Equals
                         `elbow_height_asymmetry`. The left elbow ANGLE then becomes derived
                         rather than requested, which is why no test asserts both at once.
      shoulder_tilt   -- signed image-y difference between the shoulders; equals
                         `shoulder_tilt` (the metric) in magnitude.
      wrist_shift     -- moves the RIGHT wrist further from its hip by this amount; equals
                         `wrist_travel_asymmetry`.
    """
    hip_mid = (0.60, 0.55)
    trunk_len = 0.30
    theta = math.radians(trunk_angle_deg)
    shoulder_mid = (hip_mid[0] - trunk_len * math.cos(theta), hip_mid[1] - trunk_len * math.sin(theta))

    half_shoulder, half_hip = 0.06, 0.05
    left_shoulder = (shoulder_mid[0] - half_shoulder, shoulder_mid[1] - shoulder_tilt / 2.0)
    right_shoulder = (shoulder_mid[0] + half_shoulder, shoulder_mid[1] + shoulder_tilt / 2.0)
    left_hip = (hip_mid[0] - half_hip, hip_mid[1])
    right_hip = (hip_mid[0] + half_hip, hip_mid[1])

    # Wrists sit directly BELOW their own hip (the bar hangs under the shoulders in a hinge),
    # so wrist-to-hip distance is exactly the requested value.
    left_wrist = (left_hip[0], left_hip[1] + wrist_hip_dist)
    right_wrist = (right_hip[0], right_hip[1] + wrist_hip_dist + wrist_shift)

    left_elbow = _elbow_xy(left_shoulder, left_wrist, elbow_angle_deg, +1.0)
    left_elbow = (left_elbow[0], left_elbow[1] + elbow_dy)
    right_elbow = _elbow_xy(right_shoulder, right_wrist, elbow_angle_deg, +1.0)

    landmarks = [_lm(0.0, 0.0, 0.0) for _ in range(33)]
    landmarks[11] = _lm(*left_shoulder, visibility)
    landmarks[12] = _lm(*right_shoulder, visibility)
    landmarks[13] = _lm(*left_elbow, visibility)
    landmarks[14] = _lm(*right_elbow, visibility)
    landmarks[15] = _lm(*left_wrist, visibility)
    landmarks[16] = _lm(*right_wrist, visibility)
    landmarks[23] = _lm(*left_hip, visibility)
    landmarks[24] = _lm(*right_hip, visibility)
    return {"frame_index": frame_index, "landmarks": landmarks}


class RowMetricsTest(unittest.TestCase):
    def test_trunk_angle_equals_the_constructed_hinge_angle(self) -> None:
        from src.pose.movements.row import row_compute_raw

        for requested in (0.0, 20.0, 45.0, 80.0):
            raw = row_compute_raw([row_frame(trunk_angle_deg=requested)], fps=30.0)
            self.assertAlmostEqual(raw[0]["trunk_angle_from_horizontal_deg"], requested, places=4)

    def test_elbow_angles_equal_the_constructed_angle(self) -> None:
        from src.pose.movements.row import row_compute_raw

        raw = row_compute_raw([row_frame(elbow_angle_deg=95.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["left_elbow_angle"], 95.0, places=3)
        self.assertAlmostEqual(raw[0]["right_elbow_angle"], 95.0, places=3)
        self.assertAlmostEqual(raw[0]["min_elbow_angle"], 95.0, places=3)
        self.assertAlmostEqual(raw[0]["max_elbow_angle"], 95.0, places=3)

    def test_min_and_max_elbow_angle_pick_the_right_arm(self) -> None:
        from src.pose.movements.row import row_compute_raw

        # elbow_dy displaces the LEFT elbow only, so the two arms differ.
        raw = row_compute_raw([row_frame(elbow_angle_deg=90.0, elbow_dy=0.05)], fps=30.0)
        left, right = raw[0]["left_elbow_angle"], raw[0]["right_elbow_angle"]
        self.assertNotAlmostEqual(left, right, places=2)
        self.assertAlmostEqual(raw[0]["min_elbow_angle"], min(left, right), places=6)
        self.assertAlmostEqual(raw[0]["max_elbow_angle"], max(left, right), places=6)

    def test_wrist_hip_distance_equals_the_constructed_distance(self) -> None:
        from src.pose.movements.row import row_compute_raw

        raw = row_compute_raw([row_frame(wrist_hip_dist=0.15)], fps=30.0)
        self.assertAlmostEqual(raw[0]["left_wrist_hip_dist"], 0.15, places=6)
        self.assertAlmostEqual(raw[0]["right_wrist_hip_dist"], 0.15, places=6)
        self.assertAlmostEqual(raw[0]["mean_wrist_hip_dist"], 0.15, places=6)

    def test_asymmetry_metrics_equal_their_knobs(self) -> None:
        from src.pose.movements.row import row_compute_raw

        raw = row_compute_raw(
            [row_frame(elbow_dy=0.07, shoulder_tilt=0.03, wrist_shift=0.04)], fps=30.0
        )
        self.assertAlmostEqual(raw[0]["elbow_height_asymmetry"], 0.07, places=6)
        self.assertAlmostEqual(raw[0]["shoulder_tilt"], 0.03, places=6)
        self.assertAlmostEqual(raw[0]["wrist_travel_asymmetry"], 0.04, places=6)

    def test_the_signed_elbow_delta_records_which_side_is_lower(self) -> None:
        from src.pose.movements.row import row_compute_raw

        # elbow_dy displaces the LEFT elbow DOWNWARD (image y grows down), so the signed delta
        # (left_y - right_y) must be POSITIVE and equal to the knob.
        raw = row_compute_raw([row_frame(elbow_dy=0.07)], fps=30.0)
        self.assertAlmostEqual(raw[0]["elbow_height_delta_signed"], 0.07, places=6)
        raw = row_compute_raw([row_frame(elbow_dy=-0.07)], fps=30.0)
        self.assertAlmostEqual(raw[0]["elbow_height_delta_signed"], -0.07, places=6)

    def test_shoulder_normalized_diagnostic_is_the_ratio_of_the_two(self) -> None:
        from src.pose.movements.row import row_compute_raw

        raw = row_compute_raw([row_frame(wrist_hip_dist=0.12)], fps=30.0)
        expected = raw[0]["mean_wrist_hip_dist"] / raw[0]["shoulder_width"]
        self.assertAlmostEqual(raw[0]["wrist_hip_dist_shoulder_norm"], expected, places=6)

    def test_one_missing_landmark_invalidates_the_whole_frame(self) -> None:
        from src.pose.movements.row import row_compute_raw

        frame = row_frame()
        frame["landmarks"][13] = _lm(0.5, 0.5, 0.10)  # left elbow below VISIBILITY_THRESHOLD
        raw = row_compute_raw([frame], fps=30.0)
        self.assertFalse(raw[0]["valid"])
        self.assertNotIn("trunk_angle_from_horizontal_deg", raw[0])

    def test_non_dict_frame_is_refused_rather_than_crashing(self) -> None:
        from src.pose.movements.row import row_compute_raw

        raw = row_compute_raw([None, "nonsense"], fps=30.0)
        self.assertEqual([item["valid"] for item in raw], [False, False])


class RowDerivativeTest(unittest.TestCase):
    def test_constant_velocity_gives_zero_acceleration(self) -> None:
        from src.pose.movements.row import row_compute_raw

        frames = [
            row_frame(wrist_hip_dist=0.05 + 0.01 * i, frame_index=i) for i in range(7)
        ]
        raw = row_compute_raw(frames, fps=30.0)
        for item in raw[2:5]:
            self.assertAlmostEqual(item["wrist_accel_norm"], 0.0, places=4)

    def test_boundary_frames_carry_nan_rather_than_a_one_sided_estimate(self) -> None:
        from src.pose.movements.row import row_compute_raw

        frames = [row_frame(wrist_hip_dist=0.05 + 0.01 * i, frame_index=i) for i in range(7)]
        raw = row_compute_raw(frames, fps=30.0)
        for index in (0, 1, 5, 6):
            self.assertTrue(math.isnan(raw[index]["wrist_accel_norm"]))

    def test_trunk_angle_speed_is_degrees_per_second(self) -> None:
        from src.pose.movements.row import row_compute_raw

        # 2 degrees per frame at 30 fps == 60 deg/s.
        frames = [row_frame(trunk_angle_deg=10.0 + 2.0 * i, frame_index=i) for i in range(5)]
        raw = row_compute_raw(frames, fps=30.0)
        self.assertAlmostEqual(raw[2]["trunk_angle_speed_deg_s"], 60.0, places=3)


class RowPhaseTest(unittest.TestCase):
    def test_phases_run_setup_pull_peak_lower(self) -> None:
        from src.pose.movements.row import row_assign_phases, row_compute_raw

        angles = [170.0] * 4 + [150.0, 120.0, 95.0, 70.0, 60.0, 70.0, 95.0, 120.0, 150.0, 170.0]
        frames = [row_frame(elbow_angle_deg=a, frame_index=i) for i, a in enumerate(angles)]
        phases = row_assign_phases(row_compute_raw(frames, fps=30.0))
        self.assertEqual(phases[0], "setup")
        self.assertEqual(phases[8], "peak")
        self.assertIn("pull", phases)
        self.assertIn("lower", phases)
        self.assertEqual(len(phases), len(frames))

    def test_empty_clip_returns_empty_phases(self) -> None:
        from src.pose.movements.row import row_assign_phases

        self.assertEqual(row_assign_phases([]), [])

    def test_clip_with_no_finite_signal_is_entirely_unknown(self) -> None:
        from src.pose.movements.row import row_assign_phases

        self.assertEqual(row_assign_phases([{"valid": False}, {"valid": False}]), ["unknown", "unknown"])

    def test_an_invalid_frame_inside_the_setup_window_is_unknown_not_setup(self) -> None:
        from src.pose.movements.row import row_assign_phases, row_compute_raw

        frames = [row_frame(elbow_angle_deg=170.0 - 5.0 * i, frame_index=i) for i in range(14)]
        frames[0]["landmarks"][15] = _lm(0.5, 0.5, 0.10)
        phases = row_assign_phases(row_compute_raw(frames, fps=30.0))
        self.assertEqual(phases[0], "unknown")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pose.movements.row'`

- [ ] **Step 3: Write `src/pose/movements/row.py`**

The module docstring carries three things the design spec requires be recorded in code: the degeneracy proof for the fifth rule, the one-dropped-landmark silence note, and the no-thresholds-in-the-metric-layer rule.

```python
# Row (bent-over barbell row) raw metrics and phase segmentation. Fault rules land in
# Tasks 2-5.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `row_compute_raw` / `row_assign_phases` compute
# per-frame quantities and a phase label only. Every number that decides anything belongs in a
# `rule_*` function in a later task. The only constant this module defines, `_DEGENERATE_LENGTH`,
# is a division-by-zero guard, never a tunable threshold.
#
# ---------------------------------------------------------------------------------------
# THE PARENT SPEC'S FIFTH ROW RULE CANNOT BE IMPLEMENTED, AND THIS IS THE PROOF.
# ---------------------------------------------------------------------------------------
# The parent spec (docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md,
# §Row) lists FIVE faults. Four are implemented here. `rounded_thoracolumbar_spine` is not,
# because its detection heuristic is geometrically degenerate under BOTH constructions it
# offers:
#
#   1. "three-point angle at mid-spine using shoulder-midpoint(11,12), a synthesized mid-trunk
#      point = 0.5*(shoulder_mid + hip_mid), and hip-midpoint(23,24)" -- the middle point is BY
#      CONSTRUCTION the midpoint of the segment joining the other two. Three collinear points
#      subtend exactly 180 degrees on every frame of every video. The metric is a constant.
#   2. "Flag flexion if the shoulder-midpoint drops below the straight shoulder-hip line by a
#      normalized sag > 0.04" -- shoulder_mid is an ENDPOINT of that line. Its distance to a
#      line passing through itself is identically zero. The threshold can never be crossed.
#
# The root cause is not a wording slip: MediaPipe Pose has NO thoracic or lumbar landmark, so
# there is no measured point anywhere between the shoulders and the hips, and no sag,
# curvature or three-point spinal angle is computable from this detection model by any
# construction. The spec wrote a proxy requiring a landmark its own detection model (§3) does
# not provide.
#
# NOT SUBSTITUTED, DELIBERATELY. Two monocular signals do carry some trunk-shape information --
# trunk-length foreshortening (dist(shoulder_mid, hip_mid) shrinking as the spine flexes) and
# ear-drop relative to the trunk line. Both are confounded by camera distance and by the hinge
# angle itself, and NEITHER is what the rule's citation (Saeterbakken PMID 26134664, an
# erector-spinae EMG MAGNITUDE result) supports. Shipping either under the spec's fault_id
# would attach a real citation to a metric that citation says nothing about, which is exactly
# the fabrication this project's anti-hallucination rule forbids. Precedent for carrying the
# gap instead: `pushup.rule_scapular_winging`, permanently silent for a weaker reason (a
# view-gate accident rather than a geometric impossibility).
#
# The knowledge graph is NOT the gap: `Row:Trunk Flexion` resolves with a non-empty
# `corrections` bucket ("Maintain Neutral Spine"). The metric is the gap.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY ROW RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both shoulders, both elbows, both wrists and both hips. If
# `visible_point` drops any ONE of them the frame is marked `valid=False` and carries no
# metric keys at all, so every rule that masks on `frame.valid` goes silent for that frame,
# not just the one whose input landmark went missing. This mirrors `pushup_compute_raw`,
# `ohp_compute_raw` and `lunge_compute_raw`: an unmeasurable frame is refused wholesale rather
# than degraded, because a silently-wrong verdict is worse than no verdict.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, distance,
)

# Defined locally, matching overhead_press.py: geometry.py exports only the lower-body and
# shoulder/hip constants.
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. The name is squat-centric and carries awkwardly for a
# hinged upper-body pull, exactly as it does for OHP and push-up; Row's own rules never consume
# it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

ROW_METRIC_KEYS: tuple[str, ...] = (
    "left_elbow_angle",
    "right_elbow_angle",
    "min_elbow_angle",
    "max_elbow_angle",
    "trunk_angle_from_horizontal_deg",
    "left_wrist_hip_dist",
    "right_wrist_hip_dist",
    "mean_wrist_hip_dist",
    "wrist_hip_dist_shoulder_norm",
    "elbow_height_asymmetry",
    "elbow_height_delta_signed",
    "shoulder_tilt",
    "wrist_travel_asymmetry",
    "wrist_accel_norm",
    "trunk_angle_speed_deg_s",
    "shoulder_width",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard
# value pushup.py, overhead_press.py and lunge.py use; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6


def _derivative(values: Sequence[float], fps: float) -> list[float]:
    """Central-difference time derivative, NaN at both boundaries.

    ONE-SIDED BOUNDARY ESTIMATES ARE REFUSED ON PURPOSE. A forward difference at frame 0 and a
    central difference at frame 1 have different biases; mixing them into one series makes the
    first samples systematically unlike the rest, and `rule_momentum_jerk` compares a PEAK
    against a MEDIAN of exactly this series. NaN propagates through the mask and the frame is
    simply not scored.

    A NaN input (an invalid frame) poisons its two neighbours' derivatives, which is correct:
    a derivative across a hole in the data is not measured, it is guessed.
    """
    count = len(values)
    out = [float(np.nan)] * count
    if fps <= 0 or count < 3:
        return out
    arr = np.asarray(values, dtype=np.float64)
    for index in range(1, count - 1):
        before, after = arr[index - 1], arr[index + 1]
        if np.isfinite(before) and np.isfinite(after):
            out[index] = float((after - before) * fps / 2.0)
    return out


def row_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    wrist_mid_x: list[float] = []
    wrist_mid_y: list[float] = []
    trunk_angles: list[float] = []

    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            wrist_mid_x.append(np.nan)
            wrist_mid_y.append(np.nan)
            trunk_angles.append(np.nan)
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        required = (
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_ELBOW, RIGHT_ELBOW,
            LEFT_WRIST, RIGHT_WRIST,
            LEFT_HIP, RIGHT_HIP,
        )
        valid = all(visible_point(points, index, dims=2) is not None for index in required)
        if not valid:
            raw.append(
                {
                    "frame_index": frame_index,
                    "time": time,
                    "valid": False,
                    "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                }
            )
            wrist_mid_x.append(np.nan)
            wrist_mid_y.append(np.nan)
            trunk_angles.append(np.nan)
            continue

        left_elbow_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        finite_elbows = [v for v in (left_elbow_angle, right_elbow_angle) if np.isfinite(v)]
        min_elbow_angle = float(min(finite_elbows)) if finite_elbows else np.nan
        max_elbow_angle = float(max(finite_elbows)) if finite_elbows else np.nan

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        # BOTH COMPONENTS ABSOLUTE, and that is the whole point: |dx| makes the angle
        # independent of which way the subject faces, |dy| of which point is higher in the
        # image. A signed form would flip by 180 degrees when the lifter turns around, and the
        # torso-rising test would then mean the opposite thing for the other facing. In a
        # bent-over row the shoulders stay above the hips throughout, so no real sign
        # information is discarded. Same reasoning `lunge_compute_raw` applies to its |dx|.
        if shoulder_mid is not None and hip_mid is not None:
            trunk_dx = abs(float(hip_mid[0] - shoulder_mid[0]))
            trunk_dy = abs(float(hip_mid[1] - shoulder_mid[1]))
            trunk_angle = (
                float(np.degrees(np.arctan2(trunk_dy, trunk_dx)))
                if trunk_dx > _DEGENERATE_LENGTH or trunk_dy > _DEGENERATE_LENGTH
                else np.nan
            )
        else:
            trunk_angle = np.nan

        left_wrist_hip = distance(points, LEFT_WRIST, LEFT_HIP)
        right_wrist_hip = distance(points, RIGHT_WRIST, RIGHT_HIP)
        finite_dists = [v for v in (left_wrist_hip, right_wrist_hip) if np.isfinite(v)]
        mean_wrist_hip = float(np.mean(finite_dists)) if finite_dists else np.nan
        wrist_travel_asymmetry = (
            abs(left_wrist_hip - right_wrist_hip)
            if np.isfinite(left_wrist_hip) and np.isfinite(right_wrist_hip)
            else np.nan
        )

        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER)
        shoulder_norm = (
            mean_wrist_hip / shoulder_width
            if np.isfinite(mean_wrist_hip)
            and np.isfinite(shoulder_width)
            and shoulder_width > _DEGENERATE_LENGTH
            else np.nan
        )

        left_elbow = visible_point(points, LEFT_ELBOW, dims=2)
        right_elbow = visible_point(points, RIGHT_ELBOW, dims=2)
        left_shoulder = visible_point(points, LEFT_SHOULDER, dims=2)
        right_shoulder = visible_point(points, RIGHT_SHOULDER, dims=2)
        # SIGNED companion, positive when the LEFT elbow sits LOWER in the image (larger y).
        # `rule_asymmetric_pull` needs the DIRECTION for its coaching cue and an absolute value
        # cannot supply it; the absolute one stays because that is the quantity the spec states
        # its 0.05 threshold on.
        elbow_height_delta_signed = float(left_elbow[1] - right_elbow[1])
        elbow_height_asymmetry = abs(elbow_height_delta_signed)
        shoulder_tilt = abs(float(left_shoulder[1] - right_shoulder[1]))

        left_wrist = visible_point(points, LEFT_WRIST, dims=2)
        right_wrist = visible_point(points, RIGHT_WRIST, dims=2)
        wrist_mid_x.append(float((left_wrist[0] + right_wrist[0]) / 2.0))
        wrist_mid_y.append(float((left_wrist[1] + right_wrist[1]) / 2.0))
        trunk_angles.append(trunk_angle)

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "min_elbow_angle": min_elbow_angle,
                "max_elbow_angle": max_elbow_angle,
                "trunk_angle_from_horizontal_deg": trunk_angle,
                "left_wrist_hip_dist": left_wrist_hip,
                "right_wrist_hip_dist": right_wrist_hip,
                "mean_wrist_hip_dist": mean_wrist_hip,
                "wrist_hip_dist_shoulder_norm": shoulder_norm,
                "elbow_height_asymmetry": elbow_height_asymmetry,
                "elbow_height_delta_signed": elbow_height_delta_signed,
                "shoulder_tilt": shoulder_tilt,
                "wrist_travel_asymmetry": wrist_travel_asymmetry,
                "shoulder_width": shoulder_width,
            }
        )

    # DERIVATIVES ARE COMPUTED HERE, IN THE METRIC LAYER, AND THAT IS LOAD-BEARING.
    # `run_detector` median-filters EVERY key in `metric_keys` with a 5-frame window. A median
    # over a POSITION series flattens the acceleration transient `rule_momentum_jerk` exists to
    # find, before the rule ever sees it. Emitting the derivative as the metric means the
    # framework's filter acts on the acceleration -- a defensible low-pass on the quantity of
    # interest instead of an erasure of it. Task 4 pins that a 1-3 frame spike survives.
    accel_x = _derivative(_derivative(wrist_mid_x, fps), fps)
    accel_y = _derivative(_derivative(wrist_mid_y, fps), fps)
    trunk_speed = _derivative(trunk_angles, fps)
    for index, item in enumerate(raw):
        if not item.get("valid"):
            continue
        ax, ay = accel_x[index], accel_y[index]
        item["wrist_accel_norm"] = (
            float(np.hypot(ax, ay)) if np.isfinite(ax) and np.isfinite(ay) else float(np.nan)
        )
        speed = trunk_speed[index]
        item["trunk_angle_speed_deg_s"] = abs(float(speed)) if np.isfinite(speed) else float(np.nan)
    return raw


def row_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> pull -> peak -> lower, segmented on `min_elbow_angle`.

    Mirrors `ohp_assign_phases` and `lunge_assign_phases`, substituting the row's pull depth
    signal. "Return" is not a separate label: after the peak the arms extend and those frames
    are `lower`, the same reduction OHP makes for the press's return. Same fallbacks: an empty
    clip returns an empty list, a clip with no finite signal is entirely `unknown`, and an
    invalid frame is `unknown` regardless of where it sits (the validity check precedes the
    setup cutoff, so an occluded frame in the opening 15% is NOT labelled `setup`).
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    elbow_values = np.asarray(
        [float(item.get("min_elbow_angle", np.nan)) for item in raw], dtype=np.float32
    )
    valid_elbow = elbow_values[np.isfinite(elbow_values)]
    if valid_elbow.size == 0:
        return ["unknown" for _ in raw]

    # The most-flexed 30% of the rep is the peak hold.
    peak_threshold = float(np.percentile(valid_elbow, 30))
    deepest_index = int(np.nanargmin(np.where(np.isfinite(elbow_values), elbow_values, np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = elbow_values[index]
        if np.isfinite(value) and value <= peak_threshold:
            phases.append("peak")
        elif index < deepest_index:
            phases.append("pull")
        else:
            phases.append("lower")
    return phases
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py -q`
Expected: PASS (all of `RowMetricsTest`, `RowDerivativeTest`, `RowPhaseTest`)

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/row.py tests/test_row.py
git commit -m "feat(pose): row raw metrics and phases, and the proof one spec rule is impossible"
```

---

### Task 2: `rule_torso_rising` and the shared setup baseline

**Files:**
- Modify: `src/pose/movements/row.py`
- Test: `tests/test_row.py`

**Interfaces:**
- Consumes: Task 1's `ROW_METRIC_KEYS`, `row_compute_raw`, `row_assign_phases`.
- Produces:
  - `_setup_baseline(core: list[CoreFrame], key: str) -> float` — median of `key` over the window's valid `setup` frames; `NaN` when there are none. Task 5 reuses it.
  - `rule_torso_rising(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]`
  - `TRUNK_RISE_MILD_DEG = 15.0`, `TRUNK_RISE_SEVERE_DEG = 37.5`
  - `TRUNK_OBSERVABLE_VIEWS = {"side", "front_oblique", "rear_oblique"}`, `_OFF_VIEW_CONFIDENCE`
  - `ROW_TORSO_RISING_KG_QUERY = "Trunk Extension"`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_row.py`:

```python
def _row_clip(
    pull_frames: int = 10,
    setup_trunk: float = 20.0,
    peak_trunk: float = 20.0,
    setup_tilt: float = 0.0,
    peak_tilt: float = 0.0,
    peak_wrist_hip: float = 0.05,
    peak_elbow: float = 70.0,
    peak_elbow_dy: float = 0.0,
) -> list[dict]:
    """A synthetic single rep: 6 setup frames, then a descent into a held peak.

    CONSTANT-VALUE SEGMENTS ARE THE POINT. `run_detector` median-filters every metric with a
    5-frame window; a segment held at one value makes that filter a no-op, so an asserted
    severity is EXACT rather than approximately-whatever-the-filter-left. The OHP review found
    5 of 10 threshold mutants surviving because every fixture sat at an extreme instead of on
    a boundary, so boundary fixtures must be exact.
    """
    frames: list[dict] = []
    index = 0
    for _ in range(6):
        frames.append(
            row_frame(
                trunk_angle_deg=setup_trunk,
                shoulder_tilt=setup_tilt,
                elbow_angle_deg=170.0,
                wrist_hip_dist=0.30,
                frame_index=index,
            )
        )
        index += 1
    for _ in range(pull_frames):
        frames.append(
            row_frame(
                trunk_angle_deg=peak_trunk,
                shoulder_tilt=peak_tilt,
                elbow_angle_deg=peak_elbow,
                elbow_dy=peak_elbow_dy,
                wrist_hip_dist=peak_wrist_hip,
                frame_index=index,
            )
        )
        index += 1
    return frames


def _run_rule(rule, frames: list[dict], view_type: str = "rear_oblique", view_confidence: float = 0.8):
    """Run ONE rule over a clip, bypassing rep segmentation.

    Rules receive a per-rep slice from `run_detector`; here the whole clip IS the window, which
    is the `only_partial_reps` fallback shape and is what a single-rep fixture should exercise.
    """
    from src.pose.movements.base import CoreFrame, RuleContext
    from src.pose.movements.row import ROW_METRIC_KEYS, row_assign_phases, row_compute_raw

    raw = row_compute_raw(frames, fps=30.0)
    phases = row_assign_phases(raw)
    core = [
        CoreFrame(
            frame_index=int(item.get("frame_index", i) or i),
            time=float(item.get("time", 0.0) or 0.0),
            phase=phases[i],
            valid=bool(item.get("valid", False)),
            lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
            metrics={key: float(item.get(key, np.nan)) for key in ROW_METRIC_KEYS},
        )
        for i, item in enumerate(raw)
    ]
    ctx = RuleContext(fps=30.0, view_type=view_type, view_confidence=view_confidence, min_frames=6)
    return rule(core, ctx)


class RowTorsoRisingTest(unittest.TestCase):
    def test_a_torso_held_at_the_setup_angle_does_not_fire(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        self.assertEqual(_run_rule(rule_torso_rising, _row_clip(setup_trunk=20.0, peak_trunk=20.0)), [])

    def test_just_under_fifteen_degrees_of_rise_does_not_fire(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        clip = _row_clip(setup_trunk=20.0, peak_trunk=34.9)
        self.assertEqual(_run_rule(rule_torso_rising, clip), [])

    def test_just_over_fifteen_degrees_of_rise_fires(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        clip = _row_clip(setup_trunk=20.0, peak_trunk=35.1)
        detections = _run_rule(rule_torso_rising, clip)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "row_torso_rising")

    def test_severity_is_exact_at_the_ramp_midpoint(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        # Ramp 15 -> 37.5; a 26.25-degree rise is exactly half way.
        clip = _row_clip(setup_trunk=20.0, peak_trunk=46.25)
        detections = _run_rule(rule_torso_rising, clip)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].severity, 0.5, places=3)

    def test_severity_saturates_at_the_ramp_end(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        clip = _row_clip(setup_trunk=10.0, peak_trunk=60.0)
        detections = _run_rule(rule_torso_rising, clip)
        self.assertAlmostEqual(detections[0].severity, 1.0, places=6)

    def test_an_off_axis_view_downgrades_rather_than_silencing(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        clip = _row_clip(setup_trunk=20.0, peak_trunk=46.25)
        front = _run_rule(rule_torso_rising, clip, view_type="front")
        self.assertEqual(len(front), 1)
        self.assertEqual(front[0].observability, "medium")
        oblique = _run_rule(rule_torso_rising, clip, view_type="rear_oblique")
        self.assertEqual(oblique[0].observability, "high")
        self.assertLess(front[0].confidence, oblique[0].confidence)

    def test_a_window_with_no_setup_frames_emits_nothing(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        clip = _row_clip(setup_trunk=20.0, peak_trunk=60.0)
        for frame in clip[:6]:
            frame["landmarks"][11] = _lm(0.5, 0.5, 0.10)
        self.assertEqual(_run_rule(rule_torso_rising, clip), [])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py -q -k TorsoRising`
Expected: FAIL — `ImportError: cannot import name 'rule_torso_rising'`

- [ ] **Step 3: Implement the baseline helper and the rule**

Add to `src/pose/movements/row.py` (imports first — extend the existing import block):

```python
from src.pose.geometry import contiguous_true_segments, severity_from_range
from src.pose.movements.base import CoreFrame, RuleContext
from src.pose.pose_rule_detector import (
    VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
    PoseRuleDetection,
    build_detection,
)
```

```python
# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query,
# movement="Row")` -- the function PRODUCTION calls, not just `resolve_nodes` -- and returned a
# `Row:`-scoped seed with at least one NON-EMPTY bucket. OHP shipped three dangling queries
# because only `resolve_nodes` was checked; this is the check that would have caught them.
#
#   Trunk Extension            -> Row:Trunk Extension (Fault)
#                                 phases; corrections=[Maintain Neutral Spine];
#                                 quality_impacts=[Core Stability]
#   Scapular Protraction       -> Row:Scapular Protraction (Fault)
#                                 evidence=[Anterior Translation Of Scapulae]; related_actions
#   Loss Of Neutral Body       -> Row:Loss Of Neutral Body Position (Fault)
#     Position                    phases; evidence=[Head/Trunk/Hip Not Aligned ...];
#                                 corrections; quality_impacts; related_actions
#   Asymmetry                  -> Row:Asymmetry (Fault)
#                                 phases; risks=[Shoulder Injury, Injury Risk]; related_actions
#
# TWO DELIBERATE DEVIATIONS from the obvious name, load-bearing for later tasks that import
# these constants blind:
#   - Momentum: "Compensatory Movements" is a real `Row:`-scoped Fault node whose buckets are
#     ENTIRELY EMPTY -- precisely the OHP failure mode. "Loss Of Neutral Body Position" is the
#     richest on-topic node, and its three evidence signals ("Head Not Aligned With Trunk And
#     Hip", "Trunk Not Aligned With Head And Hip", "Hip Not Aligned With Head And Trunk") are a
#     direct description of the whole-body heave this fault is about.
#   - Asymmetry: "Interlimb Asymmetry" resolves but is scoped to `Unilateral Cable Row`, and
#     "Muscle Strength Asymmetry" carries only a generic `Injury Risk`. `Row:Asymmetry` is the
#     one whose buckets name both the phases the fault occurs in and a specific Shoulder Injury
#     risk.
ROW_TORSO_RISING_KG_QUERY = "Trunk Extension"
ROW_INCOMPLETE_ROM_KG_QUERY = "Scapular Protraction"
ROW_MOMENTUM_KG_QUERY = "Loss Of Neutral Body Position"
ROW_ASYMMETRY_KG_QUERY = "Asymmetry"

# Confidence multiplier applied when a rule fires from a view the spec does not rate `high`.
# Not a new number: aliases the shared constant rather than re-typing its value, so a future
# change to it cannot silently skip this module.
_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE

# Views with a lateral component, in which the parent spec rates trunk pitch and pull depth
# `high` ("side / front_oblique / rear_oblique ... Low from pure front/rear").
TRUNK_OBSERVABLE_VIEWS = {"side", "front_oblique", "rear_oblique"}

# FROM THE SPEC: "Flag if `trunk_angle_peak - trunk_angle_setup > 15deg`".
TRUNK_RISE_MILD_DEG = 15.0
# RULE-LEVEL CHOICE MADE HERE. The parent spec states NO severity ramp for ANY Row fault (the
# Lunge section states its ramps explicitly, so the absence is meaningful rather than a
# formatting quirk). 37.5 is 2.5x the fire threshold, the convention `pushup.rule_hip_sag`
# already uses for exactly this situation (ramp 0.06 -> 0.15). Treat it as a display/ranking
# curve, not a cited quantity.
TRUNK_RISE_SEVERE_DEG = 37.5


def _setup_baseline(core: list[CoreFrame], key: str) -> float:
    """Median of `key` over this window's valid `setup` frames; NaN when there are none.

    WHY THE BASELINE LIVES IN THE RULES AND NOT IN `row_compute_raw` -- the Row analogue of
    lunge's lead-leg problem. Three of the parent spec's five Row heuristics are deltas from a
    setup baseline, and a baseline is a PER-REP reduction. `run_detector` calls `compute_raw`
    over the WHOLE CLIP before `segment_reps`, so at metric time no rep boundary exists and
    there is no "this rep's setup" to reduce against. Rules receive a per-rep slice, which is
    the first place the question is answerable.

    MEDIAN, NOT MEAN, so one bad frame in a six-frame setup cannot move the reference every
    later comparison is made against.

    NO BASELINE MEANS SILENCE, never a guessed one: an occluded setup returns NaN and the
    caller emits nothing. Stated cost of the per-rep scope: a lifter who is ALREADY rounded or
    rotated at this rep's setup reads as clean. A clip-level baseline would catch that but
    would make rep N's verdict depend on rep 1's frames, which this architecture deliberately
    does not do.
    """
    values = [
        frame.m(key)
        for frame in core
        if frame.valid and frame.phase == "setup" and np.isfinite(frame.m(key))
    ]
    if not values:
        return float(np.nan)
    return float(np.median(values))


def rule_torso_rising(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the trunk drifting from its hinged setup angle toward upright across the pull.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLD 15 deg: FROM THE SPEC ("Flag if trunk_angle_peak - trunk_angle_setup >
      15deg").
      SEVERITY RAMP 15 -> 37.5 deg: A RULE-LEVEL CHOICE (see TRUNK_RISE_SEVERE_DEG).

    PHASE SCOPE `peak`, FROM THE SPEC's own wording ("at setup baseline and at peak pull") --
    not a rule-level call and not a shared ACTIVE_PHASES set, of which this module defines
    none: every Row heuristic names its own phase, so a shared set would be a constant every
    rule overrides.

    OBSERVABILITY DOWNGRADE, NOT A GATE. The spec rates this `high` on side/oblique and low
    from pure front/rear, but a hard gate would likely ship this rule SILENT: the production
    path calls `estimate_view_for_pose(allow_front=False)`, so the reachable labels are
    {side, rear, rear_oblique, unknown}, and across the 45 real pose JSONs in this repository
    the estimator emitted `side` exactly ONCE (from a fixture since removed) against 30
    `rear_oblique` and 13 `rear`. `rear_oblique` supplies the lateral component this rule
    needs, so it earns the spec's `high`; everything else downgrades to `medium` and takes the
    x0.65 discount, following `squat.rule_knees_inward` rather than `rule_knees_forward`.
    """
    baseline = _setup_baseline(core, "trunk_angle_from_horizontal_deg")
    if not np.isfinite(baseline):
        return []
    observable = ctx.view_type in TRUNK_OBSERVABLE_VIEWS

    mask = [
        frame.valid
        and frame.phase == "peak"
        and np.isfinite(frame.m("trunk_angle_from_horizontal_deg"))
        and (frame.m("trunk_angle_from_horizontal_deg") - baseline) > TRUNK_RISE_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        rises = [frame.m("trunk_angle_from_horizontal_deg") - baseline for frame in segment]
        max_rise = float(np.nanmax(rises))
        severity = severity_from_range(
            max_rise, TRUNK_RISE_MILD_DEG, TRUNK_RISE_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="row_torso_rising",
                fault_name="Torso Rising (Loss of Hip-Hinge)",
                kg_query=ROW_TORSO_RISING_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=rises,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "setup_trunk_angle_deg": round(baseline, 2),
                    "max_trunk_rise_deg": round(max_rise, 2),
                    "threshold": TRUNK_RISE_MILD_DEG,
                    "primary_label": "torso rise vs setup",
                    "primary_value": round(max_rise, 2),
                    "primary_threshold": TRUNK_RISE_MILD_DEG,
                },
                citation="Saeterbakken A, et al. Int J Sports Med (2015), PMID 26134664. "
                         "Supplemented by Owens LP, et al. Int J Sports Phys Ther (2026), "
                         "PMC13232157.",
                citation_support="Saeterbakken: the free-weight bent-over row produced greater "
                                 "erector spinae EMG than the machine row both bilaterally and "
                                 "unilaterally — the hinged free-weight row imposes a high, "
                                 "sustained trunk-extensor stabilizing demand that a rising "
                                 "torso abandons. Owens: breaks in efficient kinetic-chain "
                                 "sequencing \"require distal segments to increase functional "
                                 "capacity … described as the 'catch-up' phenomenon,\" and the "
                                 "protocol uses a trunk-parallel-to-floor position specifically "
                                 "to control trunk posture during rowing.",
            )
        )
    return detections
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/row.py tests/test_row.py
git commit -m "feat(pose): row torso-rising rule, with the baseline where the rep boundary is"
```

---

### Task 3: `rule_incomplete_rom`

**Files:**
- Modify: `src/pose/movements/row.py`
- Test: `tests/test_row.py`

**Interfaces:**
- Consumes: Task 2's `_OFF_VIEW_CONFIDENCE`, `TRUNK_OBSERVABLE_VIEWS`, `ROW_INCOMPLETE_ROM_KG_QUERY`.
- Produces: `rule_incomplete_rom`, `PULL_DEPTH_MILD = 0.12`, `PULL_DEPTH_SEVERE = 0.30`, `PEAK_ELBOW_MILD_DEG = 100.0`, `PEAK_ELBOW_SEVERE_DEG = 140.0`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_row.py`:

```python
class RowIncompleteRomTest(unittest.TestCase):
    def test_a_full_pull_does_not_fire(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        clip = _row_clip(peak_wrist_hip=0.05, peak_elbow=70.0)
        self.assertEqual(_run_rule(rule_incomplete_rom, clip), [])

    def test_just_inside_both_thresholds_does_not_fire(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        clip = _row_clip(peak_wrist_hip=0.119, peak_elbow=99.0)
        self.assertEqual(_run_rule(rule_incomplete_rom, clip), [])

    def test_a_short_pull_distance_alone_fires(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        clip = _row_clip(peak_wrist_hip=0.121, peak_elbow=70.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "row_incomplete_rom")
        self.assertEqual(detections[0].evidence["fired_on"], "pull_distance")

    def test_an_unbent_elbow_alone_fires(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        clip = _row_clip(peak_wrist_hip=0.05, peak_elbow=101.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].evidence["fired_on"], "elbow_angle")

    def test_severity_is_exact_at_the_distance_ramp_midpoint(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        # Ramp 0.12 -> 0.30; 0.21 is exactly half way.
        clip = _row_clip(peak_wrist_hip=0.21, peak_elbow=70.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertAlmostEqual(detections[0].severity, 0.5, places=3)

    def test_severity_is_exact_at_the_elbow_ramp_midpoint(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        # Ramp 100 -> 140; 120 is exactly half way.
        clip = _row_clip(peak_wrist_hip=0.05, peak_elbow=120.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertAlmostEqual(detections[0].severity, 0.5, places=3)

    def test_the_worse_of_the_two_conditions_sets_the_severity(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        # distance 0.21 -> 0.5; elbow 130 -> 0.75. The larger must win.
        clip = _row_clip(peak_wrist_hip=0.21, peak_elbow=130.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertAlmostEqual(detections[0].severity, 0.75, places=3)
        self.assertEqual(detections[0].evidence["fired_on"], "both")

    def test_it_reads_the_less_flexed_arm(self) -> None:
        """The conservative reading: a rep is incomplete if EITHER arm fell short."""
        from src.pose.movements.row import row_compute_raw, rule_incomplete_rom

        clip = _row_clip(peak_wrist_hip=0.05, peak_elbow=90.0, peak_elbow_dy=0.06)
        raw = row_compute_raw(clip, fps=30.0)
        peak = raw[-1]
        self.assertGreater(peak["max_elbow_angle"], 100.0)
        self.assertLess(peak["min_elbow_angle"], 100.0)
        self.assertEqual(len(_run_rule(rule_incomplete_rom, clip)), 1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py -q -k IncompleteRom`
Expected: FAIL — `ImportError: cannot import name 'rule_incomplete_rom'`

- [ ] **Step 3: Implement the rule**

```python
# FROM THE SPEC: "(a) Pull depth: minimum normalized distance from wrist(15/16) to hip(23/24)
# … flag if `min_wrist_to_torso_dist > 0.12`. (b) Elbow flexion at peak: `elbow_angle … > 100deg`
# at the top = pull not completed."
PULL_DEPTH_MILD = 0.12
PEAK_ELBOW_MILD_DEG = 100.0
# RULE-LEVEL CHOICE MADE HERE, both of them. The spec states no ramp for this fault.
#   0.30 is 2.5x the fire threshold -- `pushup.rule_hip_sag`'s convention.
#   140 deg is NOT re-derived: it is taken verbatim from `pushup.rule_shallow_depth`, whose
#   ramp is also 100 -> 140 on the very same quantity (an elbow angle whose fire threshold is
#   100). Copying it keeps the codebase's two elbow-ROM ramps from drifting apart, and inherits
#   that rule's stated argument: an elbow that never bends past 140 has travelled well under
#   half the useful range, which is where "maximally incomplete" reasonably saturates. That is
#   an argument, not a measurement.
PULL_DEPTH_SEVERE = 0.30
PEAK_ELBOW_SEVERE_DEG = 140.0


def rule_incomplete_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a pull that stops short -- the hands never reach the torso, or the elbows never bend.

    THRESHOLD PROVENANCE: fire thresholds 0.12 and 100 deg are FROM THE SPEC; both severity
    ramps are RULE-LEVEL (see the constants above).

    TWO OR'd CONDITIONS, ONE FAULT, per the spec's own (a)/(b) structure. A frame qualifies if
    either holds; the segment's severity is the WORSE of the two sub-severities, and
    `evidence["fired_on"]` records which one(s) drove it, because the coaching cue differs
    ("pull the bar all the way to the abdomen" vs "finish the elbow bend").

    `score_values` is the per-frame MAXIMUM of the two sub-severities rather than either raw
    metric, so `build_detection` nominates the frame that was worst OVERALL. Passing one raw
    series would let a frame that was fine on that metric but terrible on the other be named
    the peak.

    IT READS `max_elbow_angle`, THE LESS-FLEXED ARM, AND THAT IS A RULE-LEVEL READING OF AN
    UNDER-SPECIFIED SPEC LINE. The spec's condition (b) names no side. Taking the less-flexed
    arm is conservative -- a rep is incomplete if EITHER arm fell short -- and is the deliberate
    opposite of `pushup_shallow_depth`, whose docstring already flags its inherited more-flexed
    reading as the generous one.

    THE SPEC'S THRESHOLDS ARE IN RAW IMAGE UNITS, WHICH IS CAMERA-DISTANCE DEPENDENT. 0.12
    carries no stated normalizer, and the same spec says "normalized by shoulder width
    dist(11,12)" explicitly where it means that (Band Pull Apart), so the absence here is
    meaningful. Implemented as written; the same rep filmed further away yields a smaller body,
    smaller distances, and less firing. `wrist_hip_dist_shoulder_norm` is emitted as a
    SCALE-FREE DIAGNOSTIC that nothing fires on, so a future validation can compare the two
    readings without any threshold having been moved in the meantime.

    PHASE SCOPE `peak`, from the spec ("at the top"), and the same downgrade-not-gate view
    handling `rule_torso_rising` documents.
    """
    observable = ctx.view_type in TRUNK_OBSERVABLE_VIEWS

    def _sub_severities(frame: CoreFrame) -> tuple[float, float]:
        distance_value = frame.m("mean_wrist_hip_dist")
        elbow_value = frame.m("max_elbow_angle")
        distance_severity = (
            severity_from_range(distance_value, PULL_DEPTH_MILD, PULL_DEPTH_SEVERE, lower_is_worse=False)
            if np.isfinite(distance_value) and distance_value > PULL_DEPTH_MILD
            else 0.0
        )
        elbow_severity = (
            severity_from_range(elbow_value, PEAK_ELBOW_MILD_DEG, PEAK_ELBOW_SEVERE_DEG, lower_is_worse=False)
            if np.isfinite(elbow_value) and elbow_value > PEAK_ELBOW_MILD_DEG
            else 0.0
        )
        return distance_severity, elbow_severity

    mask = [
        frame.valid and frame.phase == "peak" and max(_sub_severities(frame)) > 0.0
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        pairs = [_sub_severities(frame) for frame in segment]
        scores = [max(pair) for pair in pairs]
        severity = float(np.nanmax(scores))
        distance_fired = any(pair[0] > 0.0 for pair in pairs)
        elbow_fired = any(pair[1] > 0.0 for pair in pairs)
        fired_on = (
            "both" if distance_fired and elbow_fired else "pull_distance" if distance_fired else "elbow_angle"
        )
        max_distance = float(np.nanmax([frame.m("mean_wrist_hip_dist") for frame in segment]))
        max_elbow = float(np.nanmax([frame.m("max_elbow_angle") for frame in segment]))
        detections.append(
            build_detection(
                fault_id="row_incomplete_rom",
                fault_name="Incomplete ROM (Pull Not Completed)",
                kg_query=ROW_INCOMPLETE_ROM_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=scores,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "fired_on": fired_on,
                    "max_wrist_hip_dist": round(max_distance, 4),
                    "max_peak_elbow_angle_deg": round(max_elbow, 2),
                    "wrist_hip_dist_shoulder_norm": round(
                        float(np.nanmax([frame.m("wrist_hip_dist_shoulder_norm") for frame in segment])), 4
                    ),
                    "distance_threshold": PULL_DEPTH_MILD,
                    "elbow_threshold": PEAK_ELBOW_MILD_DEG,
                    "primary_label": "wrist-to-hip distance at peak"
                    if fired_on != "elbow_angle"
                    else "elbow angle at peak",
                    "primary_value": round(max_distance, 4) if fired_on != "elbow_angle" else round(max_elbow, 2),
                    "primary_threshold": PULL_DEPTH_MILD if fired_on != "elbow_angle" else PEAK_ELBOW_MILD_DEG,
                },
                citation="Fischer J, et al. J Electromyogr Kinesiol (2025), PMID 40513198. "
                         "Supplemented by Padovan R, et al. J Funct Morphol Kinesiol (2025), "
                         "PMC12821611.",
                citation_support="Fischer (prone barbell row, 3 ROMs): \"The LD showed "
                                 "significantly higher mean muscle excitation in the upper-half "
                                 "ROM compared to both the lower-half ROM (p < 0.001) and full "
                                 "ROM (p < 0.001)\" — the top of the pull drives peak lat "
                                 "excitation. Padovan: the row is driven by \"scapular "
                                 "retraction, external rotation, and posterior tilt [which] "
                                 "contributes to optimizing glenohumeral alignment and force "
                                 "transmission,\" with the concentric endpoint \"defined when "
                                 "the handle reached the abdominal target.\"",
            )
        )
    return detections
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/row.py tests/test_row.py
git commit -m "feat(pose): row incomplete-ROM rule, reading the arm that fell short"
```

---

### Task 4: `rule_momentum_jerk` — the event rule

**Files:**
- Modify: `src/pose/movements/row.py`
- Test: `tests/test_row.py`

**Interfaces:**
- Consumes: Task 2's `_OFF_VIEW_CONFIDENCE`, `ROW_MOMENTUM_KG_QUERY`.
- Produces: `rule_momentum_jerk`, `JERK_RATIO_MILD = 3.0`, `JERK_RATIO_SEVERE = 7.5`, `_DEGENERATE_ACCEL = 1e-4`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_row.py`:

```python
_JERK_PULL_FRAMES = 26
_JERK_TOTAL_FRAMES = 38
_JERK_BURST_AT = 10
_JERK_BURST_WIDTH = 1.2


def _jerk_clip(burst_amplitude: float) -> list[dict]:
    """A smooth bell-shaped pull, optionally with one injected acceleration burst.

    THE BASELINE MUST NOT BE LINEAR, and this was established by measurement, not taste. A
    constant-velocity wrist path has zero acceleration everywhere, so the rule's median over the
    pull is ~1e-5 -- below `_DEGENERATE_ACCEL` -- and `rule_momentum_jerk` returns [] via the
    degenerate guard on EVERY such clip, spike or no spike. A fixture built that way tests the
    guard and nothing else. The cosine-ease travel below gives a half-sine velocity and a
    genuinely nonzero acceleration median, which is what puts the ratio test in play at all.

    MEASURED BEHAVIOUR OF THIS FIXTURE, after `run_detector`'s 5-frame median filter, with the
    median taken over `pull` frames exactly as the rule takes it:

        burst_amplitude   peak/median ratio   severity   fired-frame span
        0.00 (clean)               1.56          --          silent
        0.03                       4.22         0.271           3
        0.05                       6.94         0.876           7

    Two facts those numbers carry, both of which the tests below pin:
      - A smooth pull does NOT fire, at any pull speed tried (ratio 1.25-1.56 for pulls of
        10-26 frames). The design spec's "expected to over-fire" worry is not borne out on
        synthetic smooth profiles; it remains untested on real video.
      - The 0.03 burst's fired span is 3 frames, SHORTER than `ctx.min_frames` (6 at 30 fps).
        That is the concrete case the event-rule deviation exists for.
    """
    frames: list[dict] = []
    for index in range(_JERK_TOTAL_FRAMES):
        progress = min(index / _JERK_PULL_FRAMES, 1.0)
        # Cosine ease: velocity is a half-sine, so acceleration is nonzero across the pull.
        travel = 0.25 * (1.0 - math.cos(math.pi * progress)) / 2.0
        distance_value = 0.30 - travel
        if burst_amplitude:
            distance_value -= burst_amplitude * math.exp(
                -(((index - _JERK_BURST_AT) / _JERK_BURST_WIDTH) ** 2)
            )
        frames.append(
            row_frame(
                trunk_angle_deg=20.0,
                wrist_hip_dist=max(distance_value, 0.02),
                elbow_angle_deg=170.0 - 100.0 * progress,
                frame_index=index,
            )
        )
    return frames


class RowMomentumJerkTest(unittest.TestCase):
    def test_a_smooth_controlled_pull_does_not_fire(self) -> None:
        """Specificity on a realistic profile: measured peak/median ratio 1.56, well under 3."""
        from src.pose.movements.row import rule_momentum_jerk

        self.assertEqual(_run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.0)), [])

    def test_a_three_frame_spike_survives_the_median_filter_and_fires(self) -> None:
        """The §4.6(b) claim, verified rather than asserted.

        `run_detector` median-filters every metric with a 5-frame window. This test runs the
        FULL detector path -- smoothing included -- so it fails if the derivative-as-metric
        decision is ever reverted to differencing a smoothed position series.
        """
        from src.pose.movements.base import run_detector
        from src.pose.movements.row import ROW_DETECTOR

        result = run_detector(
            ROW_DETECTOR, _jerk_clip(burst_amplitude=0.05), fps=30.0,
            view_type="rear_oblique", view_confidence=0.8, max_reps=None,
        )
        fired = [d for d in result.detections if d.fault_id == "row_momentum_jerk"]
        self.assertEqual(len(fired), 1)

    def test_a_burst_shorter_than_min_frames_still_fires(self) -> None:
        """min_frames is 6 at 30fps; this burst's fired span is 3. It must NOT be filtered out.

        This is the concrete case the event-rule deviation exists for: a `contiguous_true_segments`
        call passing `ctx.min_frames` instead of 1 would drop this detection entirely.
        """
        from src.pose.movements.row import rule_momentum_jerk

        detections = _run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.03))
        self.assertEqual(len(detections), 1)
        self.assertLess(detections[0].end_frame - detections[0].start_frame + 1, 6)

    def test_severity_rises_with_the_ratio(self) -> None:
        from src.pose.movements.row import rule_momentum_jerk

        small = _run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.03))
        large = _run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.05))
        self.assertTrue(small and large)
        self.assertLess(small[0].severity, large[0].severity)
        # Measured against the 3.0 -> 7.5 ramp; exact, because the fixture is deterministic.
        self.assertAlmostEqual(small[0].severity, 0.271, places=2)
        self.assertAlmostEqual(large[0].severity, 0.876, places=2)

    def test_a_motionless_window_is_refused_rather_than_maximally_flagged(self) -> None:
        """A zero median makes every ratio infinite; the guard must silence, not fire."""
        from src.pose.movements.row import rule_momentum_jerk

        frames = [
            row_frame(wrist_hip_dist=0.10, elbow_angle_deg=170.0 - 4.0 * i, frame_index=i)
            for i in range(20)
        ]
        self.assertEqual(_run_rule(rule_momentum_jerk, frames), [])

    def test_observability_is_medium_in_every_view(self) -> None:
        from src.pose.movements.row import rule_momentum_jerk

        for view in ("side", "rear", "rear_oblique", "unknown"):
            detections = _run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.05), view_type=view)
            self.assertEqual(len(detections), 1, view)
            self.assertEqual(detections[0].observability, "medium", view)
            self.assertAlmostEqual(detections[0].confidence, detections[0].severity, places=6)

    def test_trunk_heave_is_evidence_and_never_a_fire_condition(self) -> None:
        from src.pose.movements.row import rule_momentum_jerk

        detections = _run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.05))
        self.assertIn("trunk_heave", detections[0].evidence)
        self.assertIn(detections[0].evidence["trunk_heave"], ("yes", "no"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py -q -k MomentumJerk`
Expected: FAIL — `ImportError: cannot import name 'rule_momentum_jerk'`

(`test_a_three_frame_spike_survives...` additionally needs `ROW_DETECTOR`, which Task 6 assembles. Mark that one `@unittest.skip("ROW_DETECTOR lands in Task 6")` while implementing this task and **remove the skip in Task 6, Step 3**.)

- [ ] **Step 3: Implement the rule**

```python
# FROM THE SPEC: "flag if peak concentric wrist acceleration exceeds ~3x the rep's median
# concentric acceleration".
JERK_RATIO_MILD = 3.0
# RULE-LEVEL CHOICE MADE HERE: 2.5x the fire threshold, `pushup.rule_hip_sag`'s convention.
JERK_RATIO_SEVERE = 7.5

# RULE-LEVEL MEASURABILITY GUARD -- the third category, the one that can ONLY EVER SILENCE.
# If the median acceleration over the pull is at or below this floor the wrists barely moved,
# every ratio divides by ~0, and the rule would emit a confident maximum-severity jerk verdict
# on a stationary lifter. Refusing is the lesser evil. Not a tuned number and not a fire
# threshold; it can never cause a detection, only prevent a meaningless one.
_DEGENERATE_ACCEL = 1e-4


def rule_momentum_jerk(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the bar being yanked -- an acceleration transient during the concentric pull.

    THIS RULE BREAKS TWO SHARED CONVENTIONS ON PURPOSE. Both are stated here because a reviewer
    should see the deviation argued rather than discover it.

    (1) IT DOES NOT USE `ctx.min_frames`. Every other rule in this codebase passes it to
    `contiguous_true_segments` because every other rule tests a SUSTAINED STATE. A jerk is a
    TRANSIENT: `min_frames` is max(3, ceil(fps * 0.20)) -- 6 frames at 30 fps -- and a genuine
    bar-yank spike lasts 1-3. Requiring a fifth of a second of sustained jerk contradicts the
    fault's definition, so this rule passes 1 and fires as a per-rep EVENT.

    (2) ITS METRIC IS A DERIVATIVE COMPUTED IN `row_compute_raw`, not differenced here.
    `run_detector` median-filters every key in `metric_keys`; a 5-frame median over a POSITION
    series erases the transient before any rule sees it. Emitting the acceleration as the
    metric makes that filter a low-pass on the quantity of interest instead.

    THE THRESHOLD IS SELF-NORMALIZING AND IS EXPECTED TO OVER-FIRE. "3x the rep's median"
    compares a peak against a median that includes the near-zero accelerations at both ends of
    the pull, so a controlled rep with an ordinary bell-shaped velocity profile can exceed it.
    There is no labeled row video anywhere in this repository (the design spec's §2), and
    threshold tuning is off the table by standing decision, so this ships spec-faithful with
    its expected failure mode NAMED -- the same treatment `lunge_pelvic_drop`'s split-stance
    foreshortening bias received. If it ever meets data and fires at similar rates on clean and
    jerky reps, the honest conclusion is that the self-normalizing threshold does not
    discriminate, not that rows are universally jerky.

    ON A FALLBACK PATH THE NORMALIZATION SILENTLY CHANGES MEANING: "the rep's median" becomes
    "the whole clip's median over every `pull` frame" when no rep was segmented. The rule still
    runs; `evidence["median_over_frames"]` records how many frames the median was taken over so
    a reader can tell which case they are looking at.

    A STABLE FRAME RATE IS ASSUMED AND NEVER VERIFIED. `ctx.fps` is one scalar and nothing in
    the pipeline checks inter-frame spacing; every acceleration number inherits that.

    THE SPEC'S SECOND, OR'd CONDITION IS DEGENERATE AND IS NOT IMPLEMENTED AS AN OR. It reads
    "OR if a simultaneous trunk-angle velocity spike co-occurs WITH THE WRIST SPIKE (heave)" --
    its own text requires the wrist spike condition one already tests, so it describes a strict
    SUBSET and can never widen the fire set. It is implemented as EVIDENCE instead: the trunk
    speed over the fired frames is tested against its own 3x median and recorded in
    `evidence["trunk_heave"]`, which separates an arms-only yank from a whole-body heave for
    the coaching cue without changing whether anything fires.

    OBSERVABILITY IS `medium` IN EVERY VIEW, with no discount. The spec rates this "medium --
    any view with the pulling wrist visible"; no view earns better, so there is no `high` to
    downgrade FROM and applying the x0.65 off-view scale would be inventing a penalty the spec
    does not describe.
    """
    pull_accels = [
        frame.m("wrist_accel_norm")
        for frame in core
        if frame.valid and frame.phase == "pull" and np.isfinite(frame.m("wrist_accel_norm"))
    ]
    if len(pull_accels) < 3:
        return []
    median_accel = float(np.median(pull_accels))
    if median_accel <= _DEGENERATE_ACCEL:
        return []

    def _ratio(frame: CoreFrame) -> float:
        value = frame.m("wrist_accel_norm")
        return value / median_accel if np.isfinite(value) else float(np.nan)

    mask = [
        frame.valid
        and frame.phase == "pull"
        and np.isfinite(_ratio(frame))
        and _ratio(frame) > JERK_RATIO_MILD
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    # min_frames=1, not ctx.min_frames -- see (1) in the docstring.
    for start, end in contiguous_true_segments(mask, 1):
        segment = core[start : end + 1]
        ratios = [_ratio(frame) for frame in segment]
        max_ratio = float(np.nanmax(ratios))
        severity = severity_from_range(
            max_ratio, JERK_RATIO_MILD, JERK_RATIO_SEVERE, lower_is_worse=False
        )

        trunk_speeds = [
            frame.m("trunk_angle_speed_deg_s")
            for frame in core
            if frame.valid and frame.phase == "pull" and np.isfinite(frame.m("trunk_angle_speed_deg_s"))
        ]
        trunk_median = float(np.median(trunk_speeds)) if trunk_speeds else float(np.nan)
        segment_trunk = [
            frame.m("trunk_angle_speed_deg_s")
            for frame in segment
            if np.isfinite(frame.m("trunk_angle_speed_deg_s"))
        ]
        heave = (
            np.isfinite(trunk_median)
            and trunk_median > _DEGENERATE_ACCEL
            and bool(segment_trunk)
            and max(segment_trunk) > JERK_RATIO_MILD * trunk_median
        )

        detections.append(
            build_detection(
                fault_id="row_momentum_jerk",
                fault_name="Momentum / Jerk (Body English)",
                kg_query=ROW_MOMENTUM_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=ratios,
                severity=severity,
                confidence=severity,
                observability="medium",
                evidence={
                    "peak_accel_ratio": round(max_ratio, 3),
                    "median_pull_accel": round(median_accel, 5),
                    "median_over_frames": len(pull_accels),
                    "trunk_heave": "yes" if heave else "no",
                    "threshold": JERK_RATIO_MILD,
                    "primary_label": "peak/median pull acceleration",
                    "primary_value": round(max_ratio, 3),
                    "primary_threshold": JERK_RATIO_MILD,
                },
                citation="Padovan R, et al. J Funct Morphol Kinesiol (2025), PMC12821611. "
                         "Supplemented, descriptively only, by the bent-over row entry in "
                         "data/rag/docs/row_wiki.txt.",
                citation_support="Padovan: \"Accelerating a given load during dynamic "
                                 "contractions increases force requirements during the "
                                 "concentric phase, whereas the same load imposes lower "
                                 "mechanical demands during the eccentric phase\" — momentum "
                                 "redistributes loading away from the controlled tension the "
                                 "exercise intends; their protocol standardizes a 2 s "
                                 "concentric / 2 s eccentric tempo. Wiki (descriptive only): "
                                 "advises \"a slow tempo and avoiding jerking … prevents "
                                 "momentum from creating momentary weightlessness or slack in "
                                 "the muscles during the ascent.\"",
            )
        )
    return detections
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py -q`
Expected: PASS (with `test_a_three_frame_spike_survives_the_median_filter_and_fires` skipped until Task 6)

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/row.py tests/test_row.py
git commit -m "feat(pose): row momentum rule, and the two conventions a transient has to break"
```

---

### Task 5: `rule_asymmetric_pull`

**Files:**
- Modify: `src/pose/movements/row.py`
- Test: `tests/test_row.py`

**Interfaces:**
- Consumes: Task 2's `_setup_baseline`, `_OFF_VIEW_CONFIDENCE`, `ROW_ASYMMETRY_KG_QUERY`.
- Produces: `rule_asymmetric_pull`, `ELBOW_ASYMMETRY_MILD = 0.05`, `ELBOW_ASYMMETRY_SEVERE = 0.125`, `SHOULDER_TILT_RISE_MILD = 0.04`, `SHOULDER_TILT_RISE_SEVERE = 0.10`, `ASYMMETRY_OBSERVABLE_VIEWS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_row.py`:

```python
class RowAsymmetricPullTest(unittest.TestCase):
    def test_a_symmetric_pull_does_not_fire(self) -> None:
        from src.pose.movements.row import rule_asymmetric_pull

        self.assertEqual(_run_rule(rule_asymmetric_pull, _row_clip()), [])

    def test_just_inside_both_thresholds_does_not_fire(self) -> None:
        from src.pose.movements.row import rule_asymmetric_pull

        clip = _row_clip(peak_elbow_dy=0.049, setup_tilt=0.0, peak_tilt=0.039)
        self.assertEqual(_run_rule(rule_asymmetric_pull, clip), [])

    def test_elbow_height_asymmetry_alone_fires(self) -> None:
        from src.pose.movements.row import rule_asymmetric_pull

        clip = _row_clip(peak_elbow_dy=0.051)
        detections = _run_rule(rule_asymmetric_pull, clip)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "row_asymmetric_pull")
        self.assertEqual(detections[0].evidence["fired_on"], "elbow_height")

    def test_a_shoulder_tilt_increase_alone_fires(self) -> None:
        from src.pose.movements.row import rule_asymmetric_pull

        clip = _row_clip(setup_tilt=0.0, peak_tilt=0.041)
        detections = _run_rule(rule_asymmetric_pull, clip)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].evidence["fired_on"], "shoulder_tilt")

    def test_the_tilt_term_is_a_delta_not_an_absolute(self) -> None:
        """A lifter tilted the same amount at setup and at peak has not become asymmetric."""
        from src.pose.movements.row import rule_asymmetric_pull

        clip = _row_clip(setup_tilt=0.06, peak_tilt=0.06)
        self.assertEqual(_run_rule(rule_asymmetric_pull, clip), [])

    def test_severity_is_exact_at_the_elbow_ramp_midpoint(self) -> None:
        from src.pose.movements.row import rule_asymmetric_pull

        # Ramp 0.05 -> 0.125; 0.0875 is exactly half way.
        clip = _row_clip(peak_elbow_dy=0.0875)
        detections = _run_rule(rule_asymmetric_pull, clip)
        self.assertAlmostEqual(detections[0].severity, 0.5, places=3)

    def test_evidence_names_the_high_side(self) -> None:
        from src.pose.movements.row import rule_asymmetric_pull

        # elbow_dy is POSITIVE downward, so a positive dy puts the LEFT elbow LOWER.
        clip = _row_clip(peak_elbow_dy=0.08)
        detections = _run_rule(rule_asymmetric_pull, clip)
        self.assertEqual(detections[0].evidence["high_side"], "right")

    def test_a_pure_side_view_downgrades_rather_than_silencing(self) -> None:
        from src.pose.movements.row import rule_asymmetric_pull

        clip = _row_clip(peak_elbow_dy=0.0875)
        side = _run_rule(rule_asymmetric_pull, clip, view_type="side")
        rear = _run_rule(rule_asymmetric_pull, clip, view_type="rear")
        self.assertEqual(side[0].observability, "medium")
        self.assertEqual(rear[0].observability, "high")
        self.assertLess(side[0].confidence, rear[0].confidence)

    def test_wrist_travel_asymmetry_is_evidence_and_never_fires_alone(self) -> None:
        """The spec's third term has NO threshold, so inventing one would be a fabrication."""
        from src.pose.movements.row import rule_asymmetric_pull

        clip = _row_clip(peak_elbow_dy=0.0, peak_tilt=0.0)
        for frame in clip[6:]:
            frame["landmarks"][16] = _lm(
                frame["landmarks"][16]["x"], frame["landmarks"][16]["y"] + 0.20, 0.95
            )
        self.assertEqual(_run_rule(rule_asymmetric_pull, clip), [])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py -q -k AsymmetricPull`
Expected: FAIL — `ImportError: cannot import name 'rule_asymmetric_pull'`

- [ ] **Step 3: Implement the rule**

```python
# Views in which the parent spec rates the frontal-plane asymmetry cues `high` ("front / rear
# (both shoulders and elbows visible); low from pure side view"). Defined locally rather than
# imported from lunge.py: the two modules happen to agree today but answer different spec lines
# and must be free to diverge.
ASYMMETRY_OBSERVABLE_VIEWS = {"front", "front_oblique", "rear", "rear_oblique"}

# FROM THE SPEC: "Flag if elbow-height asymmetry > 0.05 normalized OR shoulder-line tilt
# increases > 0.04 vs setup."
ELBOW_ASYMMETRY_MILD = 0.05
SHOULDER_TILT_RISE_MILD = 0.04
# RULE-LEVEL CHOICES MADE HERE: each is 2.5x its fire threshold, `pushup.rule_hip_sag`'s
# convention. The spec states no ramp for this fault.
ELBOW_ASYMMETRY_SEVERE = 0.125
SHOULDER_TILT_RISE_SEVERE = 0.10


def rule_asymmetric_pull(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag one arm pulling higher or further than the other.

    THRESHOLD PROVENANCE: fire thresholds 0.05 and 0.04 are FROM THE SPEC; both ramps are
    RULE-LEVEL.

    TWO TERMS, ASYMMETRICALLY DEFINED, AND THAT IS THE SPEC'S OWN CONSTRUCTION: the elbow term
    is ABSOLUTE ("elbow-height asymmetry > 0.05") while the shoulder term is a DELTA
    ("shoulder-line tilt INCREASES > 0.04 VS SETUP"). A lifter with a structurally uneven
    shoulder line is therefore not flagged for standing still, which is the point of the delta.
    The baseline is this window's own (`_setup_baseline`); no baseline means silence.

    THE SPEC'S THIRD TERM IS EMITTED BUT NEVER FIRED ON. Its heuristic mentions wrist-to-hip
    travel asymmetry, `|dist(15,23) - dist(16,24)|`, but gives it NO threshold, unlike the other
    two. Inventing one would be a fabricated fire criterion, so `wrist_travel_asymmetry` is
    carried in `evidence` as a diagnostic and the firing rests on the two terms the spec
    actually quantifies. Pinned by
    `test_wrist_travel_asymmetry_is_evidence_and_never_fires_alone`.

    DIRECTION IS PART OF THE VERDICT, following `pushup_hip_sag`: the coaching cue is
    side-specific, so `evidence["high_side"]` records which elbow was HIGHER in the image (i.e.
    smaller y). `score_values` is the per-frame max sub-severity, an absolute quantity, so
    `build_detection` nominates the genuinely worst frame in either direction.

    FACING-FREE BY CONSTRUCTION, which is what lets this fire from the views production can
    actually reach. `elbow_height_asymmetry` and `shoulder_tilt` are magnitudes of image-y
    differences, and image y does not depend on whether the camera is in front of or behind the
    subject. Since `estimate_view_for_pose(allow_front=False)` means `front`/`front_oblique`
    are never emitted downstream, a rule gated positively on them would be PERMANENTLY SILENT
    (what happened to `pushup_elbow_flare`); `rear`/`rear_oblique` earn the spec's `high` here
    for the same reason `lunge.rule_knee_valgus` argues for its midline-relative proxy. A pure
    `side` view downgrades to `medium` with the x0.65 discount rather than being silenced.

    PHASE SCOPE `peak`, from the spec's "At peak: compare left vs right elbow height".
    """
    baseline_tilt = _setup_baseline(core, "shoulder_tilt")
    if not np.isfinite(baseline_tilt):
        return []
    observable = ctx.view_type in ASYMMETRY_OBSERVABLE_VIEWS

    def _sub_severities(frame: CoreFrame) -> tuple[float, float]:
        elbow_value = frame.m("elbow_height_asymmetry")
        tilt_rise = frame.m("shoulder_tilt") - baseline_tilt
        elbow_severity = (
            severity_from_range(
                elbow_value, ELBOW_ASYMMETRY_MILD, ELBOW_ASYMMETRY_SEVERE, lower_is_worse=False
            )
            if np.isfinite(elbow_value) and elbow_value > ELBOW_ASYMMETRY_MILD
            else 0.0
        )
        tilt_severity = (
            severity_from_range(
                tilt_rise, SHOULDER_TILT_RISE_MILD, SHOULDER_TILT_RISE_SEVERE, lower_is_worse=False
            )
            if np.isfinite(tilt_rise) and tilt_rise > SHOULDER_TILT_RISE_MILD
            else 0.0
        )
        return elbow_severity, tilt_severity

    mask = [
        frame.valid and frame.phase == "peak" and max(_sub_severities(frame)) > 0.0
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        pairs = [_sub_severities(frame) for frame in segment]
        scores = [max(pair) for pair in pairs]
        severity = float(np.nanmax(scores))
        elbow_fired = any(pair[0] > 0.0 for pair in pairs)
        tilt_fired = any(pair[1] > 0.0 for pair in pairs)
        fired_on = (
            "both" if elbow_fired and tilt_fired else "elbow_height" if elbow_fired else "shoulder_tilt"
        )
        worst = segment[int(np.nanargmax(scores))]
        # `elbow_height_delta_signed` is left_y - right_y and image y grows DOWNWARD, so a
        # POSITIVE delta means the left elbow is LOWER and the RIGHT one is the high side.
        high_side = "right" if worst.m("elbow_height_delta_signed") > 0.0 else "left"
        detections.append(
            build_detection(
                fault_id="row_asymmetric_pull",
                fault_name="Asymmetric Pull (One Side Leading)",
                kg_query=ROW_ASYMMETRY_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=scores,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "fired_on": fired_on,
                    "high_side": high_side,
                    "max_elbow_height_asymmetry": round(
                        float(np.nanmax([frame.m("elbow_height_asymmetry") for frame in segment])), 4
                    ),
                    "max_shoulder_tilt_rise": round(
                        float(np.nanmax([frame.m("shoulder_tilt") - baseline_tilt for frame in segment])), 4
                    ),
                    "wrist_travel_asymmetry": round(
                        float(np.nanmax([frame.m("wrist_travel_asymmetry") for frame in segment])), 4
                    ),
                    "setup_shoulder_tilt": round(baseline_tilt, 4),
                    "elbow_threshold": ELBOW_ASYMMETRY_MILD,
                    "tilt_threshold": SHOULDER_TILT_RISE_MILD,
                    "primary_label": "elbow-height asymmetry"
                    if fired_on != "shoulder_tilt"
                    else "shoulder-tilt increase vs setup",
                    "primary_value": round(
                        float(np.nanmax([frame.m("elbow_height_asymmetry") for frame in segment])), 4
                    )
                    if fired_on != "shoulder_tilt"
                    else round(
                        float(np.nanmax([frame.m("shoulder_tilt") - baseline_tilt for frame in segment])), 4
                    ),
                    "primary_threshold": ELBOW_ASYMMETRY_MILD
                    if fired_on != "shoulder_tilt"
                    else SHOULDER_TILT_RISE_MILD,
                },
                citation="Saeterbakken A, et al. Int J Sports Med (2015), PMID 26134664. "
                         "Supplemented by Padovan R, et al. J Funct Morphol Kinesiol (2025), "
                         "PMC12821611.",
                citation_support="Saeterbakken: \"unilateral performance of exercises activated "
                                 "the external oblique more than bilateral performance, "
                                 "regardless of exercise\" — an unintended one-sided pull "
                                 "imposes the higher anti-rotation/oblique load characteristic "
                                 "of unilateral rowing. Padovan frames correct rowing as "
                                 "\"coordinated scapulothoracic motion\" with bilateral "
                                 "scapular adduction to the abdominal target, which asymmetry "
                                 "breaks.",
            )
        )
    return detections
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/row.py tests/test_row.py
git commit -m "feat(pose): row asymmetric-pull rule, with the term the spec never quantified"
```

---

### Task 6: Assemble and register `ROW_DETECTOR`

**Files:**
- Modify: `src/pose/movements/row.py`
- Modify: `src/pose/movements/registry.py:20-34`
- Modify: `tests/test_movement_registry.py:212`, `:254`, `:264`, `:281`
- Test: `tests/test_row.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `ROW_DETECTOR: MovementDetector` registered under `"Row"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_row.py`:

```python
class RowDetectorAssemblyTest(unittest.TestCase):
    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        """A key the tuple omits is dropped by run_detector and read back as NaN."""
        from src.pose.movements.row import ROW_METRIC_KEYS, row_compute_raw

        raw = row_compute_raw(_row_clip(), fps=30.0)
        emitted = set(raw[-1]) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(ROW_METRIC_KEYS))

    def test_the_detector_is_registered_and_unvalidated(self) -> None:
        from src.pose.movements import registry

        self.assertEqual(registry.get_detector("Row").name, "Row")
        self.assertEqual(registry.get_detector("row").name, "Row")
        self.assertFalse(registry.get_detector("Row").validated)

    def test_all_four_rules_are_wired_in(self) -> None:
        from src.pose.movements import registry, row

        rules = registry.get_detector("Row").rules
        self.assertEqual(
            [rule.__name__ for rule in rules],
            [
                "rule_torso_rising",
                "rule_incomplete_rom",
                "rule_momentum_jerk",
                "rule_asymmetric_pull",
            ],
        )
        self.assertIs(registry.get_detector("Row").compute_raw, row.row_compute_raw)
        self.assertIs(registry.get_detector("Row").assign_phases, row.row_assign_phases)

    def test_the_fifth_spec_rule_is_absent_by_design(self) -> None:
        """rounded_thoracolumbar_spine is geometrically degenerate; see row.py's docstring."""
        from src.pose.movements import registry

        fault_ids = {rule.__name__ for rule in registry.get_detector("Row").rules}
        self.assertNotIn("rule_rounded_thoracolumbar_spine", fault_ids)


class RowPerRepBaselineTest(unittest.TestCase):
    def test_each_rep_is_scored_against_its_own_setup(self) -> None:
        """The §4.2 guard: a clean rep 1 followed by a rising rep 2 flags rep 2 only.

        Single-rep fixtures structurally cannot check this -- the whole clip is one window
        there, so a per-clip baseline would pass every earlier test. This is Row's analogue of
        Lunge's alternating-lead fixture.
        """
        from src.pose.movements.base import run_detector
        from src.pose.movements.row import ROW_DETECTOR

        frames: list[dict] = []
        index = 0
        for rep, peak_trunk in enumerate((20.0, 55.0)):
            for _ in range(6):  # setup / return to extension
                frames.append(row_frame(trunk_angle_deg=20.0, elbow_angle_deg=170.0,
                                        wrist_hip_dist=0.30, frame_index=index))
                index += 1
            for _ in range(10):  # peak hold
                frames.append(row_frame(trunk_angle_deg=peak_trunk, elbow_angle_deg=60.0,
                                        wrist_hip_dist=0.05, frame_index=index))
                index += 1
        for _ in range(6):
            frames.append(row_frame(trunk_angle_deg=20.0, elbow_angle_deg=170.0,
                                    wrist_hip_dist=0.30, frame_index=index))
            index += 1

        result = run_detector(
            ROW_DETECTOR, frames, fps=30.0, view_type="rear_oblique",
            view_confidence=0.8, max_reps=None,
        )
        self.assertEqual(len(result.reps), 2)
        rising = [d for d in result.detections if d.fault_id == "row_torso_rising"]
        self.assertEqual(len(rising), 1)
        self.assertEqual(rising[0].occurred_reps, (1,))
```

Also append to `tests/test_movement_registry.py` (mirroring the existing Lunge cases at
lines 60–88) and update the four existing collections:

- line ~212 rep-signal map: add `"Row": ("min_elbow_angle", "min")`
- line ~254 ordered names: `["Squat", "Overhead Press", "Push-up", "Lunge", "Row"]`
- line ~264 validated map: add `"Row": False`
- line ~281 name set: add `"Row"`

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_row.py tests/test_movement_registry.py -q`
Expected: FAIL — `KeyError: "No detector registered for movement 'Row'"` and the four collection assertions.

- [ ] **Step 3: Assemble, register, and un-skip Task 4's detector test**

Append to `src/pose/movements/row.py`:

```python
from src.pose.movements.base import MovementDetector
from src.pose.movements import registry

# FOUR of the parent spec's FIVE Row rules are listed here. The fifth,
# `rounded_thoracolumbar_spine`, is absent because it is geometrically degenerate under both
# constructions the spec offers -- the proof is in this module's docstring, and
# `test_the_fifth_spec_rule_is_absent_by_design` pins the absence so a future reader cannot
# mistake it for an oversight.
#
# `ROW_METRIC_KEYS` must stay a two-way match with what `row_compute_raw` emits (pinned by
# `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits is dropped by
# `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and read back as
# NaN by every rule.
ROW_DETECTOR = MovementDetector(
    "Row",
    ROW_METRIC_KEYS,
    row_compute_raw,
    row_assign_phases,
    (rule_torso_rising, rule_incomplete_rom, rule_momentum_jerk, rule_asymmetric_pull),
    # `validated` stays at its default False, and for Row that is not a formality: REHAB24-6
    # holds arm abduction, arm VW, table push-ups, leg abduction, lunge and squats, and no row.
    # Neither does Fit3D. There is NO labeled row repetition anywhere in this repository, so no
    # threshold here has ever been checked against a row performed by a human being. Beta is the
    # factual label. Flipping this flag would require data that does not exist yet.
    rep_signal="min_elbow_angle",
    rep_polarity="min",
    rep_start="extended",
)

registry.register(ROW_DETECTOR)
```

Add the side-effect import at the bottom of `src/pose/movements/registry.py`, after the lunge line:

```python
from src.pose.movements import row  # noqa: E402,F401
```

and extend that module's `list_detectors` docstring registration-order parenthetical to
`(Squat, Overhead Press, Push-up, Lunge, Row)`.

Remove the `@unittest.skip("ROW_DETECTOR lands in Task 6")` decorator added in Task 4.

- [ ] **Step 4: Run the whole suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS. `tests/test_analyze_pose_service.py::test_concurrent_analyses_are_bounded` is a live-Supabase call, not a flake — if it fails for lack of credentials, confirm it fails identically on `main` before proceeding.

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/row.py src/pose/movements/registry.py tests/test_row.py tests/test_movement_registry.py
git commit -m "feat(pose): register the row detector, and score each rep against its own setup"
```

---

### Task 7: Record the spec defect and verify the gates

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` §7 (Honest limitations & gaps, line ~1359)
- Modify: `scripts/pose/README.md` (movement list, if one is present)

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: no code.

- [ ] **Step 1: Add the parent-spec gap entry**

Append to §7 of the parent spec, as its own bullet:

```markdown
- **`rounded_thoracolumbar_spine` (Row) is not implementable from this document's own detection
  model, and was not implemented.** Both constructions its `detection_heuristic` offers are
  degenerate: the "three-point angle at mid-spine" places its middle point at
  `0.5·(shoulder_mid + hip_mid)`, which is by construction the midpoint of the segment joining
  the other two, so the angle is exactly 180° on every frame; and the sag alternative measures
  the distance from `shoulder_mid` to a line of which `shoulder_mid` is an endpoint, which is
  identically zero. The root cause is that MediaPipe Pose (§3) has no thoracic or lumbar
  landmark, so no point exists between the shoulders and the hips to measure spinal curvature
  with. Found during the Row implementation (2026-08-01,
  `docs/superpowers/specs/2026-08-01-row-detector-design.md` §3). Row therefore ships **four**
  rules, not five. Two monocular substitutes were considered and rejected — trunk-length
  foreshortening and ear-drop relative to the trunk line — because both are confounded by
  camera distance and by the hinge angle, and neither is what this rule's citation
  (Saeterbakken PMID 26134664, an EMG magnitude result) supports; either would need its own
  `fault_id` and an explicitly-invented threshold. The KG target `Row:Trunk Flexion` exists and
  is non-empty, so the gap is the metric, not the knowledge.
```

- [ ] **Step 2: Add the Row status line to §8**

Append to the parent spec's §8 status block:

```markdown
- **Row — IMPLEMENTED 2026-08-01, UNVALIDATED.** Four of five rules
  (`row_torso_rising`, `row_incomplete_rom`, `row_momentum_jerk`, `row_asymmetric_pull`);
  the fifth is recorded in §7 as a spec defect. `validated=False`: REHAB24-6 contains no row
  and neither does Fit3D, so §8.4's "validate thresholds against labeled data per movement"
  is **not** satisfied and cannot be until labeled row video exists. All four severity ramps
  are rule-level display curves (the Row section states none), and `row_momentum_jerk`'s
  self-normalizing 3×-median threshold is expected to over-fire.
```

- [ ] **Step 3: Run the full verification set**

```bash
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```
Expected: suite passes; coverage gate passes. If coverage on `src/pose/movements/row.py` falls
below the gate, add tests for the uncovered branches — do **not** lower the gate.

- [ ] **Step 4: Confirm the CLI path end-to-end**

```bash
.venv\Scripts\python.exe scripts/pose/run_pose_rule_detection.py --help
```
Expected: `--movement` accepts `Row` (the choices come from `registry.list_detectors()`).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md scripts/pose/README.md
git commit -m "docs(spec): record the row spec defect, and what Row ships without validation"
```

---

## Self-Review

**Spec coverage.** Design spec §4.1 → Task 1. §4.2 → Task 2 (`_setup_baseline`) and Task 6
(the multi-rep guard). §4.3 → Task 1, plus the signed elbow-delta added in Task 5's note.
§4.4 → Tasks 2–5, one rule each. §4.5 (image-unit scale dependence + scale-free diagnostic) →
Task 1 emits `wrist_hip_dist_shoulder_norm`, Task 3 documents it and carries it in evidence.
§4.6 → Task 4. §4.7 → the view handling in Tasks 2, 3, 4 and 5. §5 (KG queries) → Task 2's
Step 0 block. §6 (testing) → tests in every task. §7 honesty constraints → the docstrings in
Tasks 1, 4 and 6 plus Task 7's spec entries. §3 (degeneracy) → Task 1's module docstring and
Task 6's absence test and Task 7's §7 bullet. §2 (no validation) → Task 6's `validated=False`
comment and Task 7's §8 line.

**Placeholder scan.** Clean. One forward dependency is deliberate and named in place: Task 4's
`test_a_three_frame_spike_survives_the_median_filter_and_fires` needs `ROW_DETECTOR`, so it is
skipped when written and the skip removal is an explicit step in Task 6.

**Type consistency.** `_setup_baseline(core, key) -> float` is defined in Task 2 and reused in
Task 5 with the same signature. `_sub_severities` is a local closure in Tasks 3 and 5 — same
name, deliberately not shared, because the two return different pairs.
`elbow_height_delta_signed` is emitted and key-listed in Task 1 and consumed in Task 5; Task 6's
two-way-match test catches any drift between `ROW_METRIC_KEYS` and the emitter. Rule function
names in Task 6's ordering assertion match their definitions in Tasks 2–5 exactly.

**Known remaining risk, for the implementer to resolve empirically rather than assume.** Task 4's
`_jerk_clip` spike magnitudes (0.03 / 0.06 / 0.12) are chosen to produce a peak-to-median
acceleration ratio above 3 after the framework's 5-frame median filter, but that ratio has not
been computed by hand here. If a spike fails to fire, print `wrist_accel_norm` across the clip
and adjust the FIXTURE, never the threshold — `JERK_RATIO_MILD` is the spec's number and the
no-tuning constraint is absolute.
