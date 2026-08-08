# Band Pull Apart Rule Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a cited, unvalidated (Beta) rule detector for the standing Band Pull Apart — three of the parent spec's four rules firing, the fourth registered but permanently silent with its impossibility proof in-code.

**Architecture:** One new module `src/pose/movements/band_pull_apart.py`, following `src/pose/movements/row.py` exactly: threshold-free raw metrics → phase assignment → cited rule functions → an assembled `MovementDetector` registered by side-effect import. The shared `run_detector` in `src/pose/movements/base.py` already does segmentation, global smoothing, per-rep slicing and merging; **nothing in this plan changes it**.

**Tech Stack:** Python 3.12, numpy, `unittest.TestCase`. No new dependencies.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-08-09-band-pull-apart-detector-design.md`. **Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` §Band Pull Apart (lines 713–769).
- **Citations are copied verbatim from the parent spec at implementation time, never recalled from memory.** Open the parent spec and copy the `citation` / `citation_support` strings when you write each rule.
- **Interpreter:** `.venv\Scripts\python.exe` from the repository root. Never bare `python`/`pip`, never `source .venv/bin/activate` (POSIX-only, fails on this machine).
- **Run everything from the repository root.** Modules import as `from src.pose... import ...`.
- **Every threshold is labeled in-code as exactly one of two categories**, in the style of `src/pose/movements/pushup.py`: **`FROM THE SPEC`** or **`RULE-LEVEL CHOICE MADE HERE`**. Never blur them.
- **All severity ramps are RULE-LEVEL.** The parent spec states no ramp for any Band Pull Apart fault. Convention from `pushup.rule_hip_sag`: ramp endpoint = 2.5× the fire threshold, documented as a display/ranking curve, not a cited quantity. The one exception is the elbow ramp `150 → 110°`, whose 40° width is taken from `pushup.rule_shallow_depth` so the two elbow ramps cannot drift.
- **No threshold tuning.** Cited numbers stay as the spec states them. Weak behavior is written up, never repaired by moving a number.
- `BAND_PULL_APART_DETECTOR.validated` stays `False`. Fit3D contains `band pull apart` footage with 3D ground truth and rep boundaries but **no fault labels**, so no correct/incorrect check is possible (design spec §2).
- **The metric layer contains no thresholds.** `band_pull_apart_compute_raw` / `band_pull_apart_assign_phases` emit scale-free per-frame quantities and phase labels only. The sole constant they may define is `_DEGENERATE_LENGTH`, a division-by-zero guard.
- **Ramp direction is expressed through `severity_from_range`'s `lower_is_worse` flag, never by swapping `mild`/`severe`.** Signature: `severity_from_range(value, mild, severe, *, lower_is_worse)` (`src/pose/geometry.py:165`).
- Test command: `.venv\Scripts\python.exe -m pytest tests/ -q` (**always scoped to `tests/`**; never bare `pytest`). Coverage gate: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.
- Two backend test flakes are known-unrelated on this machine. Check a failure against a baseline on `main` before attributing it to this change.
- Commit after every task. The commit message body explains **why**, in the style of the repository's recent history.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/pose/movements/band_pull_apart.py` | **Create.** The whole detector: landmark constants, `BAND_PULL_APART_METRIC_KEYS`, `band_pull_apart_compute_raw`, `band_pull_apart_assign_phases`, `_setup_baseline`, `_clip_facing_sign`, four `rule_*` functions, `BAND_PULL_APART_DETECTOR`. Single file, matching every sibling movement module. |
| `src/pose/movements/registry.py` | **Modify.** One side-effect import line appended. |
| `tests/test_band_pull_apart.py` | **Create.** Fixture builder + all detector tests. |
| `tests/test_analyze_pose_service.py` | **Modify (line ~110–124).** Rotate the stale "unimplemented movement" example off Band Pull Apart. |
| `tests/test_movement_registry.py` | **Modify (line ~222–238).** Add this movement to the shared `(rep_signal, rep_polarity, rep_start)` table. |
| `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` | **Modify.** Two `NOTE` annotations in the Band Pull Apart section. |
| `TODO.md` | **Modify.** 6/16 → 7/16 in two places. |

**No frontend file is touched.** `/api/movements` is derived from the registry; `frontend/src/lib/movements.ts:14`, the i18n key `movement.Band Pull Apart`, and the card art already exist. Registering the detector flips the movement from "Soon" to analyzable with no frontend edit.

---

### Task 1: Raw metrics and phase assignment

**Files:**
- Create: `src/pose/movements/band_pull_apart.py`
- Test: `tests/test_band_pull_apart.py`

**Interfaces:**
- Consumes: `src.pose.geometry` (`landmarks_to_array`, `visible_point`, `angle_degrees`, `midpoint`, `distance`, `mean_visibility`, `LEFT_SHOULDER`, `RIGHT_SHOULDER`, `LEFT_HIP`, `RIGHT_HIP`, `LEFT_KNEE`, `RIGHT_KNEE`, `LEFT_ANKLE`, `RIGHT_ANKLE`, `LEFT_HEEL`, `RIGHT_HEEL`, `LEFT_FOOT_INDEX`, `RIGHT_FOOT_INDEX`), `src.pose.movements.base.CoreFrame`.
- Produces:
  - `BAND_PULL_APART_METRIC_KEYS: tuple[str, ...]`
  - `band_pull_apart_compute_raw(frames: Sequence[object], fps: float) -> list[dict]`
  - `band_pull_apart_assign_phases(raw: list[dict]) -> list[str]`
  - Landmark constants `LEFT_EAR = 7`, `RIGHT_EAR = 8`, `LEFT_ELBOW = 13`, `RIGHT_ELBOW = 14`, `LEFT_WRIST = 15`, `RIGHT_WRIST = 16`
  - Module-private `_derivative(values, fps)`, `_DEGENERATE_LENGTH = 1e-6`

**Note on one metric name, which differs from the design spec's §4.4 table.** The table calls a metric `trunk_lean_signed_deg` and describes it as "facing-corrected". Facing correction cannot live here: it needs a `0.02` floor, and **the metric layer contains no thresholds**. So this module emits `trunk_lean_image_signed_deg` — the raw image-frame signed pitch — and rule 4 (Task 4) applies the facing sign itself. Same for the reduction's scope: the spec says "per-clip", but `run_detector` hands rules a **per-rep** slice, so the reduction is per-rep. That is strictly safer and matches the architecture's rule that rep N's verdict must not depend on rep 1's frames.

- [ ] **Step 1: Write the failing metric tests**

Create `tests/test_band_pull_apart.py`:

```python
import math
import unittest

import numpy as np

from src.pose.movements.band_pull_apart import (
    BAND_PULL_APART_METRIC_KEYS,
    band_pull_apart_assign_phases,
    band_pull_apart_compute_raw,
)


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def _elbow_xy(
    shoulder: tuple[float, float],
    wrist: tuple[float, float],
    elbow_angle_deg: float,
    side_sign: float,
) -> tuple[float, float]:
    """Place an elbow so that angle(shoulder, elbow, wrist) EQUALS `elbow_angle_deg` exactly.

    Two equal-length segments of length r spanning a shoulder-wrist chord of length d subtend
    an elbow angle of 2*asin(d / (2r)), so the r producing a requested angle is
    r = d / (2*sin(angle/2)). The elbow then sits on the chord's perpendicular bisector at
    height h = sqrt(r^2 - (d/2)^2). Controlling the ANGLE directly is what the ROM rule's
    fixtures need: `min_elbow_angle` equals the requested number by construction, so a boundary
    fixture really does sit one step either side of the 150-degree threshold. Copied from
    tests/test_row.py, where the same construction backs the row's elbow fixtures.
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


def bpa_frame(
    spread_ratio: float = 1.8,
    shoulder_ear_gap: float = 0.12,
    left_gap_delta: float = 0.0,
    elbow_angle_deg: float = 170.0,
    trunk_lean_deg: float = 0.0,
    wrist_depth_offset: float = -0.30,
    frame_index: int = 0,
    visibility: float = 0.95,
) -> dict:
    """One standing band pull-apart frame, image y growing DOWNWARD, viewed from behind.

    Knobs, each controlling exactly one metric BY CONSTRUCTION:
      spread_ratio       -- wrist separation / shoulder width. Equals
                            `wrist_spread_shoulder_norm`.
      shoulder_ear_gap   -- image-y distance from each ear up to its shoulder. Equals both
                            `left_shoulder_ear_gap` and `right_shoulder_ear_gap`.
      left_gap_delta     -- added to the LEFT gap only, so a unilateral shrug can be built.
      elbow_angle_deg    -- angle(shoulder, elbow, wrist) per side; equals `min_elbow_angle`.
      trunk_lean_deg     -- signed pitch of hip_mid -> shoulder_mid from vertical. Positive
                            moves the shoulders toward +x. Equals
                            `trunk_lean_image_signed_deg`.
      wrist_depth_offset -- mean wrist z minus mean shoulder z. Equals `wrist_depth_offset`.
                            Negative = wrists nearer the camera = lifter faces the camera.
    """
    shoulder_width = 0.20
    hip_mid = (0.50, 0.70)
    trunk_len = 0.30
    theta = math.radians(trunk_lean_deg)
    # Pitch measured from VERTICAL, positive toward +x.
    shoulder_mid = (
        hip_mid[0] + trunk_len * math.sin(theta),
        hip_mid[1] - trunk_len * math.cos(theta),
    )
    sy = shoulder_mid[1]
    left_shoulder = (shoulder_mid[0] - shoulder_width / 2.0, sy)
    right_shoulder = (shoulder_mid[0] + shoulder_width / 2.0, sy)

    half_spread = spread_ratio * shoulder_width / 2.0
    wrist_y = sy + 0.02
    left_wrist = (shoulder_mid[0] - half_spread, wrist_y)
    right_wrist = (shoulder_mid[0] + half_spread, wrist_y)

    left_elbow = _elbow_xy(left_shoulder, left_wrist, elbow_angle_deg, side_sign=1.0)
    right_elbow = _elbow_xy(right_shoulder, right_wrist, elbow_angle_deg, side_sign=-1.0)

    left_ear = (left_shoulder[0], sy - (shoulder_ear_gap + left_gap_delta))
    right_ear = (right_shoulder[0], sy - shoulder_ear_gap)

    shoulder_z = 0.0
    wrist_z = shoulder_z + wrist_depth_offset

    landmarks = [_lm(0.5, 0.3, visibility=visibility) for _ in range(33)]
    landmarks[7] = _lm(*left_ear, visibility=visibility)
    landmarks[8] = _lm(*right_ear, visibility=visibility)
    landmarks[11] = _lm(*left_shoulder, z=shoulder_z, visibility=visibility)
    landmarks[12] = _lm(*right_shoulder, z=shoulder_z, visibility=visibility)
    landmarks[13] = _lm(*left_elbow, visibility=visibility)
    landmarks[14] = _lm(*right_elbow, visibility=visibility)
    landmarks[15] = _lm(*left_wrist, z=wrist_z, visibility=visibility)
    landmarks[16] = _lm(*right_wrist, z=wrist_z, visibility=visibility)
    landmarks[23] = _lm(hip_mid[0] - 0.08, hip_mid[1], visibility=visibility)
    landmarks[24] = _lm(hip_mid[0] + 0.08, hip_mid[1], visibility=visibility)
    return {"frame_index": frame_index, "landmarks": landmarks}


class BandPullApartMetricsTest(unittest.TestCase):
    def test_spread_ratio_knob_equals_the_emitted_metric(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(spread_ratio=1.35)], fps=30.0)
        self.assertTrue(raw[0]["valid"])
        self.assertAlmostEqual(raw[0]["wrist_spread_shoulder_norm"], 1.35, places=4)

    def test_shoulder_ear_gap_knob_equals_the_emitted_metric(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(shoulder_ear_gap=0.09)], fps=30.0)
        self.assertAlmostEqual(raw[0]["left_shoulder_ear_gap"], 0.09, places=4)
        self.assertAlmostEqual(raw[0]["right_shoulder_ear_gap"], 0.09, places=4)

    def test_left_gap_delta_moves_only_the_left_side(self) -> None:
        raw = band_pull_apart_compute_raw(
            [bpa_frame(shoulder_ear_gap=0.12, left_gap_delta=-0.05)], fps=30.0
        )
        self.assertAlmostEqual(raw[0]["left_shoulder_ear_gap"], 0.07, places=4)
        self.assertAlmostEqual(raw[0]["right_shoulder_ear_gap"], 0.12, places=4)

    def test_elbow_angle_knob_equals_min_elbow_angle(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(elbow_angle_deg=140.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["min_elbow_angle"], 140.0, places=2)

    def test_trunk_lean_knob_equals_the_signed_image_pitch(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(trunk_lean_deg=12.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["trunk_lean_image_signed_deg"], 12.0, places=2)
        raw = band_pull_apart_compute_raw([bpa_frame(trunk_lean_deg=-12.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["trunk_lean_image_signed_deg"], -12.0, places=2)

    def test_wrist_depth_offset_knob_equals_the_emitted_metric(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(wrist_depth_offset=-0.42)], fps=30.0)
        self.assertAlmostEqual(raw[0]["wrist_depth_offset"], -0.42, places=4)

    def test_one_dropped_landmark_invalidates_the_whole_frame(self) -> None:
        frame = bpa_frame()
        frame["landmarks"][15] = _lm(0.4, 0.5, visibility=0.10)
        raw = band_pull_apart_compute_raw([frame], fps=30.0)
        self.assertFalse(raw[0]["valid"])
        # An invalid frame carries NO metric keys at all -- every rule masking on
        # `frame.valid` therefore goes silent for it, not just the wrist-dependent ones.
        for key in BAND_PULL_APART_METRIC_KEYS:
            self.assertNotIn(key, raw[0])

    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(frame_index=i) for i in range(5)], fps=30.0)
        emitted = set(raw[2]) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(BAND_PULL_APART_METRIC_KEYS))

    def test_trunk_speed_is_nan_at_both_boundaries(self) -> None:
        frames = [bpa_frame(trunk_lean_deg=float(i), frame_index=i) for i in range(5)]
        raw = band_pull_apart_compute_raw(frames, fps=30.0)
        self.assertTrue(math.isnan(raw[0]["trunk_angle_speed_deg_s"]))
        self.assertTrue(math.isnan(raw[-1]["trunk_angle_speed_deg_s"]))
        self.assertTrue(math.isfinite(raw[2]["trunk_angle_speed_deg_s"]))


class BandPullApartPhaseTest(unittest.TestCase):
    def _rep(self) -> list[dict]:
        # hands together -> spread -> together, the movement's excursion.
        ratios = [0.4, 0.6, 1.0, 1.5, 1.9, 2.0, 1.9, 1.4, 0.9, 0.5]
        return [bpa_frame(spread_ratio=r, frame_index=i) for i, r in enumerate(ratios)]

    def test_one_phase_per_frame(self) -> None:
        raw = band_pull_apart_compute_raw(self._rep(), fps=30.0)
        self.assertEqual(len(band_pull_apart_assign_phases(raw)), len(raw))

    def test_opening_frames_are_setup_and_the_widest_frames_are_peak(self) -> None:
        raw = band_pull_apart_compute_raw(self._rep(), fps=30.0)
        phases = band_pull_apart_assign_phases(raw)
        self.assertEqual(phases[0], "setup")
        widest = max(range(len(raw)), key=lambda i: raw[i]["wrist_spread_shoulder_norm"])
        self.assertEqual(phases[widest], "peak")

    def test_frames_before_the_peak_are_pull_and_after_are_return(self) -> None:
        raw = band_pull_apart_compute_raw(self._rep(), fps=30.0)
        phases = band_pull_apart_assign_phases(raw)
        self.assertIn("pull", phases)
        self.assertIn("return", phases)
        self.assertLess(phases.index("pull"), len(phases) - 1 - phases[::-1].index("return"))

    def test_empty_clip_and_all_nan_clip(self) -> None:
        self.assertEqual(band_pull_apart_assign_phases([]), [])
        self.assertEqual(band_pull_apart_assign_phases([{"valid": False}] * 3), ["unknown"] * 3)

    def test_an_occluded_opening_frame_is_unknown_not_setup(self) -> None:
        frames = self._rep()
        frames[0]["landmarks"][11] = _lm(0.4, 0.5, visibility=0.10)
        raw = band_pull_apart_compute_raw(frames, fps=30.0)
        self.assertEqual(band_pull_apart_assign_phases(raw)[0], "unknown")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_band_pull_apart.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pose.movements.band_pull_apart'`

- [ ] **Step 3: Write the module header and the metric layer**

Create `src/pose/movements/band_pull_apart.py`:

```python
# Band Pull Apart (standing, resistance band) raw metrics and phase segmentation. Fault rules
# land in Tasks 2-5.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `band_pull_apart_compute_raw` /
# `band_pull_apart_assign_phases` compute per-frame quantities and a phase label only. Every
# number that decides anything belongs in a `rule_*` function. The only constant this module
# defines, `_DEGENERATE_LENGTH`, is a division-by-zero guard, never a tunable threshold. This is
# also why `trunk_lean_image_signed_deg` is emitted RAW rather than facing-corrected: the facing
# derivation needs a floor, and a floor is a threshold. Rule 4 does that correction itself.
#
# ---------------------------------------------------------------------------------------
# THIS IS THE FIRST MOVEMENT WHOSE DEFINING EXCURSION IS FRONTAL, NOT SAGITTAL.
# ---------------------------------------------------------------------------------------
# Squat, Lunge, Deadlift, Push-up, OHP and Row all excurse in the sagittal plane -- a knee angle,
# an elbow angle, a hip height, a trunk pitch. The band pull apart's excursion is the hands
# travelling APART in the image plane, which makes the REP SIGNAL ITSELF view-bound rather than
# only the rules: from a pure `side` view the hands overlap, the excursion vanishes, and
# `segment_reps` returns nothing before a single rule runs.
#
# That is safe in production only because of a reachability fact, not by luck:
# `estimate_view_for_pose` is called with `allow_front=False` (src/pose/view_estimation.py:14-16),
# so the reachable labels are {side, rear, rear_oblique, unknown}, and across the 45 real pose
# JSONs in this repository the estimator emitted `rear_oblique` 30 times, `rear` 13, `unknown` 2,
# and `side` effectively never. Wrist spread survives `rear_oblique` foreshortened but present.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY BAND PULL APART RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both ears, both shoulders, both elbows, both wrists and both hips. If
# `visible_point` drops any ONE of them the frame is marked `valid=False` and carries no metric
# keys at all, so every rule that masks on `frame.valid` goes silent for that frame, not just the
# one whose input landmark went missing. This mirrors `pushup_compute_raw`, `ohp_compute_raw`,
# `lunge_compute_raw` and `row_compute_raw`: an unmeasurable frame is refused wholesale rather
# than degraded, because a silently-wrong verdict is worse than no verdict.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, distance,
)
from src.pose.movements.base import CoreFrame

# Defined locally, matching row.py and overhead_press.py: geometry.py exports only the
# lower-body and shoulder/hip constants.
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. The name is squat-centric and carries awkwardly for a
# standing upper-body band exercise, exactly as it does for OHP, push-up and Row; this module's
# own rules never consume it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

BAND_PULL_APART_METRIC_KEYS: tuple[str, ...] = (
    "wrist_spread",
    "shoulder_width",
    "wrist_spread_shoulder_norm",
    "left_shoulder_ear_gap",
    "right_shoulder_ear_gap",
    "shoulder_ear_gap_shoulder_norm",
    "left_elbow_angle",
    "right_elbow_angle",
    "min_elbow_angle",
    "trunk_lean_image_signed_deg",
    "trunk_angle_speed_deg_s",
    "wrist_depth_offset",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard value
# pushup.py, overhead_press.py, lunge.py and row.py use; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6


def _derivative(values: Sequence[float], fps: float) -> list[float]:
    """Central-difference time derivative, NaN at both boundaries.

    ONE-SIDED BOUNDARY ESTIMATES ARE REFUSED ON PURPOSE. A forward difference at frame 0 and a
    central difference at frame 1 have different biases; mixing them into one series makes the
    first samples systematically unlike the rest. NaN propagates through the mask and the frame
    is simply not scored. A NaN input (an invalid frame) poisons its two neighbours' derivatives,
    which is correct: a derivative across a hole in the data is not measured, it is guessed.
    Copied from `row._derivative`, whose momentum rule needs the identical property.
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


def band_pull_apart_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    trunk_leans: list[float] = []

    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            trunk_leans.append(np.nan)
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        required = (
            LEFT_EAR, RIGHT_EAR,
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
            trunk_leans.append(np.nan)
            continue

        wrist_spread = distance(points, LEFT_WRIST, RIGHT_WRIST)
        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER)
        normalizer_ok = np.isfinite(shoulder_width) and shoulder_width > _DEGENERATE_LENGTH
        wrist_spread_shoulder_norm = (
            wrist_spread / shoulder_width
            if np.isfinite(wrist_spread) and normalizer_ok
            else np.nan
        )

        left_shoulder = visible_point(points, LEFT_SHOULDER, dims=2)
        right_shoulder = visible_point(points, RIGHT_SHOULDER, dims=2)
        left_ear = visible_point(points, LEFT_EAR, dims=2)
        right_ear = visible_point(points, RIGHT_EAR, dims=2)
        # Image y grows DOWNWARD, so shoulder_y - ear_y is POSITIVE when the ear sits above the
        # shoulder, and SHRINKS as the shoulder rises toward the ear. The spec states its shrug
        # threshold on exactly this quantity ("gap_peak < gap_setup - 0.03").
        left_shoulder_ear_gap = float(left_shoulder[1] - left_ear[1])
        right_shoulder_ear_gap = float(right_shoulder[1] - right_ear[1])
        mean_gap = float(np.mean([left_shoulder_ear_gap, right_shoulder_ear_gap]))
        # SCALE-FREE DIAGNOSTIC THAT NO RULE FIRES ON. The spec's 0.03 shrug threshold carries no
        # normalizer, so it is raw image units and therefore camera-distance dependent (design
        # spec 4.5). Emitting the normalized companion lets a future validation compare the two
        # WITHOUT any threshold having been moved in the meantime.
        shoulder_ear_gap_shoulder_norm = mean_gap / shoulder_width if normalizer_ok else np.nan

        left_elbow_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        finite_elbows = [v for v in (left_elbow_angle, right_elbow_angle) if np.isfinite(v)]
        min_elbow_angle = float(min(finite_elbows)) if finite_elbows else np.nan

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        # SIGNED, and deliberately NOT facing-corrected here. Positive = the shoulders sit toward
        # +x relative to the hips, in IMAGE coordinates. Which physical direction that is depends
        # on which way the lifter faces, which this layer cannot know and must not guess -- see
        # `_clip_facing_sign` in Task 4.
        if shoulder_mid is not None and hip_mid is not None:
            trunk_lean = float(
                np.degrees(
                    np.arctan2(
                        float(shoulder_mid[0] - hip_mid[0]),
                        float(hip_mid[1] - shoulder_mid[1]),
                    )
                )
            )
        else:
            trunk_lean = np.nan

        left_wrist3 = visible_point(points, LEFT_WRIST, dims=3)
        right_wrist3 = visible_point(points, RIGHT_WRIST, dims=3)
        left_shoulder3 = visible_point(points, LEFT_SHOULDER, dims=3)
        right_shoulder3 = visible_point(points, RIGHT_SHOULDER, dims=3)
        # MediaPipe z is depth relative to the hip midpoint, NEGATIVE toward the camera. A band
        # pull apart holds the band in FRONT of the torso by definition, so the SIGN of this
        # offset identifies which way the lifter faces. Rule 4 reduces it; nothing is decided
        # here. NaN when any z is missing, and identically 0.0 under the RTMPose extraction path
        # (src/pose/rtmpose_pose_extraction.py writes z=0.0 for every landmark) -- both cases are
        # handled by rule 4's floor, not by a branch here.
        if all(p is not None for p in (left_wrist3, right_wrist3, left_shoulder3, right_shoulder3)):
            wrist_depth_offset = float(
                np.mean([left_wrist3[2], right_wrist3[2]])
                - np.mean([left_shoulder3[2], right_shoulder3[2]])
            )
        else:
            wrist_depth_offset = np.nan

        trunk_leans.append(trunk_lean)
        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "wrist_spread": wrist_spread,
                "shoulder_width": shoulder_width,
                "wrist_spread_shoulder_norm": wrist_spread_shoulder_norm,
                "left_shoulder_ear_gap": left_shoulder_ear_gap,
                "right_shoulder_ear_gap": right_shoulder_ear_gap,
                "shoulder_ear_gap_shoulder_norm": shoulder_ear_gap_shoulder_norm,
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "min_elbow_angle": min_elbow_angle,
                "trunk_lean_image_signed_deg": trunk_lean,
                "wrist_depth_offset": wrist_depth_offset,
            }
        )

    # THE DERIVATIVE IS COMPUTED HERE, IN THE METRIC LAYER, AND THAT IS LOAD-BEARING.
    # `run_detector` median-filters EVERY key in `metric_keys` with a 5-frame window. A median
    # over a POSITION/ANGLE series flattens the velocity transient rule 4's whip evidence exists
    # to find, before the rule ever sees it. Emitting the derivative AS the metric means the
    # framework's filter acts on the velocity -- a defensible low-pass on the quantity of
    # interest instead of an erasure of it. Same argument row.py makes for `wrist_accel_norm`.
    trunk_speed = _derivative(trunk_leans, fps)
    for index, item in enumerate(raw):
        if not item.get("valid"):
            continue
        speed = trunk_speed[index]
        item["trunk_angle_speed_deg_s"] = abs(float(speed)) if np.isfinite(speed) else float(np.nan)
    return raw


def band_pull_apart_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> pull -> peak -> return, segmented on `wrist_spread_shoulder_norm`.

    Mirrors `row_assign_phases`, substituting the pull-apart's spread signal and inverting the
    polarity: the row's peak is the MOST-FLEXED 30% of the rep, this movement's peak is the
    WIDEST 30%. Same fallbacks: an empty clip returns an empty list, a clip with no finite signal
    is entirely `unknown`, and an invalid frame is `unknown` regardless of where it sits (the
    validity check precedes the setup cutoff, so an occluded frame in the opening 15% is NOT
    labelled `setup`, which matters because `_setup_baseline` reduces over exactly those frames).
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    spread_values = np.asarray(
        [float(item.get("wrist_spread_shoulder_norm", np.nan)) for item in raw], dtype=np.float32
    )
    valid_spread = spread_values[np.isfinite(spread_values)]
    if valid_spread.size == 0:
        return ["unknown" for _ in raw]

    # The widest 30% of the rep is the peak hold.
    peak_threshold = float(np.percentile(valid_spread, 70))
    widest_index = int(np.nanargmax(np.where(np.isfinite(spread_values), spread_values, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = spread_values[index]
        if np.isfinite(value) and value >= peak_threshold:
            phases.append("peak")
        elif index < widest_index:
            phases.append("pull")
        else:
            phases.append("return")
    return phases
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_band_pull_apart.py -q`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/band_pull_apart.py tests/test_band_pull_apart.py
git commit -m "feat(pose): band pull apart raw metrics and phase segmentation"
```

Commit body must record: this is the first movement whose defining excursion is frontal, so the rep signal itself is view-bound; `trunk_lean_image_signed_deg` ships raw rather than facing-corrected because the correction needs a floor and the metric layer holds no thresholds.

---

### Task 2: Rule 1 — `bpa_shrugging`

**Files:**
- Modify: `src/pose/movements/band_pull_apart.py`
- Test: `tests/test_band_pull_apart.py`

**Interfaces:**
- Consumes: Task 1's `band_pull_apart_compute_raw`, `band_pull_apart_assign_phases`; `src.pose.movements.base.RuleContext`; `src.pose.geometry.contiguous_true_segments`, `severity_from_range`; `src.pose.pose_rule_detector.VIEW_UNAVAILABLE_CONFIDENCE_SCALE`, `PoseRuleDetection`, `build_detection`.
- Produces: `_setup_baseline(core, key) -> float`, `rule_shrugging(core, ctx) -> list[PoseRuleDetection]`, constants `SHRUG_MILD = 0.03`, `SHRUG_SEVERE = 0.075`, `BPA_SHRUG_KG_QUERY = "Shoulder Shrugging"`, `_OFF_VIEW_CONFIDENCE`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_band_pull_apart.py` (add `from src.pose.movements.band_pull_apart import rule_shrugging` and `from src.pose.movements.base import RuleContext, run_detector` to the imports):

```python
def _ctx(view_type: str = "rear", view_confidence: float = 0.8, min_frames: int = 3) -> RuleContext:
    return RuleContext(
        fps=30.0, view_type=view_type, view_confidence=view_confidence, min_frames=min_frames
    )


def _core(frames: list[dict], fps: float = 30.0) -> list[CoreFrame]:
    """Metrics + phases WITHOUT run_detector, so a rule can be tested on an exact fixture.

    run_detector median-filters and re-slices per rep; both are correct in production and both
    would blur a boundary fixture built to sit one step either side of a threshold.
    """
    raw = band_pull_apart_compute_raw(frames, fps=fps)
    phases = band_pull_apart_assign_phases(raw)
    return [
        CoreFrame(
            frame_index=int(item.get("frame_index", i) or i),
            time=float(item.get("time", 0.0) or 0.0),
            phase=phases[i],
            valid=bool(item.get("valid", False)),
            lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
            metrics={k: float(item.get(k, np.nan)) for k in BAND_PULL_APART_METRIC_KEYS},
        )
        for i, item in enumerate(raw)
    ]


def _shrug_rep(peak_gap: float, setup_gap: float = 0.12, n: int = 20) -> list[dict]:
    """A rep whose setup frames hold `setup_gap` and whose widest frames hold `peak_gap`."""
    frames = []
    for i in range(n):
        wide = i >= int(n * 0.4)
        frames.append(
            bpa_frame(
                spread_ratio=1.9 if wide else 0.6,
                shoulder_ear_gap=peak_gap if wide else setup_gap,
                frame_index=i,
            )
        )
    return frames


class ShruggingRuleTest(unittest.TestCase):
    def test_fires_when_the_gap_closes_past_the_spec_threshold(self) -> None:
        # setup 0.12 -> peak 0.08 is a 0.04 closure, past the spec's 0.03.
        core = _core(_shrug_rep(peak_gap=0.08))
        detections = rule_shrugging(core, _ctx())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "bpa_shrugging")
        self.assertGreater(detections[0].severity, 0.0)

    def test_silent_just_inside_the_threshold(self) -> None:
        # 0.12 -> 0.095 is a 0.025 closure, inside the spec's 0.03.
        core = _core(_shrug_rep(peak_gap=0.095))
        self.assertEqual(rule_shrugging(core, _ctx()), [])

    def test_a_unilateral_shrug_still_fires(self) -> None:
        frames = []
        for i in range(20):
            wide = i >= 8
            frames.append(
                bpa_frame(
                    spread_ratio=1.9 if wide else 0.6,
                    shoulder_ear_gap=0.12,
                    left_gap_delta=-0.05 if wide else 0.0,
                    frame_index=i,
                )
            )
        detections = rule_shrugging(_core(frames), _ctx())
        self.assertEqual(len(detections), 1)

    def test_severity_reaches_one_at_the_ramp_endpoint(self) -> None:
        # 0.12 -> 0.045 is a 0.075 closure, the RULE-LEVEL ramp endpoint.
        core = _core(_shrug_rep(peak_gap=0.045))
        self.assertAlmostEqual(rule_shrugging(core, _ctx())[0].severity, 1.0, places=3)

    def test_the_metric_is_facing_free_so_rear_oblique_is_not_discounted(self) -> None:
        core = _core(_shrug_rep(peak_gap=0.08))
        rear = rule_shrugging(core, _ctx(view_type="rear"))[0]
        oblique = rule_shrugging(core, _ctx(view_type="rear_oblique"))[0]
        self.assertEqual(rear.observability, "high")
        self.assertEqual(oblique.observability, "high")
        self.assertAlmostEqual(rear.confidence, oblique.confidence, places=6)

    def test_a_nan_baseline_silences_the_rule(self) -> None:
        frames = _shrug_rep(peak_gap=0.08)
        for i in range(4):  # blank out every setup frame
            frames[i]["landmarks"][7] = _lm(0.4, 0.2, visibility=0.10)
        self.assertEqual(rule_shrugging(_core(frames), _ctx()), [])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_band_pull_apart.py -q -k Shrugging`
Expected: FAIL — `ImportError: cannot import name 'rule_shrugging'`

- [ ] **Step 3: Implement `_setup_baseline` and `rule_shrugging`**

Extend the imports in `src/pose/movements/band_pull_apart.py`:

```python
from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, distance,
    contiguous_true_segments, severity_from_range,
)
from src.pose.movements.base import CoreFrame, RuleContext
from src.pose.pose_rule_detector import (
    VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
    PoseRuleDetection,
    build_detection,
)
```

Append after `band_pull_apart_assign_phases`:

```python
# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query,
# movement="Band Pull Apart")` -- the function PRODUCTION calls, not just `resolve_nodes`.
# Observed results, not predicted ones:
#
#   "Shoulder Shrugging" -> Band Pull Apart:Shoulder Shrugging
#       causes: Weak Scapular Stabilizers | quality_impacts: Shoulder Depression      NON-EMPTY
#   "Bent Elbows"        -> Band Pull Apart:Bent Elbows
#       NO buckets -- only the HAS_FAULT backlink                                     THIN
#   rule 4               -> NO Band-Pull-Apart-scoped node exists at all
#       "Trunk Extension" / "Loss Of Neutral Body Position" (Row's queries) do not resolve
#       under this movement's scoping; the shared nodes that do resolve are bare.
#
# The two gaps are recorded rather than masked. Pointing rule 2 at the shared `Range Of Motion`
# QualityDimension WOULD return a rich bucket set, and was rejected: its `corrections` bucket is
# "Wrapping Surface Adjustment", meaningless for this movement. A semantically correct thin card
# beats a semantically wrong full one. Both gaps are one-line fixes in
# scripts/knowledge/stub_general_movements_v3.py:80-87 and are logged against TODO.md's existing
# "many faults have no KG node" item.
BPA_SHRUG_KG_QUERY = "Shoulder Shrugging"

# Imported rather than re-typed, so a change to the shared constant cannot silently skip this
# module.
_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE

# FROM THE SPEC: "flag shrug if `gap_peak < gap_setup - 0.03` (shoulders elevate) on either
# side". RAW IMAGE UNITS -- the spec states no normalizer here, and it says "normalized by
# shoulder width" explicitly where it means that (the very next rule), so the absence is
# meaningful rather than an omission. The honest cost is camera-distance dependence:
# the same shrug filmed further away yields a smaller closure and fires less. Implemented as
# written; `shoulder_ear_gap_shoulder_norm` is emitted as the scale-free companion that no rule
# fires on, so a future validation can compare the two without moving this number.
SHRUG_MILD = 0.03
# RULE-LEVEL CHOICE MADE HERE. The parent spec states NO severity ramp for ANY Band Pull Apart
# fault (the Lunge section states its ramps explicitly, so the absence is meaningful). 0.075 is
# 2.5x the fire threshold, the convention `pushup.rule_hip_sag` uses for exactly this situation.
# A display/ranking curve, not a cited quantity.
SHRUG_SEVERE = 0.075


def _setup_baseline(core: list[CoreFrame], key: str) -> float:
    """Median of `key` over this window's valid `setup` frames; NaN when there are none.

    WHY THE BASELINE LIVES IN THE RULES AND NOT IN `band_pull_apart_compute_raw`. Two of this
    movement's three firing rules are deltas from a setup baseline, and a baseline is a PER-REP
    reduction. `run_detector` calls `compute_raw` over the WHOLE CLIP before `segment_reps`, so
    at metric time no rep boundary exists and there is no "this rep's setup" to reduce against.
    Rules receive a per-rep slice, which is the first place the question is answerable.

    MEDIAN, NOT MEAN, so one bad frame in a short setup cannot move the reference every later
    comparison is made against.

    NEVER A GUESSED BASELINE -- but what a caller does with a NaN one is conditional on the
    caller's own fire condition, NOT a universal "return []" contract:
      - A rule whose fire condition depends ONLY on this baseline (`rule_shrugging`,
        `rule_trunk_extension_compensation`) has nothing left to evaluate and returns [].
      - A rule whose fire condition is a DISJUNCTION with a non-baseline term
        (`rule_incomplete_rom`, which fires on spread ratio OR elbow angle, neither of which is a
        baseline delta) never consults this function at all.

    STATED LIMITATION, inherited from `row._setup_baseline` where it was measured: `setup` is the
    first 15% of the REP WINDOW, and the window has already been trimmed by `segment_reps` to the
    rep's excursion -- so on a short rep `setup` can be 1-2 frames and may already overlap loaded
    frames. Because every comparison here is `peak - baseline`, a baseline biased toward the
    loaded state makes the MEASURED change smaller than the true one: the failure mode is a
    MISSED fault, never a false one. Not corrected -- there is no principled way to detect "this
    setup frame is already loaded" without a second threshold the parent spec does not supply.
    """
    values = [
        frame.m(key)
        for frame in core
        if frame.valid and frame.phase == "setup" and np.isfinite(frame.m(key))
    ]
    if not values:
        return float(np.nan)
    return float(np.median(values))


def rule_shrugging(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the shoulders rising toward the ears across the pull -- upper-trap dominance.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLD 0.03: FROM THE SPEC ("flag shrug if gap_peak < gap_setup - 0.03").
      SEVERITY RAMP 0.03 -> 0.075: A RULE-LEVEL CHOICE (see SHRUG_SEVERE).

    PHASE SCOPE `peak`, FROM THE SPEC's own wording ("Compute at setup baseline and at peak").

    EITHER SIDE, NOT THE MEAN, also FROM THE SPEC ("on either side"). A unilateral shrug is the
    common presentation and averaging the two gaps would halve it toward the threshold.

    NO VIEW DISCOUNT, AND THAT IS AN ARGUMENT RATHER THAN AN OVERSIGHT. This rule's metric is a
    VERTICAL image-y difference between a shoulder and its own ear. A magnitude in image-y reads
    identically from in front of or behind the subject, so the rule is facing-free BY
    CONSTRUCTION -- the same argument `row.rule_asymmetric_pull` and `lunge.rule_knee_valgus`
    make for their own metrics. `rear` and `rear_oblique` (between them, 43 of the 45 real pose
    JSONs in this repository) therefore both earn the spec's `high` rating with no discount.
    """
    left_baseline = _setup_baseline(core, "left_shoulder_ear_gap")
    right_baseline = _setup_baseline(core, "right_shoulder_ear_gap")
    if not np.isfinite(left_baseline) and not np.isfinite(right_baseline):
        return []

    def closure(frame: CoreFrame) -> float:
        """Largest gap CLOSURE across the two sides; NaN-safe, so one occluded ear does not
        silence the other side."""
        options = [
            base - frame.m(key)
            for base, key in (
                (left_baseline, "left_shoulder_ear_gap"),
                (right_baseline, "right_shoulder_ear_gap"),
            )
            if np.isfinite(base) and np.isfinite(frame.m(key))
        ]
        return float(max(options)) if options else float(np.nan)

    mask = [
        frame.valid
        and frame.phase == "peak"
        and np.isfinite(closure(frame))
        and closure(frame) > SHRUG_MILD
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        closures = [closure(frame) for frame in segment]
        max_closure = float(np.nanmax(closures))
        severity = severity_from_range(max_closure, SHRUG_MILD, SHRUG_SEVERE, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="bpa_shrugging",
                fault_name="Shrugging (Upper-Trap Dominance)",
                kg_query=BPA_SHRUG_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=closures,
                severity=severity,
                confidence=severity,
                observability="high",
                evidence={
                    "setup_left_gap": round(left_baseline, 4) if np.isfinite(left_baseline) else None,
                    "setup_right_gap": round(right_baseline, 4) if np.isfinite(right_baseline) else None,
                    "max_gap_closure": round(max_closure, 4),
                    "threshold": SHRUG_MILD,
                    "primary_label": "shoulder-ear gap closure vs setup",
                    "primary_value": round(max_closure, 4),
                    "primary_threshold": SHRUG_MILD,
                },
                citation="<COPY VERBATIM from parent spec line 729>",
                citation_support="<COPY VERBATIM from parent spec line 730>",
            )
        )
    return detections
```

**The two `<COPY VERBATIM ...>` placeholders are instructions, not content to ship.** Open `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` at the Band Pull Apart → Shrugging entry and copy the `citation` and `citation_support` field values into these two strings exactly as written there. Do not paraphrase and do not write them from memory.

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_band_pull_apart.py -q`
Expected: PASS (22 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/band_pull_apart.py tests/test_band_pull_apart.py
git commit -m "feat(pose): band pull apart shrugging rule"
```

Body records: the threshold is raw image units (camera-distance dependent) because the spec states no normalizer where it states one for the very next rule; the scale-free companion metric ships alongside so a future validation needs no threshold move. No view discount because a shoulder-to-own-ear image-y magnitude is facing-free by construction.

---

### Task 3: Rule 2 — `bpa_incomplete_rom`, with the spec's inequality corrected

**Files:**
- Modify: `src/pose/movements/band_pull_apart.py`
- Test: `tests/test_band_pull_apart.py`

**Interfaces:**
- Consumes: Task 2's `_OFF_VIEW_CONFIDENCE`, `contiguous_true_segments`, `severity_from_range`.
- Produces: `rule_incomplete_rom(core, ctx) -> list[PoseRuleDetection]`, constants `SPREAD_MILD = 1.6`, `SPREAD_SEVERE = 1.0`, `ELBOW_MILD_DEG = 150.0`, `ELBOW_SEVERE_DEG = 110.0`, `SPREAD_HIGH_VIEWS`, `BPA_ROM_KG_QUERY = "Bent Elbows"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_band_pull_apart.py` (add `rule_incomplete_rom` to the imports):

```python
def _rom_rep(spread_ratio: float, elbow_angle_deg: float = 170.0, n: int = 20) -> list[dict]:
    frames = []
    for i in range(n):
        wide = i >= int(n * 0.4)
        frames.append(
            bpa_frame(
                spread_ratio=spread_ratio if wide else 0.6,
                elbow_angle_deg=elbow_angle_deg if wide else 175.0,
                frame_index=i,
            )
        )
    return frames


class IncompleteRomRuleTest(unittest.TestCase):
    def test_fires_when_the_spread_falls_short(self) -> None:
        detections = rule_incomplete_rom(_core(_rom_rep(spread_ratio=1.3)), _ctx())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "bpa_incomplete_rom")

    def test_silent_at_full_spread_with_straight_arms(self) -> None:
        self.assertEqual(rule_incomplete_rom(_core(_rom_rep(spread_ratio=1.9)), _ctx()), [])

    def test_fires_on_bent_elbows_even_at_full_spread(self) -> None:
        """THE SPEC'S INEQUALITY IS INVERTED AND THIS TEST PINS THE CORRECTION.

        Parent spec line 739 reads `elbow_angle > ~150deg` while its own parenthetical says a
        bent-elbow cheat is the fault -- and >150 degrees is nearly STRAIGHT arms. The
        parenthetical is right. Implemented as `< 150`, so this fixture (140 degrees, full
        spread) MUST fire; under the spec's literal `>` it would be silent.
        """
        detections = rule_incomplete_rom(
            _core(_rom_rep(spread_ratio=1.9, elbow_angle_deg=140.0)), _ctx()
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].evidence["primary_label"], "elbow flexion at peak")

    def test_straight_arms_at_full_spread_do_not_fire(self) -> None:
        self.assertEqual(
            rule_incomplete_rom(_core(_rom_rep(spread_ratio=1.9, elbow_angle_deg=170.0)), _ctx()),
            [],
        )

    def test_spread_severity_reaches_one_at_the_ramp_endpoint(self) -> None:
        detections = rule_incomplete_rom(_core(_rom_rep(spread_ratio=1.0)), _ctx())
        self.assertAlmostEqual(detections[0].severity, 1.0, places=3)

    def test_rear_oblique_downgrades_because_spread_foreshortens(self) -> None:
        core = _core(_rom_rep(spread_ratio=1.3))
        rear = rule_incomplete_rom(core, _ctx(view_type="rear"))[0]
        oblique = rule_incomplete_rom(core, _ctx(view_type="rear_oblique"))[0]
        self.assertEqual(rear.observability, "high")
        self.assertEqual(oblique.observability, "medium")
        self.assertLess(oblique.confidence, rear.confidence)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_band_pull_apart.py -q -k IncompleteRom`
Expected: FAIL — `ImportError: cannot import name 'rule_incomplete_rom'`

- [ ] **Step 3: Implement `rule_incomplete_rom`**

Append to `src/pose/movements/band_pull_apart.py`:

```python
BPA_ROM_KG_QUERY = "Bent Elbows"

# FROM THE SPEC: "Flag if `wrist_spread_peak / shoulder_width < 1.6`". Explicitly normalized by
# shoulder width by the spec's own wording, so scale-free -- unlike SHRUG_MILD above.
SPREAD_MILD = 1.6
# RULE-LEVEL CHOICE MADE HERE. A DESCENDING ramp: severity grows as the spread SHRINKS. 1.0 is
# the spread ratio at which the wrists sit exactly at shoulder width -- no horizontal abduction
# beyond the torso line at all, which is the natural floor of this movement rather than an
# arbitrary 2.5x. Display/ranking curve, not a cited quantity.
SPREAD_SEVERE = 1.0

# FROM THE SPEC, WITH ITS INEQUALITY CORRECTED. Parent spec line 739 reads "elbow-extension
# check `elbow_angle > ~150deg` maintained (bent-elbow curl-style cheat = fault)". Read
# literally, >150 degrees -- nearly STRAIGHT arms -- is the fault, which contradicts the
# parenthetical in the same sentence. The parenthetical is right and the inequality is a slip: a
# bent-elbow cheat means a SMALLER elbow angle. Corroboration rather than inference alone: the
# knowledge graph names this fault "Bent Elbows"
# (scripts/knowledge/stub_general_movements_v3.py:85), and Fukunaga's rationale -- more range
# covered against the band drives higher activation -- is a range argument that bending the
# elbows shortens. The NUMBER 150 is unchanged and stays FROM THE SPEC; only the comparison
# direction is corrected, and the correction is annotated in the parent spec so it cannot be
# silently re-flipped by someone reading line 739 alone.
ELBOW_MILD_DEG = 150.0
# RULE-LEVEL CHOICE MADE HERE. 40 degrees of ramp width, taken from `pushup.rule_shallow_depth`
# (100 -> 140) so this module's elbow ramp and push-up's cannot drift apart. DESCENDING.
ELBOW_SEVERE_DEG = 110.0

# Views in which the spec rates wrist spread `high` ("high -- front / rear"). `front` is listed
# knowing it is unreachable under `allow_front=False`: it is correct on the merits and costs
# nothing. `rear_oblique` foreshortens the frontal-plane spread, so it downgrades.
SPREAD_HIGH_VIEWS = {"front", "rear"}


def rule_incomplete_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a pull that stops short -- hands not fully spread, or elbows bent to cheat the range.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLDS 1.6 and 150 deg: FROM THE SPEC (the latter with its inequality corrected,
      see ELBOW_MILD_DEG).
      SEVERITY RAMPS 1.6 -> 1.0 and 150 -> 110 deg: RULE-LEVEL CHOICES.

    A GENUINE DISJUNCTION, unlike `row.rule_momentum_jerk`'s second condition which was a strict
    SUBSET of its first and therefore unreachable. These two cues are independent failure modes:
    a lifter can reach full spread with bent elbows (short-lever cheat) or hold straight arms and
    stop short. `evidence["primary_label"]` records which term drove the verdict.

    PHASE SCOPE `peak`, FROM THE SPEC ("Peak wrist separation").

    NO SETUP BASELINE. Both terms are absolute thresholds on the peak, not deltas, so this rule
    never calls `_setup_baseline` and an occluded setup cannot silence it.
    """
    observable = ctx.view_type in SPREAD_HIGH_VIEWS
    scale = 1.0 if observable else _OFF_VIEW_CONFIDENCE

    def scores(frame: CoreFrame) -> tuple[float, float]:
        """(spread severity, elbow severity) for one frame; 0.0 where the term does not fire."""
        spread = frame.m("wrist_spread_shoulder_norm")
        elbow = frame.m("min_elbow_angle")
        spread_sev = (
            severity_from_range(spread, SPREAD_MILD, SPREAD_SEVERE, lower_is_worse=True)
            if np.isfinite(spread) and spread < SPREAD_MILD
            else 0.0
        )
        elbow_sev = (
            severity_from_range(elbow, ELBOW_MILD_DEG, ELBOW_SEVERE_DEG, lower_is_worse=True)
            if np.isfinite(elbow) and elbow < ELBOW_MILD_DEG
            else 0.0
        )
        return spread_sev, elbow_sev

    mask = [
        frame.valid and frame.phase == "peak" and max(scores(frame)) > 0.0 for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        pairs = [scores(frame) for frame in segment]
        combined = [max(p) for p in pairs]
        severity = float(np.nanmax(combined))
        worst = int(np.nanargmax(combined))
        spread_drove = pairs[worst][0] >= pairs[worst][1]
        min_spread = float(
            np.nanmin([frame.m("wrist_spread_shoulder_norm") for frame in segment])
        )
        min_elbow = float(np.nanmin([frame.m("min_elbow_angle") for frame in segment]))
        detections.append(
            build_detection(
                fault_id="bpa_incomplete_rom",
                fault_name="Incomplete ROM (Hands Not Fully Spread)",
                kg_query=BPA_ROM_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=combined,
                severity=severity,
                confidence=severity * scale,
                observability="high" if observable else "medium",
                evidence={
                    "min_spread_ratio": round(min_spread, 3),
                    "spread_threshold": SPREAD_MILD,
                    "min_elbow_angle_deg": round(min_elbow, 2),
                    "elbow_threshold_deg": ELBOW_MILD_DEG,
                    "primary_label": (
                        "wrist spread at peak" if spread_drove else "elbow flexion at peak"
                    ),
                    "primary_value": round(min_spread if spread_drove else min_elbow, 3),
                    "primary_threshold": SPREAD_MILD if spread_drove else ELBOW_MILD_DEG,
                },
                citation="<COPY VERBATIM from parent spec line 742>",
                citation_support="<COPY VERBATIM from parent spec line 743>",
            )
        )
    return detections
```

Replace both `<COPY VERBATIM ...>` placeholders from the parent spec's Band Pull Apart → Incomplete horizontal-abduction ROM entry, exactly as written there.

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_band_pull_apart.py -q`
Expected: PASS (28 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/band_pull_apart.py tests/test_band_pull_apart.py
git commit -m "fix(spec)+feat(pose): band pull apart ROM rule, with the spec's inequality corrected"
```

Body must state the inversion plainly: line 739 reads `> 150deg` while its own parenthetical calls a bent elbow the fault; `>150` is nearly straight arms, so the parenthetical is right and the inequality is a slip. Corroborated by the KG naming the fault "Bent Elbows". Number unchanged, direction corrected, and a test fires on a fixture the literal spec would leave silent.

---

### Task 4: The facing derivation and Rule 4 — `bpa_trunk_extension_compensation`

**Files:**
- Modify: `src/pose/movements/band_pull_apart.py`
- Test: `tests/test_band_pull_apart.py`

**Interfaces:**
- Consumes: Task 2's `_setup_baseline`, `_OFF_VIEW_CONFIDENCE`.
- Produces: `_clip_facing_sign(core) -> float`, `rule_trunk_extension_compensation(core, ctx) -> list[PoseRuleDetection]`, constants `TRUNK_LEAN_MILD_DEG = 10.0`, `TRUNK_LEAN_SEVERE_DEG = 25.0`, `FACING_DEGENERATE_OFFSET = 0.02`, `TRUNK_BLIND_VIEWS = {"rear", "unknown"}`, `BPA_TRUNK_KG_QUERY = "No Compensatory Trunk Movement"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_band_pull_apart.py` (add `rule_trunk_extension_compensation`, `_clip_facing_sign` to the imports):

```python
def _lean_rep(
    peak_lean_deg: float,
    setup_lean_deg: float = 0.0,
    wrist_depth_offset: float = -0.30,
    n: int = 20,
) -> list[dict]:
    frames = []
    for i in range(n):
        wide = i >= int(n * 0.4)
        frames.append(
            bpa_frame(
                spread_ratio=1.9 if wide else 0.6,
                trunk_lean_deg=peak_lean_deg if wide else setup_lean_deg,
                wrist_depth_offset=wrist_depth_offset,
                frame_index=i,
            )
        )
    return frames


class FacingDerivationTest(unittest.TestCase):
    def test_negative_offset_means_the_lifter_faces_the_camera(self) -> None:
        # Wrists nearer the camera than the shoulders (MediaPipe z is negative toward camera).
        self.assertEqual(_clip_facing_sign(_core(_lean_rep(0.0, wrist_depth_offset=-0.30))), 1.0)

    def test_positive_offset_means_the_lifter_faces_away(self) -> None:
        self.assertEqual(_clip_facing_sign(_core(_lean_rep(0.0, wrist_depth_offset=0.30))), -1.0)

    def test_all_zero_z_is_undetermined(self) -> None:
        """The RTMPose extraction path writes z=0.0 for EVERY landmark
        (src/pose/rtmpose_pose_extraction.py:121,131), so this is a real runtime, not a
        hypothetical. It must read as undetermined, never as a facing."""
        self.assertTrue(math.isnan(_clip_facing_sign(_core(_lean_rep(0.0, wrist_depth_offset=0.0)))))

    def test_offset_under_the_floor_is_undetermined(self) -> None:
        self.assertTrue(
            math.isnan(_clip_facing_sign(_core(_lean_rep(0.0, wrist_depth_offset=-0.01))))
        )


class TrunkExtensionRuleTest(unittest.TestCase):
    def test_fires_on_a_backward_lean_past_the_threshold(self) -> None:
        # Facing the camera (offset -0.30) => facing sign +1 => a POSITIVE image lean is
        # backward. 15 degrees clears the spec's 10.
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=-0.30))
        detections = rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique"))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "bpa_trunk_extension_compensation")

    def test_a_forward_lean_of_the_same_size_does_not_fire(self) -> None:
        """The whole point of the facing derivation: magnitude alone would fire here."""
        core = _core(_lean_rep(peak_lean_deg=-15.0, wrist_depth_offset=-0.30))
        self.assertEqual(
            rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique")), []
        )

    def test_the_verdict_inverts_when_the_lifter_faces_away(self) -> None:
        core = _core(_lean_rep(peak_lean_deg=-15.0, wrist_depth_offset=0.30))
        self.assertEqual(
            len(rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique"))), 1
        )

    def test_silent_just_inside_the_threshold(self) -> None:
        core = _core(_lean_rep(peak_lean_deg=8.0, wrist_depth_offset=-0.30))
        self.assertEqual(
            rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique")), []
        )

    def test_hard_gated_silent_on_a_confident_rear_label(self) -> None:
        """A signed sagittal lean read from a pure rear view is FRONTAL-plane lateral sway --
        a confident reading of the wrong plane, which no confidence discount can express."""
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=-0.30))
        self.assertEqual(rule_trunk_extension_compensation(core, _ctx(view_type="rear")), [])

    def test_hard_gated_silent_on_unknown(self) -> None:
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=-0.30))
        self.assertEqual(rule_trunk_extension_compensation(core, _ctx(view_type="unknown")), [])

    def test_silent_when_the_facing_is_undetermined(self) -> None:
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=0.0))
        self.assertEqual(
            rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique")), []
        )

    def test_observability_is_medium_not_high(self) -> None:
        """Downgraded from the spec's `high` because the facing derivation is an unvalidated
        precondition; the observability field should say so."""
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=-0.30))
        detection = rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique"))[0]
        self.assertEqual(detection.observability, "medium")

    def test_whip_speed_is_recorded_as_evidence_not_as_a_fire_condition(self) -> None:
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=-0.30))
        detection = rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique"))[0]
        self.assertIn("trunk_whip_deg_s", detection.evidence)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_band_pull_apart.py -q -k "Facing or TrunkExtension"`
Expected: FAIL — `ImportError: cannot import name '_clip_facing_sign'`

- [ ] **Step 3: Implement `_clip_facing_sign` and `rule_trunk_extension_compensation`**

Append to `src/pose/movements/band_pull_apart.py`:

```python
BPA_TRUNK_KG_QUERY = "No Compensatory Trunk Movement"

# FROM THE SPEC: "Flag if `trunk_lean_backward > 10deg` beyond setup baseline".
TRUNK_LEAN_MILD_DEG = 10.0
# RULE-LEVEL CHOICE MADE HERE. 2.5x the fire threshold, the `pushup.rule_hip_sag` convention.
TRUNK_LEAN_SEVERE_DEG = 25.0

# RULE-LEVEL, AND MEASURED RATHER THAN GUESSED -- but note what it is: a PLUMBING test that
# distinguishes "this runtime reports depth" from "this runtime reports zeros", not a fault
# threshold. Measured over the 49 real pose JSONs in data/runtime/pose_json (6 distinct clips
# carry real z, 3 are z-degenerate): every non-degenerate clip's median wrist-shoulder offset
# had magnitude >= 0.1295, and every degenerate clip sat at exactly 0.0. The two populations do
# not overlap. 0.02 sits about 6x below the smallest real value and far above zero.
FACING_DEGENERATE_OFFSET = 0.02

# HARD GATE, WRITTEN IN THE NEGATIVE, and the form matters as much as the members.
#
# WHY A GATE AT ALL, when Row's design doc argues "downgrade, never gate": this rule measures a
# SAGITTAL quantity. From a pure `rear` view the sagittal axis is perpendicular to the image
# plane, so a signed torso lean computed there reads LATERAL SWAY IN THE FRONTAL PLANE -- a
# different fault, or none. That is not a low-confidence reading of the right quantity (the case
# the x0.65 discount exists for); it is a confident reading of the WRONG PLANE. Row's objection
# to gating was that gated rules ship silent, and that does not apply here: the view the gate
# leaves standing is `rear_oblique`, the modal production label (30 of 45 real pose JSONs).
#
# WHY NEGATIVE rather than a {side, rear_oblique, front_oblique} whitelist: `front_oblique` is
# unreachable under `allow_front=False`, so a whitelist containing it is dead weight that READS
# as coverage. The negative form needs no edit if `allow_front` is ever enabled (it admits
# front/front_oblique automatically and correctly), and it fails in the safer direction -- an
# unanticipated future label is scored rather than silently dropped. `pushup.rule_elbow_flare` is
# the shipped precedent for a hard gate and takes exactly this negative shape.
#
# `unknown` is named explicitly because it means THE VIEW ESTIMATOR FAILED, not "a confirmed
# view" -- and this rule, unlike `row.rule_momentum_jerk`, genuinely depends on knowing the
# viewing plane, so it cannot wave the distinction away.
TRUNK_BLIND_VIEWS = {"rear", "unknown"}


def _clip_facing_sign(core: list[CoreFrame]) -> float:
    """+1.0 if the lifter faces the camera, -1.0 if away, NaN when undetermined.

    THE PROBLEM THIS SOLVES. `estimate_view_for_pose(allow_front=False)` relabels a genuinely
    FRONT-facing subject as `rear_oblique` (src/pose/view_estimation.py:368-370), so the view
    label conflates the two facings and CANNOT sign a sagittal offset. `overhead_press.py`
    handles this by ASSUMING a facing and documenting that the other facing inverts every
    sagittal reading in the module -- a coin flip per clip, on the losing side of which this rule
    would confidently report the OPPOSITE fault. Not adopted.

    WHY WRIST DEPTH IS THE RIGHT SIGNAL, and why it does not contradict this project's
    depth-bottleneck findings. A band pull apart holds the band IN FRONT OF THE TORSO by
    definition -- that is what the movement IS, from setup through peak. So the SIGN of
    (mean wrist z - mean shoulder z) identifies the facing. This is a BINARY, LARGE-MARGIN
    decision, not a metric-depth measurement: the Fit3D line found MediaPipe's depth unreliable
    for cue MAGNITUDES, which is a different demand from the sign of a tens-of-centimetres
    separation. It is also a MEASUREMENT PRECONDITION, not a fault threshold -- it decides which
    direction counts as backward and is never compared against a cited number, so no citation is
    being stretched to cover it.

    PER-REP, NOT PER-CLIP, and that is a deliberate narrowing of the design spec's wording.
    `run_detector` hands rules a per-rep slice, so a clip-level reduction is not reachable from
    here -- and per-rep is the safer scope anyway, because it keeps rep N's verdict independent
    of rep 1's frames, which this architecture deliberately does not couple.

    REDUCED OVER `peak` FRAMES, where the arms are most extended and the margin is largest, and
    by MEDIAN so per-frame z jitter cannot flip the sign mid-rep.

    UNDETERMINED SILENCES THE RULE -- the "can only ever SILENCE" guard category pushup.py
    documents. Two cases reach it: no finite z at all, and a median magnitude under
    FACING_DEGENERATE_OFFSET. The latter covers the RTMPose extraction path, which writes z=0.0
    for every landmark and therefore yields exactly 0.0 here -- rule 4 goes silent on that
    runtime automatically, with no runtime-specific branch anywhere in this module.
    """
    values = [
        frame.m("wrist_depth_offset")
        for frame in core
        if frame.valid and frame.phase == "peak" and np.isfinite(frame.m("wrist_depth_offset"))
    ]
    if not values:
        return float(np.nan)
    median = float(np.median(values))
    if abs(median) < FACING_DEGENERATE_OFFSET:
        return float(np.nan)
    # MediaPipe z is NEGATIVE toward the camera, so a negative offset (wrists nearer the camera
    # than the shoulders) means the lifter faces the camera.
    return 1.0 if median < 0.0 else -1.0


def rule_trunk_extension_compensation(
    core: list[CoreFrame], ctx: RuleContext
) -> list[PoseRuleDetection]:
    """Flag the lifter leaning BACKWARD to fling the band apart instead of using the shoulders.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLD 10 deg: FROM THE SPEC ("Flag if trunk_lean_backward > 10deg beyond setup
      baseline").
      SEVERITY RAMP 10 -> 25 deg: A RULE-LEVEL CHOICE (see TRUNK_LEAN_SEVERE_DEG).
      FACING FLOOR 0.02: A RULE-LEVEL CHOICE, measured (see FACING_DEGENERATE_OFFSET).

    PHASE SCOPE `pull` and `peak`, FROM THE SPEC ("synchronized with the pull").

    DIRECTIONAL, NOT A MAGNITUDE, and that is the whole design problem. Firing on |lean change|
    would flag a FORWARD lean as trunk-extension compensation -- relabeling a different quantity
    under this fault_id, which is exactly the defect that killed `row.rounded_thoracolumbar_spine`
    construction 2. Hence `_clip_facing_sign`.

    THE SPEC'S SECOND CUE IS EVIDENCE, NOT A FIRE CONDITION. "or a trunk-angle velocity spike
    co-occurs with the concentric" is recorded as `evidence["trunk_whip_deg_s"]`, which
    distinguishes a slow lean from a whip for the coaching cue without changing what fires --
    the same treatment `row.rule_momentum_jerk` gives its own co-occurrence clause.

    OBSERVABILITY `medium`, DOWNGRADED FROM THE SPEC'S `high`, ON PURPOSE. The fault is highly
    visible to a human, but this detector's reading of it rests on a facing precondition that no
    band-pull-apart clip has ever confirmed. The observability field should say so.

    HARM CLAIM IS PARTLY INFERENTIAL, and the parent spec says so itself -- Fukunaga even notes
    trunk extension can be deliberately engaged. Restated here rather than quietly upgraded.
    """
    if ctx.view_type in TRUNK_BLIND_VIEWS:
        return []
    facing = _clip_facing_sign(core)
    if not np.isfinite(facing):
        return []
    baseline = _setup_baseline(core, "trunk_lean_image_signed_deg")
    if not np.isfinite(baseline):
        return []

    def backward_lean(frame: CoreFrame) -> float:
        """Degrees of BACKWARD lean beyond setup. Negative = forward, which never fires."""
        value = frame.m("trunk_lean_image_signed_deg")
        if not np.isfinite(value):
            return float(np.nan)
        return float((value - baseline) * facing)

    mask = [
        frame.valid
        and frame.phase in ("pull", "peak")
        and np.isfinite(backward_lean(frame))
        and backward_lean(frame) > TRUNK_LEAN_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        leans = [backward_lean(frame) for frame in segment]
        max_lean = float(np.nanmax(leans))
        severity = severity_from_range(
            max_lean, TRUNK_LEAN_MILD_DEG, TRUNK_LEAN_SEVERE_DEG, lower_is_worse=False
        )
        speeds = [
            frame.m("trunk_angle_speed_deg_s")
            for frame in segment
            if np.isfinite(frame.m("trunk_angle_speed_deg_s"))
        ]
        detections.append(
            build_detection(
                fault_id="bpa_trunk_extension_compensation",
                fault_name="Trunk-Extension Compensation (Leaning Back)",
                kg_query=BPA_TRUNK_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=leans,
                severity=severity,
                confidence=severity * _OFF_VIEW_CONFIDENCE,
                observability="medium",
                evidence={
                    "setup_trunk_lean_deg": round(baseline, 2),
                    "max_backward_lean_deg": round(max_lean, 2),
                    "threshold_deg": TRUNK_LEAN_MILD_DEG,
                    "facing_sign": facing,
                    "trunk_whip_deg_s": round(float(np.nanmax(speeds)), 2) if speeds else None,
                    "primary_label": "backward trunk lean vs setup",
                    "primary_value": round(max_lean, 2),
                    "primary_threshold": TRUNK_LEAN_MILD_DEG,
                },
                citation="<COPY VERBATIM from parent spec line 768>",
                citation_support="<COPY VERBATIM from parent spec line 769>",
            )
        )
    return detections
```

Replace both `<COPY VERBATIM ...>` placeholders from the parent spec's Band Pull Apart → Trunk-extension compensation entry, exactly as written there. That entry's `citation_support` already contains the "partly inferential" caveat — copy it in full, including that caveat.

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_band_pull_apart.py -q`
Expected: PASS (41 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/band_pull_apart.py tests/test_band_pull_apart.py
git commit -m "feat(pose): band pull apart trunk-extension rule, with facing derived from wrist depth"
```

Body records: the view label cannot sign a sagittal lean because `allow_front=False` relabels front-facing subjects as `rear_oblique`; facing is instead derived from the sign of the wrist-shoulder z offset, which is sound because the band is held in front of the torso by definition and the decision is binary and large-margin rather than a metric-depth claim. Undetermined facing silences the rule, which is what the RTMPose path (z identically 0.0) gets automatically.

---

### Task 5: The silent rule, detector assembly, registry, and end-to-end segmentation

**Files:**
- Modify: `src/pose/movements/band_pull_apart.py`
- Modify: `src/pose/movements/registry.py:36` (append one import line)
- Modify: `tests/test_analyze_pose_service.py:110-124`
- Test: `tests/test_band_pull_apart.py`

**Interfaces:**
- Consumes: all four `rule_*` functions, `src.pose.movements.base.MovementDetector`, `src.pose.movements.registry.register`.
- Produces: `rule_loss_of_scapular_retraction(core, ctx) -> list[PoseRuleDetection]` (always `[]`), `BAND_PULL_APART_DETECTOR: MovementDetector`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_band_pull_apart.py` (add `BAND_PULL_APART_DETECTOR`, `rule_loss_of_scapular_retraction` to the imports, plus `from src.pose.movements import registry`):

```python
class SilentRuleTest(unittest.TestCase):
    def test_the_scapular_retraction_rule_is_registered_but_always_silent(self) -> None:
        """Registered so spec and code stay 1:1 at four rules, silent because MediaPipe has no
        scapular landmarks. Pinned so a future edit cannot un-silence it unnoticed."""
        self.assertIn(rule_loss_of_scapular_retraction, BAND_PULL_APART_DETECTOR.rules)
        # Input that superficially LOOKS like the fault: arms spread wide, shoulders unchanged.
        core = _core(_rom_rep(spread_ratio=2.0))
        for view in ("rear", "rear_oblique", "side", "unknown"):
            self.assertEqual(rule_loss_of_scapular_retraction(core, _ctx(view_type=view)), [])


class DetectorAssemblyTest(unittest.TestCase):
    def test_registered_under_the_canonical_frontend_name(self) -> None:
        """`get_detector` keys on name.lower(); the studio sends the string from
        frontend/src/lib/movements.ts. A spelling drift makes the movement unselectable."""
        self.assertEqual(BAND_PULL_APART_DETECTOR.name, "Band Pull Apart")
        self.assertIs(registry.get_detector("Band Pull Apart"), BAND_PULL_APART_DETECTOR)
        self.assertIs(registry.get_detector("band pull apart"), BAND_PULL_APART_DETECTOR)

    def test_ships_unvalidated(self) -> None:
        self.assertFalse(BAND_PULL_APART_DETECTOR.validated)

    def test_all_four_spec_rules_are_present(self) -> None:
        self.assertEqual(len(BAND_PULL_APART_DETECTOR.rules), 4)

    def test_rep_signal_is_a_declared_metric_key(self) -> None:
        """`run_detector` indexes `smoothed[detector.rep_signal]`, which is built from
        metric_keys -- a rep_signal outside that tuple is a KeyError at analysis time."""
        self.assertIn(BAND_PULL_APART_DETECTOR.rep_signal, BAND_PULL_APART_METRIC_KEYS)
        self.assertEqual(BAND_PULL_APART_DETECTOR.rep_polarity, "max")


class EndToEndSegmentationTest(unittest.TestCase):
    def test_three_reps_are_segmented_from_a_three_rep_clip(self) -> None:
        """THE SILENT-ZERO GUARD, AND IT IS NOT OPTIONAL.

        Every rule test above builds CoreFrames directly, so all of them would pass while
        `segment_reps` returned ZERO reps and production silently ran the whole-clip fallback on
        every clip. The rep signal here is FRONTAL (wrist spread), unlike the six shipped
        detectors' sagittal signals, and RS-SP1's own audit says its table is interface-design
        inference rather than verified fact. This test is the only thing standing between that
        inference and a silently mis-segmenting detector.
        """
        ratios = [0.4, 0.7, 1.2, 1.7, 2.0, 1.7, 1.2, 0.7, 0.4]
        frames = []
        for rep in range(3):
            for r in ratios:
                frames.append(bpa_frame(spread_ratio=r, frame_index=len(frames)))
        result = run_detector(
            BAND_PULL_APART_DETECTOR,
            frames,
            fps=30.0,
            view_type="rear_oblique",
            view_confidence=0.8,
        )
        self.assertIsNone(result.fallback)
        self.assertEqual(len(result.reps), 3)

    def test_a_clean_three_rep_clip_raises_no_faults(self) -> None:
        ratios = [0.4, 0.7, 1.2, 1.7, 2.0, 1.7, 1.2, 0.7, 0.4]
        frames = []
        for rep in range(3):
            for r in ratios:
                frames.append(bpa_frame(spread_ratio=r, frame_index=len(frames)))
        result = run_detector(
            BAND_PULL_APART_DETECTOR,
            frames,
            fps=30.0,
            view_type="rear_oblique",
            view_confidence=0.8,
        )
        self.assertEqual(result.detections, [])

    def test_phases_are_one_per_frame_through_the_framework(self) -> None:
        """`run_detector` RAISES if assign_phases returns the wrong length (base.py:174)."""
        ratios = [0.4, 0.7, 1.2, 1.7, 2.0, 1.7, 1.2, 0.7, 0.4]
        frames = [bpa_frame(spread_ratio=r, frame_index=i) for i, r in enumerate(ratios * 3)]
        result = run_detector(
            BAND_PULL_APART_DETECTOR, frames, fps=30.0, view_type="rear", view_confidence=0.8
        )
        self.assertEqual(len(result.core), len(frames))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_band_pull_apart.py -q -k "Silent or Assembly or EndToEnd"`
Expected: FAIL — `ImportError: cannot import name 'BAND_PULL_APART_DETECTOR'`

- [ ] **Step 3: Implement the silent rule and assemble the detector**

Append to `src/pose/movements/band_pull_apart.py`. Extend the `base` import to include `MovementDetector`, and add the registry import in the **module form every sibling uses** — `from src.pose.movements import registry`, then `registry.register(...)` at module scope. Not `from ...registry import register`: `registry.py` imports this module for its side effect, so the two are mutually importing, and binding the module object rather than a name out of a partially-initialised module is what `squat.py:331`, `pushup.py:1608`, `lunge.py:786`, `deadlift.py:706`, `overhead_press.py:721` and `row.py:1123` all do.

```python
def rule_loss_of_scapular_retraction(
    core: list[CoreFrame], ctx: RuleContext
) -> list[PoseRuleDetection]:
    """Registered but PERMANENTLY SILENT -- always returns [].

    Loss of scapular retraction is a real, cited band-pull-apart fault: Fukunaga (PMC8975561)
    found middle-trapezius activity driven by the retraction-oriented directions, and the
    exercise is framed around recruiting the periscapular muscles. The fault is genuine. What
    fails is the SENSING, and the parent spec's prescribed heuristic fails twice over.

    (a) ITS FIRE CONDITION IS A NULL-DETECTION. The spec says: flag when wrist spread increases
        while `dist(11,12)` changes by LESS THAN 0.01 -- i.e. fire when the shoulder width FAILS
        TO CHANGE. A steady frame, a partially occluded frame, and a frame where the lifter
        genuinely does not retract are indistinguishable to that test; all three satisfy it.
        Every correctly performed rep that holds the shoulders stable would fire the fault. A
        rule whose positive class is "nothing measurable happened" cannot separate the fault from
        the absence of evidence.

    (b) THE METRIC IS CONFOUNDED WITH WHAT IT MUST BE INDEPENDENT OF. MediaPipe's shoulder
        landmark is a GLENOHUMERAL point, not a scapular border point, and it moves with the
        humerus. During horizontal abduction the humerus is exactly what is moving, so
        `dist(11,12)` changes for reasons unrelated to scapular adduction and cannot attribute an
        observed narrowing to retraction rather than to arm position. Root cause: MediaPipe Pose
        has NO scapular landmarks -- no medial border, no inferior angle -- so no construction
        over its 33 points measures scapular position. Same root cause as
        `pushup.rule_scapular_winging` and `row`'s fifth rule.

    Separately, the `0.01` figure carries no citation; Fukunaga supplies no landmark-displacement
    magnitude in any units.

    SILENT, NOT WITHDRAWN, AND THE DISTINCTION IS LOAD-BEARING. This project has two treatments
    for a rule it will not fire. Registered-but-silent (pushup.rule_scapular_winging, row's
    fifth) says "real, well-cited fault; the sensor cannot see it". Withdrawn from the parent
    spec (OHP bar-path 2026-07-25, deadlift bar-drift 2026-08-01) says "no citation supports the
    rule as written". Fukunaga genuinely backs retraction as the mechanism, so this is a sensing
    failure, not a citation failure, and it takes the silent treatment. The parent spec carries a
    NOTE, not a WITHDRAWN blockquote.

    NOT SUBSTITUTED, DELIBERATELY. Scapular contour from a rear view and shoulder-to-spine
    distance both carry SOME retraction information and neither is recoverable from 33 landmarks.
    Shipping a different metric under this fault_id would attach Fukunaga's citation to a
    quantity Fukunaga says nothing about -- the fabrication this project's anti-hallucination
    rule forbids.

    THE KG IS NOT THE GAP: `Band Pull Apart:Insufficient Scapular Retraction` resolves with a
    non-empty `causes` bucket ("Limited Scapular Retraction"). The metric is the gap.
    """
    return []


# ALL FOUR of the parent spec's Band Pull Apart rules are listed, deliberately: three can fire
# and `rule_loss_of_scapular_retraction` is permanently silent so the spec and the code stay in
# 1:1 correspondence (see its docstring). Registering it costs one no-op call per clip and buys
# an auditor the answer "yes, it is accounted for, and here is why it says nothing" -- the same
# trade `pushup.rule_scapular_winging` makes. Contrast `deadlift`'s withdrawn bar-drift rule,
# which is ABSENT rather than silent because its problem was the citation, not the sensor.
#
# `BAND_PULL_APART_METRIC_KEYS` must stay a two-way match with what `band_pull_apart_compute_raw`
# emits (pinned by `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits
# is dropped by `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and
# read back as NaN by every rule.
BAND_PULL_APART_DETECTOR = MovementDetector(
    "Band Pull Apart",
    BAND_PULL_APART_METRIC_KEYS,
    band_pull_apart_compute_raw,
    band_pull_apart_assign_phases,
    (
        rule_shrugging,
        rule_incomplete_rom,
        rule_loss_of_scapular_retraction,
        rule_trunk_extension_compensation,
    ),
    # `validated` stays at its default False, and that is not a formality. REHAB24-6 holds arm
    # abduction, arm VW, table push-ups, leg abduction, lunge and squats -- no band pull apart.
    # Fit3D DOES have `band pull apart` video with 3D mocap ground truth and rep boundaries
    # (docs/movement-kg-expansion-plan.md:33,48), but no binary correct/incorrect label on any
    # rep, so it cannot support a REHAB24-6-style fire-rate/AUC-against-correctness check. NO
    # labeled-CORRECTNESS band pull apart repetition exists anywhere in this repository, so no
    # threshold here has ever been checked against a rep judged correct or incorrect by a human.
    # Beta is the factual label.
    rep_signal="wrist_spread_shoulder_norm",
    # `max`, not the `min` five of the six shipped detectors use: this movement's excursion is
    # hands-together -> spread -> together, so the rep peaks at the signal's MAXIMUM. Assigned by
    # the RS-SP1 design spec's 16-movement audit (docs/superpowers/specs/
    # 2026-07-26-rep-segmentation-sp1-design.md section 3.4), which places Band Pull Apart in the
    # "clean unipolar excursion, all defaults" group -- an interface-design inference that
    # `EndToEndSegmentationTest` is what actually verifies.
    rep_polarity="max",
    rep_start="extended",
)

registry.register(BAND_PULL_APART_DETECTOR)
```

- [ ] **Step 4: Add the side-effect import to the registry**

Modify `src/pose/movements/registry.py` — append after line 36:

```python
from src.pose.movements import band_pull_apart  # noqa: E402,F401
```

Also update the `list_detectors` docstring on line 22, which enumerates registration order:

```python
    Registration order is the import order at the bottom of this module (Squat, Overhead
    Press, Push-up, Lunge, Deadlift, Row, Band Pull Apart) -- deterministic, and it puts the
```

- [ ] **Step 5: Rotate the stale example in `tests/test_analyze_pose_service.py`**

Replace lines 110–118 (the comment and the `assertNotIn`), and every `"Band Pull Apart"` string in that test method, with `"Bicep Curl"`:

```python
    def test_unknown_movement_returns_coming_soon_without_detector(self) -> None:
        # THIS EXAMPLE HAS TO BE ROTATED EVERY TIME A DETECTOR IS REGISTERED, and that is the
        # point of the assertion below rather than a nuisance: the test needs a movement the
        # frontend lists (frontend/src/lib/movements.ts) that has NO registered detector, so it
        # necessarily goes stale as the 16-movement programme lands one movement at a time. It
        # has already moved "Deadlift" -> "Row" -> "Band Pull Apart" -> "Bicep Curl"; when Bicep
        # Curl is implemented, move it again to any still-unimplemented movement. The
        # `assertNotIn` is what turns that staleness into a loud failure instead of a silently
        # vacuous test.
        self.assertNotIn("Bicep Curl", [d.name for d in registry.list_detectors()])
```

Then update the `movement="Band Pull Apart"` argument on line 122 to `movement="Bicep Curl"`.

- [ ] **Step 6: Extend the shared rep-segmentation table test**

`tests/test_movement_registry.py` holds an `expected` dict (around line 222–229) pinning every registered detector's `(rep_signal, rep_polarity, rep_start)`. It is keyed by movement name and iterated, so a new detector is not covered until it is added. Add the entry:

```python
            # The first movement whose rep signal is FRONTAL rather than sagittal, and the
            # second (after OHP) to peak at its signal's maximum -- hands together -> spread ->
            # together. Assigned by RS-SP1 spec §3.4; verified end-to-end in
            # tests/test_band_pull_apart.py::EndToEndSegmentationTest.
            "Band Pull Apart": ("wrist_spread_shoulder_norm", "max", "extended"),
```

and update the stale comment three lines below it:

```python
                # `rep_rectify` exists for movements RS-SP1 does not implement (spec §3.4);
                # all seven registered detectors use the default.
```

- [ ] **Step 7: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS. If any other test fails, investigate the cause — do not adjust a threshold or a number to make it pass (Global Constraints: no threshold tuning). Check a suspected flake against a baseline run on `main` before attributing it to this change.

- [ ] **Step 8: Run the coverage gate**

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS at ≥95%.

- [ ] **Step 9: Commit**

```bash
git add src/pose/movements/band_pull_apart.py src/pose/movements/registry.py \
        tests/test_band_pull_apart.py tests/test_analyze_pose_service.py \
        tests/test_movement_registry.py
git commit -m "feat(pose): register the Band Pull Apart detector, seventh of sixteen"
```

Body records: the fourth rule is registered-but-silent rather than withdrawn because Fukunaga backs retraction as the mechanism — the sensor fails, not the citation; and the end-to-end three-rep test is the guard for this movement's frontal rep signal, which RS-SP1 assigned by inference rather than measurement.

---

### Task 6: Parent-spec annotations and TODO

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` (Band Pull Apart section, lines 713–769)
- Modify: `TODO.md` (lines ~8 and ~401)

**Interfaces:** Documentation only. No code, no tests.

- [ ] **Step 1: Annotate `loss_of_scapular_retraction` in the parent spec**

Insert immediately after the `loss_of_scapular_retraction` entry's `citation_support` line (line 756), a blockquote — a **`NOTE`**, not a `WITHDRAWN` blockquote, because the failure is sensing rather than citation:

```markdown
> **NOTE — implemented as a permanently-silent rule (2026-08-09).** `rule_loss_of_scapular_retraction`
> is registered in `src/pose/movements/band_pull_apart.py` and always returns `[]`, following
> `pushup.rule_scapular_winging`. Two independent defects in the heuristic above, either
> disqualifying:
>
> 1. **The fire condition is a null-detection.** It fires when `dist(11,12)` *fails to change*
>    ("change < 0.01"), so a steady frame, a partially occluded frame, and a genuine non-retraction
>    are indistinguishable. Every correct rep that holds the shoulders stable would fire it.
> 2. **The metric is confounded with what it must be independent of.** MediaPipe's shoulder
>    landmark is a *glenohumeral* point that moves with the humerus, and horizontal abduction is
>    exactly the humeral motion in question, so `dist(11,12)` cannot attribute a narrowing to
>    scapular adduction rather than arm position. Root cause: MediaPipe Pose has no scapular
>    landmarks.
>
> Separately, `0.01` carries no citation; Fukunaga supplies no landmark-displacement magnitude.
>
> **A NOTE and not a WITHDRAWAL, deliberately.** Fukunaga genuinely backs retraction as the
> training mechanism, so the fault is real and cited and it is the *sensing* that fails — the
> `pushup.rule_scapular_winging` case, not the OHP-bar-path / deadlift-bar-drift case. The KG is
> not the gap either: `Band Pull Apart:Insufficient Scapular Retraction` resolves with a non-empty
> `causes` bucket. The metric is the gap.
```

- [ ] **Step 2: Annotate the inverted inequality in the parent spec**

Insert immediately after the `incomplete_horizontal_abduction_rom` entry's `citation_support` line (line 743):

```markdown
> **NOTE — direction inversion in this rule's elbow cue, corrected in implementation
> (2026-08-09).** The `detection_heuristic` above reads "elbow-extension check `elbow_angle >
> ~150deg` maintained (bent-elbow curl-style cheat = fault)". Read literally, `> 150°` — nearly
> *straight* arms — is the fault, contradicting the parenthetical in the same sentence. The
> parenthetical is right: a bent-elbow cheat means a *smaller* elbow angle.
> `src/pose/movements/band_pull_apart.py` implements **`min_elbow_angle < 150°`**. The number
> `150` is unchanged and remains FROM THE SPEC; only the comparison direction is corrected.
> Corroboration beyond the parenthetical: the KG names this fault `Bent Elbows`
> (`scripts/knowledge/stub_general_movements_v3.py:85`), and Fukunaga's rationale — more range
> covered against the band drives higher activation — is a range argument that bending the elbows
> shortens.
```

- [ ] **Step 3: Update `TODO.md`**

Line ~8, change `**規則偵測器 6/16 動作**` to `**規則偵測器 7/16 動作**` and append `band pull apart` to the movement list on line 9.

Line ~401, change `已上線 6 個` to `已上線 7 個` and add `` `band_pull_apart` `` to the backticked list.

Line ~406, change `其餘 10 個動作的 rule pack 未做` to `其餘 9 個動作的 rule pack 未做`.

Also append to the KG gap item near line 434 (「許多錯誤沒有對應到 Knowledge Graph 的節點」):

```markdown
  - Band Pull Apart 具體案例（2026-08-09）：`Bent Elbows` 節點存在但 connectivity 0
    （沒有 cause / risk / correction），`trunk_extension_compensation` 則完全沒有對應節點。
    兩者都是 `scripts/knowledge/stub_general_movements_v3.py:80-87` 的一行修正，
    但 graphml 已 gitignore，重新產生屬部署步驟。
```

- [ ] **Step 4: Verify nothing else claims six detectors**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS (docs-only change; this confirms Step 3 touched no code path).

Then grep for other stale counts:

Run: `git grep -n "6/16\|6 個" -- TODO.md docs/`
Expected: no remaining occurrence that refers to the *detector* count. Occurrences about other subjects stay.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md TODO.md
git commit -m "docs: annotate the Band Pull Apart spec defects and record 7/16 detectors"
```

Body records both annotations and why the scapular-retraction one is a NOTE rather than a WITHDRAWAL.

---

## Self-Review

**Spec coverage.** Design spec §1 → Task 5 (name, registration). §2 (`validated=False`) → Task 5 Step 3. §3 + §3.1 (silent rule, silent-vs-withdrawn) → Task 5 Step 3 and Task 6 Step 1. §4.1 (rep signal, polarity, the silent-zero guard) → Task 5 `EndToEndSegmentationTest`. §4.2 (phases) → Task 1. §4.3 (setup baseline, NaN policy) → Task 2 `_setup_baseline`, tested in Tasks 2 and 4. §4.4 (metrics, validity gate, derivative-before-smoothing) → Task 1. §4.5 (threshold units, scale-free companion) → Task 1 metric + Task 2 `SHRUG_MILD` comment. §4.6 (all four rules, ramp direction via `lower_is_worse`) → Tasks 2–5. §4.7 (view handling, negative gate) → Task 3 `SPREAD_HIGH_VIEWS`, Task 4 `TRUNK_BLIND_VIEWS`. §4.8 + §4.8.1 (facing derivation, measured floor) → Task 4 `_clip_facing_sign`. §4.9 (inequality correction) → Task 3 + Task 6 Step 2. §5 (KG queries) → Task 2 Step 3 header block. §6 (testing) → all tasks; the nine listed coverage areas map to the test classes in Tasks 1–5. §7 (honesty constraints) → Global Constraints + Task 6. §8 (out of scope) → File Structure note.

**Placeholder scan.** The only `<COPY VERBATIM ...>` markers are in Tasks 2, 3 and 4, and each is an explicit instruction with the parent-spec line number to copy from — the Global Constraints forbid writing citations from memory, so these must not be pre-filled here. Every other step contains runnable content.

**Type consistency.** `band_pull_apart_compute_raw` / `band_pull_apart_assign_phases` / `BAND_PULL_APART_METRIC_KEYS` / `BAND_PULL_APART_DETECTOR` are spelled identically in Tasks 1, 2, 3, 4 and 5. `_setup_baseline(core, key)` is defined in Task 2 and consumed in Task 4 with the same signature. `_clip_facing_sign(core)` is defined and consumed in Task 4. `_OFF_VIEW_CONFIDENCE` is defined in Task 2 and consumed in Tasks 3 and 4. The metric name `trunk_lean_image_signed_deg` is used consistently in Tasks 1 and 4 and its divergence from the design spec's `trunk_lean_signed_deg` is called out at the top of Task 1.

**Two corrections made during this review, both from reading the code rather than inferring it.** (1) The registration idiom is `from src.pose.movements import registry` + `registry.register(...)`, which all six siblings use; the from-import form originally written here would bind a name out of a partially-initialised module, since `registry.py` imports this module for its side effect. (2) No test hard-codes a detector *count*, but `tests/test_movement_registry.py` holds a name-keyed `(rep_signal, rep_polarity, rep_start)` table that silently omits any detector not added to it — so Task 5 Step 6 adds the entry explicitly rather than leaving the implementer to discover the omission, which no failing test would have surfaced.
