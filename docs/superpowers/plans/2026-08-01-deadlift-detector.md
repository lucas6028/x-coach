# Deadlift Rule Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `src/pose/movements/deadlift.py` with three cited fault rules, registered as the 5th of 16 movement detectors, plus the parent-spec amendments that record one withdrawn rule and two KG gaps.

**Architecture:** A new movement module following `src/pose/movements/lunge.py` exactly: `deadlift_compute_raw` emits threshold-free per-frame metrics, `deadlift_assign_phases` labels five phases per rep, three `rule_*` functions own every number that decides anything, and `DEADLIFT_DETECTOR` binds them together and self-registers. `run_detector` in `base.py` already drives all of this; the only framework feature newly exercised is `rep_start="flexed"`, which `base.py:55` was written for.

**Tech Stack:** Python 3.11/3.12, numpy, stdlib. `unittest.TestCase` under `tests/`. No new dependencies.

**Design spec:** `docs/superpowers/specs/2026-08-01-deadlift-detector-design.md`
**Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` §Deadlift (lines 213–263)

## Global Constraints

- **Interpreter is always `.venv\Scripts\python.exe` from the repo root.** This machine has no `python` on PATH. Never `source .venv/bin/activate`. In this worktree, invoke it by absolute path: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe`.
- **Run everything from the worktree root** — modules import by absolute package path (`from src.pose... import ...`).
- **Tests are always scoped to `tests/`**: `... -m pytest tests/`. Never bare `pytest`.
- **The metric layer contains no thresholds.** `deadlift_compute_raw` and `deadlift_assign_phases` emit scale-free metrics and phase labels only. Every deciding number lives in a `rule_*` function. The sole exception is `_DEGENERATE_LENGTH = 1e-6`, a division guard, not a tunable.
- **Every threshold is spec-derived and unvalidated.** No labeled deadlift data exists. `DEADLIFT_DETECTOR.validated` stays `False`. Do not flip it.
- **`DEADLIFT_METRIC_KEYS` must be a two-way match with what `deadlift_compute_raw` emits.** `run_detector` builds each `CoreFrame.metrics` *from* this tuple, so a key the tuple omits is silently dropped and read back as NaN. Task 1 pins this with a test.
- **Confidence discount for an unavailable view is `VIEW_UNAVAILABLE_CONFIDENCE_SCALE`** (0.65), imported from `src.pose.pose_rule_detector`. Never hardcode 0.65.
- **The side-view confidence floor is `SIDE_VIEW_CONF_THRESHOLD`** (0.20), same import. No new number.
- **Coverage gates:** `... -m pytest tests/` all green, and `... scripts/run_backend_coverage.py --fail-under 95`.
- **No frontend change.** `frontend/src/lib/movements.ts` already lists Deadlift, and analyzability is derived at runtime from `GET /api/movements`. `pages.Movements.test.tsx` mocks `getMovements` with a hardcoded fixture, so no frontend test is affected.
- **Every rule needs ONE test with per-frame VARYING metrics, not just the constant-fixture ones written below.** Added 2026-08-01 after the Task 4 review. `_frames()` builds identical `CoreFrame`s, so every segment in these test blocks has a constant score across frames — which means `build_detection`'s `nanargmax` returns index 0 whatever the sign convention, and `nanmin`/`nanmax` are indistinguishable. Concretely: in `rule_lumbar_flexion`, flipping `score_values=[-r for r in ratios]` to `score_values=ratios` failed **zero** tests. Each rule must therefore also assert `peak_frame` (and `start_frame`, to pin the slice offset) against a window whose metric varies across frames. Prove the test discriminates by making the mutation, watching it fail, and reverting — an untripped tripwire is not known to work. This applies to `rule_incomplete_lockout` and `rule_hips_shoot_up` as much as to `rule_lumbar_flexion`.
- **`kg_query` strings are VERIFIED RETRIEVAL INPUTS — copy them verbatim, never paraphrase.** `pose_rule_detector.py:753` feeds `kg_query` to `retrieve_graph_context` in `kg` mode and to `query_vector_db` in `rag` mode, so the two modes need different *kinds* of string: a node name for `kg`, a natural-language search phrase for `rag`. All three were checked by execution against the real graph and the real vector DB on 2026-08-01, and near-misses were measured, not guessed — node-style phrasing grounded two of these rules in row and leg-abduction papers. Rewording any of them silently changes what the coaching layer cites.

---

## Execution order — Task 4 runs SECOND

Tasks are executed **1 → 4 → 2 → 3 → 5 → 6**. Numbering below is unchanged; only the order
differs. Reason, found while executing Task 1:

`tests/test_kg_query_resolution.py` is a pre-existing corpus gate that keys off **file
existence**. Any `.py` in `src/pose/movements/` must appear in its `MODULE_MOVEMENTS` map (or
`test_every_module_is_covered` fails) *and* must contain at least one `kg`-mode `kg_query` (or
`test_queries_were_actually_found` fails). Task 1 creates `deadlift.py` with no rules at all,
so **no ordering of these six tasks leaves Task 1's commit green** — the gate is unsatisfiable
until the module's one `kg`-mode rule exists.

`rule_lumbar_flexion` (Task 4) is that rule — the only one of the three grounded in the KG
rather than RAG. Running it immediately after Task 1 reduces the red window to a single
commit. That one commit (`cde4d63f`) is knowingly red on
`test_queries_were_actually_found`; the branch tip is green and CI runs on the PR, not on
intermediate commits. This is recorded rather than hidden, and the gate is deliberately NOT
weakened to accommodate the split — it exists because Overhead Press shipped three broken
queries.

## File Structure

| File | Responsibility |
|---|---|
| `src/pose/movements/deadlift.py` (create) | Metrics, phases, three rules, detector assembly, self-registration |
| `src/pose/movements/registry.py` (modify, line 34) | One side-effect import |
| `tests/test_deadlift.py` (create) | Metrics/phase/rule unit tests |
| `tests/test_movements_endpoint.py` (modify) | Five-detector list and `validated` map |
| `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` (modify) | `bar_drift` withdrawal, §7 gaps, KG divergence |

One module, matching every other movement. `lunge.py` is 786 lines and `pushup.py` 1608; a three-rule Deadlift module lands well inside the established pattern, so no split is warranted.

---

## Task 1: Metrics and phase assignment

**Files:**
- Create: `src/pose/movements/deadlift.py`
- Create: `tests/test_deadlift.py`

**Interfaces:**
- Consumes: `src.pose.geometry` (`landmarks_to_array`, `visible_point`, `angle_degrees`, `midpoint`, `mean_visibility`, `distance`, `line_angle_from_vertical`), `src.pose.movements.base.CoreFrame`
- Produces:
  - `DEADLIFT_METRIC_KEYS: tuple[str, ...]`
  - `deadlift_compute_raw(frames: Sequence[object], fps: float) -> list[dict]`
  - `deadlift_assign_phases(raw: list[dict]) -> list[str]`
  - `DEADLIFT_ACTIVE_PHASES: set[str]`, `LOWER_BODY_LANDMARKS: tuple[int, ...]`, `_DEGENERATE_LENGTH: float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_deadlift.py`:

```python
import unittest

import numpy as np

from src.pose.movements.base import CoreFrame
from src.pose.movements.deadlift import (
    DEADLIFT_METRIC_KEYS,
    deadlift_assign_phases,
    deadlift_compute_raw,
)


def _landmarks(overrides: dict[int, tuple[float, float]]) -> list[dict]:
    """33 fully-visible landmarks in a plausible standing pose, with overrides applied."""
    base = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99} for _ in range(33)]
    defaults = {
        11: (0.50, 0.30), 12: (0.52, 0.30),   # shoulders
        23: (0.50, 0.55), 24: (0.52, 0.55),   # hips
        25: (0.50, 0.75), 26: (0.52, 0.75),   # knees
        27: (0.50, 0.95), 28: (0.52, 0.95),   # ankles
        29: (0.49, 0.96), 30: (0.51, 0.96),   # heels
        31: (0.55, 0.97), 32: (0.57, 0.97),   # foot index
    }
    defaults.update(overrides)
    for index, (x, y) in defaults.items():
        base[index] = {"x": x, "y": y, "z": 0.0, "visibility": 0.99}
    return base


def _frame(index: int, overrides: dict[int, tuple[float, float]] | None = None) -> dict:
    return {"frame_index": index, "landmarks": _landmarks(overrides or {})}


class DeadliftMetricTests(unittest.TestCase):
    def test_metric_keys_match_the_emitted_metrics(self):
        """A key the tuple omits is dropped by run_detector and read back as NaN."""
        raw = deadlift_compute_raw([_frame(0)], fps=30.0)
        emitted = set(raw[0]) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(DEADLIFT_METRIC_KEYS))

    def test_an_upright_lockout_reads_near_180_degrees(self):
        raw = deadlift_compute_raw([_frame(0)], fps=30.0)
        self.assertTrue(raw[0]["valid"])
        self.assertGreater(raw[0]["hip_angle_deg"], 170.0)
        self.assertGreater(raw[0]["knee_angle_deg"], 170.0)
        self.assertLess(raw[0]["torso_pitch_deg"], 10.0)

    def test_a_pitched_trunk_reads_a_large_torso_pitch(self):
        # Shoulders driven forward of the hips: trunk near horizontal.
        raw = deadlift_compute_raw(
            [_frame(0, {11: (0.75, 0.50), 12: (0.77, 0.50)})], fps=30.0
        )
        self.assertGreater(raw[0]["torso_pitch_deg"], 60.0)

    def test_one_missing_landmark_invalidates_the_whole_frame(self):
        landmarks = _landmarks({})
        landmarks[24] = {"x": 0.52, "y": 0.55, "z": 0.0, "visibility": 0.01}
        raw = deadlift_compute_raw([{"frame_index": 0, "landmarks": landmarks}], fps=30.0)
        self.assertFalse(raw[0]["valid"])
        self.assertNotIn("hip_angle_deg", raw[0])

    def test_a_non_dict_frame_is_invalid_rather_than_raising(self):
        self.assertEqual(deadlift_compute_raw([None], fps=30.0), [{"valid": False}])


class DeadliftPhaseTests(unittest.TestCase):
    @staticmethod
    def _rep(hip_angles: list[float]) -> list[dict]:
        return [
            {"frame_index": i, "valid": True, "hip_angle_deg": a}
            for i, a in enumerate(hip_angles)
        ]

    @staticmethod
    def _pull(n: int, low: float, high: float) -> list[float]:
        """A flexed-start rep: floor -> lockout -> floor."""
        up = list(np.linspace(low, high, n // 2))
        return up + list(np.linspace(high, low, n - n // 2))

    def test_an_empty_clip_returns_no_phases(self):
        self.assertEqual(deadlift_assign_phases([]), [])

    def test_a_clip_with_no_finite_signal_is_entirely_unknown(self):
        raw = self._rep([np.nan] * 20)
        self.assertEqual(set(deadlift_assign_phases(raw)), {"unknown"})

    def test_an_invalid_frame_is_unknown_even_inside_the_setup_prefix(self):
        raw = self._rep(self._pull(40, 60.0, 178.0))
        raw[0]["valid"] = False
        self.assertEqual(deadlift_assign_phases(raw)[0], "unknown")

    def test_a_full_rep_produces_all_five_phases(self):
        raw = self._rep(self._pull(60, 60.0, 178.0))
        self.assertEqual(
            set(deadlift_assign_phases(raw)),
            {"setup", "lift_off", "mid_pull", "lockout", "lowering"},
        )

    def test_the_rep_opens_in_setup_because_a_flexed_start_begins_on_the_floor(self):
        raw = self._rep(self._pull(60, 60.0, 178.0))
        self.assertEqual(deadlift_assign_phases(raw)[0], "setup")

    def test_a_rep_that_never_locks_out_still_has_a_lockout_phase(self):
        """The fault IS failing to reach extension, so the phase must not vanish with it.

        The lockout threshold is a PERCENTILE of this rep's own hip-angle excursion, not an
        absolute angle, so a rep peaking at 150 degrees still yields a lockout phase for
        `rule_incomplete_lockout` to score. An absolute cutoff would silence the rule on
        exactly the reps it exists to catch.
        """
        raw = self._rep(self._pull(60, 60.0, 150.0))
        phases = deadlift_assign_phases(raw)
        self.assertGreaterEqual(phases.count("lockout"), 6)

    def test_phase_count_always_equals_frame_count(self):
        """run_detector raises if assign_phases returns a different length."""
        for n in (1, 7, 40, 61):
            raw = self._rep(self._pull(n, 60.0, 178.0))
            self.assertEqual(len(deadlift_assign_phases(raw)), n)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/test_deadlift.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pose.movements.deadlift'`

- [ ] **Step 3: Write `src/pose/movements/deadlift.py`**

```python
# Deadlift raw metrics and phase segmentation. Fault rules land in Tasks 2-4.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `deadlift_compute_raw` / `deadlift_assign_phases`
# compute scale-free per-frame metrics and a phase label only. Every number that decides
# anything belongs in a `rule_*` function, not here. `_DEGENERATE_LENGTH` is a
# division-by-zero guard, never a tunable threshold.
#
# ---------------------------------------------------------------------------------------
# THE REP STARTS FLEXED, AND THAT IS WHY A SETUP BASELINE MEANS ANYTHING HERE.
# ---------------------------------------------------------------------------------------
# `DEADLIFT_DETECTOR` sets `rep_start="flexed"` (Task 5) -- the hook `base.py:55` names
# deadlift as the motivating case for. A rep therefore runs floor -> lockout -> floor, so
# the window's OPENING frames are genuinely the bar-on-the-floor setup. Two rules
# (`rule_hips_shoot_up`, `rule_lumbar_flexion`) reference a per-rep setup baseline, which is
# only meaningful because of this. For a movement whose rep starts standing, the same
# baseline would be measuring the wrong end of the lift.
#
# The window also contains the ECCENTRIC. The parent spec's four phases cover only the
# concentric, so a fifth phase `lowering` exists here; without it, return-to-floor frames
# would be labelled `lockout` and `rule_incomplete_lockout` would score the descent.
# `lowering` is excluded from `DEADLIFT_ACTIVE_PHASES`: no rule has literature backing for a
# claim about the eccentric.
#
# ---------------------------------------------------------------------------------------
# EVERY METRIC IS BUILT FROM MIDPOINTS, AND EVERY RULE WANTS A SAGITTAL VIEW.
# ---------------------------------------------------------------------------------------
# Parent spec section 7 item 3 records that `_visible_midpoint` needs BOTH landmarks of a
# pair above 0.35 visibility, and that one occluded shoulder silently reverts body-extent
# measurement to a vertical fallback -- "exactly in the view most likely to trigger it: a
# sagittal (side) view is precisely where far-side landmarks are most often occluded." This
# detector sits squarely in that failure mode. `required` below therefore refuses the frame
# wholesale when any input landmark is missing, matching lunge/pushup/OHP: an unmeasurable
# frame is refused rather than degraded, because a silently-wrong verdict is worse than none.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, midpoint, mean_visibility,
    line_angle_from_vertical,
)

LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

DEADLIFT_METRIC_KEYS: tuple[str, ...] = (
    "hip_angle_deg",
    "knee_angle_deg",
    "torso_pitch_deg",
    "hip_y",
    "torso_len",
)

# `shoulder_y` is deliberately absent. An earlier design emitted it for a hip-vs-shoulder
# rise differential in `rule_hips_shoot_up`; that term was shown to be algebraically
# identical to a trunk-pitch change (see the rule's docstring in Task 3), so nothing consumes
# it.

_DEGENERATE_LENGTH = 1e-6

# Phases in which the deadlift is under load. `lowering` and `setup` are excluded.
DEADLIFT_ACTIVE_PHASES = {"lift_off", "mid_pull", "lockout"}


def deadlift_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        required = (
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE,
            LEFT_ANKLE, RIGHT_ANKLE,
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
            continue

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        knee_mid = midpoint(points, LEFT_KNEE, RIGHT_KNEE, dims=2)
        ankle_mid = midpoint(points, LEFT_ANKLE, RIGHT_ANKLE, dims=2)

        hip_angle_deg = _angle_between(shoulder_mid, hip_mid, knee_mid)
        knee_angle_deg = _angle_between(hip_mid, knee_mid, ankle_mid)
        # `line_angle_from_vertical(top, bottom)` takes abs() of both deltas, so this is an
        # UNSIGNED angle in [0, 90] -- it cannot distinguish a forward from a backward lean.
        # Correct for the deadlift, where the trunk only ever pitches forward, and it is why
        # `rule_hips_shoot_up` can compare magnitudes without resolving the subject's facing.
        torso_pitch_deg = line_angle_from_vertical(shoulder_mid, hip_mid)
        torso_len = (
            float(np.linalg.norm(shoulder_mid - hip_mid))
            if shoulder_mid is not None and hip_mid is not None
            else np.nan
        )

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "hip_angle_deg": hip_angle_deg,
                "knee_angle_deg": knee_angle_deg,
                "torso_pitch_deg": torso_pitch_deg,
                "hip_y": float(hip_mid[1]) if hip_mid is not None else np.nan,
                "torso_len": torso_len,
            }
        )
    return raw


def _angle_between(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float:
    """Interior angle at `b`, in degrees. NaN when any point is missing or degenerate.

    `geometry.angle_degrees` takes LANDMARK INDICES, not points; these vertices are computed
    midpoints with no index, so the arithmetic is done here rather than reaching for it.
    """
    if a is None or b is None or c is None:
        return float(np.nan)
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na < _DEGENERATE_LENGTH or nc < _DEGENERATE_LENGTH:
        return float(np.nan)
    cosine = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def deadlift_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> lift_off -> mid_pull -> lockout -> lowering, segmented on `hip_angle_deg`.

    Mirrors `lunge_assign_phases`, substituting hip angle for knee angle and inverting the
    sense: a lunge rep is deepest in the middle, a deadlift rep is most EXTENDED in the
    middle.

    THE PHASE CUTOFFS ARE PERCENTILES OF THIS REP'S OWN EXCURSION, NOT ABSOLUTE ANGLES, and
    that is load-bearing rather than stylistic. `rule_incomplete_lockout` scores the `lockout`
    phase, and the fault it detects IS failing to reach extension. An absolute cutoff (say
    "lockout = hip angle above 165 degrees") would give a shallow-finishing rep NO lockout
    frames at all, so the rule would go silent on precisely the reps it exists to catch. A
    percentile guarantees the phase exists for every rep, however badly performed. Same
    reasoning as lunge's `bottom_threshold = np.percentile(valid_knee, 30)`.

    The lockout test precedes the post-peak test deliberately: a lifter standing at lockout
    produces high-angle frames on BOTH sides of the peak frame, and those are lockout, not
    lowering. Checking `index > peak` first would discard half the lockout plateau.
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    hip_values = np.asarray(
        [float(item.get("hip_angle_deg", np.nan)) for item in raw], dtype=np.float32
    )
    finite = hip_values[np.isfinite(hip_values)]
    if finite.size == 0:
        return ["unknown" for _ in raw]

    lockout_threshold = float(np.percentile(finite, 75))
    mid_pull_threshold = float(np.percentile(finite, 40))
    peak_index = int(np.nanargmax(np.where(np.isfinite(hip_values), hip_values, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.10))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = hip_values[index]
        if np.isfinite(value) and value >= lockout_threshold:
            phases.append("lockout")
        elif index > peak_index:
            phases.append("lowering")
        elif np.isfinite(value) and value >= mid_pull_threshold:
            phases.append("mid_pull")
        else:
            phases.append("lift_off")
    return phases
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/test_deadlift.py -v`
Expected: PASS, 13 tests.

If `test_metric_keys_match_the_emitted_metrics` fails, the tuple and the dict have drifted — fix the tuple, never the test.

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/deadlift.py tests/test_deadlift.py
git commit -m "feat(pose): deadlift metrics and the five phases a flexed-start rep needs"
```

---

## Task 2: `rule_incomplete_lockout`

**Files:**
- Modify: `src/pose/movements/deadlift.py`
- Modify: `tests/test_deadlift.py`

**Interfaces:**
- Consumes: Task 1's `DEADLIFT_ACTIVE_PHASES`; `base.CoreFrame`, `base.RuleContext`
- Produces: `rule_incomplete_lockout(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]`, `DEADLIFT_LOCKOUT_MILD_DEG = 165.0`, `DEADLIFT_LOCKOUT_SEVERE_DEG = 140.0`, `DEADLIFT_LOCKOUT_KG_QUERY`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_deadlift.py`:

```python
from src.pose.movements.base import RuleContext
from src.pose.movements.deadlift import (
    DEADLIFT_LOCKOUT_MILD_DEG,
    rule_incomplete_lockout,
)


def _ctx(view: str = "side", conf: float = 0.9, min_frames: int = 6) -> RuleContext:
    return RuleContext(fps=30.0, view_type=view, view_confidence=conf, min_frames=min_frames)


def _frames(metrics: dict, count: int = 12, phase: str = "lockout") -> list[CoreFrame]:
    """A window of `count` identical CoreFrames carrying `metrics`."""
    return [
        CoreFrame(
            frame_index=i,
            time=i / 30.0,
            phase=phase,
            valid=True,
            lower_body_visibility=0.9,
            metrics=dict(metrics),
        )
        for i in range(count)
    ]


class IncompleteLockoutTests(unittest.TestCase):
    LOCKED = {"hip_angle_deg": 178.0, "knee_angle_deg": 176.0}

    def test_a_locked_out_rep_is_silent(self):
        self.assertEqual(rule_incomplete_lockout(_frames(self.LOCKED), _ctx()), [])

    def test_a_soft_hip_fires(self):
        out = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 150.0, "knee_angle_deg": 176.0}), _ctx()
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].fault_id, "deadlift_incomplete_lockout")

    def test_a_soft_knee_fires(self):
        out = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 178.0, "knee_angle_deg": 150.0}), _ctx()
        )
        self.assertEqual(len(out), 1)

    def test_a_hip_only_failure_still_scores_the_hip_ramp(self):
        """The OHP mis-attribution regression: selecting the ramp by which reading is finite
        scored the wrong axis when a segment fired on one criterion alone."""
        soft = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 145.0, "knee_angle_deg": 178.0}), _ctx()
        )[0]
        softer = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 141.0, "knee_angle_deg": 178.0}), _ctx()
        )[0]
        self.assertGreater(softer.severity, soft.severity)

    def test_the_worse_of_the_two_axes_drives_severity(self):
        both = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 160.0, "knee_angle_deg": 141.0}), _ctx()
        )[0]
        knee_only = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 178.0, "knee_angle_deg": 141.0}), _ctx()
        )[0]
        self.assertAlmostEqual(both.severity, knee_only.severity, places=4)

    def test_severity_saturates_at_the_severe_endpoint(self):
        out = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 100.0, "knee_angle_deg": 178.0}), _ctx()
        )[0]
        self.assertAlmostEqual(out.severity, 1.0, places=4)

    def test_only_the_lockout_phase_is_scored(self):
        soft = {"hip_angle_deg": 120.0, "knee_angle_deg": 120.0}
        for phase in ("setup", "lift_off", "mid_pull", "lowering", "rest"):
            self.assertEqual(
                rule_incomplete_lockout(_frames(soft, phase=phase), _ctx()), [],
                msg=f"{phase} must not be scored",
            )

    def test_nan_metrics_are_silent_rather_than_firing(self):
        out = rule_incomplete_lockout(
            _frames({"hip_angle_deg": np.nan, "knee_angle_deg": np.nan}), _ctx()
        )
        self.assertEqual(out, [])

    def test_an_off_view_reading_is_discounted_but_not_suppressed(self):
        soft = {"hip_angle_deg": 150.0, "knee_angle_deg": 178.0}
        on = rule_incomplete_lockout(_frames(soft), _ctx(view="side"))[0]
        off = rule_incomplete_lockout(_frames(soft), _ctx(view="rear"))[0]
        self.assertEqual(off.observability, "medium")
        self.assertLess(off.confidence, on.confidence)

    def test_a_run_shorter_than_min_frames_is_not_a_detection(self):
        soft = {"hip_angle_deg": 150.0, "knee_angle_deg": 178.0}
        self.assertEqual(rule_incomplete_lockout(_frames(soft, count=3), _ctx()), [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/test_deadlift.py -k Lockout -v`
Expected: FAIL — `ImportError: cannot import name 'rule_incomplete_lockout'`

- [ ] **Step 3: Implement the rule**

Add these imports to the top of `src/pose/movements/deadlift.py` (`CoreFrame` and `RuleContext`
were not needed by Task 1's metric layer and are added here, where the first rule needs them):

```python
from src.pose.movements.base import CoreFrame, RuleContext
from src.pose.geometry import contiguous_true_segments, severity_from_range
from src.pose.pose_rule_detector import (
    SIDE_VIEW_CONF_THRESHOLD,
    VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
    PoseRuleDetection,
    build_detection,
)

_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE

# Views in which a sagittal angle reads at full confidence. Per parent spec section 7 item 2,
# `front_oblique` is unreachable in the production path (`allow_front=False`), so in practice
# this is `side`; it is listed because the spec names it and a test can reach it.
SAGITTAL_VIEWS = {"side", "front_oblique"}
```

Then append the rule:

```python
# STEP 0 -- KG QUERY RESOLUTION, recorded before the rule was written. Checked against
# data/kg/sports_kg_v3.graphml via BOTH `resolve_nodes` and `retrieve_graph_context` (the
# latter is what production calls, and is what OHP's three-blank-queries defect would have
# been caught by).
#
# NO KG NODE EXISTS FOR THIS FAULT, so it takes the `rag` fallback. The 5-node Deadlift stub
# carries exactly one lockout node -- `Deadlift:Hyperextension At Lockout` -- which is the
# LITERAL OPPOSITE fault: too much extension, not too little. `Incomplete Lockout` resolves to
# nothing and `Incomplete Range Of Motion` resolves only to the generic shared-layer
# `Range Of Motion` concept node, which would ground a coaching explanation on an abstraction
# rather than on an error. Grounding this rule on any of them would retrieve advice for a
# different problem, so per the lunge Step-0 rule -- "do NOT invent a near-miss" -- it does not.
#
# IN `rag` MODE THIS STRING IS A VECTOR-DB SEARCH PHRASE, NOT A NODE NAME
# (`pose_rule_detector.py:756` passes it straight to `query_vector_db`), so it is written as
# one and was verified by running it. The node-style "Incomplete Lockout" retrieves a ROW
# suspension-EMG paper and a LEG ABDUCTION paper -- the wrong movement entirely. The phrasing
# below returns PMC12148905, this rule's cross-support citation, at ranks 1 and 3. Verified
# 2026-08-01; re-run before changing it.
DEADLIFT_LOCKOUT_KG_QUERY = "deadlift incomplete lockout hip and knee extension"

# Spec-derived, unvalidated. The 180-degree TARGET is measured -- Moreira PMC12225233 recorded
# the three key positions at lift-off 95 deg, mid-pull 126 deg and lock-out 180 deg, with
# "180 degrees ... equivalent to full extension" -- but the 165-degree tolerance below which a
# rep is called incomplete is the parent spec's number and no source states it.
DEADLIFT_LOCKOUT_MILD_DEG = 165.0
DEADLIFT_LOCKOUT_SEVERE_DEG = 140.0


# SUPERSEDED 2026-08-01 by the whole-branch review. The version below is what SHIPPED; the
# original plan text specified a PER-FRAME mask (`phase == "lockout" and angle < 165`) handed to
# `contiguous_true_segments`, and that was wrong. `lockout` is the 75th PERCENTILE of the rep's
# own hip-angle excursion -- a rank cutoff, not an angle -- so a rep that spends under 25% of its
# frames above 165 gets a lockout band reaching BELOW 165, and the per-frame mask fired
# "incomplete lockout" on a rep peaking at 178 deg (severity 0.66, observability "high",
# reproduced on the segmented production path). The fix scores the rep's PEAK extension, which
# introduces no new number and matches both the parent spec's "at rep end" phrasing and
# `overhead_press.rule_incomplete_lockout`'s long-standing `nanmax` aggregate. See the design
# spec section 4.2 amendment box.
def _peak_extension(segment: list[CoreFrame], key: str) -> float:
    """GREATEST finite value of `key` across the segment; NaN when the axis is wholly missing.

    "How far did this rep extend" is a `nanmax`, not a `nanmin`. The guard is not decoration:
    the rule flags on either axis independently, so a segment can be flagged entirely by one
    axis while the other is NaN on every frame. Bare `np.nanmax` over an all-NaN list warns and
    returns NaN, and a NaN in `evidence` survives `dataclasses.asdict()` into a postgrest write
    with `allow_nan=False` -- a ValueError this codebase documents as silently swallowed,
    dropping the analysis from the user's history. Mirrors `overhead_press.py:321-328`.
    """
    values = [frame.m(key) for frame in segment]
    return float(np.nanmax(values)) if any(np.isfinite(v) for v in values) else float(np.nan)


def rule_incomplete_lockout(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Rep ends without full hip AND knee extension.

    Scores the rep's PEAK extension over the `lockout` phase: fires when the best hip extension
    reached is under 165 deg OR the best knee extension is. Scores BOTH ramps, taking the worse.
    Selecting the ramp by "which reading is finite" is the OHP mis-attribution bug recorded in
    the parent spec's section 8 status note; this scores both unconditionally instead.

    View policy is DEGRADE, not gate. An extension angle seen head-on is foreshortened, so it
    under-reads -- the off-view failure mode is a missed fault, not a false one. Contrast
    `rule_lumbar_flexion`, which inverts off-view and is therefore hard-gated.
    """
    observable = ctx.view_type in SAGITTAL_VIEWS

    # The mask is the PHASE ALONE; the `< 165` test moved onto the segment's peak, below.
    mask = [frame.valid and frame.phase == "lockout" for frame in core]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        peak_hip = _peak_extension(segment, "hip_angle_deg")
        peak_knee = _peak_extension(segment, "knee_angle_deg")
        flagged = (np.isfinite(peak_hip) and peak_hip < DEADLIFT_LOCKOUT_MILD_DEG) or (
            np.isfinite(peak_knee) and peak_knee < DEADLIFT_LOCKOUT_MILD_DEG
        )
        if not flagged:
            continue

        hip_sev = severity_from_range(
            peak_hip, DEADLIFT_LOCKOUT_MILD_DEG, DEADLIFT_LOCKOUT_SEVERE_DEG, lower_is_worse=True
        )
        knee_sev = severity_from_range(
            peak_knee, DEADLIFT_LOCKOUT_MILD_DEG, DEADLIFT_LOCKOUT_SEVERE_DEG, lower_is_worse=True
        )
        severity = float(max(hip_sev, knee_sev))
        driver = "hip" if hip_sev >= knee_sev else "knee"
        driver_peak = peak_hip if driver == "hip" else peak_knee
        # `build_detection` takes `argmax(score_values)`, so the driver axis's raw angles point
        # `peak_frame` at the frame that achieved the reported peak -- the frame the evidence
        # quotes. Same convention as `overhead_press`, which passes its raw `wrist_values`.
        score_values = [frame.m(f"{driver}_angle_deg") for frame in segment]

        detections.append(
            build_detection(
                fault_id="deadlift_incomplete_lockout",
                fault_name="Incomplete Lockout",
                kg_query=DEADLIFT_LOCKOUT_KG_QUERY,
                retrieval_mode="rag",
                segment_metrics=segment,
                score_values=score_values,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "peak_hip_angle_deg": round(peak_hip, 2) if np.isfinite(peak_hip) else 0.0,
                    "peak_knee_angle_deg": round(peak_knee, 2) if np.isfinite(peak_knee) else 0.0,
                    "threshold": DEADLIFT_LOCKOUT_MILD_DEG,
                    "driver": driver,
                    "primary_label": f"peak {driver} angle at lockout",
                    "primary_value": round(driver_peak, 2) if np.isfinite(driver_peak) else 0.0,
                    "primary_threshold": DEADLIFT_LOCKOUT_MILD_DEG,
                },
                citation="Moreira VM, et al. \"Analysis of Muscle Strength and Electromyographic "
                         "Activity during Different Deadlift Positions.\" Muscles (2023). "
                         "PMC12225233. Cross-support: Hanen NC, et al. PMC12148905 (2025).",
                citation_support="PMC12225233 measured the three key positions at "
                                 "\"approximately 95 deg, 126 deg, and 180 deg\" for lift-off, "
                                 "mid-pull and lock-out, with \"180 deg ... equivalent to full "
                                 "extension\" -- so full triple extension is a measured target, "
                                 "not an assumption. PMC12148905: \"lift completion[] is "
                                 "achieved when the athlete assumes a fully upright position "
                                 "with extended hips and knees, with scapular retraction.\" "
                                 "Verified in RAG docs. The 165 deg flag point is spec-derived.",
            )
        )
    return detections
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/test_deadlift.py -v`
Expected: PASS, all Task 1 + Task 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/deadlift.py tests/test_deadlift.py
git commit -m "feat(pose): deadlift lockout, scoring both ramps so neither axis is mis-attributed"
```

---

## Task 3: `rule_hips_shoot_up`

**Files:**
- Modify: `src/pose/movements/deadlift.py`
- Modify: `tests/test_deadlift.py`

**Interfaces:**
- Consumes: Task 1's `DEADLIFT_ACTIVE_PHASES`; Task 2's `SAGITTAL_VIEWS`, `_OFF_VIEW_CONFIDENCE`
- Produces: `rule_hips_shoot_up(core, ctx) -> list[PoseRuleDetection]`, `setup_baseline(core, key) -> float`, `DEADLIFT_PITCH_MILD_DEG = 55.0`, `DEADLIFT_PITCH_SEVERE_DEG = 75.0`, `DEADLIFT_HIPS_KG_QUERY`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_deadlift.py`:

```python
from src.pose.movements.deadlift import rule_hips_shoot_up, setup_baseline


def _rep_window(setup: dict, active: dict, setup_n: int = 8, active_n: int = 12):
    """A window whose opening frames are `setup` phase and whose remainder is `mid_pull`."""
    return (
        _frames(setup, count=setup_n, phase="setup")
        + _frames(active, count=active_n, phase="mid_pull")
    )


class SetupBaselineTests(unittest.TestCase):
    def test_the_baseline_is_the_median_of_the_setup_frames_only(self):
        window = _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 80.0})
        self.assertAlmostEqual(setup_baseline(window, "torso_pitch_deg"), 50.0, places=4)

    def test_a_window_with_no_setup_frames_has_no_baseline(self):
        window = _frames({"torso_pitch_deg": 60.0}, phase="mid_pull")
        self.assertTrue(np.isnan(setup_baseline(window, "torso_pitch_deg")))


class HipsShootUpTests(unittest.TestCase):
    def test_a_trunk_that_flattens_past_the_gate_fires(self):
        window = _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 65.0})
        out = rule_hips_shoot_up(window, _ctx())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].fault_id, "deadlift_hips_shoot_up")

    def test_a_rep_that_stays_flat_without_flattening_further_is_silent(self):
        """Setting up flat is not the sequencing fault the citation describes."""
        window = _rep_window({"torso_pitch_deg": 70.0}, {"torso_pitch_deg": 68.0})
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_a_trunk_that_flattens_but_stays_upright_is_silent(self):
        window = _rep_window({"torso_pitch_deg": 20.0}, {"torso_pitch_deg": 40.0})
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_a_good_hinge_that_becomes_more_upright_is_silent(self):
        window = _rep_window({"torso_pitch_deg": 60.0}, {"torso_pitch_deg": 25.0})
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_severity_ramps_with_peak_pitch(self):
        mild = rule_hips_shoot_up(
            _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 60.0}), _ctx()
        )[0]
        worse = rule_hips_shoot_up(
            _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 72.0}), _ctx()
        )[0]
        self.assertGreater(worse.severity, mild.severity)

    def test_severity_saturates_at_the_severe_endpoint(self):
        out = rule_hips_shoot_up(
            _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 85.0}), _ctx()
        )[0]
        self.assertAlmostEqual(out.severity, 1.0, places=4)

    def test_a_window_without_a_setup_baseline_is_silent(self):
        window = _frames({"torso_pitch_deg": 80.0}, phase="mid_pull")
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_nan_pitch_is_silent(self):
        window = _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": np.nan})
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_lowering_frames_are_never_scored(self):
        window = (
            _frames({"torso_pitch_deg": 50.0}, count=8, phase="setup")
            + _frames({"torso_pitch_deg": 80.0}, count=12, phase="lowering")
        )
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_an_off_view_reading_is_discounted_but_not_suppressed(self):
        window = _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 65.0})
        on = rule_hips_shoot_up(window, _ctx(view="side"))[0]
        off = rule_hips_shoot_up(window, _ctx(view="rear"))[0]
        self.assertEqual(off.observability, "medium")
        self.assertLess(off.confidence, on.confidence)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/test_deadlift.py -k "Baseline or ShootUp" -v`
Expected: FAIL — `ImportError: cannot import name 'rule_hips_shoot_up'`

- [ ] **Step 3: Implement the baseline helper and the rule**

```python
def setup_baseline(core: list[CoreFrame], key: str) -> float:
    """Median of `key` over the window's `setup` frames; NaN when there are none.

    A per-rep baseline cannot live in `deadlift_compute_raw`, which `run_detector` calls over
    the WHOLE CLIP before any rep boundary exists. It belongs here, where the window IS one
    rep -- the same split lunge uses for lead-side resolution and squat uses for its heel
    baseline. Median rather than mean so one mis-tracked setup frame cannot move it.
    """
    values = [
        frame.m(key) for frame in core if frame.valid and frame.phase == "setup"
    ]
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float(np.nan)


# NO KG NODE EXISTS FOR THIS FAULT -- `rag` fallback, resolved before the rule was written.
# The nearest Deadlift-scoped candidate, `Deadlift:Insufficient Hip Hinge`, is a near-miss
# POINTING THE WRONG WAY: insufficient hinge means failing to push the hips back, a
# knee-dominant squat-like pull, whereas this fault is excessive hip dominance with the trunk
# flattening. Its only edge is `AFFECTS_QUALITY -> Hip Hinge` -- no risk, no correction. The
# other candidates (`Hips Rise Before Shoulders`, `Trunk Over Inclination`, `Anterior Trunk
# Tilt`, `Excessive Forward Lean`) resolve to nothing or to the bare `Hip` anatomy node.
#
# IN `rag` MODE THIS STRING IS A VECTOR-DB SEARCH PHRASE, NOT A NODE NAME, and it was chosen by
# running candidates rather than by writing something plausible. The corpus holds only 2
# deadlift documents among 85, so semantic search drifts badly: "Hips Rise Before Shoulders"
# returns a row EMG paper, and four different mechanism-keyword phrasings ("...erector spinae
# trunk flexion barbell shear force", "...lever arm lower back barbell", and two more) each
# returned 0/3 deadlift documents, mostly Overhead Press. The phrasing below returns
# PMC12225233 -- this rule's primary citation -- at ranks 1, 2 AND 3. Verified 2026-08-01;
# re-run before changing it, because near-miss phrasings silently ground this fault in the
# wrong movement's literature.
DEADLIFT_HIPS_KG_QUERY = "deadlift trunk position electromyographic activity lift-off mid-pull lockout"

# Spec-derived, UNVALIDATED AND UNSOURCED. Neither deadlift RAG document reports a trunk
# inclination in degrees -- the only degree value in PMC12148905 is an unrelated 8 deg knee
# adduction. What the citation backs is the MECHANISM and the DIRECTION (a flatter trunk means
# more spinal flexion torque), which is what the two-clause criterion encodes; these endpoints
# are the parent spec's numbers.
DEADLIFT_PITCH_MILD_DEG = 55.0
DEADLIFT_PITCH_SEVERE_DEG = 75.0


def rule_hips_shoot_up(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Hips out-run the shoulders off the floor, flattening the trunk into a back-dominant pull.

    THIS IS NOT WRITTEN AS A HIP-VS-SHOULDER RISE DIFFERENTIAL, and the omission is deliberate.
    The parent spec phrases the signal as "Delta(hip_y) rises faster than Delta(shoulder_y)",
    and an earlier draft implemented that literally as

        hip_lead_ratio = ((hip_y0 - hip_y) - (shoulder_y0 - shoulder_y)) / torso_len0 > 0

    That term was checked numerically before any code was written and is ALGEBRAICALLY
    IDENTICAL to a trunk-pitch change. Since `shoulder_y - hip_y = -torso_len*cos(pitch)`, a
    rigid torso gives

        hip_lead_ratio == cos(pitch_0) - cos(pitch_t)

    exact to machine precision on a sagittal stick model. It depends ONLY on pitch and carries
    no information about how far the hips actually travelled -- two landmarks dressing up a
    single-angle test. Writing it as a differential would have implied this rule corroborates
    trunk pitch with an independent kinematic signal, which is false. The parent spec's own
    "i.e." equating the two phrasings turns out to be correct, so stating the rule in pitch
    terms is faithful to it rather than a deviation.

    The relative-to-setup clause is kept because it is what separates the SEQUENCING fault the
    citation describes from a lifter who merely sets up flat and stays there; the absolute
    55-degree gate alone cannot tell those apart.

    View policy is DEGRADE, not gate: head-on, a pitched trunk projects short and near-vertical
    so the angle UNDER-reads, making the off-view failure mode silence rather than a wrong
    claim.
    """
    baseline = setup_baseline(core, "torso_pitch_deg")
    if not np.isfinite(baseline):
        return []
    observable = ctx.view_type in SAGITTAL_VIEWS

    mask = [
        frame.valid
        and frame.phase in DEADLIFT_ACTIVE_PHASES
        and np.isfinite(frame.m("torso_pitch_deg"))
        and frame.m("torso_pitch_deg") > baseline
        and frame.m("torso_pitch_deg") > DEADLIFT_PITCH_MILD_DEG
        for frame in core
    ]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        pitches = [frame.m("torso_pitch_deg") for frame in segment]
        peak = float(np.nanmax(pitches))
        severity = severity_from_range(
            peak, DEADLIFT_PITCH_MILD_DEG, DEADLIFT_PITCH_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="deadlift_hips_shoot_up",
                fault_name="Hips Rise Before Shoulders / Trunk Over-Inclination",
                kg_query=DEADLIFT_HIPS_KG_QUERY,
                retrieval_mode="rag",
                segment_metrics=segment,
                score_values=pitches,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "peak_torso_pitch_deg": round(peak, 2),
                    "setup_torso_pitch_deg": round(baseline, 2),
                    "threshold": DEADLIFT_PITCH_MILD_DEG,
                    "primary_label": "peak trunk pitch from vertical",
                    "primary_value": round(peak, 2),
                    "primary_threshold": DEADLIFT_PITCH_MILD_DEG,
                },
                citation="Moreira VM, et al. PMC12225233 (2023). Cross-support: Hanen NC, "
                         "et al. PMC12148905 (2025).",
                citation_support="PMC12225233: \"leaning the trunk forward results in higher "
                                 "spinal flexion torque generated by the barbell. Therefore, "
                                 "ERE [erector spinae] requires higher activation and higher "
                                 "strength to avoid trunk flexion, reducing shear.\" "
                                 "PMC12148905 frames \"a significantly reduced trunk "
                                 "inclination angle\" as the low-back-sparing state. Verified "
                                 "in RAG docs. Both ramp endpoints are spec-derived: neither "
                                 "source reports a trunk inclination in degrees.",
            )
        )
    return detections
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/test_deadlift.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/deadlift.py tests/test_deadlift.py
git commit -m "feat(pose): deadlift trunk flattening, stated as the angle it actually measures"
```

---

## Task 4: `rule_lumbar_flexion`

**Files:**
- Modify: `src/pose/movements/deadlift.py`
- Modify: `tests/test_deadlift.py`

**Interfaces:**
- Consumes: Task 3's `setup_baseline`; Task 2's `SAGITTAL_VIEWS`, `_OFF_VIEW_CONFIDENCE`
- Produces: `rule_lumbar_flexion(core, ctx) -> list[PoseRuleDetection]`, `DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED = 0.95`, `DEADLIFT_TORSO_SHORTENING_SEVERE_UNSOURCED = 0.85`, `DEADLIFT_HIP_STATIONARY_BAND_UNSOURCED = 0.10`, `DEADLIFT_LUMBAR_KG_QUERY`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_deadlift.py`:

```python
from src.pose.movements.deadlift import rule_lumbar_flexion


class LumbarFlexionTests(unittest.TestCase):
    SETUP = {"torso_len": 0.25, "hip_y": 0.60}

    def test_a_shortening_torso_over_stationary_hips_fires_at_low_observability(self):
        window = _rep_window(self.SETUP, {"torso_len": 0.22, "hip_y": 0.60})
        out = rule_lumbar_flexion(window, _ctx())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].fault_id, "deadlift_lumbar_flexion")
        self.assertEqual(out[0].observability, "low")

    def test_a_rigid_torso_is_silent(self):
        window = _rep_window(self.SETUP, {"torso_len": 0.25, "hip_y": 0.50})
        self.assertEqual(rule_lumbar_flexion(window, _ctx()), [])

    def test_shortening_while_the_hips_travel_is_silent(self):
        """Hips moving means the shortening is the lift, not the spine."""
        window = _rep_window(self.SETUP, {"torso_len": 0.22, "hip_y": 0.40})
        self.assertEqual(rule_lumbar_flexion(window, _ctx()), [])

    def test_confidence_carries_the_low_observability_discount(self):
        out = rule_lumbar_flexion(
            _rep_window(self.SETUP, {"torso_len": 0.22, "hip_y": 0.60}), _ctx()
        )[0]
        # Discounted even in its own view: a proxy is never a measurement.
        self.assertAlmostEqual(out.confidence, out.severity * 0.65, places=4)

    def test_an_off_view_window_emits_nothing_at_all(self):
        """HARD GATE, unlike the other two rules: off-view, trunk pitch alone shortens the
        projected segment, so the proxy produces FALSE POSITIVES rather than silence."""
        window = _rep_window(self.SETUP, {"torso_len": 0.22, "hip_y": 0.60})
        for view in ("front", "rear", "rear_oblique", "unknown"):
            self.assertEqual(rule_lumbar_flexion(window, _ctx(view=view)), [], msg=view)

    def test_a_weakly_classified_side_view_emits_nothing(self):
        window = _rep_window(self.SETUP, {"torso_len": 0.22, "hip_y": 0.60})
        self.assertEqual(rule_lumbar_flexion(window, _ctx(view="side", conf=0.05)), [])

    def test_a_window_without_a_setup_baseline_is_silent(self):
        window = _frames({"torso_len": 0.22, "hip_y": 0.60}, phase="mid_pull")
        self.assertEqual(rule_lumbar_flexion(window, _ctx()), [])

    def test_a_degenerate_setup_torso_length_is_silent_rather_than_dividing_by_zero(self):
        window = _rep_window({"torso_len": 0.0, "hip_y": 0.60}, {"torso_len": 0.0, "hip_y": 0.60})
        self.assertEqual(rule_lumbar_flexion(window, _ctx()), [])

    def test_nan_metrics_are_silent(self):
        window = _rep_window(self.SETUP, {"torso_len": np.nan, "hip_y": np.nan})
        self.assertEqual(rule_lumbar_flexion(window, _ctx()), [])

    def test_severity_saturates_at_the_severe_endpoint(self):
        out = rule_lumbar_flexion(
            _rep_window(self.SETUP, {"torso_len": 0.20, "hip_y": 0.60}), _ctx()
        )[0]
        self.assertAlmostEqual(out.severity, 1.0, places=4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/test_deadlift.py -k Lumbar -v`
Expected: FAIL — `ImportError: cannot import name 'rule_lumbar_flexion'`

- [ ] **Step 3: Implement the rule**

```python
# The ONLY Deadlift rule whose kg_query resolves. Verified through the production path:
# `retrieve_graph_context("Lumbar Flexion", movement="Deadlift")` returns the seed
# `Deadlift:Lumbar Flexion` with a NON-EMPTY bucket -- `INCREASES_RISK_OF -> Lumbar Spine
# Injury`, `CORRECTED_BY -> Maintain Neutral Spine`, `HAS_FAULT <- Deadlift`. Checking
# resolution alone was not enough: OHP shipped queries that resolved but returned nothing.
DEADLIFT_LUMBAR_KG_QUERY = "Lumbar Flexion"

# ---------------------------------------------------------------------------------------
# THESE THREE NUMBERS ARE UNSOURCED. The suffix is not decoration.
# ---------------------------------------------------------------------------------------
# No source anywhere gives a segment-shortening-to-lumbar-flexion figure. 0.95 says "5%
# shortening", chosen to sit above frame-to-frame landmark jitter WITHOUT ANY MEASUREMENT OF
# WHAT THAT JITTER IS; 0.85 is a doubling of it; 0.10 of a torso length is a loose "the hips
# have not really moved yet" band. The fault itself IS cited (see citation_support) -- what
# is unsupported is the detection, which is why this rule emits at observability `low` with
# the off-view discount and `run_detector` sorts it last. Calibrating the gate against a
# measured landmark-jitter floor is the known upgrade path; see the design spec section 4.3.
DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED = 0.95
DEADLIFT_TORSO_SHORTENING_SEVERE_UNSOURCED = 0.85
DEADLIFT_HIP_STATIONARY_BAND_UNSOURCED = 0.10


def rule_lumbar_flexion(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Lower back rounds under load -- PROXY ONLY, and the weakest rule in this module.

    MediaPipe has no lumbar landmarks, so true rounded-vs-neutral spine is not recoverable;
    the parent spec rates this `low` observability and says "Do NOT assert precision here".
    The proxy: in a sagittal view a rigid hip hinge holds the PROJECTED shoulder-to-hip length
    constant, because the trunk rotates within the image plane. Shortening against the rep's
    own setup baseline, while the hips are not themselves travelling, is consistent with the
    trunk curling.

    HARD VIEW GATE, unlike this module's other two rules. Off-view, trunk pitch alone shortens
    the projected segment, so the proxy produces FALSE POSITIVES rather than silence. Where the
    off-view failure mode is a wrong claim rather than a missed one, the OHP precedent
    (`ohp_forward_head`) gates instead of discounting. The `SIDE_VIEW_CONF_THRESHOLD` floor
    follows squat's `rule_knees_forward` and OHP -- no new number.
    """
    if ctx.view_type not in SAGITTAL_VIEWS or ctx.view_confidence < SIDE_VIEW_CONF_THRESHOLD:
        return []

    torso_0 = setup_baseline(core, "torso_len")
    hip_0 = setup_baseline(core, "hip_y")
    if not np.isfinite(torso_0) or not np.isfinite(hip_0) or torso_0 < _DEGENERATE_LENGTH:
        return []

    def _ratio(frame: CoreFrame) -> float:
        value = frame.m("torso_len")
        return value / torso_0 if np.isfinite(value) else float(np.nan)

    def _hips_still(frame: CoreFrame) -> bool:
        value = frame.m("hip_y")
        if not np.isfinite(value):
            return False
        return abs(value - hip_0) / torso_0 < DEADLIFT_HIP_STATIONARY_BAND_UNSOURCED

    mask = [
        frame.valid
        and frame.phase in DEADLIFT_ACTIVE_PHASES
        and np.isfinite(_ratio(frame))
        and _ratio(frame) < DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED
        and _hips_still(frame)
        for frame in core
    ]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        ratios = [_ratio(frame) for frame in segment]
        worst = float(np.nanmin(ratios))
        severity = severity_from_range(
            worst,
            DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED,
            DEADLIFT_TORSO_SHORTENING_SEVERE_UNSOURCED,
            lower_is_worse=True,
        )
        detections.append(
            build_detection(
                fault_id="deadlift_lumbar_flexion",
                fault_name="Rounded Lower Back / Lumbar Flexion",
                kg_query=DEADLIFT_LUMBAR_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                # Severity rises as the ratio FALLS, so the peak frame is the smallest ratio;
                # negate so build_detection's argmax finds it.
                score_values=[-r for r in ratios],
                severity=severity,
                # ALWAYS discounted: this is a proxy, not a measurement, even in its own view.
                confidence=severity * _OFF_VIEW_CONFIDENCE,
                observability="low",
                evidence={
                    "min_torso_length_ratio": round(worst, 4),
                    "setup_torso_length": round(torso_0, 4),
                    "threshold": DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED,
                    "proxy": "projected torso shortening; MediaPipe has no lumbar landmarks",
                    "primary_label": "torso length vs setup",
                    "primary_value": round(worst, 4),
                    "primary_threshold": DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED,
                },
                citation="Moreira VM, et al. \"Analysis of Muscle Strength and Electromyographic "
                         "Activity during Different Deadlift Positions.\" Muscles (2023). "
                         "PMC12225233.",
                citation_support="PMC12225233: \"The lift-off position in DL, using the "
                                 "powerlift posture, generates greater lumbar spine shear "
                                 "force,\" and erector-spinae activation peaks at "
                                 "lift-off/mid-pull because \"ERE requires higher activation "
                                 "and higher strength to avoid trunk flexion, reducing shear.\" "
                                 "Verified in RAG doc. NOTE what this does and does not "
                                 "support: the FAULT is cited, loaded and mechanistically "
                                 "understood; the source says nothing about detecting it from "
                                 "pose, and the detection threshold here is unsourced.",
            )
        )
    return detections
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/test_deadlift.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/deadlift.py tests/test_deadlift.py
git commit -m "feat(pose): deadlift lumbar flexion as an explicitly unsourced sagittal proxy"
```

---

## Task 5: Assemble and register `DEADLIFT_DETECTOR`

**Files:**
- Modify: `src/pose/movements/deadlift.py`
- Modify: `src/pose/movements/registry.py:34`
- Modify: `tests/test_deadlift.py`
- Modify: `tests/test_movements_endpoint.py`

**Interfaces:**
- Consumes: Tasks 1–4
- Produces: `DEADLIFT_DETECTOR: MovementDetector`, registered under `"Deadlift"`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_deadlift.py`:

```python
from src.pose.movements import registry
from src.pose.movements.deadlift import DEADLIFT_DETECTOR


class DeadliftDetectorTests(unittest.TestCase):
    def test_it_is_registered_under_its_canonical_name(self):
        self.assertIs(registry.get_detector("Deadlift"), DEADLIFT_DETECTOR)

    def test_lookup_is_case_insensitive(self):
        self.assertIs(registry.get_detector("deadlift"), DEADLIFT_DETECTOR)

    def test_it_ships_unvalidated_because_no_labeled_deadlift_data_exists(self):
        self.assertFalse(DEADLIFT_DETECTOR.validated)

    def test_the_rep_signal_is_a_declared_metric_key(self):
        self.assertIn(DEADLIFT_DETECTOR.rep_signal, DEADLIFT_DETECTOR.metric_keys)

    def test_the_rep_starts_flexed_because_the_bar_starts_on_the_floor(self):
        self.assertEqual(DEADLIFT_DETECTOR.rep_start, "flexed")

    def test_all_three_surviving_rules_are_wired(self):
        names = {rule.__name__ for rule in DEADLIFT_DETECTOR.rules}
        self.assertEqual(
            names,
            {"rule_hips_shoot_up", "rule_incomplete_lockout", "rule_lumbar_flexion"},
        )

    def test_bar_drift_is_absent_because_it_was_withdrawn(self):
        self.assertNotIn(
            "rule_bar_drift", {rule.__name__ for rule in DEADLIFT_DETECTOR.rules}
        )
```

Modify `tests/test_movements_endpoint.py` — replace the two assertions (currently at lines ~22 and ~30) with:

```python
        self.assertEqual(names, ["Squat", "Overhead Press", "Push-up", "Lunge", "Deadlift"])
```

```python
        self.assertEqual(
            validated,
            {
                "Squat": True,
                "Overhead Press": False,
                "Push-up": False,
                "Lunge": False,
                # Deadlift ships Beta: there is no labeled deadlift data anywhere in this
                # repository, so its thresholds have never been checked against ground truth.
                "Deadlift": False,
            },
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/test_deadlift.py tests/test_movements_endpoint.py -v`
Expected: FAIL — `ImportError: cannot import name 'DEADLIFT_DETECTOR'`, and the endpoint list assertion fails on the 4-entry list.

- [ ] **Step 3: Assemble the detector**

Add `MovementDetector` and `registry` to the imports at the top of `deadlift.py`:

```python
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
```

Append to `src/pose/movements/deadlift.py`:

```python
# THREE of the parent spec's FOUR Deadlift rules are listed. `deadlift_bar_drift` is absent
# because it is WITHDRAWN, not because it was forgotten -- see the boxed note in the parent
# spec's Deadlift section. Briefly: its citation (Hanen PMC12148905) contains no bar-path
# measurement and explicitly defers one ("Analyzing the bar path would be valuable to validate
# this hypothesis"), and its `midfoot_x` reference is the invented construct that the OHP
# bar-path withdrawal already ruled out. Unlike push-up's `rule_scapular_winging`, it is not
# registered-but-silent: a silent rule says "real fault, unmeasurable", whereas this one says
# "no citation supports the rule as written", which is a spec problem, not a sensing problem.
#
# `DEADLIFT_METRIC_KEYS` must stay a two-way match with what `deadlift_compute_raw` emits --
# pinned by `test_metric_keys_match_the_emitted_metrics`.
DEADLIFT_DETECTOR = MovementDetector(
    "Deadlift",
    DEADLIFT_METRIC_KEYS,
    deadlift_compute_raw,
    deadlift_assign_phases,
    (rule_hips_shoot_up, rule_incomplete_lockout, rule_lumbar_flexion),
    # `validated` stays at its default False. No labeled deadlift data exists anywhere in this
    # repository, so unlike Lunge there is not even a validation pass to defer to; flipping
    # this would need evidence that cannot currently be obtained.
    rep_signal="hip_angle_deg",
    # The signal bottoms out at the floor and peaks at lockout, so a rep is an excursion in
    # hip EXTENSION that starts and ends flexed -- `rep_start="flexed"`, the case base.py:55
    # names deadlift for.
    rep_polarity="min",
    rep_start="flexed",
)

registry.register(DEADLIFT_DETECTOR)
```

Modify `src/pose/movements/registry.py`, adding after line 34:

```python
from src.pose.movements import deadlift  # noqa: E402,F401
```

Also update the `list_detectors` docstring, which currently names only three movements:

```python
    """Every registered detector, in registration order.

    Registration order is the import order at the bottom of this module (Squat, Overhead
    Press, Push-up, Lunge, Deadlift) -- deterministic, and it puts the validated detector
    first without encoding a UI preference in the ML layer. Backs GET /api/movements, which is
    why the frontend needs no hand-maintained list of analyzable movements.
    """
```

- [ ] **Step 4: Run the full suite**

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: PASS, no failures. `tests/test_movement_registry.py` and `tests/test_analyze_movement.py` exercise the registry generically and must stay green.

Then the coverage gate:

Run: `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/deadlift.py src/pose/movements/registry.py tests/test_deadlift.py tests/test_movements_endpoint.py
git commit -m "feat(pose): register the deadlift detector as the fifth of sixteen"
```

---

## Task 6: Parent-spec amendments

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`

No code, no tests — this task records decisions in the document that governs every detector. It is separate because a reviewer could reasonably accept the implementation and reject the wording, or vice versa.

- [ ] **Step 1: Insert the withdrawal note**

In the `### Deadlift` section, immediately after the `#### Bar Drifting Away From the Body` entry's `citation_support` line (currently ~line 240), insert:

```markdown
> **WITHDRAWN — bar drift.** This rule is **withdrawn** (2026-08-01) and is NOT implemented in
> `src/pose/movements/deadlift.py`, for three reasons:
>
> 1. **The citation contains no bar-path measurement.** Hanen PMC12148905 was read in full. Its
>    only bar-position statement is qualitative — "keeping the barbell closer to the body during
>    the SDL reduces the lever arm stress." No distance, no threshold, no units. The
>    `0.5·foot_len` figure above has no source.
> 2. **The citation explicitly disclaims it.** The paper states: *"Analyzing the bar path would
>    be valuable to validate this hypothesis."* It did not analyze bar path and says so. A rule
>    cannot cite a source for a measurement that source declares un-performed.
> 3. **The mid-foot reference is the construct already forbidden.** The OHP bar-path withdrawal
>    (above) rejected referencing the bar to mid-foot because it "would require an invented
>    mid-foot proxy — forbidden by this project's every-threshold-literature-backed premise."
>    This rule prescribes exactly that construct.
>
> **Open spec question:** does the Deadlift rule set want a genuine bar-path fault? It would need
> (a) a base-of-support reference MediaPipe can resolve and (b) a citation that measures bar
> displacement with a number. Neither exists today. This is a withdrawal pending a decision, not
> a silent deletion.
```

- [ ] **Step 2: Extend §7 with the two new honest gaps**

In `## 7. Honest limitations & gaps`, after the existing `**Deadlift lumbar flexion**` bullet, add:

```markdown
- **Deadlift lumbar-flexion detection thresholds are UNSOURCED** (2026-08-01). The implemented
  proxy — projected torso shortening against the rep's own setup baseline while the hips stay
  stationary — uses `0.95` / `0.85` ratio endpoints and a `0.10` hip-stationary band. No source
  gives a segment-shortening-to-lumbar-flexion figure; 0.95 was chosen to sit above landmark
  jitter *without any measurement of what that jitter is*. The constants carry `UNSOURCED` in
  their names. The fault is cited; the detection is not. Calibrating against a measured jitter
  floor is the known upgrade path.
- **Deadlift `hips_shoot_up` ramp endpoints are unsourced** (2026-08-01): neither deadlift RAG
  document reports a trunk inclination in degrees, so 55°/75° rest on the spec alone. The
  mechanism and direction are cited; the numbers are not.
- **Two of three Deadlift rules have no KG node** (2026-08-01) and take the `rag` fallback. The
  5-node `Deadlift:` stub (9 nodes counting its shared 1-hop neighbours, e.g.
  `Lumbar Spine Injury`, `Hip Hinge`) was authored independently of this rule catalog and does
  not agree with it: it carries nodes for two faults the catalog has no rule for (`Hyperextension At
  Lockout`, `Insufficient Hip Hinge`), lacks nodes for `deadlift_hips_shoot_up` and
  `deadlift_incomplete_lockout`, and its one exactly-matching fault node (`Bar Drift From Body`)
  belongs to the rule withdrawn above. Only `Deadlift:Lumbar Flexion` grounds a shipped rule.
  Near-misses were rejected rather than used: `Insufficient Hip Hinge` describes a
  knee-dominant pull where `hips_shoot_up` is hip-dominant, and `Hyperextension At Lockout` is
  the literal opposite of `incomplete_lockout`.
```

- [ ] **Step 3: Record the status in §8**

Append to `## 8. Next steps`, after the 2026-07-25 status block:

```markdown
**Status (2026-08-01):** **Deadlift** implemented in `src/pose/movements/deadlift.py` and
registered as the 5th of 16 — `deadlift_hips_shoot_up`, `deadlift_incomplete_lockout`,
`deadlift_lumbar_flexion`. `deadlift_bar_drift` is WITHDRAWN (see the boxed note in §Deadlift).
**Thresholds are spec-derived and UNVALIDATED**: no labeled deadlift data exists in this
repository, so unlike Lunge there is no validation pass to defer to, and §8.4 remains
unsatisfied for this movement. Deviations from the heuristics written above, deliberate and
documented in-code:

- `deadlift_hips_shoot_up` — the spec's "Δ(hip_y) rises faster than Δ(shoulder_y)" is
  **implemented as a trunk-pitch test**, not a two-landmark differential. The differential was
  checked numerically first and is algebraically identical to a pitch change: since
  `shoulder_y − hip_y = −torso_len·cos(pitch)`, a rigid torso gives
  `hip_lead_ratio ≡ cos(pitch₀) − cos(pitch_t)` exactly. It depends only on pitch and says
  nothing about hip travel, so writing it as a differential would falsely imply independent
  corroboration. The spec's own "i.e." equating the two phrasings is correct.
- Phase cutoffs are **percentiles of each rep's own hip-angle excursion**, not absolute angles.
  `deadlift_incomplete_lockout` scores the `lockout` phase and the fault *is* failing to reach
  extension, so an absolute cutoff would delete the phase on exactly the reps the rule exists
  to catch.
- `deadlift_lumbar_flexion` is **hard-gated** to sagittal views while the other two rules
  **degrade** off-view. The asymmetry is deliberate: an angle magnitude under-reads head-on
  (failure mode = silence), whereas the torso-shortening proxy is corrupted by trunk pitch
  head-on (failure mode = a false positive). Gate where a wrong claim is possible, discount
  where only a missed one is — the OHP `ohp_forward_head` precedent.
```

- [ ] **Step 4: Verify the spec still renders and nothing else changed**

Run: `git diff --stat docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`
Expected: one file changed, insertions only, no deletions.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md
git commit -m "docs(pose): record the deadlift withdrawal, the unsourced numbers, and the KG divergence"
```

---

## Final verification

- [ ] `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe -m pytest tests/ -q` — all green
- [ ] `C:/Users/ttsh1/code/x-coach/.venv/Scripts/python.exe scripts/run_backend_coverage.py --fail-under 95` — passes
- [ ] `cd frontend && yarn test:coverage` — unchanged and green (no frontend edit was made; this confirms it)
- [ ] `git log --oneline origin/main..HEAD` — six implementation commits plus the two design commits
- [ ] Rename the branch for the PR: `git branch -m feat/deadlift-detector`
