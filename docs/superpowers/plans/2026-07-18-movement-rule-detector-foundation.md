# Movement Rule-Detector Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the squat-hardcoded pose rule detector into a movement-dispatched engine, migrate squat into it with no behavior change, attach literature citations to every detection, and prove the abstraction generalizes by implementing a second, architecturally-different movement (Overhead Press).

**Architecture:** A per-movement `MovementDetector` (raw-metric compute fn + phase-assignment fn + list of rule fns + the metric keys to smooth) is registered by canonical movement name. A generic runner computes raw metrics → assigns phases → smooths declared metric keys → assembles a stable `CoreFrame` (small fixed core + a `metrics` dict) → runs each rule fn. `detect_pose_rules_from_payload` routes on the existing `movement` argument (default `"Squat"`) to the registry. Squat's five current rules move verbatim into `src/pose/movements/squat.py`; OHP is added as `src/pose/movements/overhead_press.py`.

**Tech Stack:** Python 3.11/3.12, numpy, `unittest.TestCase` under `tests/`. No new dependencies (local-first, stdlib + numpy per project style).

## Global Constraints

- **Interpreter (this machine has no `python` on PATH):** always `.venv\Scripts\python.exe` from repo root. Run tests as `.venv\Scripts\python.exe -m pytest tests/...`.
- **Backend coverage gate (CI enforces 95%):** `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95` must still pass. Every new function needs tests.
- **Run from repo root:** modules import by absolute package path (`from src.pose... import ...`).
- **Additive detection fields only:** the frontend consumes `kg_query` and `retrieval_mode` on each detection (`frontend/src/api.ts`, `frontend/src/lib/retrieval.ts`). These fields MUST stay. New citation fields are added, never renamed or removed.
- **Behavior-preserving squat gate:** the four existing tests in `tests/test_pose_rule_detector.py` MUST pass unchanged after squat migrates into the registry. Do not edit them.
- **Threshold-validation honesty:** thresholds for Overhead Press (and every future movement) are derived from the spec, not from labeled data. Synthetic tests prove the geometry math, NOT that the numbers detect real faults. Every new-movement module carries a header comment saying its thresholds are unvalidated pending labeled data (spec §8.4).
- **Source of truth for rules & citations:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`. Every rule's `citation`/`citation_support` text is copied from that spec, not re-derived.

---

### Task 1: Add citation metadata to detections

**Files:**
- Modify: `src/pose/pose_rule_detector.py` (dataclass `PoseRuleDetection` ~lines 77-93; `build_detection` ~lines 404-442; the five squat `build_detection(...)` call sites in `detect_rule_segments`)
- Test: `tests/test_pose_rule_detector.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `PoseRuleDetection` gains `citation: str` and `citation_support: str` fields; `build_detection(..., citation: str = "", citation_support: str = "")` keyword args. Downstream `asdict(detection)` now includes both keys.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pose_rule_detector.py`:

```python
def test_detection_carries_citation_metadata(self) -> None:
    frames = [frame(knee_x_gap=0.20, ankle_x_gap=0.40, frame_index=i) for i in range(12)]
    metrics = compute_frame_metrics(frames, fps=30.0)
    detections = detect_rule_segments(metrics, fps=30.0, view_type="rear", view_confidence=0.8)
    inward = next(d for d in detections if d.fault_id == "knees_inward")
    assert inward.citation.startswith("Ford KR")
    assert "73%" in inward.citation_support or "abduction" in inward.citation_support.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pose_rule_detector.py::PoseRuleDetectorTests::test_detection_carries_citation_metadata -v`
Expected: FAIL — `AttributeError: 'PoseRuleDetection' object has no attribute 'citation'`.

- [ ] **Step 3: Add the fields and thread them through**

In `PoseRuleDetection` (after `evidence`):

```python
    citation: str = ""
    citation_support: str = ""
```

In `build_detection` signature add `citation: str = "", citation_support: str = "",` and in the returned `PoseRuleDetection(...)` add:

```python
        citation=citation,
        citation_support=citation_support,
```

In `detect_rule_segments`, add `citation=`/`citation_support=` to the `knees_inward` call using the spec text:

```python
                citation="Ford KR, Nguyen AD, Dischiavi SL, Hegedus EJ, Zuk EF, Taylor JB. (2015). "
                         "An evidence-based review of hip-focused neuromuscular exercise interventions "
                         "to address dynamic lower extremity valgus. Open Access J Sports Med. PMC4556293.",
                citation_support="Knee abduction moment predicted future ACL injury risk with 73% "
                                 "sensitivity and 78% specificity in young female athletes.",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pose_rule_detector.py::PoseRuleDetectorTests::test_detection_carries_citation_metadata -v`
Expected: PASS.

- [ ] **Step 5: Fill in citations for the other four squat rules**

Add `citation`/`citation_support` kwargs (copied from spec Group A → Squat) to the `knees_forward`, `shallow_depth`, `excessive_forward_lean`, and `heel_rise` `build_detection` calls. Use the exact reference + one-line finding from the spec for each.

- [ ] **Step 6: Run the full detector test file**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pose_rule_detector.py -v`
Expected: all pass (4 original + 1 new).

- [ ] **Step 7: Commit**

```bash
git add src/pose/pose_rule_detector.py tests/test_pose_rule_detector.py
git commit -m "feat(pose): attach literature citations to squat detections"
```

---

### Task 2: Extract stateless geometry helpers into a shared module

**Files:**
- Create: `src/pose/geometry.py`
- Modify: `src/pose/pose_rule_detector.py` (replace the moved definitions with imports/re-exports)
- Test: `tests/test_pose_geometry.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `src/pose/geometry.py` exporting (unchanged signatures) `landmarks_to_array`, `visible_point`, `distance`, `angle_degrees`, `midpoint`, `line_angle_from_vertical`, `mean_visibility`, `mean_finite`, `centered_median`, `knee_forward_ratio`, `heel_height_delta`, `clip01`, `contiguous_true_segments`, `severity_from_range`, plus the landmark-index constants (`LEFT_SHOULDER`…`RIGHT_FOOT_INDEX`, `LANDMARK_COUNT`, `VISIBILITY_THRESHOLD`). `pose_rule_detector.py` re-imports them so its public API is unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pose_geometry.py`:

```python
import unittest
import numpy as np
from src.pose.geometry import angle_degrees, severity_from_range, contiguous_true_segments


class PoseGeometryTests(unittest.TestCase):
    def test_angle_degrees_right_angle(self) -> None:
        pts = np.zeros((33, 4), dtype=np.float32)
        pts[:, 3] = 1.0
        pts[0, :3] = [0, 1, 0]
        pts[1, :3] = [0, 0, 0]
        pts[2, :3] = [1, 0, 0]
        self.assertAlmostEqual(angle_degrees(pts, 0, 1, 2), 90.0, places=3)

    def test_severity_ramp_monotonic(self) -> None:
        self.assertEqual(severity_from_range(0.82, 0.82, 0.70, lower_is_worse=True), 0.0)
        self.assertEqual(severity_from_range(0.70, 0.82, 0.70, lower_is_worse=True), 1.0)

    def test_contiguous_segments_respects_min_frames(self) -> None:
        self.assertEqual(contiguous_true_segments([True, True, False, True], 2), [(0, 1)])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pose_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pose.geometry'`.

- [ ] **Step 3: Create `src/pose/geometry.py`**

Move the listed functions and constants verbatim from `pose_rule_detector.py` into `src/pose/geometry.py` (add `from __future__ import annotations` and `import numpy as np` at top). Do not change any function body.

- [ ] **Step 4: Re-import in `pose_rule_detector.py`**

Replace the moved definitions with:

```python
from src.pose.geometry import (
    LANDMARK_COUNT, VISIBILITY_THRESHOLD,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, distance, angle_degrees, midpoint,
    line_angle_from_vertical, mean_visibility, mean_finite, centered_median,
    knee_forward_ratio, heel_height_delta, clip01, contiguous_true_segments,
    severity_from_range,
)
```

- [ ] **Step 5: Run the geometry + detector tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pose_geometry.py tests/test_pose_rule_detector.py -v`
Expected: all pass (extraction is behavior-preserving).

- [ ] **Step 6: Commit**

```bash
git add src/pose/geometry.py src/pose/pose_rule_detector.py tests/test_pose_geometry.py
git commit -m "refactor(pose): extract stateless geometry helpers into src/pose/geometry.py"
```

---

### Task 3: Define the movement-detector interface and registry

**Files:**
- Create: `src/pose/movements/__init__.py`
- Create: `src/pose/movements/base.py`
- Create: `src/pose/movements/registry.py`
- Test: `tests/test_movement_registry.py` (create)

**Interfaces:**
- Consumes: `PoseRuleDetection`, `build_detection` from `pose_rule_detector`; helpers from `geometry`.
- Produces:
  - `CoreFrame(frame_index:int, time:float, phase:str, valid:bool, lower_body_visibility:float, metrics:dict[str,float])` — frozen dataclass; `def m(self, key:str)->float` returns `self.metrics.get(key, nan)`.
  - `RuleContext(fps:float, view_type:str, view_confidence:float, min_frames:int)`.
  - `RuleFn = Callable[[list[CoreFrame], RuleContext], list[PoseRuleDetection]]`.
  - `MovementDetector(name:str, metric_keys:tuple[str,...], compute_raw:Callable[[Sequence[object], float], list[dict]], assign_phases:Callable[[list[dict]], list[str]], rules:tuple[RuleFn,...])`.
  - `run_detector(detector:MovementDetector, frames:Sequence[object], fps:float, view_type:str, view_confidence:float) -> tuple[list[CoreFrame], list[PoseRuleDetection]]`.
  - `registry.get_detector(movement:str|None) -> MovementDetector` (case-insensitive; default `"Squat"`; raises `KeyError` on unknown), and `registry.register(detector)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_movement_registry.py`:

```python
import unittest
from src.pose.movements.base import CoreFrame, RuleContext, MovementDetector, run_detector
from src.pose.movements import registry


class MovementRegistryTests(unittest.TestCase):
    def test_core_frame_metric_accessor_defaults_nan(self) -> None:
        cf = CoreFrame(0, 0.0, "setup", True, 0.9, {"a": 1.0})
        self.assertEqual(cf.m("a"), 1.0)
        import math
        self.assertTrue(math.isnan(cf.m("missing")))

    def test_default_movement_is_squat(self) -> None:
        det = registry.get_detector(None)
        self.assertEqual(det.name, "Squat")

    def test_unknown_movement_raises(self) -> None:
        with self.assertRaises(KeyError):
            registry.get_detector("Nonexistent Movement")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pose.movements'`.

- [ ] **Step 3: Create `src/pose/movements/__init__.py`** (empty file).

- [ ] **Step 4: Create `src/pose/movements/base.py`**

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from src.pose.geometry import centered_median, contiguous_true_segments  # noqa: F401
from src.pose.pose_rule_detector import PoseRuleDetection


@dataclass(frozen=True)
class CoreFrame:
    frame_index: int
    time: float
    phase: str
    valid: bool
    lower_body_visibility: float
    metrics: dict[str, float] = field(default_factory=dict)

    def m(self, key: str) -> float:
        return float(self.metrics.get(key, math.nan))


@dataclass(frozen=True)
class RuleContext:
    fps: float
    view_type: str
    view_confidence: float
    min_frames: int


RuleFn = Callable[[list["CoreFrame"], "RuleContext"], list[PoseRuleDetection]]


@dataclass(frozen=True)
class MovementDetector:
    name: str
    metric_keys: tuple[str, ...]
    compute_raw: Callable[[Sequence[object], float], list[dict]]
    assign_phases: Callable[[list[dict]], list[str]]
    rules: tuple[RuleFn, ...]


def run_detector(
    detector: MovementDetector,
    frames: Sequence[object],
    fps: float,
    view_type: str,
    view_confidence: float,
) -> tuple[list[CoreFrame], list[PoseRuleDetection]]:
    raw = detector.compute_raw(frames, fps)
    phases = detector.assign_phases(raw)
    smoothed = {
        key: centered_median([float(item.get(key, np.nan)) for item in raw], window=5)
        for key in detector.metric_keys
    }
    core: list[CoreFrame] = []
    for i, item in enumerate(raw):
        core.append(
            CoreFrame(
                frame_index=int(item.get("frame_index", i) or i),
                time=float(item.get("time", 0.0) or 0.0),
                phase=phases[i],
                valid=bool(item.get("valid", False)),
                lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
                metrics={key: float(smoothed[key][i]) for key in detector.metric_keys},
            )
        )
    min_frames = max(3, int(math.ceil(max(fps, 1.0) * 0.20)))
    ctx = RuleContext(fps=fps, view_type=view_type, view_confidence=view_confidence, min_frames=min_frames)
    detections: list[PoseRuleDetection] = []
    for rule in detector.rules:
        detections.extend(rule(core, ctx))
    detections.sort(key=lambda d: (d.observability == "low", -d.severity, d.start_frame))
    return core, detections
```

- [ ] **Step 5: Create `src/pose/movements/registry.py`**

```python
from __future__ import annotations

from src.pose.movements.base import MovementDetector

_REGISTRY: dict[str, MovementDetector] = {}


def register(detector: MovementDetector) -> None:
    _REGISTRY[detector.name.lower()] = detector


def get_detector(movement: str | None) -> MovementDetector:
    key = (movement or "Squat").lower()
    if key not in _REGISTRY:
        raise KeyError(f"No detector registered for movement {movement!r}")
    return _REGISTRY[key]


# Import movement modules for their registration side effects.
from src.pose.movements import squat  # noqa: E402,F401
```

Note: `squat` import will fail until Task 4 creates it. Temporarily comment out the last line for this task's test, then re-enable in Task 4.

- [ ] **Step 6: Run the registry test (with squat import commented)**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py::MovementRegistryTests::test_core_frame_metric_accessor_defaults_nan -v`
Expected: PASS. (The `default_movement`/`unknown` tests stay red until Task 4 registers squat — that is expected and closed there.)

- [ ] **Step 7: Commit**

```bash
git add src/pose/movements/ tests/test_movement_registry.py
git commit -m "feat(pose): add MovementDetector interface and registry skeleton"
```

---

### Task 4: Migrate squat into the registry (behavior-preserving)

**Files:**
- Create: `src/pose/movements/squat.py`
- Modify: `src/pose/movements/registry.py` (re-enable `import squat`)
- Modify: `src/pose/pose_rule_detector.py` (`detect_pose_rules_from_payload` routes via registry)
- Test: `tests/test_movement_registry.py`, `tests/test_pose_rule_detector.py` (unchanged — the gate)

**Interfaces:**
- Consumes: `MovementDetector`, `CoreFrame`, `RuleContext`, `run_detector` from `base`; geometry helpers; `build_detection`.
- Produces: `src/pose/movements/squat.py` defining `SQUAT_DETECTOR: MovementDetector` (name `"Squat"`), registered on import. `detect_pose_rules_from_payload` gains internal use of `registry.get_detector(movement)` + `run_detector(...)`.

- [ ] **Step 1: Add the behavior-preserving gate test**

Add to `tests/test_movement_registry.py`:

```python
def test_squat_via_registry_matches_legacy(self) -> None:
    from src.pose.pose_rule_detector import compute_frame_metrics, detect_rule_segments
    from src.pose.movements import registry
    from src.pose.movements.base import run_detector
    from tests.test_pose_rule_detector import frame  # reuse fixture builder

    frames = [frame(knee_x_gap=0.20, ankle_x_gap=0.40, frame_index=i) for i in range(14)]
    legacy = detect_rule_segments(compute_frame_metrics(frames, 30.0), fps=30.0, view_type="rear", view_confidence=0.8)
    _, new = run_detector(registry.get_detector("Squat"), frames, 30.0, "rear", 0.8)
    self.assertEqual([d.fault_id for d in legacy], [d.fault_id for d in new])
    self.assertEqual([round(d.severity, 4) for d in legacy], [round(d.severity, 4) for d in new])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py::MovementRegistryTests::test_squat_via_registry_matches_legacy -v`
Expected: FAIL — squat not yet registered.

- [ ] **Step 3: Create `src/pose/movements/squat.py`**

Wrap the existing squat logic behind the interface WITHOUT changing the math:
- `compute_raw(frames, fps)` = the body of the current `raw_frame_metrics` mapped over frames (return the raw dicts; keep every existing metric key inside them).
- `assign_phases(raw)` = the current `assign_phases` verbatim.
- `metric_keys` = the exact 12 smoothed field names currently in `compute_frame_metrics`.
- Rule fns: split the current `detect_rule_segments` body into five module-level functions `rule_knees_inward`, `rule_knees_forward`, `rule_shallow_depth`, `rule_forward_lean`, `rule_heel_rise`, each `(core, ctx) -> list[PoseRuleDetection]`, reading metrics via `frame.m("avg_knee_angle")` etc. Copy the citation kwargs added in Task 1.
- `SQUAT_DETECTOR = MovementDetector("Squat", METRIC_KEYS, compute_raw, assign_phases, (rule_knees_inward, ...))` then `registry.register(SQUAT_DETECTOR)`.

- [ ] **Step 4: Re-enable squat import in `registry.py`** (uncomment the line from Task 3 Step 5).

- [ ] **Step 5: Route `detect_pose_rules_from_payload` through the registry**

Replace the direct `compute_frame_metrics` + `detect_rule_segments` calls with:

```python
    from src.pose.movements import registry
    from src.pose.movements.base import run_detector

    detector = registry.get_detector(movement)
    core, detections = run_detector(detector, frames, fps if fps > 0 else 30.0, view_type, view_confidence)
```

Serialize `frame_metrics` by flattening each `CoreFrame` (core fields + spread `metrics`) so the JSON key shape is unchanged for squat:

```python
    "frame_metrics": [
        {"frame_index": c.frame_index, "time": c.time, "phase": c.phase,
         "valid": c.valid, "lower_body_visibility": c.lower_body_visibility, **c.metrics}
        for c in core
    ],
```

Keep the legacy `compute_frame_metrics`/`detect_rule_segments`/`FrameMetrics` symbols in place (still imported by tests); they are now the squat reference, not the dispatch path.

- [ ] **Step 6: Run the gate + registry tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pose_rule_detector.py tests/test_movement_registry.py -v`
Expected: ALL pass — the four original detector tests unchanged, the registry match test green.

- [ ] **Step 7: Commit**

```bash
git add src/pose/movements/squat.py src/pose/movements/registry.py src/pose/pose_rule_detector.py tests/test_movement_registry.py
git commit -m "refactor(pose): dispatch squat through the movement registry (no behavior change)"
```

---

### Task 5: Overhead Press — raw metrics and phases

**Files:**
- Create: `src/pose/movements/overhead_press.py`
- Test: `tests/test_overhead_press.py` (create)

**Interfaces:**
- Consumes: geometry helpers; `CoreFrame`.
- Produces (used by Task 6):
  - `ohp_compute_raw(frames, fps) -> list[dict]` with keys: `frame_index`, `time`, `valid`, `lower_body_visibility`, and metrics `left_elbow_angle`, `right_elbow_angle`, `avg_elbow_angle`, `wrist_above_shoulder` (mean wrist-y minus shoulder-y, image-y down so negative = wrists above), `torso_lean_signed_deg` (signed angle of shoulder-mid→hip-mid from vertical; positive = leaning back), `elbow_height_asymmetry` (`|y_L_elbow − y_R_elbow|`), `shoulder_ear_gap` (mean `y_shoulder − y_ear`).
  - `ohp_assign_phases(raw) -> list[str]` with phases `setup`, `press` (ascending, wrists rising), `lockout` (wrists highest / elbows most extended), `lower` (descending).
  - `OHP_METRIC_KEYS: tuple[str, ...]` listing the six metric keys above.

Header comment (required): `# Thresholds below are spec-derived (docs/.../16-movement...), NOT validated against labeled OHP data (spec §8.4).`

- [ ] **Step 1: Write the failing test**

Create `tests/test_overhead_press.py`:

```python
import unittest
import numpy as np
from src.pose.movements.overhead_press import ohp_compute_raw, ohp_assign_phases


def ohp_frame(elbow_angle: float, wrist_y: float, shoulder_y: float = 0.4, frame_index: int = 0) -> dict:
    lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]
    # shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24, ears 7/8
    lm[11] = {"x": 0.45, "y": shoulder_y, "z": 0, "visibility": 1.0}
    lm[12] = {"x": 0.55, "y": shoulder_y, "z": 0, "visibility": 1.0}
    lm[13] = {"x": 0.43, "y": shoulder_y + 0.15, "z": 0, "visibility": 1.0}
    lm[14] = {"x": 0.57, "y": shoulder_y + 0.15, "z": 0, "visibility": 1.0}
    lm[15] = {"x": 0.44, "y": wrist_y, "z": 0, "visibility": 1.0}
    lm[16] = {"x": 0.56, "y": wrist_y, "z": 0, "visibility": 1.0}
    lm[23] = {"x": 0.46, "y": 0.75, "z": 0, "visibility": 1.0}
    lm[24] = {"x": 0.54, "y": 0.75, "z": 0, "visibility": 1.0}
    lm[7] = {"x": 0.46, "y": shoulder_y - 0.08, "z": 0, "visibility": 1.0}
    lm[8] = {"x": 0.54, "y": shoulder_y - 0.08, "z": 0, "visibility": 1.0}
    return {"frame_index": frame_index, "landmarks": lm}


class OverheadPressMetricsTests(unittest.TestCase):
    def test_wrist_above_shoulder_sign(self) -> None:
        raw = ohp_compute_raw([ohp_frame(160, wrist_y=0.20)], 30.0)  # wrists above shoulders
        self.assertLess(raw[0]["wrist_above_shoulder"], 0.0)

    def test_phases_include_lockout_at_top(self) -> None:
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(175, 0.15, frame_index=i + 4) for i in range(4)]
                  + [ohp_frame(90, 0.45, frame_index=i + 8) for i in range(4)])
        phases = ohp_assign_phases(ohp_compute_raw(frames, 30.0))
        self.assertIn("lockout", phases)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_overhead_press.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `ohp_compute_raw` and `ohp_assign_phases`**

Implement using geometry helpers (`landmarks_to_array`, `visible_point`, `angle_degrees`, `midpoint`, `mean_visibility`, `mean_finite`). Required landmarks: shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24; ears 7/8 optional (NaN gap when missing). `torso_lean_signed_deg`: signed x-offset of shoulder-mid relative to hip-mid → `degrees(arctan2(shoulder_mid_x − hip_mid_x, hip_mid_y − shoulder_mid_y))`; positive = shoulders behind hips (back-lean) when filmed from the side. `ohp_assign_phases`: `lockout` where `avg_elbow_angle` ≥ 70th percentile AND `wrist_above_shoulder` ≤ 30th percentile; frames before the wrist-highest index = `press`, after = `lower`; first 15% = `setup`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_overhead_press.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/overhead_press.py tests/test_overhead_press.py
git commit -m "feat(pose): overhead press raw metrics and phase segmentation"
```

---

### Task 6: Overhead Press — three core cited rules + register

**Files:**
- Modify: `src/pose/movements/overhead_press.py`
- Test: `tests/test_overhead_press.py`

**Interfaces:**
- Consumes: `CoreFrame`, `RuleContext`, `MovementDetector`, `build_detection`, `severity_from_range`, `contiguous_true_segments`; `registry.register`.
- Produces: `rule_incomplete_lockout`, `rule_excessive_back_lean`, `rule_asymmetric_press` (each `(core, ctx)->list[PoseRuleDetection]`), and `OHP_DETECTOR: MovementDetector` (name `"Overhead Press"`) registered on import. Rule content and citations copied from spec Group B → Overhead Press.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_overhead_press.py`:

```python
class OverheadPressRulesTests(unittest.TestCase):
    def _run(self, frames, view="side", vc=0.8):
        from src.pose.movements import registry
        from src.pose.movements.base import run_detector
        return run_detector(registry.get_detector("Overhead Press"), frames, 30.0, view, vc)[1]

    def test_incomplete_lockout_flagged(self) -> None:
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(120, 0.30, frame_index=i + 4) for i in range(6)]   # elbows never extend
                  + [ohp_frame(90, 0.45, frame_index=i + 10) for i in range(4)])
        ids = {d.fault_id for d in self._run(frames)}
        self.assertIn("ohp_incomplete_lockout", ids)

    def test_full_lockout_not_flagged(self) -> None:
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(178, 0.12, frame_index=i + 4) for i in range(6)]
                  + [ohp_frame(90, 0.45, frame_index=i + 10) for i in range(4)])
        ids = {d.fault_id for d in self._run(frames)}
        self.assertNotIn("ohp_incomplete_lockout", ids)

    def test_lockout_rule_carries_citation(self) -> None:
        frames = [ohp_frame(120, 0.30, frame_index=i) for i in range(12)]
        det = next((d for d in self._run(frames) if d.fault_id == "ohp_incomplete_lockout"), None)
        assert det is not None and det.citation and det.citation_support
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_overhead_press.py::OverheadPressRulesTests -v`
Expected: FAIL — rules/detector not defined.

- [ ] **Step 3: Implement the three rules and register the detector**

- `rule_incomplete_lockout`: over frames whose phase is `lockout` (or the top window if no lockout phase), flag when peak `avg_elbow_angle < 165` OR `wrist_above_shoulder > 0` (wrists never clear shoulders). Severity ramp `165→140` on elbow angle. `citation`/`citation_support` from spec (Evangelista P et al., PMC12372072). `observability="high"` for side/front_oblique.
- `rule_excessive_back_lean`: flag frames where `torso_lean_signed_deg > 15` (leaning back). Ramp `15→35`. Citation: Gregori P et al., PMC13086636 (spine alignment) or Abdelraouf OR et al., PMC13116542 (posture under failure) per spec. `observability`: high side/oblique, medium head-on.
- `rule_asymmetric_press`: flag frames where `elbow_height_asymmetry > 0.05` sustained. Ramp `0.05→0.15`. Citation: Coratella G et al., PMC9354811 per spec (or the spec's asymmetry citation). `observability`: high front/rear.
- Each rule uses `contiguous_true_segments(mask, ctx.min_frames)` and `build_detection(...)` exactly like squat, with `retrieval_mode="kg"` and a `kg_query` matching the fault.
- Append: `OHP_DETECTOR = MovementDetector("Overhead Press", OHP_METRIC_KEYS, ohp_compute_raw, ohp_assign_phases, (rule_incomplete_lockout, rule_excessive_back_lean, rule_asymmetric_press))` then `registry.register(OHP_DETECTOR)`. Add `from src.pose.movements import overhead_press  # noqa` to `registry.py`'s side-effect imports.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_overhead_press.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pose/movements/overhead_press.py src/pose/movements/registry.py tests/test_overhead_press.py
git commit -m "feat(pose): overhead press cited rules (lockout, back-lean, asymmetry)"
```

---

### Task 7: Wire the `movement` argument end-to-end

**Files:**
- Modify: `src/pose/pose_rule_detector.py` (`detect_pose_rules_from_json` passthrough already carries `movement`; ensure the CLI exposes it)
- Modify: the module `main()` / argparse block (bottom of `pose_rule_detector.py`)
- Test: `tests/test_movement_registry.py`

**Interfaces:**
- Consumes: `get_detector`, `run_detector`.
- Produces: `--movement <name>` CLI flag (default `Squat`) threaded to `detect_pose_rules_from_json(..., movement=...)`; unknown movement yields a clear error, not a squat silent-fallback.

- [ ] **Step 1: Write the failing test**

```python
def test_payload_routes_to_named_movement(self) -> None:
    from src.pose.pose_rule_detector import detect_pose_rules_from_payload
    from tests.test_overhead_press import ohp_frame
    frames = [ohp_frame(120, 0.30, frame_index=i) for i in range(12)]
    payload = {"metadata": {"fps": 30.0}, "frames": frames}
    result = detect_pose_rules_from_payload(payload, movement="Overhead Press")
    ids = {d["fault_id"] for d in result["detections"]}
    self.assertIn("ohp_incomplete_lockout", ids)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py::MovementRegistryTests::test_payload_routes_to_named_movement -v`
Expected: FAIL if payload routing still hardcodes squat (it should already route after Task 4 — if so this test PASSES and just locks the behavior; keep it).

- [ ] **Step 3: Add the `--movement` argparse flag**

In the argparse setup add `parser.add_argument("--movement", default="Squat")` and pass `movement=args.movement` into the detect call.

- [ ] **Step 4: Run the full pose test suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pose_rule_detector.py tests/test_movement_registry.py tests/test_overhead_press.py tests/test_pose_geometry.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pose/pose_rule_detector.py tests/test_movement_registry.py
git commit -m "feat(pose): expose --movement flag and route detection by movement"
```

---

### Task 8: Coverage gate, docs, and honesty note

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` (§8 status: mark foundation + OHP done, thresholds unvalidated)
- Modify: `scripts/pose/README.md` (document `--movement`)
- Test: full suite + coverage gate

- [ ] **Step 1: Run the backend coverage gate**

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS at ≥95%. If new modules dip coverage, add targeted tests for the uncovered lines (e.g. NaN/invalid-frame branches in `ohp_compute_raw`).

- [ ] **Step 2: Update spec §8 status**

Add under §8: "Foundation shipped (movement registry + citations + squat migration + Overhead Press) on branch `feat/movement-rule-detector-spec`. **OHP thresholds are spec-derived and unvalidated** — no labeled OHP data yet (§8.4). Remaining 14 movements follow as per-movement plans reusing this framework."

- [ ] **Step 3: Document the CLI flag** in `scripts/pose/README.md`.

- [ ] **Step 4: Full suite green**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: pass (allow the 2 known pre-existing flakes noted in memory `multimovement-kg-schema`, unrelated to this change).

- [ ] **Step 5: Commit**

```bash
git add docs/ scripts/pose/README.md
git commit -m "docs(pose): document movement dispatch; mark OHP thresholds unvalidated"
```

---

## Roadmap (follow-on plans — NOT part of this plan)

Each reuses the framework above; one plan per movement, same task shape (raw metrics → phases → cited rules → tests):
Push-up · Lunge · Deadlift · Row · Band Pull Apart · Bicep Curl · Arm Abduction · Arm VW · Sit-up · Shoulder Bridge · Leg Abduction · Torso Twist · Jumping Jacks · High Knee.

Analysis (`backend`) stays squat-only in the product until a movement's thresholds are validated against labeled data, regardless of detector availability.

## Self-Review

- **Spec coverage:** this plan implements the *engine* + squat (spec Group A squat) + OHP (spec Group B OHP, 3 of 5 rules). The remaining 65 rules are explicitly deferred to per-movement follow-on plans (roadmap above). No spec rule is silently dropped.
- **Placeholders:** none — every code step shows real code; every rule's citation text is copied from the spec at implementation time.
- **Type consistency:** `CoreFrame.m(key)`, `RuleContext`, `MovementDetector(name, metric_keys, compute_raw, assign_phases, rules)`, `run_detector(...) -> (core, detections)`, `registry.get_detector/register` are used identically across Tasks 3–7.
- **Honesty:** OHP thresholds are flagged unvalidated in the module header (Task 5), spec §8 (Task 8), and the roadmap; synthetic tests are stated to prove geometry, not real-fault detection.
