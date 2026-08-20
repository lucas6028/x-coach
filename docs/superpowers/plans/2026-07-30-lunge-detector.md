# Lunge Detector + First Labeled Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Lunge rule detector (4th of 16 movements) and, for the first time in this
project, measure a detector's rules against human-labeled ground truth.

**Architecture:** Phase 0 extracts pose for REHAB24-6 Ex5 and answers one question that
changes what Phase 1 may claim (does a true sagittal clip read `side`?). Phase 1 adds
`src/pose/movements/lunge.py` following the existing `MovementDetector` contract — raw
metrics for **both legs**, phases, four cited rules that resolve the lead leg over their own
per-rep window. Phase 2 adds a validation harness that replays 174 labeled reps through the
production rules and reports per-subject separation.

**Tech Stack:** Python 3.11/3.12, numpy, stdlib. Tests are `unittest.TestCase` under
`tests/`. No new dependencies.

**Branch:** `feat/lunge-detector` (already created; the design spec is committed on it).

## Global Constraints

- **Interpreter:** `.venv\Scripts\python.exe` from the repo root. NEVER bare `python`/`pip`,
  never `source .venv/bin/activate`. This machine has NO `python` on PATH.
- **Tests:** `.venv\Scripts\python.exe -m pytest tests/` — always scope to `tests/`, never
  bare `pytest`.
- **Coverage gate (CI-enforced):** `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.
- **Source of truth for every rule:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`
  §Lunge (lines 163–211). The design spec for THIS work is
  `docs/superpowers/specs/2026-07-30-lunge-detector-design.md`. If this plan and either spec
  disagree, **the spec wins** — stop and report the conflict.
- **Anti-hallucination (the project's whole premise):** every rule carries a real `citation` +
  `citation_support` copied from the parent spec **at implementation time, by reading the
  file** — never recalled from memory. NEVER invent a threshold, an author, or an
  anthropometric constant. A **substitution is not a unit conversion**; if a spec heuristic is
  not implementable with MediaPipe's 33 landmarks, say so explicitly.
- **Threshold provenance must be labeled in-code, in the categories push-up established:**
  (a) FIRE THRESHOLDS FROM THE SPEC, (b) SEVERITY RAMPS FROM THE SPEC (Lunge, unlike Push-up,
  states these — see Task 2), (c) RULE-LEVEL CHOICES. Exactly two category-(c) numbers exist
  in this plan and both are named where they appear. Do not add a third silently.
- **No threshold tuning from validation results** (user decision, design spec §6.2). Weak
  separation is written up as a finding. A threshold tuned to a metric is no longer the cited
  number.
- **Never bend production code to make a test pass.** If a fixture cannot produce a
  condition, fix the fixture.
- **Squat is production.** `backend/app/services/analysis.py` and `library.py` hardcode
  `movement="Squat"`. Any change that moves a squat verdict is a regression unless explicitly
  approved. The byte-for-byte squat gate in `tests/test_movement_registry.py` must keep passing.
- **MediaPipe coords:** normalized image space, `x,y ∈ [0,1]`, **y grows DOWNWARD**.
  Landmarks: 11/12 shoulders, 23/24 hips, 25/26 knees, 27/28 ankles, 29/30 heels,
  31/32 foot index.
- **All pose data under `data/` is gitignored.** Committed tests must not depend on it;
  data-backed checks must `skipUnless` the files exist.
- Commit messages end with: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

## File Structure

| File | Responsibility |
|---|---|
| `notes/lunge-view-reconnaissance.md` (create, Task 1) | The Phase 0 finding: does cam18 read `side`, and is the transpose baked in |
| `src/pose/movements/lunge.py` (create, Tasks 2–6) | Lunge raw metrics (both legs), phases, lead-side resolution, 4 cited rules, detector assembly |
| `src/pose/movements/registry.py` (modify, Task 6) | Side-effect import of `lunge` |
| `tests/test_lunge.py` (create, Tasks 2–5) | Fixture + metric, phase, lead-resolution, rule firing/non-firing, boundary, severity tests |
| `tests/test_movement_registry.py` (modify, Task 6) | Lunge resolution; squat gate unchanged |
| `src/rehab24/lunge_rule_validation.py` (create, Task 7) | Pure functions: rep-window slicing, camera routing, contingency, AUC |
| `tests/test_lunge_rule_validation.py` (create, Task 7) | Synthetic tests for those pure functions — no data dependency |
| `scripts/rehab24/validate_lunge_rules.py` (create, Task 8) | Thin CLI entry point; runs the harness over the real data |
| `notes/lunge-rule-validation.md` (create, Task 8) | The results writeup |
| Both specs + `scripts/pose/README.md` (modify, Tasks 6, 8) | Deviations, `--movement "Lunge"`, §8 status block |

---

# PHASE 0 — Reconnaissance (must complete before Phase 1)

### Task 1: Extract Ex5 pose and answer the side-gate question

**Files:**
- Create: `notes/lunge-view-reconnaissance.md`
- Writes (gitignored): `data/REHAB24-6/processed/lunge_pose_json/*.json`

**Interfaces:**
- Consumes: `scripts/pose/run_pose_extraction.py` (`--dataset unlabeled` rglobs a directory),
  `src.pose.view_estimation.estimate_view_for_pose(Path) -> ViewEstimate`,
  `src.rehab24.dataset.read_segmentation(Path) -> list[Segment]` and `camera_orientation(segment, camera) -> str`
- Produces: a committed findings note. **No production code changes in this task.**

**Why this is first:** `lunge_knee_past_toes` is hard-gated on `view_type == "side"` +
`view_confidence >= 0.20`. Across the 45 real pose JSONs already in this repo the estimator
emitted `side` exactly **once**, and that verdict was a fabricated degenerate since removed.
If cam18's genuinely sagittal Ex5 clips also read `rear_oblique`, that rule fires zero times
in production and the design spec's §3.5 claim (that Lunge dodges the `pushup_elbow_flare`
permanently-silent trap) is only half true. Discovering that after writing the rule is the
failure this task prevents.

- [ ] **Step 1: Extract pose for the 18 Ex5 clips**

```
.venv\Scripts\python.exe scripts/pose/run_pose_extraction.py --dataset unlabeled --video-dir data/REHAB24-6/Ex5 --output-dir data/REHAB24-6/processed/lunge_pose_json --no-video
```

Expected: 18 JSON files (9 video ids × cam17/cam18). `data/REHAB24-6/Ex5` contains exactly
those 18 mp4s, so the rglob needs no filtering. If the count is not 18, STOP and report.

- [ ] **Step 2: Verify the cam18 transpose is baked into the pixels**

This is a STOP condition, not a footnote: cam18 files are named
`*-Camera18-30fps-transposed.mp4`, and if that rotation is NOT already applied to the pixels
then every cam18 metric is silently rotated 90° and every sagittal result in Phase 2 is
garbage.

```
.venv\Scripts\python.exe -c "import json; from pathlib import Path; import numpy as np
for name in ('PM_021-Camera17-30fps', 'PM_021-Camera18-30fps-transposed'):
    p = Path('data/REHAB24-6/processed/lunge_pose_json') / (name + '.json')
    frames = json.loads(p.read_text(encoding='utf-8'))['frames']
    pts = [f['landmarks'] for f in frames if f.get('landmarks')]
    sh = np.array([[lm[11]['x'], lm[11]['y']] for lm in pts])
    an = np.array([[lm[27]['x'], lm[27]['y']] for lm in pts])
    d = np.nanmean(an - sh, axis=0)
    print(f'{name}: mean shoulder->ankle dx={d[0]:+.3f} dy={d[1]:+.3f}')"
```

Expected: BOTH lines show `dy` clearly dominant and positive (an upright person: ankles below
shoulders, y grows downward), with `|dy| > |dx|`. If cam18 instead shows `|dx| > |dy|`, the
transpose is NOT baked in — **STOP, report, and do not proceed to Phase 2.** Phase 1 can still
proceed (the detector does not depend on this dataset), but the validation design needs
revisiting.

- [ ] **Step 3: Run view estimation over all 18 clips and tabulate against the labels**

```
.venv\Scripts\python.exe -c "from pathlib import Path
import sys; sys.path.insert(0, '.')
from src.pose.view_estimation import estimate_view_for_pose
from src.rehab24.dataset import read_segmentation, camera_orientation
segs = [s for s in read_segmentation(Path('data/REHAB24-6/Segmentation.csv')) if s.exercise_id == '5']
labels = {}
for s in segs:
    for cam, suffix in (('cam17', '-Camera17-30fps'), ('cam18', '-Camera18-30fps-transposed')):
        labels[s.video_id + suffix] = camera_orientation(s, cam)
table = {}
for p in sorted(Path('data/REHAB24-6/processed/lunge_pose_json').glob('*.json')):
    est = estimate_view_for_pose(p)
    key = (labels.get(p.stem, '?'), est.view_type)
    table.setdefault(key, []).append(round(est.view_confidence, 2))
for (truth, got), confs in sorted(table.items()):
    print(f'label={truth:14s} -> estimated={got:14s} n={len(confs):2d} conf={confs}')"
```

- [ ] **Step 4: Record the answer**

Write `notes/lunge-view-reconnaissance.md` containing, in plain language:

1. The transpose verdict from Step 2, with the printed numbers.
2. The Step 3 table verbatim.
3. **The gate answer:** on reps whose cam18 orientation the dataset calls `side`, does
   `estimate_view_for_pose` return `side` with `view_confidence >= 0.20`? Give the count.
4. **The consequence, stated explicitly.** If the answer is "no", write: *"`lunge_knee_past_toes`
   will not fire in production on this dataset; it is validatable only in the oracle pass
   (design spec §4.4), and the design spec's §3.5 escape claim covers the two frontal rules
   only."* If "yes", write that the side gate opens and on how many reps.
5. **Do NOT change any threshold or gate in response.** This note records a fact; Task 4
   consumes it as documentation, not as a tuning signal.

Frame this note as a measurement, not a recommendation. It is allowed to report bad news.

- [ ] **Step 5: Commit**

```bash
git add notes/lunge-view-reconnaissance.md
git commit -m "measure(pose): check whether REHAB24-6 sagittal lunge clips read as a side view"
```

---

# PHASE 1 — The Lunge detector

Parent spec rules: `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`
lines 163–211. **Four rules, all four implementable.**

| fault_id | Fire threshold | Severity ramp | Provenance of both |
|---|---|---|---|
| `lunge_knee_past_toes` | ratio > 0.10 | 0.10 → 0.30 | spec, via the shared `KNEE_FORWARD_MILD`/`KNEE_FORWARD_SEVERE` constants |
| `lunge_knee_valgus` | medial offset > 0.10 | 0.10 → 0.25 | spec, both stated |
| `lunge_insufficient_depth` | min lead-knee angle > 100° | 100° → 130° | spec, both stated |
| `lunge_pelvic_drop` | pelvis tilt > 8° | 8° → 20° | spec, both stated |

**Unlike Push-up, the Lunge section states its severity ramps**, so no ramp here is a
rule-level invention. Exactly **two** rule-level numbers exist in this whole phase, both
introduced in Task 3 and both labeled there: `LEAD_SIDE_MIN_SEPARATION_DEG` (an ambiguity
guard that can only silence) and the choice of `{descent, bottom, ascent}` as the active phase
set (following the squat detector; the spec scopes only `lunge_knee_past_toes` to phases).

### Task 2: Lunge raw metrics and phase segmentation

**Files:**
- Create: `src/pose/movements/lunge.py`
- Create: `tests/test_lunge.py`

**Interfaces:**
- Consumes: `src.pose.geometry` (`landmarks_to_array`, `visible_point`, `midpoint`,
  `distance`, `angle_degrees`, `knee_forward_ratio`, `mean_visibility`, and the landmark
  index constants), `src.pose.movements.base` (`CoreFrame`, `RuleContext`, `MovementDetector`,
  `run_detector`)
- Produces:
  - `LUNGE_METRIC_KEYS: tuple[str, ...]`
  - `lunge_compute_raw(frames: Sequence[object], fps: float) -> list[dict]`
  - `lunge_assign_phases(raw: list[dict]) -> list[str]` with phases in
    `{"setup", "descent", "bottom", "ascent", "unknown"}`

**The design decision this task exists to encode:** `compute_raw` emits **both legs
symmetrically and resolves nothing**. It runs over the whole clip *before* `segment_reps`, so
no rep boundary and therefore no bottom frame exists here; a per-frame "more flexed knee"
would flicker through setup and recovery where both knees sit near extension, swapping legs
mid-clip and letting `centered_median` blend two legs into one meaningless ratio. Lead-side
resolution belongs in the rules (Task 3), which receive a per-rep slice.

**Metrics:**

| key | definition |
|---|---|
| `left_knee_angle` / `right_knee_angle` | `angle_degrees(hip, knee, ankle)` per side |
| `min_knee_angle` | more-flexed (smaller) of the two finite sides; NaN if neither finite. The rep signal |
| `left_knee_forward_ratio` / `right_knee_forward_ratio` | `geometry.knee_forward_ratio(points, knee, ankle, foot_index)` per side |
| `left_knee_medial_offset_ratio` / `right_knee_medial_offset_ratio` | signed offset of that knee from its own hip→ankle line, **positive = toward the mid-hip**, divided by hip width |
| `pelvis_tilt_signed_deg` | `degrees(atan2(right_hip_y − left_hip_y, abs(right_hip_x − left_hip_x)))`; **positive = RIGHT hip lower**, facing-independent by construction |
| `trunk_lateral_lean_deg` | `degrees(atan2(shoulder_mid_x − hip_mid_x, abs(shoulder_mid_y − hip_mid_y)))`; positive = shoulders right of hips |
| `hip_width` | `distance(points, LEFT_HIP, RIGHT_HIP)` — the normalizer, emitted for diagnostics |

- [ ] **Step 1: Write the failing metric tests**

Create `tests/test_lunge.py`. The fixture must control every asserted metric **by
construction** — never hardcode an expected value that the fixture does not actually produce.

```python
import math
import unittest

import numpy as np

from src.pose.movements.base import RuleContext, run_detector


def _lm(x: float, y: float, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _knee_at(
    hip_xy: tuple[float, float],
    ankle_xy: tuple[float, float],
    along: float,
    perpendicular: float,
) -> tuple[float, float]:
    """Place a knee at (`along`, `perpendicular`) in the leg's own frame.

    `along` is the fraction of the way from hip to ankle; `perpendicular` is the signed
    displacement off the hip-ankle line in ABSOLUTE image units, positive along the leg's
    left-hand normal.

    WHY NOT the perpendicular-bisector angle construction that tests/test_pushup.py::_elbow_xy
    uses: for THIS metric that construction is the wrong control. `_medial_offset_ratio`
    measures exactly the perpendicular displacement, so a fixture that requests a knee ANGLE
    is implicitly requesting a perpendicular offset -- and a 90-degree in-image knee bend over
    a 0.40-long leg puts the knee 0.20 off the line, which is 1.7 HIP WIDTHS, an order of
    magnitude past the spec's 0.10 fire threshold. Controlling the offset directly makes
    `left_knee_medial_offset_ratio` equal the requested value BY CONSTRUCTION, which is the
    property a fixture is supposed to have; the knee angle is then derived and asserted as
    measured rather than requested.
    """
    hx, hy = hip_xy
    ax, ay = ankle_xy
    dx, dy = ax - hx, ay - hy
    norm = math.hypot(dx, dy)
    ux, uy = (0.0, 1.0) if norm < 1e-9 else (dx / norm, dy / norm)
    px, py = -uy, ux
    return (hx + along * dx + perpendicular * px, hy + along * dy + perpendicular * py)


def lunge_frame(
    lead: str = "left",
    lead_medial: float = 0.0,
    trail_medial: float = 0.0,
    lead_anterior: float = 0.60,
    pelvis_tilt_deg: float = 0.0,
    lead_offset: float = 0.10,
    frame_index: int = 0,
) -> dict:
    """One OBLIQUE-view lunge frame, y growing DOWNWARD. Split stance along image x.

    Knobs:
      lead          -- "left" or "right"; which leg is forward. The lead ankle is displaced
                       by `lead_offset` along image x, which is what gives the lead leg a
                       genuinely different in-image knee angle from the trailing leg.
      lead_medial   -- lead-knee displacement toward the midline, IN HIP WIDTHS.
      trail_medial  -- the same for the trailing leg.
      lead_anterior -- lead-knee displacement in the ANTERIOR (step) direction, in hip widths.
                       This is what bends the lead knee in-image; the default 0.60 gives a
                       clearly-flexed lead leg against a straight trailing one.
      pelvis_tilt_deg -- rotates the hip pair about the mid-hip; POSITIVE = RIGHT hip lower,
                       matching `pelvis_tilt_signed_deg`'s convention exactly.

    ------------------------------------------------------------------------------------
    TWO PROJECTION FACTS THIS FIXTURE ENCODES. Both are properties of monocular geometry,
    not fixture conveniences, and both must be carried into `rule_knee_valgus`'s docstring.
    ------------------------------------------------------------------------------------

    (1) IN A STRICTLY FRONTAL VIEW, in-image knee flexion and medial offset are the SAME
        degree of freedom. A knee on the hip-ankle line has an interior angle of exactly
        180 degrees, so the ONLY way to bend a knee in-image is to move it off that line --
        which is precisely what `_medial_offset_ratio` measures. A lunge's real flexion is
        sagittal and projects onto the leg line frontally, contributing nothing to either.
        Consequence: `resolve_lead_side` and `rule_knee_valgus` read the same quantity in a
        pure frontal view. This fixture uses an OBLIQUE stance to break the degeneracy, and
        `front_oblique`/`rear_oblique` are the labels production actually reaches anyway.

    (2) OBLIQUELY, ANTERIOR KNEE TRAVEL CONTAMINATES THE VALGUS PROXY. `lead_anterior` and
        `lead_medial` add to the SAME perpendicular axis, because an oblique camera gives the
        anterior direction an in-image component. So `_medial_offset_ratio` cannot separate
        "knee travelled forward" from "knee caved inward" off-axis; it is clean only in a true
        frontal view, which is the view production never emits. That is a genuine limitation
        of the spec's frontal-plane proxy under this pipeline's reachable view labels, and it
        is documented rather than corrected -- correcting it needs depth this pipeline lacks.

    THE OPEN QUESTIONS THESE RAISE, which Phase 2 must answer on real data: does
    `resolve_lead_side` work on the 88 cam17 reps the dataset calls `front` (if in-image knee
    angles there are near-symmetric, the ambiguity guard fires and the frontal rules go silent
    on exactly the camera routed to them), and does `lunge_knee_valgus` fire in proportion to
    step depth rather than to correctness (the signature of contamination (2))? Task 8 Step 3
    reports the unresolved rate and Step 4 reads the valgus/depth relationship for exactly
    these reasons -- do not treat either as a harness bug.
    """
    half_hip = 0.06
    tilt = math.radians(pelvis_tilt_deg)
    left_hip = (0.50 - half_hip * math.cos(tilt), 0.50 - half_hip * math.sin(tilt))
    right_hip = (0.50 + half_hip * math.cos(tilt), 0.50 + half_hip * math.sin(tilt))
    # The lead ankle steps forward along image x; the trailing ankle stays under its hip.
    lead_shift = lead_offset if lead == "left" else -lead_offset
    left_ankle = (0.44 + (lead_shift if lead == "left" else 0.0), 0.90)
    right_ankle = (0.56 + (lead_shift if lead == "right" else 0.0), 0.90)

    hip_width = math.dist(left_hip, right_hip)
    # Medial is toward the mid-hip. `_knee_at`'s +perpendicular is the leg's left-hand normal,
    # which points toward the midline for one leg and away for the other, so the sign flips.
    left_sign = 1.0 if left_hip[0] < right_hip[0] else -1.0
    left_medial = lead_medial if lead == "left" else trail_medial
    right_medial = lead_medial if lead == "right" else trail_medial
    # Anterior travel lands on the same perpendicular axis -- see fact (2) in the docstring.
    # It is applied to the LEAD leg only; the trailing leg stays straight.
    left_anterior = lead_anterior if lead == "left" else 0.0
    right_anterior = lead_anterior if lead == "right" else 0.0

    lm = [_lm(0.50, 0.50) for _ in range(33)]
    lm[11], lm[12] = _lm(0.44, 0.25), _lm(0.56, 0.25)
    lm[23], lm[24] = _lm(*left_hip), _lm(*right_hip)
    lm[25] = _lm(*_knee_at(left_hip, left_ankle, 0.5,
                           left_sign * (left_medial + left_anterior) * hip_width))
    lm[26] = _lm(*_knee_at(right_hip, right_ankle, 0.5,
                           -left_sign * (right_medial + right_anterior) * hip_width))
    lm[27], lm[28] = _lm(*left_ankle), _lm(*right_ankle)
    lm[29], lm[30] = _lm(left_ankle[0] - 0.02, 0.92), _lm(right_ankle[0] - 0.02, 0.92)
    lm[31], lm[32] = _lm(left_ankle[0] + 0.04, 0.94), _lm(right_ankle[0] + 0.04, 0.94)
    return {"frame_index": frame_index, "landmarks": lm}


def mirrored(frame: dict) -> dict:
    """The same body FACING THE OTHER WAY: left/right landmark CONTENTS swapped, then the
    whole thing reflected about x=0.5.

    Reflecting x alone is not a facing flip -- it moves landmark 23 to the right-hand side of
    the image while leaving it the "left hip", and since `pelvis_tilt_signed_deg` reads
    `right_hip[1] - left_hip[1]` and reflection does not touch y, such a test passes trivially
    without exercising anything. Swapping the contents of every left/right pair is what a real
    turn-around does.
    """
    lm = [dict(item) for item in frame["landmarks"]]
    for left, right in ((11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (7, 8)):
        lm[left], lm[right] = lm[right], lm[left]
    for item in lm:
        item["x"] = 1.0 - item["x"]
    return {"frame_index": frame.get("frame_index", 0), "landmarks": lm}


class LungeMetricTests(unittest.TestCase):
    def test_medial_offset_equals_the_requested_displacement(self) -> None:
        # Controlled by construction: `lead_medial` is in hip widths and the metric normalizes
        # by hip width, so the two must agree to within the fixture's rounding.
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw(
            [lunge_frame(lead="left", lead_medial=0.18, lead_anterior=0.0)], 30.0
        )
        self.assertTrue(raw[0]["valid"])
        self.assertAlmostEqual(raw[0]["left_knee_medial_offset_ratio"], 0.18, delta=0.01)

    def test_medial_offset_means_toward_the_midline_on_the_right_leg_too(self) -> None:
        # The sign convention is the whole point: "medial" is toward the mid-hip for BOTH
        # legs, which is opposite image-x directions for left and right.
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw(
            [lunge_frame(lead="right", lead_medial=0.18, lead_anterior=0.0)], 30.0
        )
        self.assertAlmostEqual(raw[0]["right_knee_medial_offset_ratio"], 0.18, delta=0.01)

    def test_a_knee_tracking_outside_reads_negative(self) -> None:
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw(
            [lunge_frame(lead="left", lead_medial=-0.18, lead_anterior=0.0)], 30.0
        )
        self.assertLess(raw[0]["left_knee_medial_offset_ratio"], 0.0)

    def test_a_well_tracked_lunge_sits_below_the_fire_threshold(self) -> None:
        # THE SCALE CHECK. Without it, the rule-level boundary tests (which inject the metric
        # directly) would prove `severity_from_range` works while never establishing that
        # `_medial_offset_ratio` produces spec-scale values from an actual body. A correct
        # lunge must land well under the spec's 0.10; a caved one inside its 0.10-0.25 ramp.
        from src.pose.movements.lunge import lunge_compute_raw

        good = lunge_compute_raw(
            [lunge_frame(lead="left", lead_medial=0.02, lead_anterior=0.0)], 30.0
        )[0]
        self.assertLess(abs(good["left_knee_medial_offset_ratio"]), 0.10)

        caved = lunge_compute_raw(
            [lunge_frame(lead="left", lead_medial=0.18, lead_anterior=0.0)], 30.0
        )[0]
        self.assertGreater(caved["left_knee_medial_offset_ratio"], 0.10)
        self.assertLess(caved["left_knee_medial_offset_ratio"], 0.25)

    def test_anterior_knee_travel_contaminates_the_valgus_proxy(self) -> None:
        # PINNED ON PURPOSE -- this documents a limitation, it does not endorse it. Off-axis,
        # anterior travel and medial collapse land on the same perpendicular measurement, so a
        # deep, perfectly-tracked lunge reads as valgus. The rule ships with this stated in its
        # docstring; if someone later separates the two (it needs depth this pipeline lacks),
        # this test should fail and force a spec conversation rather than silently changing
        # what a stored `lunge_knee_valgus` severity meant.
        from src.pose.movements.lunge import lunge_compute_raw

        deep_but_clean = lunge_compute_raw(
            [lunge_frame(lead="left", lead_medial=0.0, lead_anterior=0.60)], 30.0
        )[0]
        self.assertGreater(deep_but_clean["left_knee_medial_offset_ratio"], 0.10)

    def test_the_lead_leg_reads_a_smaller_knee_angle_than_the_trailing_leg(self) -> None:
        # Derived, not requested: the split stance is what makes the lead knee measurably more
        # flexed in-image. Asserting the ORDERING (not a specific angle) is what
        # `resolve_lead_side` actually depends on.
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw([lunge_frame(lead="left")], 30.0)
        self.assertLess(raw[0]["left_knee_angle"], raw[0]["right_knee_angle"])
        self.assertAlmostEqual(raw[0]["min_knee_angle"], raw[0]["left_knee_angle"], places=6)

    def test_pelvis_tilt_is_positive_when_the_right_hip_is_lower(self) -> None:
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw([lunge_frame(pelvis_tilt_deg=12.0)], 30.0)
        self.assertAlmostEqual(raw[0]["pelvis_tilt_signed_deg"], 12.0, delta=1.0)

    def test_pelvis_tilt_sign_does_not_depend_on_facing(self) -> None:
        # A real turn-around (see `mirrored`): left/right landmark CONTENTS swap and the image
        # reflects. Which hip is physically lower is unchanged, so the metric must not flip.
        # A tilt built with the RIGHT hip lower stays positive after the subject turns around.
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw([mirrored(lunge_frame(pelvis_tilt_deg=12.0))], 30.0)
        self.assertAlmostEqual(raw[0]["pelvis_tilt_signed_deg"], 12.0, delta=1.0)

    def test_a_frame_missing_a_required_landmark_is_invalid_and_carries_no_metrics(self) -> None:
        from src.pose.movements.lunge import lunge_compute_raw

        frame = lunge_frame()
        frame["landmarks"][25] = _lm(0.44, 0.70, visibility=0.10)
        raw = lunge_compute_raw([frame], 30.0)
        self.assertFalse(raw[0]["valid"])
        self.assertNotIn("left_knee_angle", raw[0])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lunge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pose.movements.lunge'`.

- [ ] **Step 3: Implement `lunge_compute_raw`**

Create `src/pose/movements/lunge.py`. Open with a module docstring stating, in the style of
`pushup.py`'s header: that the metric layer contains **no thresholds**; that every side-specific
metric is emitted for **both legs** because `compute_raw` runs before `segment_reps` and cannot
know the lead leg; that `required` includes both hips, knees, ankles, foot indices and
shoulders, so one dropped landmark silences **every** lunge rule for that frame.

```python
LUNGE_METRIC_KEYS: tuple[str, ...] = (
    "left_knee_angle",
    "right_knee_angle",
    "min_knee_angle",
    "left_knee_forward_ratio",
    "right_knee_forward_ratio",
    "left_knee_medial_offset_ratio",
    "right_knee_medial_offset_ratio",
    "pelvis_tilt_signed_deg",
    "trunk_lateral_lean_deg",
    "hip_width",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard
# value pushup.py and overhead_press.py use; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6


def _medial_offset_ratio(
    points, hip_index: int, knee_index: int, ankle_index: int, mid_hip, hip_width: float
) -> float:
    """Signed offset of one knee from its own hip->ankle line, POSITIVE = toward the mid-hip.

    The frontal-plane knee-abduction proxy the spec asks for ("signed medial offset of the
    knee from the hip-ankle line, normalised by hip width"). No true 3-D abduction angle is
    recoverable from monocular pose, and none is claimed.

    WHY THIS IS FACING-INDEPENDENT, which is what lets the rule avoid gating on `front` /
    `front_oblique` (unreachable in production under allow_front=False): "medial" is defined
    as "toward the mid-hip", and the mid-hip is the midline whether the camera is in front of
    or behind the subject. Nothing here consults `signed_orientation`.
    """
    hip = visible_point(points, hip_index, dims=2)
    knee = visible_point(points, knee_index, dims=2)
    ankle = visible_point(points, ankle_index, dims=2)
    if hip is None or knee is None or ankle is None or mid_hip is None:
        return np.nan
    if not np.isfinite(hip_width) or hip_width <= _DEGENERATE_LENGTH:
        return np.nan

    leg = np.asarray(ankle, dtype=np.float64) - np.asarray(hip, dtype=np.float64)
    leg_length = float(np.linalg.norm(leg))
    if leg_length <= _DEGENERATE_LENGTH:
        return np.nan
    normal = np.asarray([-leg[1], leg[0]], dtype=np.float64) / leg_length

    # Orient the normal toward the midline, so a positive projection means "medial" for
    # whichever leg this is -- the left and right legs point in opposite image-x directions.
    toward_midline = np.asarray(mid_hip, dtype=np.float64) - np.asarray(hip, dtype=np.float64)
    if float(np.dot(normal, toward_midline)) < 0.0:
        normal = -normal

    offset = float(np.dot(np.asarray(knee, dtype=np.float64) - np.asarray(hip, dtype=np.float64), normal))
    return offset / float(hip_width)


def lunge_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        # Foot indices are required because `knee_forward_ratio` needs the toe-ankle vector;
        # shoulders because the trunk lean does. See the module docstring: one dropped
        # landmark silences EVERY lunge rule for this frame, not just the dependent one.
        required = (
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE,
            LEFT_ANKLE, RIGHT_ANKLE,
            LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
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

        left_knee_angle = angle_degrees(points, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        right_knee_angle = angle_degrees(points, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        finite_knees = [v for v in (left_knee_angle, right_knee_angle) if np.isfinite(v)]
        min_knee_angle = float(min(finite_knees)) if finite_knees else np.nan

        hip_width = distance(points, LEFT_HIP, RIGHT_HIP)
        mid_hip = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)

        left_hip = visible_point(points, LEFT_HIP, dims=2)
        right_hip = visible_point(points, RIGHT_HIP, dims=2)
        # atan2 over |dx|: the magnitude of the horizontal hip separation, never its sign.
        # Using signed dx would flip the whole angle by 180 degrees when the subject turns
        # around, making the metric mean "which hip is lower" only for one facing.
        if left_hip is not None and right_hip is not None:
            dx = abs(float(right_hip[0] - left_hip[0]))
            dy = float(right_hip[1] - left_hip[1])
            pelvis_tilt_signed_deg = (
                float(np.degrees(np.arctan2(dy, dx))) if dx > _DEGENERATE_LENGTH else np.nan
            )
        else:
            pelvis_tilt_signed_deg = np.nan

        if shoulder_mid is not None and mid_hip is not None:
            lean_dy = abs(float(shoulder_mid[1] - mid_hip[1]))
            lean_dx = float(shoulder_mid[0] - mid_hip[0])
            trunk_lateral_lean_deg = (
                float(np.degrees(np.arctan2(lean_dx, lean_dy))) if lean_dy > _DEGENERATE_LENGTH else np.nan
            )
        else:
            trunk_lateral_lean_deg = np.nan

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_knee_angle": left_knee_angle,
                "right_knee_angle": right_knee_angle,
                "min_knee_angle": min_knee_angle,
                "left_knee_forward_ratio": knee_forward_ratio(
                    points, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX
                ),
                "right_knee_forward_ratio": knee_forward_ratio(
                    points, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX
                ),
                "left_knee_medial_offset_ratio": _medial_offset_ratio(
                    points, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, mid_hip, hip_width
                ),
                "right_knee_medial_offset_ratio": _medial_offset_ratio(
                    points, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, mid_hip, hip_width
                ),
                "pelvis_tilt_signed_deg": pelvis_tilt_signed_deg,
                "trunk_lateral_lean_deg": trunk_lateral_lean_deg,
                "hip_width": hip_width,
            }
        )
    return raw
```

Import from `src.pose.geometry`: `landmarks_to_array`, `visible_point`, `midpoint`,
`distance`, `angle_degrees`, `knee_forward_ratio`, `mean_visibility`, `LEFT_SHOULDER`,
`RIGHT_SHOULDER`, `LEFT_HIP`, `RIGHT_HIP`, `LEFT_KNEE`, `RIGHT_KNEE`, `LEFT_ANKLE`,
`RIGHT_ANKLE`, `LEFT_FOOT_INDEX`, `RIGHT_FOOT_INDEX`.

`LOWER_BODY_LANDMARKS` is **defined locally in each movement module**, not shared — see
`src/pose/movements/pushup.py:173` and `overhead_press.py:48`. Follow that pattern and define
it locally in `lunge.py` too, with a comment noting that the name is squat-centric but that
the field it feeds (`CoreFrame.lower_body_visibility`) is genuinely lower-body for this
movement, unlike the upper-body detectors that inherited the name awkwardly.

- [ ] **Step 4: Run the metric tests → PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lunge.py -v`

- [ ] **Step 5: Write the failing phase tests**

```python
class LungePhaseTests(unittest.TestCase):
    def _descend_and_rise(self) -> list[dict]:
        # 30 frames: the lead knee bends in and back out, i.e. one clean rep on the left leg.
        # Depth is driven by `lead_anterior`, which is what bends the knee in-image.
        depths = list(np.linspace(0.0, 0.80, 15)) + list(np.linspace(0.80, 0.0, 15))
        return [
            lunge_frame(lead="left", lead_anterior=float(d), frame_index=i)
            for i, d in enumerate(depths)
        ]

    def test_phases_run_setup_descent_bottom_ascent(self) -> None:
        from src.pose.movements.lunge import lunge_assign_phases, lunge_compute_raw

        phases = lunge_assign_phases(lunge_compute_raw(self._descend_and_rise(), 30.0))
        self.assertEqual(len(phases), 30)
        self.assertEqual(phases[0], "setup")
        self.assertIn("descent", phases)
        self.assertIn("bottom", phases)
        self.assertIn("ascent", phases)
        self.assertLess(phases.index("descent"), phases.index("ascent"))

    def test_an_empty_clip_returns_no_phases(self) -> None:
        from src.pose.movements.lunge import lunge_assign_phases

        self.assertEqual(lunge_assign_phases([]), [])

    def test_a_clip_with_no_finite_signal_is_entirely_unknown(self) -> None:
        from src.pose.movements.lunge import lunge_assign_phases

        raw = [{"valid": False} for _ in range(10)]
        self.assertEqual(lunge_assign_phases(raw), ["unknown"] * 10)
```

- [ ] **Step 6: Implement `lunge_assign_phases`**

Mirror `pushup_assign_phases` exactly, substituting `min_knee_angle` for `min_elbow_angle`:
empty clip → `[]`; no finite signal → all `unknown`; invalid frame → `unknown` **before** the
setup cutoff is consulted; first 15% → `setup`; frames at or below the 30th percentile of the
knee angle → `bottom`; before the deepest index → `descent`; otherwise → `ascent`. Docstring
must say it mirrors `pushup_assign_phases`/`ohp_assign_phases` and why.

- [ ] **Step 7: Run all tests → PASS, then commit**

```bash
.venv\Scripts\python.exe -m pytest tests/test_lunge.py -v
git add src/pose/movements/lunge.py tests/test_lunge.py
git commit -m "feat(pose): lunge raw metrics for both legs, and phase segmentation"
```

---

### Task 3: Lead-side resolution and `lunge_insufficient_depth`

**Files:**
- Modify: `src/pose/movements/lunge.py`, `tests/test_lunge.py`

**Interfaces:**
- Produces:
  - `resolve_lead_side(window: list[CoreFrame]) -> str | None` — `"left"`, `"right"`, or
    `None` when unresolvable
  - `rule_insufficient_depth(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]`
  - `LUNGE_ACTIVE_PHASES: set[str]`, `LEAD_SIDE_MIN_SEPARATION_DEG: float`,
    `DEPTH_OBSERVABLE_VIEWS: set[str]`, `_OFF_VIEW_CONFIDENCE: float`

**Both of this phase's rule-level numbers appear here and must be labeled as such:**
`LEAD_SIDE_MIN_SEPARATION_DEG` and `LUNGE_ACTIVE_PHASES`.

- [ ] **Step 0: Resolve ALL FOUR `kg_query` strings against the live graph, before any rule is written**

Done once, here, so that no rule is ever committed carrying an unresolved placeholder. The OHP
detector shipped with three `kg_query` strings that resolve to **no KG node at all**, recorded
as an open item in the parent spec — this step is what stops that recurring.

First read the signature (`grep -n "def resolve_nodes" -A 10 src/knowledge/graph_retrieval.py`)
and adjust the call below to match it. Then:

```
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.')
from src.knowledge.graph_retrieval import resolve_nodes
for q in ['Knee Valgus', 'Lead Knee Extends Beyond Toe', 'Excessive Knee Flexion',
          'Anterior Trunk Tilt', 'Poor Dynamic Stability', 'Knee Anterior Displacement',
          'Reduced Hip Flexion Angles', 'Compensatory Trunk Lean']:
    print(f'{q!r:38s} ->', resolve_nodes(q, movement='Lunge'))"
```

Pick, for each of the four faults, a query that **resolves to a real `Lunge:`-scoped node**,
and define the four as module constants next to the threshold constants so a reader sees at a
glance which node each rule grounds against:

```python
# Each string was checked against data/kg/sports_kg_v3.graphml via
# graph_retrieval.resolve_nodes BEFORE being written here -- see this task's report for the
# resolution output. The OHP detector shipped three queries that resolve to nothing; these do
# not repeat that.
LUNGE_PAST_TOES_KG_QUERY = "..."      # fill from the command's output
LUNGE_VALGUS_KG_QUERY = "..."
LUNGE_DEPTH_KG_QUERY = "..."
LUNGE_PELVIC_DROP_KG_QUERY = "..."
```

Record the four choices in the task report; Tasks 4 and 5 import them without re-deriving.

If no node exists for a fault, do **NOT** invent a near-miss: set that rule's
`retrieval_mode` to the codebase's no-retrieval value (check what `build_detection` callers
use) and record the gap in the report so Task 6 Step 7 can write it into the spec.

- [ ] **Step 1: Write the failing lead-resolution tests**

```python
def _rule_frames(metrics: dict, count: int = 12, phase: str = "bottom") -> list:
    """A window of `count` identical CoreFrames carrying `metrics`.

    Constant values on purpose: `run_detector`'s median smoothing is a no-op over constants,
    so an asserted severity is EXACT rather than approximately right.
    """
    from src.pose.movements.base import CoreFrame

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


def _ctx(view_type: str = "front_oblique", *, min_frames: int = 6, view_confidence: float = 0.9):
    """min_frames=6 is what `run_detector` computes at 30 fps -- max(3, ceil(30 * 0.20)) --
    so a segment-length mutant cannot hide behind an artificially permissive 1."""
    return RuleContext(fps=30.0, view_type=view_type, view_confidence=view_confidence,
                       min_frames=min_frames)


class LeadSideResolutionTests(unittest.TestCase):
    def test_resolves_the_more_flexed_leg_at_the_windows_bottom(self) -> None:
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 85.0, "left_knee_angle": 85.0,
                               "right_knee_angle": 165.0})
        self.assertEqual(resolve_lead_side(window), "left")

    def test_resolves_the_right_leg_when_it_is_the_flexed_one(self) -> None:
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 85.0, "left_knee_angle": 165.0,
                               "right_knee_angle": 85.0})
        self.assertEqual(resolve_lead_side(window), "right")

    def test_returns_none_when_both_knees_are_within_the_ambiguity_guard(self) -> None:
        # A near-symmetric bottom is not a lunge. Guessing a side here would mis-attribute
        # every fault in the rep, so the rules go silent instead.
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 100.0, "left_knee_angle": 100.0,
                               "right_knee_angle": 102.0})
        self.assertIsNone(resolve_lead_side(window))

    def test_returns_none_when_no_frame_is_valid(self) -> None:
        from src.pose.movements.base import CoreFrame
        from src.pose.movements.lunge import resolve_lead_side

        window = [
            CoreFrame(frame_index=i, time=i / 30.0, phase="unknown", valid=False,
                      lower_body_visibility=0.0, metrics={})
            for i in range(12)
        ]
        self.assertIsNone(resolve_lead_side(window))

    def test_resolution_uses_the_bottom_frame_not_the_first(self) -> None:
        # The rep opens with the RIGHT knee incidentally more flexed, but the bottom is
        # unambiguously a LEFT-lead lunge. Resolving on frame 0 would answer "right".
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 160.0, "left_knee_angle": 168.0,
                               "right_knee_angle": 160.0}, count=6)
        window += _rule_frames({"min_knee_angle": 85.0, "left_knee_angle": 85.0,
                                "right_knee_angle": 150.0}, count=6)
        self.assertEqual(resolve_lead_side(window), "left")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lunge.py -k lead_side -v`
Expected: FAIL — `cannot import name 'resolve_lead_side'`.

- [ ] **Step 3: Implement the constants and `resolve_lead_side`**

```python
# Phases in which a lunge is under load. RULE-LEVEL CHOICE, not a spec quantity: the parent
# spec scopes only `lunge_knee_past_toes` to phases ("during descent/bottom/ascent") and
# scopes the other three to none. Applying that same set to all four follows the squat
# detector's ACTIVE_PHASES precedent (src/pose/movements/squat.py) rather than a spec
# requirement. Cost, stated: a fault that appears only during `setup` or `recovery` is missed.
LUNGE_ACTIVE_PHASES = {"descent", "bottom", "ascent"}

# Minimum left/right knee-angle difference at the bottom before a lead leg is claimed.
# RULE-LEVEL CHOICE -- the parent spec defines the lead leg ("the more flexed / more anterior
# foot") but names no separation below which the answer is unsafe. 5 degrees is chosen as the
# scale at which a landmark-noise-driven difference could flip the answer; below it the two
# legs are doing the same thing, which is not a lunge. This constant can ONLY SILENCE: an
# unresolved lead side emits no detections at all, never a guessed one. A coin-flip here would
# mis-attribute every fault in the rep to the wrong leg, which is worse than saying nothing.
LEAD_SIDE_MIN_SEPARATION_DEG = 5.0

# Views in which the parent spec rates lunge depth `high` ("high on side / front_oblique;
# medium head-on"). Defined locally rather than imported from pushup.py: the two modules
# happen to agree today but answer different spec lines and must be free to diverge.
DEPTH_OBSERVABLE_VIEWS = {"side", "front_oblique"}

# Views in which the parent spec rates the frontal-plane cues `high`. Matches the set
# `squat.rule_knees_inward` already uses for the same fault family.
ALIGNMENT_OBSERVABLE_VIEWS = {"front", "front_oblique", "rear", "rear_oblique"}

# Confidence multiplier applied when a rule fires from a view the spec does not rate `high`.
# The same 0.65 already used across squat, OHP and push-up -- not a new number.
_OFF_VIEW_CONFIDENCE = 0.65


def resolve_lead_side(window: list[CoreFrame]) -> str | None:
    """Which leg led this repetition: "left", "right", or None when unresolvable.

    SUBSTITUTION, NOT A RESTATEMENT -- record it as one. The parent spec defines the lead leg
    as "the more flexed / more anterior foot". `more anterior` is exactly the axis that
    collapses in a frontal view, which is where two of the four lunge rules live, so the
    anterior half of that definition is unusable where it is most needed. This uses the
    more-flexed half only, evaluated at the window's bottom frame.

    WHY THIS LIVES IN THE RULES AND NOT IN `lunge_compute_raw`: `run_detector` calls
    `compute_raw` over the WHOLE CLIP before `segment_reps`, so at metric time there is no rep
    boundary and therefore no bottom frame. A per-frame "whichever knee is more flexed right
    now" flickers through `setup` and `recovery`, where both knees sit near extension within
    landmark noise of each other; every lead-relative quantity would then swap legs mid-clip
    and `centered_median` would blend two legs into a number describing neither. Rules receive
    a per-rep slice (`run_detector` slices `core[rep.start:rep.end + 1]`), which is the first
    place the question is answerable.

    On a fallback path (`no_reps_detected`, `only_partial_reps`, `segmentation_disabled`) the
    rules receive the whole clip, so `window` is the whole clip and this resolves once for it.
    That degrades exactly as everything else on the fallback path does; it is stated, not
    hidden.
    """
    bottom: CoreFrame | None = None
    bottom_value = np.inf
    for frame in window:
        if not frame.valid:
            continue
        value = frame.m("min_knee_angle")
        if np.isfinite(value) and value < bottom_value:
            bottom_value, bottom = value, frame
    if bottom is None:
        return None

    left = bottom.m("left_knee_angle")
    right = bottom.m("right_knee_angle")
    if not np.isfinite(left) or not np.isfinite(right):
        return None
    if abs(left - right) < LEAD_SIDE_MIN_SEPARATION_DEG:
        return None
    return "left" if left < right else "right"
```

- [ ] **Step 4: Run the lead-side tests → PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lunge.py -k lead_side -v`

- [ ] **Step 5: Write the failing depth-rule tests**

Boundary tests sit just inside and just outside the threshold. The OHP review found 5 of 10
threshold mutants surviving because every fixture sat at an extreme — do not repeat that.

```python
class LungeDepthRuleTests(unittest.TestCase):
    def _window(self, lead_angle: float, view: str = "side", **kwargs):
        from src.pose.movements.lunge import rule_insufficient_depth

        window = _rule_frames({"min_knee_angle": lead_angle, "left_knee_angle": lead_angle,
                               "right_knee_angle": 170.0})
        return rule_insufficient_depth(window, _ctx(view, **kwargs))

    def test_fires_on_a_shallow_lunge(self) -> None:
        detections = self._window(120.0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "lunge_insufficient_depth")

    def test_silent_on_a_deep_lunge(self) -> None:
        self.assertEqual(self._window(88.0), [])

    def test_silent_just_inside_the_threshold(self) -> None:
        self.assertEqual(self._window(99.0), [])

    def test_fires_just_outside_the_threshold(self) -> None:
        self.assertEqual(len(self._window(101.0)), 1)

    def test_severity_is_exact_at_the_ramp_midpoint(self) -> None:
        # ramp 100 -> 130, so 115 is exactly half way.
        self.assertAlmostEqual(self._window(115.0)[0].severity, 0.5, places=3)

    def test_severity_saturates_at_the_ramp_end(self) -> None:
        self.assertAlmostEqual(self._window(130.0)[0].severity, 1.0, places=3)

    def test_off_view_is_downgraded_not_silenced(self) -> None:
        detections = self._window(120.0, view="rear")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].observability, "medium")
        self.assertAlmostEqual(
            detections[0].confidence, round(detections[0].severity * 0.65, 4), places=3
        )

    def test_silent_when_the_lead_side_is_unresolvable(self) -> None:
        from src.pose.movements.lunge import rule_insufficient_depth

        window = _rule_frames({"min_knee_angle": 120.0, "left_knee_angle": 120.0,
                               "right_knee_angle": 121.0})
        self.assertEqual(rule_insufficient_depth(window, _ctx()), [])

    def test_silent_outside_the_active_phases(self) -> None:
        from src.pose.movements.lunge import rule_insufficient_depth

        window = _rule_frames({"min_knee_angle": 120.0, "left_knee_angle": 120.0,
                               "right_knee_angle": 170.0}, phase="setup")
        self.assertEqual(rule_insufficient_depth(window, _ctx()), [])
```

- [ ] **Step 6: Run to verify failure, then implement `rule_insufficient_depth`**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lunge.py -k Depth -v` → FAIL.

```python
def rule_insufficient_depth(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a lunge whose lead knee never reaches roughly a right angle.

    THRESHOLD PROVENANCE -- BOTH FROM THE SPEC, unlike every push-up rule. The parent spec's
    Lunge entry states the fire threshold ("Flag when the minimum lead-knee angle across the
    rep > 100 degrees") AND the ramp ("Severity ramp 100 degrees -> 130 degrees (more extended
    = worse)"). Neither number is chosen here.

    The lead side is resolved over THIS window (see `resolve_lead_side`); an unresolved side
    emits nothing.
    """
    lead = resolve_lead_side(core)
    if lead is None:
        return []
    lead_key = f"{lead}_knee_angle"
    observable = ctx.view_type in DEPTH_OBSERVABLE_VIEWS

    mask = [
        frame.valid
        and frame.phase in LUNGE_ACTIVE_PHASES
        and np.isfinite(frame.m(lead_key))
        and frame.m(lead_key) > LUNGE_DEPTH_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        angles = [frame.m(lead_key) for frame in segment]
        max_angle = float(np.nanmax(angles))
        severity = severity_from_range(
            max_angle, LUNGE_DEPTH_MILD_DEG, LUNGE_DEPTH_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="lunge_insufficient_depth",
                fault_name="Insufficient Depth",
                kg_query=LUNGE_DEPTH_KG_QUERY,   # the string resolved in Step 0
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=angles,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "lead_side": lead,
                    "max_lead_knee_angle_deg": round(max_angle, 2),
                    "threshold": LUNGE_DEPTH_MILD_DEG,
                    "primary_label": "lead knee angle",
                    "primary_value": round(max_angle, 2),
                    "primary_threshold": LUNGE_DEPTH_MILD_DEG,
                },
                citation=...,          # COPY VERBATIM from the parent spec, lines ~197
                citation_support=...,  # COPY VERBATIM from the parent spec, lines ~198
            )
        )
    return detections
```

Define alongside the other constants:

```python
# FROM THE SPEC: "Flag when the minimum lead-knee angle across the rep > 100 degrees.
# Severity ramp 100 degrees -> 130 degrees (more extended = worse)."
LUNGE_DEPTH_MILD_DEG = 100.0
LUNGE_DEPTH_SEVERE_DEG = 130.0
```

**Before writing the `citation` / `citation_support` strings, open
`docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` and copy them from the
`lunge_insufficient_depth` entry.** Do not type them from memory — that is the exact failure
mode this project's premise forbids.

- [ ] **Step 7: Run → PASS, then commit**

```bash
.venv\Scripts\python.exe -m pytest tests/test_lunge.py -v
git add src/pose/movements/lunge.py tests/test_lunge.py
git commit -m "feat(pose): resolve the lunge lead leg per rep, and flag insufficient depth"
```

---

### Task 4: `lunge_knee_past_toes` and `lunge_knee_valgus`

**Files:**
- Modify: `src/pose/movements/lunge.py`, `tests/test_lunge.py`

**Interfaces:**
- Consumes: `resolve_lead_side`, `LUNGE_ACTIVE_PHASES`, `ALIGNMENT_OBSERVABLE_VIEWS`,
  `_OFF_VIEW_CONFIDENCE` (Task 3); `KNEE_FORWARD_MILD`, `KNEE_FORWARD_SEVERE`,
  `SIDE_VIEW_CONF_THRESHOLD` from `src.pose.pose_rule_detector`
- Produces: `rule_knee_past_toes(core, ctx) -> list[PoseRuleDetection]`,
  `rule_knee_valgus(core, ctx) -> list[PoseRuleDetection]`,
  `LUNGE_VALGUS_MILD`, `LUNGE_VALGUS_SEVERE`

**No new numbers.** `lunge_knee_past_toes` **reuses** `KNEE_FORWARD_MILD` (0.10) and
`KNEE_FORWARD_SEVERE` (0.30) from `src/pose/pose_rule_detector.py` rather than restating them:
the parent spec's Lunge and Squat entries carry identical wording ("flag when > 0.10 …
severe ≥ 0.30"), this repo already reads that as a 0.10 → 0.30 ramp, and single-sourcing stops
the two from drifting apart.

- [ ] **Step 1: Write the failing tests for both rules**

```python
class LungeKneePastToesRuleTests(unittest.TestCase):
    def _fire(self, ratio: float, view: str = "side", conf: float = 0.9):
        from src.pose.movements.lunge import rule_knee_past_toes

        window = _rule_frames({"min_knee_angle": 90.0, "left_knee_angle": 90.0,
                               "right_knee_angle": 170.0, "left_knee_forward_ratio": ratio,
                               "right_knee_forward_ratio": 0.0})
        return rule_knee_past_toes(window, _ctx(view, view_confidence=conf))

    def test_fires_when_the_lead_knee_travels_past_the_toes(self) -> None:
        detections = self._fire(0.20)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "lunge_knee_past_toes")
        self.assertEqual(detections[0].evidence["lead_side"], "left")

    def test_silent_when_the_knee_stays_behind_the_toes(self) -> None:
        self.assertEqual(self._fire(0.02), [])

    def test_silent_just_inside_the_threshold(self) -> None:
        self.assertEqual(self._fire(0.09), [])

    def test_fires_just_outside_the_threshold(self) -> None:
        self.assertEqual(len(self._fire(0.11)), 1)

    def test_severity_is_exact_at_the_ramp_midpoint(self) -> None:
        self.assertAlmostEqual(self._fire(0.20)[0].severity, 0.5, places=3)

    def test_hard_gated_to_silence_off_the_sagittal_view(self) -> None:
        # Not a downgrade: sagittal knee travel is not resolvable head-on, and the squat's
        # rule_knees_forward sets the precedent of silence rather than a low-confidence claim.
        self.assertEqual(self._fire(0.20, view="rear_oblique"), [])

    def test_hard_gated_to_silence_on_a_weakly_classified_side_view(self) -> None:
        self.assertEqual(self._fire(0.20, view="side", conf=0.10), [])

    def test_reads_the_lead_legs_ratio_not_the_trailing_legs(self) -> None:
        # Trailing leg way past its toes, lead leg fine -> nothing fires. The whole point of
        # per-window lead resolution.
        from src.pose.movements.lunge import rule_knee_past_toes

        window = _rule_frames({"min_knee_angle": 90.0, "left_knee_angle": 90.0,
                               "right_knee_angle": 170.0, "left_knee_forward_ratio": 0.01,
                               "right_knee_forward_ratio": 0.90})
        self.assertEqual(rule_knee_past_toes(window, _ctx("side")), [])


class LungeKneeValgusRuleTests(unittest.TestCase):
    def _fire(self, offset: float, view: str = "front_oblique"):
        from src.pose.movements.lunge import rule_knee_valgus

        window = _rule_frames({"min_knee_angle": 90.0, "left_knee_angle": 90.0,
                               "right_knee_angle": 170.0,
                               "left_knee_medial_offset_ratio": offset,
                               "right_knee_medial_offset_ratio": 0.0})
        return rule_knee_valgus(window, _ctx(view))

    def test_fires_when_the_lead_knee_caves_medially(self) -> None:
        detections = self._fire(0.18)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "lunge_knee_valgus")

    def test_silent_when_the_knee_tracks_over_the_foot(self) -> None:
        self.assertEqual(self._fire(0.01), [])

    def test_silent_when_the_knee_tracks_laterally(self) -> None:
        # A NEGATIVE offset is the knee bowing outward -- the opposite fault, not this one.
        self.assertEqual(self._fire(-0.30), [])

    def test_silent_just_inside_the_threshold(self) -> None:
        self.assertEqual(self._fire(0.09), [])

    def test_fires_just_outside_the_threshold(self) -> None:
        self.assertEqual(len(self._fire(0.11)), 1)

    def test_severity_is_exact_at_the_ramp_midpoint(self) -> None:
        # ramp 0.10 -> 0.25, midpoint 0.175
        self.assertAlmostEqual(self._fire(0.175)[0].severity, 0.5, places=2)

    def test_off_view_is_downgraded_not_silenced(self) -> None:
        detections = self._fire(0.18, view="side")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].observability, "medium")

    def test_fires_from_a_rear_view_at_full_observability(self) -> None:
        # `front` is unreachable in production (allow_front=False), so a rule that only rated
        # `front` highly would be permanently downgraded. Medial-vs-midline reads the same
        # from behind, so `rear` earns the same rating -- matching squat.rule_knees_inward.
        self.assertEqual(self._fire(0.18, view="rear")[0].observability, "high")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lunge.py -k "PastToes or Valgus" -v` → FAIL.

- [ ] **Step 3: Implement `rule_knee_past_toes`**

```python
def rule_knee_past_toes(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the lead knee translating well in front of the toes.

    THRESHOLD PROVENANCE: the parent spec's Lunge entry says "Flag when > 0.10 during
    descent/bottom/ascent; severe >= 0.30" -- word-for-word the Squat entry's wording, which
    this repo already reads as a 0.10 -> 0.30 ramp via KNEE_FORWARD_MILD / KNEE_FORWARD_SEVERE
    in src/pose/pose_rule_detector.py. Those constants are IMPORTED here rather than restated,
    so the two movements cannot drift apart. No new number is introduced.

    HARD VIEW GATE, not a downgrade. The spec rates this `high` on `side` and `low` head-on
    ("sagittal knee travel not resolvable"). `squat.rule_knees_forward` sets the precedent:
    outside a confidently-classified `side` view the rule emits NOTHING rather than a
    low-confidence claim, because the projection that produces the number is the thing that
    has failed. SIDE_VIEW_CONF_THRESHOLD is the same 0.20 floor squat already applies.
    """
    lead = resolve_lead_side(core)
    if lead is None:
        return []
    observable_side = (
        ctx.view_type == "side" and ctx.view_confidence >= SIDE_VIEW_CONF_THRESHOLD
    )
    lead_key = f"{lead}_knee_forward_ratio"

    mask = [
        frame.valid
        and frame.phase in LUNGE_ACTIVE_PHASES
        and observable_side
        and np.isfinite(frame.m(lead_key))
        and frame.m(lead_key) > KNEE_FORWARD_MILD
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        ratios = [frame.m(lead_key) for frame in segment]
        max_ratio = float(np.nanmax(ratios))
        severity = severity_from_range(
            max_ratio, KNEE_FORWARD_MILD, KNEE_FORWARD_SEVERE, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="lunge_knee_past_toes",
                fault_name="Lead Knee Past Toes / Anterior Knee Translation",
                kg_query=LUNGE_PAST_TOES_KG_QUERY,   # resolved in Task 3 Step 0
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=ratios,
                severity=severity,
                confidence=severity,
                observability="high",
                evidence={
                    "lead_side": lead,
                    "max_knee_forward_ratio": round(max_ratio, 4),
                    "threshold": KNEE_FORWARD_MILD,
                    "primary_label": "lead knee past toes",
                    "primary_value": round(max_ratio, 4),
                    "primary_threshold": KNEE_FORWARD_MILD,
                },
                citation=...,          # COPY VERBATIM from the parent spec (Zellmer, PMC6523035)
                citation_support=...,  # COPY VERBATIM from the parent spec
            )
        )
    return detections
```

- [ ] **Step 4: Implement `rule_knee_valgus`**

```python
# FROM THE SPEC: "Flag when medial offset > ~0.10 * hip_width toward the midline;
# ramp 0.10 -> 0.25."
LUNGE_VALGUS_MILD = 0.10
LUNGE_VALGUS_SEVERE = 0.25


def rule_knee_valgus(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the lead knee caving medially relative to its own hip-ankle line.

    THRESHOLD PROVENANCE: both numbers FROM THE SPEC (fire > 0.10 of hip width, ramp
    0.10 -> 0.25). This is a frontal-plane knee-abduction PROXY; monocular pose yields no true
    3-D abduction angle and none is claimed.

    OBSERVABILITY DOWNGRADE, NOT A GATE -- and deliberately not gated on `front`. The
    production path calls estimate_view_for_pose(allow_front=False), so `front` and
    `front_oblique` are never emitted downstream; a rule gated positively on them would be
    PERMANENTLY SILENT, which is what happened to `pushup_elbow_flare`. This rule does not need
    them: `_medial_offset_ratio` defines medial as "toward the mid-hip", and the mid-hip is
    the midline from in front of the subject or behind. So `rear`/`rear_oblique` -- the labels
    production actually reaches -- earn the same `high` rating, matching
    `squat.rule_knees_inward`, which resolves the same fault family the same way.

    KNOWN CONTAMINATION, NOT CORRECTED HERE -- carry this over verbatim from the projection
    facts in tests/test_lunge.py::lunge_frame, and do not let the `high` rating above be read
    as a claim of cleanliness. A knee's perpendicular displacement from its hip-ankle line is
    the sum of its MEDIAL travel and its ANTERIOR travel projected into the image. In a true
    frontal view the anterior component projects onto the leg line and vanishes, leaving the
    proxy clean -- but `front` is exactly the label production can never emit. In the oblique
    views it does reach, a deep, perfectly-tracked lunge produces a positive reading with no
    valgus present (pinned by test_anterior_knee_travel_contaminates_the_valgus_proxy).
    Separating the two needs a depth estimate this pipeline does not have, so the limitation is
    documented rather than corrected, and Phase 2 checks whether firing tracks step depth
    rather than correctness.
    """
    lead = resolve_lead_side(core)
    if lead is None:
        return []
    observable = ctx.view_type in ALIGNMENT_OBSERVABLE_VIEWS
    lead_key = f"{lead}_knee_medial_offset_ratio"

    mask = [
        frame.valid
        and frame.phase in LUNGE_ACTIVE_PHASES
        and np.isfinite(frame.m(lead_key))
        and frame.m(lead_key) > LUNGE_VALGUS_MILD
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        offsets = [frame.m(lead_key) for frame in segment]
        max_offset = float(np.nanmax(offsets))
        severity = severity_from_range(
            max_offset, LUNGE_VALGUS_MILD, LUNGE_VALGUS_SEVERE, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="lunge_knee_valgus",
                fault_name="Lead Knee Valgus / Medial Collapse",
                kg_query=LUNGE_VALGUS_KG_QUERY,   # resolved in Task 3 Step 0
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=offsets,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "lead_side": lead,
                    "max_medial_offset_ratio": round(max_offset, 4),
                    "threshold": LUNGE_VALGUS_MILD,
                    "primary_label": "lead knee medial offset",
                    "primary_value": round(max_offset, 4),
                    "primary_threshold": LUNGE_VALGUS_MILD,
                },
                citation=...,          # COPY VERBATIM from the parent spec (Ford, PMC4556293)
                citation_support=...,  # COPY VERBATIM from the parent spec
            )
        )
    return detections
```

- [ ] **Step 5: Run → PASS, then commit**

```bash
.venv\Scripts\python.exe -m pytest tests/test_lunge.py -v
git add src/pose/movements/lunge.py tests/test_lunge.py
git commit -m "feat(pose): lunge knee-past-toes and lead-knee valgus cited rules"
```

---

### Task 5: `lunge_pelvic_drop`

**Files:**
- Modify: `src/pose/movements/lunge.py`, `tests/test_lunge.py`

**Interfaces:**
- Produces: `rule_pelvic_drop(core, ctx) -> list[PoseRuleDetection]`,
  `LUNGE_PELVIC_TILT_MILD_DEG`, `LUNGE_PELVIC_TILT_SEVERE_DEG`


- [ ] **Step 1: Write the failing pelvic-drop tests**

```python
class LungePelvicDropRuleTests(unittest.TestCase):
    def _fire(self, tilt: float, lead: str = "left", view: str = "front_oblique"):
        from src.pose.movements.lunge import rule_pelvic_drop

        angles = ({"left_knee_angle": 90.0, "right_knee_angle": 170.0} if lead == "left"
                  else {"left_knee_angle": 170.0, "right_knee_angle": 90.0})
        window = _rule_frames({"min_knee_angle": 90.0, "pelvis_tilt_signed_deg": tilt, **angles})
        return rule_pelvic_drop(window, _ctx(view))

    def test_fires_when_the_contralateral_hip_drops_on_a_left_lead(self) -> None:
        # Left lead -> contralateral is the RIGHT hip -> positive tilt is the fault.
        detections = self._fire(14.0, lead="left")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "lunge_pelvic_drop")

    def test_fires_when_the_contralateral_hip_drops_on_a_right_lead(self) -> None:
        # Right lead -> contralateral is the LEFT hip -> NEGATIVE tilt is the fault.
        self.assertEqual(len(self._fire(-14.0, lead="right")), 1)

    def test_silent_when_the_LEAD_side_hip_drops(self) -> None:
        # This is the sign error the rule exists to avoid: an ipsilateral drop is not
        # Trendelenburg, and reporting it would invert the coaching cue.
        self.assertEqual(self._fire(-14.0, lead="left"), [])

    def test_silent_on_a_level_pelvis(self) -> None:
        self.assertEqual(self._fire(1.0), [])

    def test_silent_just_inside_the_threshold(self) -> None:
        self.assertEqual(self._fire(7.0), [])

    def test_fires_just_outside_the_threshold(self) -> None:
        self.assertEqual(len(self._fire(9.0)), 1)

    def test_severity_is_exact_at_the_ramp_midpoint(self) -> None:
        # ramp 8 -> 20, midpoint 14
        self.assertAlmostEqual(self._fire(14.0)[0].severity, 0.5, places=3)

    def test_silent_from_a_pure_side_view(self) -> None:
        # The parent spec: "not observable from a pure side view". A frontal-plane tilt has no
        # meaning in the sagittal projection, so this is silence, not a downgrade.
        self.assertEqual(self._fire(14.0, view="side"), [])
```

- [ ] **Step 2: Run to verify failure, then implement**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lunge.py -k PelvicDrop -v` → FAIL.

```python
# FROM THE SPEC: "Flag when pelvis_tilt_deg > 8 degrees (contralateral hip lower) sustained
# through bottom/ascent; ramp 8 -> 20."
LUNGE_PELVIC_TILT_MILD_DEG = 8.0
LUNGE_PELVIC_TILT_SEVERE_DEG = 20.0


def rule_pelvic_drop(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the NON-lead-side pelvis dropping -- the Trendelenburg signature of hip-abductor
    insufficiency on the lead leg.

    THRESHOLD PROVENANCE: both numbers FROM THE SPEC (fire > 8 degrees, ramp 8 -> 20).

    SIGN, and why it takes two facts to get right. `pelvis_tilt_signed_deg` is positive when
    the RIGHT hip is lower, in a fixed convention that does not depend on which way the
    subject faces (it is built on |dx|, never signed dx). "Contralateral" then depends on the
    lead leg: a LEFT-lead lunge drops the RIGHT hip (positive), a RIGHT-lead lunge drops the
    LEFT hip (negative). Reading the magnitude alone would report an IPSILATERAL drop -- a
    different postural fault -- as Trendelenburg and invert the coaching cue.

    SILENT FROM `side`: the spec rates this "not observable from a pure side view". A
    frontal-plane tilt has no meaning in the sagittal projection, so this is silence rather
    than a discounted claim, following `rule_knee_past_toes`'s reasoning in the mirror image.

    KNOWN MEASUREMENT BIAS, NOT CORRECTED HERE. In a frontal view of a SPLIT STANCE the
    L_hip -> R_hip vector is rotated in the transverse plane, so its image projection shortens
    and atan2(dy, |dx|) INFLATES the apparent tilt -- the deeper the lunge, the worse. The
    expected failure mode is therefore FALSE POSITIVES on deep, correctly-performed reps, not
    silence. Correcting it would require a depth estimate this pipeline does not have, so it
    is documented rather than papered over; Phase 2 reads specificity on correct reps first
    for exactly this reason.
    """
    lead = resolve_lead_side(core)
    if lead is None:
        return []
    if ctx.view_type == "side":
        return []
    observable = ctx.view_type in ALIGNMENT_OBSERVABLE_VIEWS
    # Left lead -> the contralateral (right) hip dropping is a POSITIVE tilt; right lead -> negative.
    sign = 1.0 if lead == "left" else -1.0

    mask = [
        frame.valid
        and frame.phase in LUNGE_ACTIVE_PHASES
        and np.isfinite(frame.m("pelvis_tilt_signed_deg"))
        and sign * frame.m("pelvis_tilt_signed_deg") > LUNGE_PELVIC_TILT_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        drops = [sign * frame.m("pelvis_tilt_signed_deg") for frame in segment]
        max_drop = float(np.nanmax(drops))
        severity = severity_from_range(
            max_drop, LUNGE_PELVIC_TILT_MILD_DEG, LUNGE_PELVIC_TILT_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="lunge_pelvic_drop",
                fault_name="Pelvic Drop / Contralateral Trunk Lean (Trendelenburg)",
                kg_query=LUNGE_PELVIC_DROP_KG_QUERY,   # resolved in Task 3 Step 0
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=drops,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "lead_side": lead,
                    "max_contralateral_drop_deg": round(max_drop, 2),
                    "threshold": LUNGE_PELVIC_TILT_MILD_DEG,
                    "primary_label": "contralateral pelvic drop",
                    "primary_value": round(max_drop, 2),
                    "primary_threshold": LUNGE_PELVIC_TILT_MILD_DEG,
                },
                citation=...,          # COPY VERBATIM from the parent spec (Ford PMC4556293)
                citation_support=...,  # COPY VERBATIM from the parent spec
            )
        )
    return detections
```

- [ ] **Step 3: Run → PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lunge.py -k PelvicDrop -v`

- [ ] **Step 4: Commit**

```bash
git add src/pose/movements/lunge.py tests/test_lunge.py
git commit -m "feat(pose): lunge contralateral pelvic drop, with the sign the fault actually needs"
```

---

### Task 6: Assemble, register, guard, document

**Files:**
- Modify: `src/pose/movements/lunge.py` (detector assembly + the `kg_query` strings left as
  `...` in Tasks 3–5), `src/pose/movements/registry.py`, `tests/test_movement_registry.py`,
  `scripts/pose/README.md`, both specs

**Interfaces:**
- Produces: `LUNGE_DETECTOR: MovementDetector`, registered under `"Lunge"`

- [ ] **Step 1: Audit the citations and KG queries already in place**

The four `kg_query` constants were resolved against the live graph in Task 3 Step 0, so
nothing is unresolved by now. This step is the audit, not the derivation:

1. Re-run Task 3 Step 0's `resolve_nodes` command and confirm all four constants still
   resolve to real `Lunge:`-scoped nodes.
2. Open `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`, read the
   four Lunge entries, and **diff each `citation` / `citation_support` string in the code
   against the spec text**. They must match the source, not a paraphrase of it. This is the
   project's anti-hallucination premise and it is worth the five minutes.
3. Record any fault left without a resolving KG node so Step 7 writes it into the spec, the
   way the parent spec records the OHP gap.

- [ ] **Step 2: Write the alternating-lead multi-rep regression test**

This is the single most important test in the plan: it covers the defect the Phase 2 harness
structurally cannot see, because that harness feeds one rep per clip. It needs the assembled
detector, so it lands here rather than with the rules. Write it **before** Step 3 and watch it
fail on the missing `LUNGE_DETECTOR`, then go green once Step 3 assembles it.

```python
class LungeAlternatingLeadTests(unittest.TestCase):
    """The regression guard on lead-side resolution living in the RULES, not in compute_raw.

    The Phase 2 validation harness feeds ONE REP PER CLIP, so a clip-level lead side is
    per-rep by construction there and this defect would be invisible to it. Lunges are
    normally performed alternating legs, so production sees exactly this shape.
    """

    def _alternating_clip(self) -> list[dict]:
        # Three reps: lead left, lead right, lead left. Depth is driven by `lead_anterior`,
        # which is what bends the lead knee in-image.
        frames: list[dict] = []
        index = 0
        for lead in ("left", "right", "left"):
            depths = list(np.linspace(0.0, 0.80, 12)) + list(np.linspace(0.80, 0.0, 12))
            for depth in depths:
                frames.append(
                    lunge_frame(lead=lead, lead_anterior=float(depth), frame_index=index)
                )
                index += 1
        return frames

    def test_each_rep_attributes_its_fault_to_the_leg_that_actually_led_it(self) -> None:
        from src.pose.movements.lunge import LUNGE_DETECTOR

        result = run_detector(LUNGE_DETECTOR, self._alternating_clip(), 30.0, "front_oblique", 0.9)
        self.assertGreaterEqual(len(result.reps), 2, "segmentation did not find the reps")
        # Whatever fires, no detection may name a lead side whose knee was the EXTENDED one:
        # that is the signature of a lead side resolved over the wrong window.
        for detection in result.detections:
            lead = detection.evidence.get("lead_side")
            if lead is None:
                continue
            peak = next(f for f in result.core if f.frame_index == detection.peak_frame)
            self.assertLess(
                peak.m(f"{lead}_knee_angle"),
                peak.m("left_knee_angle" if lead == "right" else "right_knee_angle"),
                f"detection {detection.fault_id} named {lead} as the lead leg, but that leg "
                f"was the more EXTENDED one at its peak frame",
            )
```

If segmentation does not find the reps, fix the FIXTURE (deepen the excursion or lengthen the
reps), never the detector — a rep-segmentation change would move squat behavior.

- [ ] **Step 3: Assemble and register the detector**

```python
LUNGE_DETECTOR = MovementDetector(
    "Lunge",
    LUNGE_METRIC_KEYS,
    lunge_compute_raw,
    lunge_assign_phases,
    (rule_knee_past_toes, rule_knee_valgus, rule_insufficient_depth, rule_pelvic_drop),
    # `validated` is left at its default False: these thresholds have not been checked against
    # labeled data at the time this detector is registered, so it surfaces as Beta. Phase 2
    # measures them; flipping this flag is a SEPARATE, evidence-backed decision, not part of
    # shipping the rules.
    rep_signal="min_knee_angle",
    rep_polarity="min",
    rep_start="extended",
)

registry.register(LUNGE_DETECTOR)
```

Add the side-effect import to `src/pose/movements/registry.py`, next to the existing three:

```python
from src.pose.movements import lunge  # noqa: E402,F401
```

- [ ] **Step 4: Run the Step 2 alternating-lead test → it must now PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lunge.py -k AlternatingLead -v`

- [ ] **Step 5: Add registry tests**

```python
    def test_lunge_detector_resolves_case_insensitively(self) -> None:
        from src.pose.movements.registry import get_detector

        self.assertEqual(get_detector("Lunge").name, "Lunge")
        self.assertEqual(get_detector("lunge").name, "Lunge")

    def test_lunge_is_not_marked_validated(self) -> None:
        # Thresholds are spec-derived; Phase 2 measures them. Beta until evidence says otherwise.
        from src.pose.movements.registry import get_detector

        self.assertFalse(get_detector("Lunge").validated)
```

Follow the file's existing style for these — read `tests/test_movement_registry.py` first and
match how the Push-up equivalents are written.

- [ ] **Step 6: Run the FULL verification set**

```
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```

Both must pass. The squat byte-for-byte gate in `tests/test_movement_registry.py` is the one
that matters most — Squat is production. Known pre-existing flake:
`tests/test_analyze_endpoint.py::test_concurrent_analyses_are_bounded` is a live-Supabase call,
not a flake in the usual sense; if ONLY that fails, note it and move on.

- [ ] **Step 7: Document**

1. `scripts/pose/README.md` — add `Lunge` to the `--movement` values.
2. `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` §8 — add a
   **Status (2026-07-30) — Lunge detector registered** block recording: 4 of 4 rules live;
   the lead-leg substitution and **why it lives in the rules rather than in `compute_raw`**;
   the two rule-level numbers (`LEAD_SIDE_MIN_SEPARATION_DEG`, the active-phase scope) and
   that every threshold and ramp is otherwise the spec's; the `KNEE_FORWARD_MILD`/`SEVERE`
   reuse; the `rule_pelvic_drop` split-stance foreshortening bias; the Task 1 view-gate
   finding and what it means for `lunge_knee_past_toes` in production; any unresolved
   `kg_query` from Step 1; and that **thresholds are spec-derived and UNVALIDATED at this
   point** — Phase 2 is what changes that.
3. Mirror the same block into the `.zh-TW.md` spec.
4. Confirm and state explicitly: `backend/app/config.py`'s `DEFAULT_ANALYSIS_MOVEMENT` is
   still `"Squat"` and `frontend/src/lib/movements.ts` still lists
   `ANALYZABLE_MOVEMENTS = ["Squat"]`. Lunge is CLI-only.

- [ ] **Step 8: Commit**

```bash
git add src/pose/movements/lunge.py src/pose/movements/registry.py tests/ scripts/pose/README.md docs/superpowers/specs/
git commit -m "feat(pose): register the lunge detector and expose it via --movement"
```

---

# PHASE 2 — Validation against REHAB24-6 Ex5

### Task 7: The harness's pure functions

**Files:**
- Create: `src/rehab24/lunge_rule_validation.py`
- Create: `tests/test_lunge_rule_validation.py`

**Interfaces:**
- Consumes: `src.rehab24.dataset` (`read_segmentation`, `Segment`, `camera_orientation` —
  which **already implements** the cam17→cam18 mapping, so do not reimplement it)
- Produces:
  - `RULE_CAMERAS: dict[str, str]` — fault_id → `"cam17"` / `"cam18"`
  - `ORACLE_VIEWS: dict[str, str]` and `ORACLE_VIEW_CONFIDENCE: float` — the dataset-orientation
    → view-label mapping the oracle pass feeds the rules, pinned in Task 8 Step 1
  - `slice_rep(frames: list[dict], first_frame: int, last_frame: int) -> list[dict]`
  - `contingency(fired: Sequence[bool], correct: Sequence[bool]) -> dict[str, int]` with keys
    `tp, fp, tn, fn` where "positive" means **the rep is incorrect**
  - `rank_auc(scores: Sequence[float], positive: Sequence[bool]) -> float` — NaN when either
    class is empty
  - `per_subject(records, key_fn, value_fn) -> dict[str, float]`

**Everything in this module is a pure function with no filesystem access**, so it is fully
CI-testable while `data/` stays gitignored.

- [ ] **Step 1: Write the failing tests**

```python
import math
import unittest


class SliceRepTests(unittest.TestCase):
    def test_slices_inclusive_of_both_bounds(self) -> None:
        from src.rehab24.lunge_rule_validation import slice_rep

        frames = [{"frame_index": i} for i in range(100)]
        window = slice_rep(frames, 10, 19)
        self.assertEqual(len(window), 10)
        self.assertEqual(window[0]["frame_index"], 10)
        self.assertEqual(window[-1]["frame_index"], 19)

    def test_clamps_a_last_frame_beyond_the_clip(self) -> None:
        # Labels come from the mocap timeline; a video can be a few frames shorter.
        from src.rehab24.lunge_rule_validation import slice_rep

        frames = [{"frame_index": i} for i in range(20)]
        self.assertEqual(len(slice_rep(frames, 15, 40)), 5)

    def test_returns_empty_when_the_window_starts_past_the_clip(self) -> None:
        from src.rehab24.lunge_rule_validation import slice_rep

        self.assertEqual(slice_rep([{"frame_index": i} for i in range(5)], 90, 99), [])


class ContingencyTests(unittest.TestCase):
    def test_counts_positives_as_incorrect_reps(self) -> None:
        from src.rehab24.lunge_rule_validation import contingency

        # rep 1: incorrect + fired  -> tp
        # rep 2: correct   + fired  -> fp
        # rep 3: correct   + silent -> tn
        # rep 4: incorrect + silent -> fn
        table = contingency(fired=[True, True, False, False],
                            correct=[False, True, True, False])
        self.assertEqual(table, {"tp": 1, "fp": 1, "tn": 1, "fn": 1})

    def test_rejects_mismatched_lengths(self) -> None:
        from src.rehab24.lunge_rule_validation import contingency

        with self.assertRaises(ValueError):
            contingency(fired=[True], correct=[True, False])


class RankAucTests(unittest.TestCase):
    def test_perfect_separation_scores_one(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(
            rank_auc([1.0, 2.0, 8.0, 9.0], [False, False, True, True]), 1.0, places=6
        )

    def test_inverted_separation_scores_zero(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(
            rank_auc([8.0, 9.0, 1.0, 2.0], [False, False, True, True]), 0.0, places=6
        )

    def test_ties_score_one_half(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(rank_auc([5.0, 5.0], [True, False]), 0.5, places=6)

    def test_is_nan_when_one_class_is_empty(self) -> None:
        # A rule the dataset never exercises must report NaN, not a misleading 0.5.
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertTrue(math.isnan(rank_auc([1.0, 2.0], [True, True])))

    def test_ignores_non_finite_scores(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(
            rank_auc([float("nan"), 1.0, 9.0], [True, False, True]), 1.0, places=6
        )


class CameraRoutingTests(unittest.TestCase):
    def test_sagittal_rules_read_cam18_and_frontal_rules_read_cam17(self) -> None:
        from src.rehab24.lunge_rule_validation import RULE_CAMERAS

        self.assertEqual(RULE_CAMERAS["lunge_knee_past_toes"], "cam18")
        self.assertEqual(RULE_CAMERAS["lunge_insufficient_depth"], "cam18")
        self.assertEqual(RULE_CAMERAS["lunge_knee_valgus"], "cam17")
        self.assertEqual(RULE_CAMERAS["lunge_pelvic_drop"], "cam17")

    def test_every_registered_lunge_rule_has_a_camera(self) -> None:
        # A rule added later without a routing entry would be silently dropped from the report.
        from src.pose.movements.lunge import LUNGE_DETECTOR
        from src.rehab24.lunge_rule_validation import RULE_CAMERAS

        emitted = {
            "lunge_knee_past_toes", "lunge_knee_valgus",
            "lunge_insufficient_depth", "lunge_pelvic_drop",
        }
        self.assertEqual(len(LUNGE_DETECTOR.rules), len(emitted))
        self.assertEqual(set(RULE_CAMERAS), emitted)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lunge_rule_validation.py -v` → FAIL.

- [ ] **Step 3: Implement the module**

```python
"""Pure helpers for replaying REHAB24-6 Ex5's labeled repetitions through the production
lunge rules.

WHAT THIS CAN AND CANNOT MEASURE -- read before quoting any number it produces. REHAB24-6
labels each repetition `correct` or `incorrect` and NEVER states which fault occurred, so a
rule firing on an incorrect rep is not evidence it found that rep's actual error. Everything
here therefore measures whether a rule's signal CARRIES INFORMATION ABOUT REP CORRECTNESS --
not per-fault precision.

Nothing in this module touches the filesystem, so it is fully testable in CI while the pose
corpus under `data/` stays gitignored.
"""

# Which camera affords each rule's required view. Segmentation.txt documents that a rep filmed
# `front` in cam17 is `side` in cam18 (the cameras are orthogonal and simultaneous), so the
# same repetition supplies both a frontal and a sagittal view. `src.rehab24.dataset
# .camera_orientation` already implements that mapping -- use it, do not restate it.
RULE_CAMERAS: dict[str, str] = {
    "lunge_knee_past_toes": "cam18",      # hard-gated to `side`
    "lunge_insufficient_depth": "cam18",  # spec rates the knee angle `high` on side
    "lunge_knee_valgus": "cam17",         # frontal-plane cue
    "lunge_pelvic_drop": "cam17",         # frontal-plane cue
}


def slice_rep(frames: list[dict], first_frame: int, last_frame: int) -> list[dict]:
    """Frames `[first_frame, last_frame]` INCLUSIVE, clamped to what the clip actually holds.

    Clamping rather than raising: the labels are indexed on the mocap timeline and a video can
    run a few frames short, which is a truncated rep, not a corrupt one. A window that starts
    past the end of the clip yields [] and is reported as a skipped rep.
    """
    if first_frame < 0 or last_frame < first_frame:
        raise ValueError(f"invalid rep window [{first_frame}, {last_frame}]")
    return frames[first_frame : last_frame + 1]


def contingency(fired, correct) -> dict[str, int]:
    """2x2 table with POSITIVE = the repetition is INCORRECT.

    `correct` is the dataset's `correctness` column as a bool (True == performed correctly), so
    a positive is `not correct`. Stated explicitly because getting this backwards silently
    inverts sensitivity and specificity in the writeup.
    """
    if len(fired) != len(correct):
        raise ValueError(f"length mismatch: {len(fired)} fired vs {len(correct)} labels")
    table = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for did_fire, is_correct in zip(fired, correct):
        positive = not is_correct
        if did_fire and positive:
            table["tp"] += 1
        elif did_fire and not positive:
            table["fp"] += 1
        elif not did_fire and positive:
            table["fn"] += 1
        else:
            table["tn"] += 1
    return table


def rank_auc(scores, positive) -> float:
    """Threshold-free AUC of `scores` against `positive`, by the rank-sum identity, ties = 0.5.

    Threshold-free on purpose: it says whether the underlying metric ORDERS incorrect reps
    above correct ones at all, independently of where the spec's cut happens to sit. That is
    what distinguishes "this cue carries no signal" from "this cue carries signal but the cited
    threshold sits in the wrong part of its distribution" -- and only the first of those is a
    reason to doubt the rule.

    NaN when either class is empty: a rule the dataset never exercises must not report 0.5,
    which reads as "no better than chance" rather than "not measured".
    """
    pairs = [(s, bool(p)) for s, p in zip(scores, positive) if math.isfinite(s)]
    pos = [s for s, p in pairs if p]
    neg = [s for s, p in pairs if not p]
    if not pos or not neg:
        return math.nan
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))
```

Also implement `per_subject(records, key_fn, value_fn) -> dict[str, float]`, grouping records
by `key_fn` (the person id) and applying `value_fn` per group. Keep it a plain grouping helper;
its docstring should say why per-subject reporting is mandatory: 174 reps from 8 people are not
independent, and a pooled statistic lets one subject's separation masquerade as a population
result.

- [ ] **Step 4: Run → PASS, then commit**

```bash
.venv\Scripts\python.exe -m pytest tests/test_lunge_rule_validation.py -v
git add src/rehab24/lunge_rule_validation.py tests/test_lunge_rule_validation.py
git commit -m "feat(rehab24): pure helpers for replaying labeled lunge reps through the rules"
```

---

### Task 8: Run the validation and write it up

**Files:**
- Create: `scripts/rehab24/validate_lunge_rules.py`
- Create: `notes/lunge-rule-validation.md`
- Modify: both specs (§8 status), `scripts/rehab24/README.md`

**Interfaces:**
- Consumes: everything from Tasks 1–7; `run_detector`, `LUNGE_DETECTOR`, `resolve_lead_side`
- Produces: `data/REHAB24-6/processed/lunge_rule_validation.json` (gitignored) and the
  committed writeup

- [ ] **Step 1: Write the CLI entry point**

`scripts/rehab24/validate_lunge_rules.py` — a thin entry point per the repo's
scripts-are-thin-entry-points architecture: bootstrap the repo root onto `sys.path`, parse
args (`--pose-dir`, `--segmentation`, `--out`), and call into
`src.rehab24.lunge_rule_validation`. Put the orchestration (loading pose JSON, looping
segments, calling `run_detector`) in the `src/` module, not the script.

For each Ex5 segment and each of the two cameras:

1. Load that camera's pose JSON, `slice_rep` the labeled window.
2. Estimate the view **per rep window**, NOT per clip.

   **Corrected 2026-07-31, measured — the original instruction here said "whole clip" and was
   wrong.** Every Ex5 video contains BOTH `front` and `half-profile` repetitions, roughly 50/50
   (`PM_021` 10/10, `PM_028` 11/10, `PM_042` 13/12, `PM_112` 12/13, …): the subject reorients
   partway through each recording. A whole-clip view estimate would therefore be derived from a
   mixture of two orientations and be wrong for about half the reps in every video — and the
   production pass exists precisely to report the view label a rule would really receive.

   `estimate_view_for_pose` only takes a path, but its aggregation is reproducible from two
   public functions with no production change. Mirror it over the window's frames:

```python
from src.pose.view_estimation import frame_view_signals, mean_finite, score_view

def estimate_view_for_window(window_frames: list[dict]) -> tuple[str, float]:
    """The view label a rule would really receive for THIS repetition.

    Mirrors `estimate_view_for_pose`'s aggregation (view_estimation.py:390-414) over a rep
    window instead of a whole file, including its `allow_front=False` production default and
    its deliberate NaN — not 0.0 — default for `torso_width_ratio`, which exists because a 0.0
    ratio reads as "maximally narrow" and manufactures a high-confidence `side` verdict from
    clips carrying no width evidence at all.
    """
    signals = [frame_view_signals(f) for f in window_frames]
    valid = [s for s in signals if s is not None]
    total = len(window_frames)
    valid_frame_ratio = len(valid) / total if total else 0.0
    view_type, confidence, *_ = score_view(
        orientation_score=mean_finite([s["orientation_score"] for s in valid], default=0.0),
        face_visibility=mean_finite([s["face_visibility"] for s in valid], default=0.0),
        torso_width_ratio=mean_finite([s["torso_width_ratio"] for s in valid], default=np.nan),
        z_asymmetry_value=mean_finite([s["z_asymmetry"] for s in valid], default=0.0),
        valid_frame_ratio=valid_frame_ratio,
        allow_front=False,
    )
    return view_type, confidence
```

   Put this in `src/rehab24/lunge_rule_validation.py` (it is pure — it takes frames, not a
   path) and unit-test it against a synthetic window.
3. **Production pass:** `run_detector(LUNGE_DETECTOR, window, 30.0, estimated_view, estimated_conf)`.
4. **Oracle pass:** the same call with the dataset's orientation substituted. The mapping is
   **fixed here, not chosen at implementation time** — put it in `lunge_rule_validation.py` as
   a named constant with this comment:

```python
# Dataset orientation -> the view label the ORACLE pass feeds the rules. `view_confidence` is
# pinned at 1.0 alongside it, since the premise of this pass is that the view is known.
#
# `front` DELIBERATELY maps to "front", a label production can NEVER emit: the production path
# calls estimate_view_for_pose(allow_front=False). That is the whole point of the oracle pass
# -- it asks "would this rule fire if the view label were correct?", which requires bypassing
# the gate rather than reproducing it. Any oracle-pass result on a `front` rep is therefore a
# statement about the RULE, never about what a user would see, and the writeup must say so
# wherever it quotes one.
#
# `half-profile` maps to "front_oblique" rather than "rear_oblique" because the dataset does
# not record which way the subject faced, and the two are equivalent for every lunge rule:
# `rule_knee_valgus` and `rule_pelvic_drop` treat both as fully observable, and the other two
# rules ignore the oblique labels entirely. Stated so nobody reads significance into the pick.
ORACLE_VIEWS: dict[str, str] = {
    "front": "front",
    "side": "side",
    "half-profile": "front_oblique",
    "profile": "side",   # unused by Ex5 (0 reps), present for completeness
}
ORACLE_VIEW_CONFIDENCE = 1.0
```
5. Record per rep: `person_id`, `correctness`, `exercise_subtype`, which faults fired in each
   pass with their severities, the `RunResult.fallback` path, the frame-validity rate, the
   resolved lead side, and the per-rule continuous score (the rule's `primary_value` when it
   fired, and the window's extreme of that metric when it did not — the AUC needs a score for
   **every** rep, not only the ones that fired).

- [ ] **Step 2: Run it**

```
.venv\Scripts\python.exe scripts/rehab24/validate_lunge_rules.py --pose-dir data/REHAB24-6/processed/lunge_pose_json --segmentation data/REHAB24-6/Segmentation.csv --out data/REHAB24-6/processed/lunge_rule_validation.json
```

Expected: 174 reps processed. If materially fewer, find out why (missing pose JSON, windows
past the end of a clip) and report the count and reason rather than quietly analyzing a subset.

- [ ] **Step 3: Compute and print the report**

Per rule, in both passes, restricted to the camera `RULE_CAMERAS` assigns it:

- **Per-subject AUC — median and range across the 8 subjects — as the HEADLINE.** Pooled AUC
  secondary. **No p-value on pooled reps**; the independence assumption it needs does not hold.
- 2×2 contingency with sensitivity and specificity, per subject and pooled.
- Where the spec threshold falls in the metric's distribution (percentile).
- Stratified: `front` reps (88) and `half-profile` reps (86) reported **separately**, never
  pooled.
- cam17 rules reported **both with and without** the 40 reps at extra-person level 2/3.

Once for the dataset:

- **Lead-leg accuracy vs `exercise_subtype`**, all 174 reps, plus the unresolved rate
  (`resolve_lead_side` returning `None`).
- Fallback-path distribution and frame-validity rates.

- [ ] **Step 4: Write `notes/lunge-rule-validation.md`**

Structure it so the caveat cannot be missed:

1. **What this measures — and what it does not.** Near the top, in the doc's own words:
   REHAB24-6 labels reps correct/incorrect and never names the fault, so a rule firing on an
   incorrect rep is not evidence it found that rep's error. This measures whether a rule's
   signal carries information about rep correctness — **not per-fault precision**.
2. Dataset summary and the Task 1 reconnaissance finding (including the side-gate answer).
3. Per-rule results, per-subject-first as above.
4. Lead-leg accuracy.
5. **Honest conclusions**, per the design spec §6:
   - A rule that separates well is validated **on this dataset** — a lab recording with fixed
     cameras, controlled lighting and instructed errors. Scope the claim.
   - A rule that does not separate may be real and simply invisible here. Say that, and do
     **not** move its threshold.
   - For `lunge_knee_valgus`, report **whether firing tracks step depth rather than
     correctness**. The proxy sums medial and anterior knee travel in every view production
     reaches (Task 4 docstring), so a rule that fires on deep reps of both classes equally is
     showing contamination, not valgus. Correlate its metric against the lead knee's flexion
     within the **correct** reps only: a strong relationship there is the contamination
     signature, since correct reps by definition have no valgus to find.
   - For `lunge_pelvic_drop` specifically, read **specificity on correct reps first**: the
     split-stance foreshortening documented in its docstring predicts false positives on deep
     correct reps, so a low specificity there is the expected failure, not a surprise. If its
     fire rate is near zero on **both** classes, the honest conclusion is "not exercised by
     this dataset", **not** "the rule works".
   - If the production and oracle passes disagree, say plainly which failed: the **gate** or
     the **rule**.
6. What would be needed to validate further (per-fault labels; a second dataset).

- [ ] **Step 5: Update both specs' §8**

Add a **Status (2026-07-30) — Lunge validated against REHAB24-6 Ex5** block to
`docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` and its `.zh-TW.md`
mirror, stating per rule what was measured and what remains unvalidated, and linking
`notes/lunge-rule-validation.md`. This is the **first** movement to carry such a block —
§8.4 has been outstanding since 2026-07-18. Say so, and be precise about how far it goes.

**Do not flip `LUNGE_DETECTOR.validated` to True in this task.** That flag drives the Beta
badge in the UI and its meaning ("checked against labeled ground truth") is a product claim.
Propose it to the user with the numbers in hand; let them decide.

- [ ] **Step 6: Final verification**

```
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```

- [ ] **Step 7: Commit**

```bash
git add scripts/rehab24/validate_lunge_rules.py src/rehab24/lunge_rule_validation.py notes/lunge-rule-validation.md docs/superpowers/specs/ scripts/rehab24/README.md
git commit -m "measure(rehab24): validate the lunge rules against 174 labeled repetitions"
```

---

## Self-Review

**Spec coverage.** Every section of `docs/superpowers/specs/2026-07-30-lunge-detector-design.md`
maps to a task: §3.1 phases → Task 2; §3.2 lead leg → Task 3 (+ the Task 6 Step 2
alternating-lead guard); §3.3 metrics → Task 2; §3.4 the four rules → Tasks 3–5; §3.5 view gating → Tasks 4–5
docstrings; §4.1 extraction + the side-gate gate → Task 1; §4.2 harness → Tasks 7–8; §4.3
camera routing → `RULE_CAMERAS`, Task 7; §4.4 two passes → Task 8 Step 1; §4.5 reporting →
Task 8 Step 3; §5 testing → Tasks 2–6, 7; §6 honesty → Task 8 Step 4; §7 risks → Task 1
Step 2 (transpose STOP), Task 3 (lead ambiguity), Task 5 (foreshortening), Task 6 Step 7
(CLI-only confirmation). All four parent-spec Lunge rules are implemented; none is dropped.

**Placeholder scan.** The `...` markers in Tasks 3–5 for `kg_query`, `citation` and
`citation_support` are deliberate and are **not** plan placeholders: they mark values that
must be read out of the parent spec and the live KG at implementation time rather than typed
from memory, and Task 6 Steps 1–2 are the explicit step that fills them. Every other step
carries real, runnable content.

**Type consistency.** `CoreFrame.m(key)`, `RuleContext(fps, view_type, view_confidence,
min_frames)`, `MovementDetector(name, metric_keys, compute_raw, assign_phases, rules, ...)`,
`run_detector(...) -> RunResult`, `build_detection(*, fault_id, fault_name, kg_query,
retrieval_mode, segment_metrics, score_values, severity, confidence, observability, evidence,
citation, citation_support)`, `severity_from_range(value, mild, severe, *, lower_is_worse)`,
`contiguous_true_segments(mask, min_frames)` and `knee_forward_ratio(points, knee, ankle, toe)`
are all used with the signatures verified by reading the source. Metric key names are
consistent between Task 2's `LUNGE_METRIC_KEYS`, the `f"{lead}_..."` lookups in Tasks 3–5, and
the fixtures. `resolve_lead_side` returns `"left"`/`"right"`/`None` everywhere it is used.

**Honesty.** Phase 1 ships thresholds that are spec-derived and, at registration time,
unvalidated — Task 6 Step 7 says so in the spec, and `validated=False` keeps the UI honest.
Phase 2 measures them but cannot measure per-fault precision, and Task 8 Step 4 requires that
limit stated up front rather than buried. No task permits a threshold to move in response to
a result.

**Late addition, recorded rather than smoothed over.** Rebuilding the test fixture on realistic
geometry surfaced two projection facts that were not in the design spec: (1) in a strictly
frontal view a knee's in-image flexion and its medial offset are the same degree of freedom,
so `resolve_lead_side` and `rule_knee_valgus` read the same quantity there; and (2) in the
oblique views production actually reaches, anterior knee travel adds to the valgus proxy, so a
deep well-tracked lunge reads as valgus. Neither is correctable without depth. Both are now
documented in `lunge_frame`, pinned by `test_anterior_knee_travel_contaminates_the_valgus_proxy`,
carried into `rule_knee_valgus`'s docstring, and turned into a question Phase 2 answers. They
should be added to the design spec's §7 risk table when Task 6 Step 7 updates the specs.

**Risk register.**
- Task 1 Step 2 (cam18 transpose) is a hard STOP for Phase 2. Phase 1 may still proceed.
- `lunge_knee_valgus` may prove to be measuring step depth rather than valgus (see above).
  Task 8 Step 4 tests for it; if confirmed, that is a finding about the spec's frontal-plane
  proxy under monocular projection, and the threshold still does not move.
- Task 1 may find the `side` gate never opens. That is a recorded finding, not a reason to
  weaken the gate.
- Task 6 Step 1 may find no KG node for a fault. Leave it unretrieved and record the gap —
  never re-point a query at a near-miss node.
- The alternating-lead test lives in Task 6 Step 2, with the detector it needs. Writing it in
  Task 5 would have committed a knowingly-failing test, which a task review would rightly
  reject.
