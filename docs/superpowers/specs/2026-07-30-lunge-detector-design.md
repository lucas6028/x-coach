# Lunge Rule Detector + First Labeled Validation — Design

**Status:** design spec · **Date:** 2026-07-30 · **Movement:** Lunge (4th of 16)
**Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` §Lunge (lines 163–211)

---

## 1. Purpose and why Lunge is next

Three of sixteen movement detectors ship today — Squat (validated), Overhead Press and
Push-up (both explicitly **spec-derived and UNVALIDATED**). Lunge is next for three
independent reasons:

1. It is next in the parent spec's own order (Group A: Squat → Lunge → Deadlift).
2. **It is the only unimplemented movement with labeled ground truth in this repository.**
   `data/REHAB24-6/` `Ex5` is "leg lunge": **174 labeled repetitions, 8 persons,
   96 incorrect / 78 correct**, with 30 fps video from two orthogonal cameras.
3. `data/kg/sports_kg_v3.graphml` already carries **168 `Lunge:`-scoped nodes**, so the
   `retrieval_mode="kg"` queries have real targets — unlike the three dangling OHP queries
   recorded as an open item in the parent spec.

The parent spec's §8.4 says "validate thresholds against labeled data per movement before
shipping analysis for that movement." No movement has ever satisfied that. Lunge can.

**Scope decision (user-approved):** detector **plus** validation, staying CLI-only.
Promotion to the web app is explicitly NOT in this pass — that decision should be made
after seeing the validation numbers, not before.

---

## 2. What the labeled data actually contains

Measured from `data/REHAB24-6/Segmentation.csv` (Ex5 rows) and
`data/REHAB24-6/Segmentation.txt` (its column semantics), 2026-07-30:

| Fact | Value |
|---|---|
| Reps | 174 across 9 video ids (`PM_021, PM_028, PM_037, PM_042, PM_104, PM_112, PM_117a, PM_117b, PM_125`) |
| Persons | 8 |
| Correctness | 96 incorrect (`0`) / 78 correct (`1`) |
| cam17 orientation | 88 `front` / 86 `half-profile` / 0 `profile` |
| Balance | front 49 bad / 39 good; half-profile 47 bad / 39 good |
| `exercise_subtype` | names the **lead leg** ("front leg right" / "front leg left") on all 174 |
| Extra-person contamination | cam17: 117 clean, 17 level-1, 38 level-2, 2 level-3. **cam18: level 0 on all 174** |

**The finding that shapes the whole validation design.** `Segmentation.txt` documents the
camera geometry explicitly:

```
cam17 orientation -> cam18 orientation
  front        -> side
  half-profile -> half-profile
  side         -> front
```

The two cameras are orthogonal and record simultaneously, so for the 88 `front` reps the
**same repetition** is available in both a frontal view (cam17) and a **true sagittal view**
(cam18). Every lunge rule can therefore be validated on a camera that actually affords its
required view — including `lunge_knee_past_toes`, which is side-only and which an earlier
reading of this dataset assumed would go unvalidated.

**What the labels do NOT contain.** `correctness` is binary per repetition and never states
*which* fault occurred. `exercise_subtype` names the lead leg, not the error. This bounds
every claim the validation can make; see §6.

---

## 3. Detector design

New module `src/pose/movements/lunge.py`, following the structure of
`src/pose/movements/pushup.py` and `overhead_press.py`: raw metrics → phase assignment →
cited rules → assembled `LUNGE_DETECTOR`, registered under `"Lunge"` via a side-effect
import in `src/pose/movements/registry.py`. Reachable from
`scripts/pose/run_pose_rule_detection.py --movement "Lunge"`. No new dependencies
(stdlib + numpy, per the ML library's local-first style).

### 3.1 Rep phases and segmentation

Phases per the parent spec: `setup → descent → bottom → ascent → recovery`, plus the
shared `rest` phase for frames outside any repetition. Segmentation uses the existing
`MovementDetector` knobs: `rep_signal="min_knee_angle"`, `rep_polarity="min"`,
`rep_start="extended"` (a lunge starts standing). Phase assignment mirrors
`ohp_assign_phases` / `pushup_assign_phases`: percentile-based on the rep signal, with the
`unknown` fallback when no frames are valid.

### 3.2 Lead-leg identification — the one gap in the parent spec

All four rules reference "the lead leg". The parent spec defines it as "the more flexed /
more anterior foot". **Anterior is exactly the axis that collapses in a frontal view**, and
two of the four rules are frontal, so the anterior half of that definition is not usable
where it is most needed.

**Implementation:** lead leg = **the leg whose knee is more flexed at the rep's bottom frame**
(smaller knee angle), which is measurable from every view. This is a **substitution, not a
restatement**, and is documented in-code as such per the project's anti-hallucination rule.

**Where that reduction happens is load-bearing, and the obvious placement is wrong.**
`run_detector` calls `compute_raw(frames, fps)` over the **whole clip, before**
`segment_reps` — so at metric-computation time there are no rep boundaries and therefore no
"bottom frame" to resolve the lead side against. A per-frame "whichever knee is more flexed
right now" is all `compute_raw` could produce, and during `setup` and `recovery` both knees
sit near extension within noise of each other, so the chosen side **flickers frame to frame**.
Every lead-relative metric would then swap legs mid-clip, and `centered_median` would blend
two different legs into a single ratio that describes neither. A `lead_side` in `metric_keys`
would additionally be median-smoothed as though `±1` were a continuous quantity.

This defect would have been **invisible to the validation harness**, which hands each rep
window in as its own clip (§4.2) — a clip-level lead side is per-rep by construction there.
Validation would pass while production broke on any clip alternating legs, which is how
lunges are normally performed.

**Therefore:** `compute_raw` emits **both sides** symmetrically and resolves nothing
(`left_/right_knee_forward_ratio`, `left_/right_knee_medial_offset_ratio`, and so on). Each
**rule** resolves the lead side **over its own window**, which `run_detector` already hands it
as a per-rep slice. The reduction belongs where the rep boundary exists, not upstream of it.
On a fallback path the rule sees the whole clip and the resolution degrades exactly as
everything else on that path does — stated, not hidden.

Lead-leg identification is also the one design choice in this document that is directly
checkable: `exercise_subtype` gives the true lead leg on all 174 reps, so **lead-leg accuracy
is a measured number** in the validation output, not an assumption. If it is poor, every
downstream lunge rule inherits the error, and the validation must say so.

### 3.3 Metrics

All normalized and scale-free; MediaPipe normalized image coordinates, **y grows DOWNWARD**.

Per §3.2, every side-specific metric is emitted **for both legs**; no metric names a "lead"
leg, because `compute_raw` cannot know which leg that is.

| key | definition |
|---|---|
| `left_knee_angle` / `right_knee_angle` | `angle_degrees(hip, knee, ankle)` per side |
| `min_knee_angle` | the more-flexed (smaller) of the two finite sides; the rep signal |
| `left_knee_forward_ratio` / `right_knee_forward_ratio` | `(proj(knee−ankle onto (toe−ankle)) − foot_len) / foot_len` per side — the squat detector's existing construction |
| `left_knee_medial_offset_ratio` / `right_knee_medial_offset_ratio` | signed offset of that knee from its own hip→ankle line, **positive = toward the mid-hip (medial)**, normalized by hip width |
| `pelvis_tilt_signed_deg` | `angle_from_horizontal(L_hip(23) → R_hip(24))`, signed in a **fixed** left/right convention — the rule converts it to "contralateral hip lower" once it knows the lead side |
| `trunk_lateral_lean_deg` | `angle_from_vertical(shoulder_mid → hip_mid)` in the x–y plane |
| `hip_width` | `distance(23, 24)` — the normalizer, emitted for diagnostics |

`compute_raw` gates frame validity on a `required` landmark tuple (both hips, knees, ankles,
foot indices, and shoulders for the trunk lean). As with push-up, **one dropped landmark
marks the frame invalid and silences every lunge rule for it**; this is deliberate and gets
the same module-level documentation push-up received.

### 3.4 Rules

Four rules, all four implementable. Unlike the Push-up section — which states no severity
ramp at all, forcing five rule-level choices — the Lunge section states its ramps, so
**nothing here is a rule-level invention**. Three ramps are given as explicit `ramp a → b`
lines (`lunge_knee_valgus` 0.10 → 0.25, `lunge_insufficient_depth` 100° → 130°,
`lunge_pelvic_drop` 8° → 20°). The fourth, `lunge_knee_past_toes`, is written as
"flag when > 0.10 … severe ≥ 0.30" — identical wording to the squat's `knees_forward` entry,
which this repo already reads as a 0.10 → 0.30 ramp via `KNEE_FORWARD_MILD` /
`KNEE_FORWARD_SEVERE` in `src/pose/pose_rule_detector.py`. **Lunge reuses those two constants
rather than restating the numbers**, so the reading stays single-sourced and no new number
enters the codebase.

| fault_id | Fire threshold | Severity ramp | View handling | Precedent followed |
|---|---|---|---|---|
| `lunge_knee_past_toes` | ratio > 0.10 | 0.10 → 0.30 | **hard gate** `side` + `view_confidence ≥ 0.20` | `squat.rule_knees_forward` |
| `lunge_knee_valgus` | medial offset > 0.10·hip_width | 0.10 → 0.25 | observability downgrade (×0.65), no gate | `squat.rule_knees_inward` |
| `lunge_insufficient_depth` | min lead-knee angle > 100° | 100° → 130° | downgrade head-on | `squat.rule_shallow_depth` |
| `lunge_pelvic_drop` | pelvis tilt > 8° | 8° → 20° | downgrade; not observable pure-`side` | — |

Phase scope for all four: `{descent, bottom, ascent}`, following the squat detector's
`ACTIVE_PHASES`. `lunge_insufficient_depth` additionally reads its minimum at `bottom`.

**Citations** are copied verbatim from the parent spec at implementation time, never recalled
from memory:

- `lunge_knee_past_toes` — Zellmer M, et al. "Patellar tendon stress between two variations
  of the forward step lunge." J Sport Health Sci (2019). PMC6523035.
- `lunge_knee_valgus` — Ford KR, et al. "An evidence-based review of hip-focused
  neuromuscular exercise interventions to address dynamic lower extremity valgus."
  PMC4556293 (2015).
- `lunge_insufficient_depth` — Alkjær T, et al. "Forward lunge before and after anterior
  cruciate ligament reconstruction." PLoS One (2020), PMC6980669; supplemented by
  Escamilla R, et al. IJSPT (2022), PMC8805090.
- `lunge_pelvic_drop` — Ford KR, et al. PMC4556293 (2015); cross-support Alkjær T, et al.
  PMC6980669 (2020).

`kg_query` strings must be resolved against `data/kg/sports_kg_v3.graphml` via
`graph_retrieval.resolve_nodes` **before** being committed, so Lunge does not repeat the
three dangling OHP queries. Candidate targets exist (`Lunge:Knee Valgus`,
`Lunge:Lead Knee Extends Beyond Toe`, `Lunge:Anterior Trunk Tilt`, and the
`Lunge:`-scoped fault set generally); each must be verified to resolve, not assumed.

### 3.5 Why the two frontal rules are NOT gated on `front` / `front_oblique`

`estimate_view_for_pose` is called with `allow_front=False` in the production path, so the
only reachable labels are `{side, rear, rear_oblique, unknown}` — `front` and `front_oblique`
are never emitted. A positive gate on those labels would make a rule **permanently silent**,
which is exactly the trap documented for `pushup_elbow_flare`.

Both frontal lunge rules avoid it because **neither needs to know which way the subject is
facing**:

- **Valgus** is medial-relative-to-midline, and the midline is the mid-hip. That reads
  identically whether the camera is in front of or behind the subject.
- **Pelvic drop** needs "contralateral", which resolves from the lead side (§3.2) plus image
  `y`. Also facing-free.

So both follow the squat `rule_knees_inward` precedent — fire in all views, downgrade
observability and apply the ×0.65 confidence discount outside
`{front, front_oblique, rear, rear_oblique}` — rather than a hard gate. `lunge_knee_past_toes`
does need the sagittal plane and does take the hard gate, matching `rule_knees_forward`.

---

## 4. Validation harness design

### 4.1 Pose extraction — and it happens FIRST, before any rule is written

**Sequencing is deliberate.** Extraction and a view-estimation reconnaissance run come before
detector implementation, because their result changes what `lunge_knee_past_toes` can claim.

The measured precedent is discouraging: across the 45 real pose JSONs in this repo, the view
estimator emitted `side` **exactly once**, and that one verdict was the fabricated degenerate
fixture since removed — the corpus is 30 `rear_oblique`, 13 `rear`, 2 `unknown`. If cam18's
genuinely sagittal Ex5 clips also come back `rear_oblique`, then `lunge_knee_past_toes` —
hard-gated on `side` + `view_confidence ≥ 0.20` (§3.4) — **fires zero times in production**.
§3.5 argues the two frontal rules dodge the `pushup_elbow_flare` permanently-silent trap; if
that happens, the sagittal rule walks straight into it, and the spec would be claiming the
escape while shipping the trap.

**This is one cheap run over the extracted pose JSON and it must precede rule implementation.**
It decides whether `lunge_knee_past_toes` ships production-live or is honestly recorded as
oracle-validatable-only. Either outcome is fine; discovering it after the rule is written is
not.

**Verified extraction route** (interface read, not assumed —
`scripts/pose/run_pose_extraction.py` `--dataset unlabeled` rglobs arbitrary mp4 directories
and shells out per video):

```
.venv\Scripts\python.exe scripts/pose/run_pose_extraction.py ^
  --dataset unlabeled ^
  --video-dir data/REHAB24-6/Ex5 ^
  --output-dir data/REHAB24-6/processed/lunge_pose_json ^
  --no-video
```

`data/REHAB24-6/Ex5` holds exactly the 18 target mp4s (9 ids × cam17/cam18), so the rglob
needs no filtering.

The result is standard pose JSON **with the visibility channel**. The existing
`data/REHAB24-6/processed/mediapipe_landmarks_cache/*.npz` cannot be reused: it stores only
`image` `(N,33,2)` and `world` `(N,33,3)` arrays, with no visibility, and the detector's
frame-validity gate requires it.

**Verify before trusting any cam18 number:** the cam18 files are named
`*-Camera18-30fps-transposed.mp4`. If that rotation is not already baked into the pixels,
every cam18 metric is silently rotated 90° and every sagittal result is garbage. This is a
STOP condition, not a footnote.

### 4.2 Harness

New `src/rehab24/lunge_rule_validation.py` with a thin CLI entry point
`scripts/rehab24/validate_lunge_rules.py`, matching the repo's scripts-are-thin-entry-points
architecture.

For each Ex5 row: slice frames `[first_frame, last_frame]` from the video's pose JSON and
hand that window to `run_detector` **as its own clip**. Ground-truth rep boundaries mean
`segment_reps` is bypassed entirely, which **isolates rule quality from segmentation
quality** — if separation is weak, you know which subsystem failed. No scoring logic is
duplicated: production rules, production severity, production merge. The harness records
which `RunResult.fallback` path each window took as a diagnostic.

### 4.3 Camera routing

Using the documented cam17→cam18 mapping (§2), per rule, for the 88 `front` reps:

| Rule | Camera read | View there |
|---|---|---|
| `lunge_knee_valgus` | cam17 | front |
| `lunge_pelvic_drop` | cam17 | front |
| `lunge_knee_past_toes` | cam18 | **side** |
| `lunge_insufficient_depth` | cam18 | side |

The 86 `half-profile` reps are oblique in both cameras and are reported as **their own
stratum**, not folded into the frontal or sagittal numbers.

### 4.4 Two passes — production view vs oracle view

Every rule is run twice:

- **Production pass** — the rule receives the view label `view_estimation` actually produces.
  This is what a user would get.
- **Oracle pass** — the rule receives the dataset's ground-truth orientation instead.

The gap between them is the point. If `lunge_knee_past_toes` never fires in the production
pass, that pass alone cannot distinguish "the rule is wrong" from "the `side` gate never
opened". Running both separates a **gate failure** from a **rule failure**.

**By-product — and it is narrower than it first looks.** The dataset's orientation vocabulary
is `{front, half-profile, profile}`; the estimator's reachable labels under
`allow_front=False` are `{side, rear, rear_oblique, unknown}`. There is no `front` for the
estimator to hit and no `rear` in the data, so these are not two labelings of the same space
and **no 174-rep confusion matrix exists**. What is genuinely checkable is one cell: *does
cam18, on a rep the dataset calls cam17-`front` and therefore cam18-`side`, actually read
`side`?* That single question is the §4.1 gate. Report it as that, not as "view estimation
validated against ground truth" — `notes/` must not inherit the larger sentence.

### 4.5 What gets reported

Per rule:

- **Per-subject AUC of the underlying continuous metric against correctness — median and
  range across the 8 subjects — reported as the headline.** Pooled AUC is secondary. The 174
  reps are **not independent**: they are ~22 reps from each of 8 people, and pooling them
  lets one subject's separation masquerade as a population result. This project has already
  been burned twice by exactly this shape of optimism (a fixed-λ ridge fabricating a null; a
  1-sequence oracle-debiased preview running 2.2× optimistic), so the conservative statistic
  leads. **No p-value is computed on pooled reps** — the independence assumption it needs
  does not hold here.
- AUC is threshold-free, which is what explains *why* a threshold did or did not separate —
  e.g. a genuinely informative metric whose spec threshold happens to sit in the tail.
- 2×2 contingency of fired/not-fired against correct/incorrect, with sensitivity and
  specificity, per subject and pooled.
- Where the spec threshold falls in that metric's distribution (percentile).

Plus, once for the whole dataset:

- **Lead-leg accuracy** against `exercise_subtype`, all 174 reps (§3.2).
- Fallback-path distribution and frame-validity rates (how often the detector could measure
  anything at all).
- cam17 results **both with and without** the 40 reps carrying level-2/3 extra-person
  contamination, since MediaPipe's single-person extraction may lock onto the wrong body
  there. cam18 is level 0 throughout and needs no such split.

**Outputs:** `notes/lunge-rule-validation.md` (committed, human-readable) plus raw JSON under
`data/` (gitignored), plus a status block appended to the parent spec's §8 recording exactly
what was and was not validated.

---

## 5. Testing

Matching the repo's existing split:

- **CI-visible:** `tests/test_lunge.py` — synthetic-fixture tests for metrics (each metric
  controlled by construction, not hardcoded), phase assignment, rule firing AND non-firing,
  **boundary tests just inside and just outside every threshold**, and one exact severity
  assertion per rule. The OHP review found 5 of 10 threshold mutants surviving because every
  fixture sat at an extreme; boundary fixtures use constant-value frames so `run_detector`'s
  median smoothing is a no-op and asserted severities are exact.
- **CI-visible:** pure-function tests for the harness — rep-window slicing, camera routing,
  contingency math — with synthetic inputs and no data dependency.
- **Local-only:** the data-backed validation run is `skipUnless` the files exist, following
  `tests/test_view_regression_corpus.py`, because `data/` is gitignored.
- **CI-visible, and the one test that must exist:** an **alternating-lead multi-rep fixture** —
  a synthetic clip whose reps lead left, right, left — asserting each rep's fault is attributed
  to the leg that actually led it. This is the test the harness structurally cannot provide
  (§3.2: it feeds one rep per clip), and it is the regression guard on the lead-side reduction
  living in the rules rather than in `compute_raw`.
- **Regression:** `tests/test_movement_registry.py` gains Lunge resolution
  (`get_detector("Lunge")` and `get_detector("lunge")`), and the existing squat
  byte-for-byte gate must still pass — Squat is production.

Verification set before claiming done: `.venv\Scripts\python.exe -m pytest tests/ -q`, then
`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.

---

## 6. Honesty constraints

These are binding on how results are written up, not optional caveats.

1. **The labels are binary, not per-fault.** REHAB24-6 says a rep was correct or incorrect and
   never says which fault occurred. A rule firing on an incorrect rep is **not** evidence it
   found that rep's actual error. The validation therefore measures whether a rule's signal
   **carries information about rep correctness** — not per-fault precision. The results doc
   must state this in its own words, near the top, not in a closing footnote.
2. **No threshold tuning (user decision).** Cited thresholds stay exactly as the literature
   and parent spec state them. Weak separation is written up as a finding, not repaired by
   moving a number — a threshold tuned to a metric is no longer the cited number, and the
   citation would become a false provenance claim. Severity ramps are also spec-stated for
   Lunge, so they do not move either.
3. **8 subjects is small.** Per-person breakdowns are mandatory. A rule that fails here may be
   real and simply invisible in this dataset; that is a stated possibility, not a hidden one.
4. **A validated rule is validated on THIS dataset.** REHAB24-6 is a lab recording with fixed
   cameras, controlled lighting and instructed errors. Any claim must be scoped to it.
5. **`lunge_pelvic_drop`'s likely failure is false positives, not silence.** In a frontal view
   of a **split stance** the `L_hip → R_hip` vector is rotated in the transverse plane, so its
   image projection **shortens**, and `atan2(dy, dx)` on a shortened `dx` **inflates** the
   apparent tilt — the deeper the lunge, the worse. So the first number to read for this rule
   is **specificity on correct reps**, not sensitivity: the failure mode to expect is firing
   on deep, correctly-performed reps. Separately, nothing guarantees the dataset's instructed
   errors include a Trendelenburg pattern at all; if the fire rate is near zero on **both**
   classes the honest conclusion is "not exercised by this dataset", **not** "the rule works".

---

## 7. Risks

| Risk | Handling |
|---|---|
| cam18 `-transposed` rotation not baked into pixels | Verify first; STOP condition, invalidates all sagittal results |
| Lead-leg heuristic inaccurate | Measured directly against `exercise_subtype`; if poor, every rule inherits it and the writeup says so |
| **Lead side resolved in the wrong place** (`compute_raw`, where no rep boundary exists) → flickering side, two legs blended by `centered_median`, and a defect the harness cannot see | §3.2 puts the reduction in the rules; the alternating-lead multi-rep fixture (§5) is the guard |
| `side` gate never opens on cam18 → `lunge_knee_past_toes` permanently silent in production | Checked **before** the rule is written (§4.1); the oracle pass then separates gate failure from rule failure |
| `lunge_pelvic_drop` inflated by split-stance foreshortening | Read specificity on correct reps first (§6.5) |
| MediaPipe locks onto the wrong person on contaminated cam17 clips | cam17 reported with and without the 40 level-2/3 reps; cam18 clean throughout |
| Extraction cost | 18 clips ≈ 1300 frames each; CPU MediaPipe, one-time |
| Scope creep into shipping Lunge to users | Explicitly out of scope; `DEFAULT_ANALYSIS_MOVEMENT` stays `"Squat"` and `ANALYZABLE_MOVEMENTS` stays `["Squat"]`, confirmed as a final step |

---

## 8. Out of scope

- Promoting Lunge to the web app (backend `DEFAULT_ANALYSIS_MOVEMENT`, frontend
  `ANALYZABLE_MOVEMENTS`). Decide after seeing the numbers.
- Deadlift, or any other movement.
- Re-validating Squat, OHP or Push-up against labeled data, though the harness built here is
  reusable for Squat (REHAB24-6 `Ex6`) and should be written with that in mind without being
  generalized speculatively.
- Fixing the parent spec's known `heel_rise` sign defect, or the three dangling OHP
  `kg_query` strings.
