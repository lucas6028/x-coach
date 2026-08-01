# Row (Bent-over Barbell Row) Rule Detector — Design

**Status:** design spec · **Date:** 2026-08-01 · **Movement:** Row (5th of 16)
**Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`
§Group C → Row (lines 621–689)

---

## 1. Purpose and why Row is next

Four of sixteen detectors ship today: Squat (`validated=True`), Overhead Press, Push-up and
Lunge (all `validated=False` → Beta). Deadlift, the remaining Group A movement, is being
implemented in parallel by the repository owner. Row is therefore the next movement in the
parent spec's own order — the first of Group C (upper-body pull) — and is the first
detector whose whole rule set sits outside the sagittal lower-body geometry the previous four
share.

Row also has the KG backing the parent spec assumes: `data/kg/sports_kg_v3.graphml` carries
**130 `Row:`-scoped nodes**, including 24 labeled `Fault`, so every `kg_query` in §4 resolves
to a real node with a non-empty bucket (verified, §5).

**Scope (user-approved):** detector **plus** tests. **No validation harness** — see §2. The
detector is registered in `src/pose/movements/registry.py`, which makes it reachable both from
`scripts/pose/run_pose_rule_detection.py --movement "Row"` and, because `GET /api/movements` is
registry-driven, immediately selectable in the web app with the **Beta** tag that
`validated=False` produces. That user-facing consequence was confirmed before this document was
written, not discovered afterwards.

---

## 2. There is no labeled-correctness ground truth for Row, and that is a hard constraint

REHAB24-6 — the only per-rep correct/incorrect labeled dataset in this repository — contains
six exercises: `Ex1` arm abduction, `Ex2` arm VW, `Ex3` table push-ups, `Ex4` leg abduction,
`Ex5` leg lunge, `Ex6` squats (`notes/dataset-summary.md`). **None is a row.**

Fit3D is different: it **does** contain row video — `barbell_row`, `barbell_dead_row` and
`one_arm_row` (`data/Fit3D/fit3d_info.json`), with 3D mocap ground truth under
`train/*/joints3d_25/` and rep boundaries in `rep_ann.json`, across all 8 train subjects. What
Fit3D does **not** carry is a binary correct/incorrect label on any rep — it is 3D truth, not a
fault judgment. That distinction is the whole point of this section: the *kind* of validation
each dataset can support differs, and neither supports the Lunge pass's kind for Row.

So the Lunge pass's second half — replay labeled repetitions through the production rules and
report per-subject AUC against correctness — has no analogue here for either dataset and is not
attempted. Building a harness with nothing to point it at would produce the *appearance* of
validation, which is worse than its absence.

What Fit3D's 3D truth **could** support instead — a 2D-cue-vs-3D-truth fidelity comparison, the
shape this project has already run for other movements (e.g. the Fit3D 2D-vs-3D findings in
`notes/`) — is possible for Row and is simply **not attempted in this pass**. That is future
work, not work blocked on absent data, and it is named here so the distinction is not lost:
"no validation was run" is a scoping decision for Row; "no data exists to validate against" is
true only for the REHAB24-6-style correctness check, not for a Fit3D-style fidelity check. One
caveat that bounds even that future work: Fit3D's rig is 4 cameras, all oblique, with no true
side view, which matters for any Row rule that needs a lateral component.

Consequences, binding on how this work is described:

1. `ROW_DETECTOR.validated` stays `False`. The Beta tag is the honest label, not a formality.
2. Every threshold in §4 is either **spec-stated** or a **rule-level choice**, and no threshold
   in either category has been checked against a row performed by a human being. The in-code
   docstrings must keep those two categories separate the way `pushup.py` does.
3. The parent spec's §8.4 ("validate thresholds against labeled data per movement before
   shipping analysis for that movement") is **not** satisfied for Row. Closing it needs either
   labeled row video with per-rep correctness labels (REHAB24-6-style, which does not exist for
   Row), or a fidelity-style pass against Fit3D's existing 3D row data (which exists but was not
   run here). Recorded as an open item either way, not quietly skipped.

---

## 3. The spec's `rounded_thoracolumbar_spine` rule is geometrically degenerate

**This is a defect in the parent spec found during implementation design, and it is the one
finding here that changes the deliverable: Row ships four rules, not five.**

The parent spec's heuristic offers two constructions, and both collapse to a constant:

- *"three-point angle at mid-spine using shoulder-midpoint(11,12), a synthesized mid-trunk
  point = 0.5·(shoulder_mid + hip_mid), and hip-midpoint(23,24)"* — the middle point is by
  construction the midpoint of the segment joining the other two. Three collinear points
  subtend exactly 180° on every frame of every video. The metric carries no information at all.
- *"Flag flexion if the shoulder-midpoint drops below the straight shoulder–hip line by a
  normalized sag > 0.04"* — `shoulder_mid` is an **endpoint** of that line. Its distance to a
  line through itself is identically zero. The threshold can never be crossed.

The root cause is not a wording slip. **MediaPipe Pose has no thoracic or lumbar landmark.**
There is no measured point anywhere between the shoulders and the hips, so no sag, curvature or
three-point spinal angle can be computed from this detection model, by any construction. The
spec wrote a proxy that requires a landmark the spec's own §3 detection model does not provide.

**Decision (user-approved): document, do not substitute.** No rule with `fault_id`
`rounded_thoracolumbar_spine` is implemented. The degeneracy proof above goes in `row.py`'s
module docstring, and the parent spec's §7 (Honest limitations & gaps) gains an entry recording
it as a **spec defect**, not as a Row implementation gap.

Precedent: `pushup.rule_scapular_winging` is already carried as a permanently-silent rule for a
related reason. Row's case is the stronger one — push-up's silence is a view-gate accident,
Row's is a geometric impossibility.

**What was rejected, and why it is recorded.** Two monocular signals do carry *some* trunk-shape
information: trunk-length foreshortening (`dist(shoulder_mid, hip_mid)` shrinking as the spine
flexes) and ear-drop relative to the trunk line. Both are confounded by camera distance and by
the hinge angle itself, neither is what the rule's citation (Saeterbakken PMID 26134664,
erector-spinae EMG magnitude) actually supports, and shipping either under the spec's `fault_id`
would attach a real citation to a metric that citation says nothing about. If one is ever
implemented it must take a **different** `fault_id` and carry an explicitly-invented threshold.

`Row:Trunk Flexion` exists in the KG with a non-empty bucket (`corrections: Maintain Neutral
Spine`), so the knowledge-graph target for this fault is present and waiting. The gap is the
metric, not the knowledge.

---

## 4. Detector design

New module `src/pose/movements/row.py`, structured exactly like `src/pose/movements/lunge.py`:
raw metrics (containing **no thresholds**) → phase assignment → cited rules → assembled
`ROW_DETECTOR` → `registry.register`. stdlib + numpy only.

### 4.1 Rep segmentation and phases

The parent spec's Row phases are *setup (hip-hinge) → concentric pull → peak hold → eccentric
lower → return*. Implemented as `setup → pull → peak → lower`, plus the framework's shared
`rest` phase for frames outside any repetition. "Return" is not a separate label: after the
peak the arms extend and the frames are `lower`, which is the same reduction
`ohp_assign_phases` makes for the press's return.

| knob | value | why |
|---|---|---|
| `rep_signal` | `"min_elbow_angle"` | the elbow flexes to pull and extends to return; this is the row's depth analogue of push-up's `min_elbow_angle` |
| `rep_polarity` | `"min"` | the rep's extremum is minimum elbow angle (peak pull) |
| `rep_start` | `"extended"` | a bent-over row starts with the arms hanging extended |
| `rep_rectify` | `False` | the signal is unipolar |
| `min_rep_seconds` | default | rows are not a fast cyclic movement |

`row_assign_phases` mirrors `ohp_assign_phases` / `lunge_assign_phases`: `setup` is the first
`max(1, int(n * 0.15))` frames, `peak` is the most-flexed 30% by percentile, frames before the
deepest index are `pull`, the rest are `lower`; an invalid frame is `unknown` and the validity
check precedes the setup cutoff; an empty clip returns `[]` and a clip with no finite signal
returns all-`unknown`.

**No shared `ACTIVE_PHASES` set is defined.** Squat, Push-up and Lunge each define one because
their spec entries mostly scope faults to "descent/bottom/ascent" as a block; the Row entries do
not. Each Row heuristic names its own phase — "at setup baseline and at peak pull", "elbow
flexion at peak", "peak concentric wrist acceleration", "at peak: compare left vs right" — so
each rule scopes to exactly the phase its own spec line names (§4.4), and a set every rule would
have to override would be a dead constant. Stated cost: a fault visible only during `lower` (the
eccentric) is not scored by any rule, because no Row spec line asks for one.

### 4.2 The setup baseline is Row's version of Lunge's lead-leg problem

Three of the parent spec's five Row heuristics are specified as **deltas from a setup
baseline** ("trunk_angle at setup baseline and at peak pull"; "shoulder-line tilt increases
> 0.04 vs setup"). A baseline is a per-rep reduction, and — exactly as with the lunge's lead
leg — `run_detector` calls `compute_raw` over the **whole clip, before** `segment_reps`, so at
metric time no rep boundary exists and there is no "this rep's setup" to reduce against.

**Therefore:** `row_compute_raw` emits only per-frame quantities and computes no baseline. Each
**rule** computes its own baseline over its own window, which `run_detector` hands it as a
per-rep slice: the **median** of the metric over that window's `setup`-phase valid frames
(median, not mean, so a single bad frame cannot move it). A window with no usable `setup` frame
yields no baseline and the rule emits **nothing** — silence, never a guessed baseline.

Two costs, stated rather than hidden:

- The baseline is **per rep**, so a lifter who is already rounded/rotated at rep 1's setup reads
  as clean. A clip-level baseline would fix that but would make rep *N*'s verdict depend on
  rep 1's frames, which the per-rep architecture deliberately does not do.
- On the fallback paths (`no_reps_detected`, `only_partial_reps`, `segmentation_disabled`) the
  rule receives the **whole clip**, so "setup" is the clip's first 15% and the baseline is
  whatever the lifter was doing then. That degrades exactly as everything else on the fallback
  path does.

### 4.3 Metrics

All in MediaPipe normalized image coordinates, **y grows downward**. Nothing is reduced to a
"the" side; left and right are emitted symmetrically.

| key | definition |
|---|---|
| `left_elbow_angle` / `right_elbow_angle` | `angle_degrees(shoulder, elbow, wrist)` per side |
| `min_elbow_angle` | the more-flexed (smaller) finite side; the rep signal |
| `max_elbow_angle` | the less-flexed finite side; the conservative reading for the ROM rule (§4.4) |
| `trunk_angle_from_horizontal_deg` | `degrees(atan2(|dy|, |dx|))` of `shoulder_mid → hip_mid`; 0° = perfectly hinged (torso horizontal), 90° = upright |
| `left_wrist_hip_dist` / `right_wrist_hip_dist` | `distance(wrist, same-side hip)` in image units |
| `mean_wrist_hip_dist` | mean of the two finite sides — the bilateral pull-depth signal |
| `wrist_hip_dist_shoulder_norm` | `mean_wrist_hip_dist / shoulder_width` — **diagnostic only, no rule fires on it** (§4.5) |
| `elbow_height_asymmetry` | `abs(y_left_elbow − y_right_elbow)` |
| `shoulder_tilt` | `abs(y_left_shoulder − y_right_shoulder)` |
| `wrist_travel_asymmetry` | `abs(left_wrist_hip_dist − right_wrist_hip_dist)` — **diagnostic only** (§4.4) |
| `wrist_accel_norm` | magnitude of the second time-derivative of the two-wrist midpoint, image units · s⁻² (§4.6) |
| `trunk_angle_speed_deg_s` | `abs(d/dt trunk_angle_from_horizontal_deg)`, deg · s⁻¹ (§4.6) |
| `shoulder_width` | `distance(11, 12)` — normalizer, emitted for diagnostics |

**Why `trunk_angle_from_horizontal_deg` uses `|dy|` and `|dx|` rather than signed components.**
Both absolutes make the metric independent of which way the subject faces *and* of which of the
two points is higher in the image — the same reasoning `lunge_compute_raw` applies to
`pelvis_tilt_signed_deg`'s `|dx|`. In a bent-over row the shoulders are above the hips
throughout, so no real sign information is discarded. A signed form would flip by 180° when the
lifter turns around and would make the rule's "torso became more upright" test mean the opposite
thing for the other facing.

**Frame validity — one dropped landmark silences every Row rule for that frame.** `required` is
both shoulders, both elbows, both wrists, both hips. As in `pushup_compute_raw`,
`ohp_compute_raw` and `lunge_compute_raw`, a frame missing any one of them is marked
`valid=False` and carries no metrics at all, so every rule masking on `frame.valid` goes silent
for it. An unmeasurable frame is refused wholesale rather than degraded. This gets the same
module-level documentation the other three received.

**Landmark indices** (`LEFT_ELBOW = 13`, `RIGHT_ELBOW = 14`, `LEFT_WRIST = 15`,
`RIGHT_WRIST = 16`) are defined locally, matching `overhead_press.py`, because `geometry.py`
exports only the lower-body and shoulder/hip constants.

### 4.4 Rules

Four rules. Every fire threshold below is the parent spec's; **every severity ramp is a
rule-level choice**, because — unlike the Lunge section, which states its ramps explicitly —
the Row section states **no severity ramp for any of its five faults**. This is the Push-up
situation, and it takes Push-up's already-argued convention rather than inventing a new
rationale per rule: *the ramp endpoint is 2.5× the fire threshold, documented as a
display/ranking curve rather than a cited quantity* (`pushup.rule_hip_sag`, ramp 0.06 → 0.15).

| fault_id | fire threshold (SPEC) | severity ramp (RULE-LEVEL) | phase scope | view handling |
|---|---|---|---|---|
| `row_torso_rising` | trunk angle at peak − baseline > 15° | 15° → 37.5° | `peak` | downgrade ×0.65 outside `{side, front_oblique, rear_oblique}` |
| `row_incomplete_rom` | `mean_wrist_hip_dist` > 0.12 **or** `max_elbow_angle` > 100° at peak | 0.12 → 0.30; 100° → 140° | `peak` | downgrade ×0.65 outside `{side, front_oblique, rear_oblique}` |
| `row_momentum_jerk` | peak `wrist_accel_norm` > 3× this rep's median over `pull` | ratio 3 → 7.5 | `pull` | no gate, observability `medium` in all views |
| `row_asymmetric_pull` | `elbow_height_asymmetry` > 0.05 **or** `shoulder_tilt` − baseline > 0.04 | 0.05 → 0.125; 0.04 → 0.10 | `peak` | downgrade ×0.65 outside `{front, front_oblique, rear, rear_oblique}` |

Notes on individual rules:

- **`row_incomplete_rom` reads `max_elbow_angle`, the LESS-flexed arm.** The spec's condition (b)
  is "elbow flexion at peak: elbow_angle > 100° at the top = pull not completed" and names no
  side. Taking the less-flexed arm is the conservative reading — a rep is incomplete if *either*
  arm fell short — and it is the deliberate opposite of `pushup_shallow_depth`'s inherited
  more-flexed reading, which that rule's docstring already flags as the generous one. The choice
  is recorded in-code as a rule-level reading of an under-specified spec line, and 100 → 140° is
  taken verbatim from `pushup_shallow_depth` rather than re-derived, so the two elbow-ROM ramps
  in this codebase cannot drift apart.
- **`row_asymmetric_pull` does not fire on `wrist_travel_asymmetry`.** The spec's heuristic
  mentions the wrist-travel term but gives it **no threshold**, unlike the other two. Inventing
  one would be a fabricated fire criterion, so the metric is emitted and carried in the
  detection's `evidence` as a diagnostic while the firing rests on the two terms the spec
  actually quantifies.
- **Direction is part of the verdict for `row_asymmetric_pull`**: `evidence` records which side
  was high, since the coaching cue is side-specific. Following `pushup_hip_sag`'s precedent,
  `score_values` is the absolute series so `build_detection` nominates the genuinely worst frame.

**Citations** are copied verbatim from the parent spec at implementation time, never recalled
from memory:

- `row_torso_rising` — Saeterbakken A et al. Int J Sports Med (2015) PMID 26134664; Owens LP et
  al. IJSPT (2026) PMC13232157.
- `row_incomplete_rom` — Fischer J et al. J Electromyogr Kinesiol (2025) PMID 40513198; Padovan
  R et al. J Funct Morphol Kinesiol (2025) PMC12821611.
- `row_momentum_jerk` — Padovan R et al. PMC12821611; `data/rag/docs/row_wiki.txt` as
  supplementary descriptive support only.
- `row_asymmetric_pull` — Saeterbakken A et al. PMID 26134664; Padovan R et al. PMC12821611.

### 4.5 The spec's thresholds are in raw image units, and that is scale-dependent

`0.12` (wrist-to-hip), `0.05` (elbow asymmetry) and `0.04` (shoulder tilt) carry no stated
normalizer. The reading taken here is that they are **raw MediaPipe normalized image units**,
on internal evidence: the same parent spec says "normalized by shoulder width `dist(11,12)`"
explicitly where it means that (Band Pull Apart's `incomplete_horizontal_abduction_rom`), so
the absence of a normalizer in the Row entries is meaningful rather than an omission.

The honest cost: image-unit thresholds are **camera-distance dependent**. The same rep filmed
from further away produces a smaller body, smaller distances, and a rule that fires less. That
is the spec's construction, implemented as written; `wrist_hip_dist_shoulder_norm` is emitted
alongside as a **scale-free diagnostic that no rule fires on**, so a future validation can read
whether the scale-dependent threshold or the scale-free one separates better without any
threshold having been moved in the meantime.

### 4.6 `row_momentum_jerk` is an event rule, and it breaks two shared conventions

The other three rules test a *sustained state*. A jerk is a **transient**, and two pieces of the
shared framework are built for the former.

**(a) `ctx.min_frames` would filter out exactly the event being detected.** `run_detector` sets
`min_frames = max(3, ceil(fps × 0.20))` — 6 frames at 30 fps — and every existing rule passes it
to `contiguous_true_segments`. A genuine bar-yank spike lasts 1–3 frames. Requiring a fifth of a
second of *sustained* jerk contradicts the fault's definition, so this rule fires as a **per-rep
event**: it locates the peak-ratio frame and emits one detection over the contiguous
above-threshold frames around it, with **no minimum-duration gate**. The deviation is deliberate
and gets its own in-code paragraph naming `min_frames` and why it is not used.

**(b) Median smoothing must act on the derivative, not on the position.** `run_detector` applies
`centered_median(..., window=5)` to **every** key in `metric_keys`. A 5-frame median over a
*position* series flattens the acceleration transient before the rule ever sees it. So
`wrist_accel_norm` and `trunk_angle_speed_deg_s` are computed as derivatives **inside
`row_compute_raw`, from raw per-frame landmark positions**, and the framework's median filter
then acts on the derivative series — a defensible low-pass on the quantity of interest rather
than an erasure of it. Derivatives are central differences over the `fps` that
`row_compute_raw` already receives, and the first and last frames are `NaN` rather than
one-sided (a one-sided estimate at a boundary carries a different bias and would be silently
mixed into the same series).

Three limitations, stated up front:

1. **The threshold is self-normalizing and is expected to over-fire.** "3× the rep's median
   concentric acceleration" compares a peak against a median that includes the near-zero
   accelerations at both ends of the pull. A perfectly controlled rep with an ordinary
   bell-shaped velocity profile can exceed 3×. There is no labeled row data to measure this
   against (§2) and threshold tuning is off the table by standing decision, so the rule ships
   spec-faithful with the expected failure mode named — the same treatment
   `lunge_pelvic_drop`'s split-stance foreshortening bias received.
2. **On a fallback path the normalization silently changes meaning.** "The rep's median" becomes
   "the whole clip's median over all `pull` frames" when no rep was segmented. The rule still
   runs; the evidence payload records the frame count the median was taken over so a reader can
   tell which case they are looking at.
3. **A stable frame rate is assumed and never verified.** `ctx.fps` is a single scalar and
   nothing in the pipeline checks inter-frame spacing. Every acceleration number inherits that
   assumption.

**The spec's second, OR'd condition is degenerate and is not implemented as an OR.** It reads
"OR if a simultaneous trunk-angle velocity spike co-occurs **with the wrist spike** (heave)" —
i.e. its own text requires the wrist spike that the first condition already tests, so it
describes a strict *subset* of what condition one fires on and can never widen the fire set.
Implementing it as a second disjunct would add a branch that is unreachable by construction.
It is therefore implemented as **evidence, not as a fire condition**: when the rule fires,
`trunk_angle_speed_deg_s` is tested against its own 3× median over the same frames and the
result is recorded as `evidence["trunk_heave"]`, which distinguishes an arms-only yank from a
whole-body heave for the coaching cue without changing whether anything fires.

One rule-level measurability guard, in the category `pushup.py` documents as
"can only ever SILENCE": if the median acceleration over the `pull` frames is at or below a
degenerate floor (a window in which the wrists barely moved), the ratio is meaningless — every
frame divides by ~0 — and the rule emits nothing rather than a confident maximum-severity
detection on a stationary lifter.

### 4.7 View handling: downgrade, never gate

`estimate_view_for_pose` is called with `allow_front=False` in the production path
(`src/pose/view_estimation.py:341`), so the only reachable labels are
`{side, rear, rear_oblique, unknown}` — `front` and `front_oblique` are **never emitted**. A
rule gated positively on an unreachable label is permanently silent, which is what happened to
`pushup_elbow_flare`.

**No Row rule takes a hard view gate**, and the reachability facts are why:

- The measured precedent recorded in the Lunge design doc is that across the 45 real pose JSONs
  in this repo the estimator emitted `side` **exactly once**, and that verdict came from a
  fixture since removed; the corpus is 30 `rear_oblique`, 13 `rear`, 2 `unknown`. Anything
  hard-gated on `side` would ship silent.
- `row_torso_rising` and `row_incomplete_rom` need a lateral component to read trunk pitch and
  pull depth, but `rear_oblique` — the label production actually produces most often — supplies
  it. They follow `squat.rule_knees_inward`'s downgrade precedent (observability `high` → `medium`,
  confidence ×`VIEW_UNAVAILABLE_CONFIDENCE_SCALE`) rather than `rule_knees_forward`'s gate.
- `row_asymmetric_pull` is **facing-free by construction**: `|y13 − y14|` and `|y11 − y12|` are
  magnitudes of image-y differences, which read identically from in front of or behind the
  subject. `rear` and `rear_oblique` therefore earn the spec's `high` rating, exactly as
  `lunge.rule_knee_valgus` argues for the midline-relative valgus proxy.
- `row_momentum_jerk` needs only a visible wrist; the spec rates it `medium` in any view and no
  view earns better, so its observability is `medium` unconditionally and no discount applies.

  **Challenged in Task 4's review, and the objection is worth stating rather than burying.** The
  reviewer's point: `unknown` in production does not mean "a confirmed non-lateral view", it
  means **the view estimator failed**, and `rule_torso_rising`'s own docstring draws exactly that
  distinction while this rule does not — so treating `unknown` identically to a confirmed
  `rear_oblique` conflates "a view where the wrist happens to be visible" with "we do not know
  what we are looking at."

  **The reason the behavior stands anyway:** this rule's stated precondition — the pulling wrist
  is visible — is **not** inferred from the view label at all. It is enforced upstream by
  `row_compute_raw`'s validity gate, which lists BOTH wrists in `required` and marks the frame
  `valid=False` if either drops below the visibility threshold. Every frame this rule scores
  therefore has both wrists visible **by construction**, whatever the view estimator did or did
  not manage to classify. A discount keyed on `unknown` would be penalizing a confidence the rule
  never depended on.

  What that argument does NOT cover, and what an eventual validation should measure: whether an
  `unknown` verdict CORRELATES with degraded landmark quality generally, such that accelerations
  computed on those clips are noisier even with both wrists nominally visible. That is an
  empirical question about the view estimator, not a geometric one about this rule, and no data
  in this repository can answer it today.

The `×0.65` multiplier is `pose_rule_detector.VIEW_UNAVAILABLE_CONFIDENCE_SCALE`, **imported**
rather than re-typed, so a change to the shared constant cannot silently skip this module.

---

## 5. KG queries — resolved before being written, not after

The parent spec's OHP pass shipped three `kg_query` strings that resolved to nothing, because
only `resolve_nodes` was checked and not `retrieve_graph_context` (the function production
actually calls). Every string below was checked with
`retrieve_graph_context(query, movement="Row")` against `data/kg/sports_kg_v3.graphml` on
2026-08-01 and returned a `Row:`-scoped seed with at least one **non-empty** bucket:

| rule | `kg_query` | resolved seed | non-empty buckets |
|---|---|---|---|
| `row_torso_rising` | `Trunk Extension` | `Row:Trunk Extension` (Fault) | phases, corrections (`Maintain Neutral Spine`), quality_impacts (`Core Stability`) |
| `row_incomplete_rom` | `Scapular Protraction` | `Row:Scapular Protraction` (Fault) | evidence (`Anterior Translation Of Scapulae`), related_actions |
| `row_momentum_jerk` | `Loss Of Neutral Body Position` | `Row:Loss Of Neutral Body Position` (Fault) | phases, evidence (×3 alignment signals), corrections, quality_impacts, related_actions |
| `row_asymmetric_pull` | `Asymmetry` | `Row:Asymmetry` (Fault) | phases, risks (`Shoulder Injury`, `Injury Risk`), related_actions |

Two of these are **deliberate deviations from the most obvious name**, recorded here so the
in-code Step 0 comment is a restatement rather than a new claim:

- `row_momentum_jerk`: the obvious candidate `Compensatory Movements` resolves to a real
  `Row:`-scoped Fault node whose buckets are **entirely empty** — the exact OHP failure mode.
  `Loss Of Neutral Body Position` is the richest on-topic node, and its evidence signals ("Head
  Not Aligned With Trunk And Hip", "Trunk Not Aligned With Head And Hip", "Hip Not Aligned With
  Head And Trunk") are a direct description of a whole-body heave.
- `row_asymmetric_pull`: `Interlimb Asymmetry` and `Muscle Strength Asymmetry` both resolve, but
  the first is scoped to `Unilateral Cable Row` and the second carries only a generic
  `Injury Risk`. `Row:Asymmetry` is the one whose buckets name the phases the fault occurs in
  and a specific `Shoulder Injury` risk.

The resolution transcript is reproduced in `row.py`'s Step 0 comment block, following
`lunge.py`'s precedent, so the reasoning survives without the reader re-running the graph.

---

## 6. Testing

Mirroring `tests/test_lunge.py`, in a new `tests/test_row.py`:

- **Metrics**: each metric controlled by fixture construction, not hardcoded — a fixture that
  hinges the torso a known number of degrees must produce that number.
- **Phases**: `setup`/`pull`/`peak`/`lower` on a synthetic rep, plus the empty-clip,
  no-finite-signal and invalid-frame fallbacks.
- **Every rule, firing AND not firing**, with **boundary fixtures just inside and just outside
  each threshold**. Boundary fixtures use constant-value frames so `run_detector`'s median
  smoothing is a no-op and asserted severities are exact — the OHP review found 5 of 10
  threshold mutants surviving because every fixture sat at an extreme.
- **One exact severity assertion per rule**, pinning the rule-level ramps of §4.4 so a silent
  change to one is a test failure.
- **The baseline test that the single-rep fixtures structurally cannot provide**: a multi-rep
  fixture whose rep 1 is clean and rep 2 rises/tilts, asserting the fault is attributed to rep 2
  and that rep 2's baseline came from rep 2's own `setup` frames. This is the §4.2 analogue of
  Lunge's alternating-lead guard.
- **`row_momentum_jerk` specifically**: a test proving a 1–3 frame spike **survives**
  `run_detector`'s median filter and fires (the §4.6(b) claim, verified rather than asserted),
  and a test proving a smooth controlled rep does **not** fire, which is the §4.6(1) over-fire
  risk pinned as a live expectation rather than a hope.
- **Registry**: `tests/test_movement_registry.py` gains `get_detector("Row")` and
  `get_detector("row")`, and `ROW_METRIC_KEYS` is pinned as a two-way match with what
  `row_compute_raw` emits (a key the tuple omits is dropped by `run_detector` and read back as
  `NaN`).
- **Regression**: the existing squat byte-for-byte gate must still pass. Squat is production.

Verification set before claiming done: `.venv\Scripts\python.exe -m pytest tests/ -q`, then
`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.

---

## 7. Honesty constraints

Binding on the writeup and the in-code documentation, not optional caveats.

1. **Nothing here is validated.** §2. "Beta" is a factual statement about this detector.
2. **Four rules, not five.** The fifth is unimplementable from MediaPipe landmarks (§3) and is
   reported as a parent-spec defect, with the rejected substitutes named.
3. **All four severity ramps are invented.** They are display/ranking curves following
   Push-up's convention; no literature or spec line fixes any of them. In-code docstrings keep
   "FROM THE SPEC" and "RULE-LEVEL CHOICE" as separate, labeled categories.
4. **`row_momentum_jerk` is expected to over-fire** (§4.6). If it ever gets data and fires on
   both clean and jerky reps at similar rates, the honest conclusion is that the self-normalizing
   threshold does not discriminate — not that rows are universally jerky.
5. **The image-unit thresholds are camera-distance dependent** (§4.5) and were implemented that
   way on purpose, with a scale-free diagnostic emitted beside them.

---

## 8. Out of scope

- Deadlift (in progress by the repository owner) and Band Pull Apart, Group C's other movement.
- Any validation harness, or extraction of new video (§2).
- Flipping `validated` to `True` for Row, or for any other movement.
- Implementing a substitute rounded-spine metric under a new `fault_id` (§3).
- Fixing the parent spec's known `heel_rise` sign defect or the three dangling OHP `kg_query`
  strings.
