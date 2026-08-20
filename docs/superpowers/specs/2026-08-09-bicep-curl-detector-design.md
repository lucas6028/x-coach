# Bicep Curl Rule Detector — Design Spec

**Status:** design spec · **Date:** 2026-08-09
**Movement:** Bicep Curl (standing, dumbbell in each hand) · **Detectors after this one:** 8/16
**Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`
§Bicep Curl (lines 843–888)

---

## 1. Purpose and why Bicep Curl is next

The parent spec's Group D opens with Bicep Curl, and Groups A–C are exactly the seven
detectors already registered (Squat, Overhead Press, Push-up, Lunge, Deadlift, Row, Band
Pull Apart). This is the eighth, and it ships the same way every one since Push-up has:
cited rules, `validated=False`, Beta in the UI, no frontend edit.

The parent spec gives Bicep Curl four rules. **Three are implemented and one is withdrawn**
(§3). The three that ship are:

| # | fault_id | Cue |
|---|---|---|
| 1 | `curl_elbow_drift_forward` | Upper arm leaves the vertical hang — the elbow stops being a fixed pivot |
| 2 | `curl_trunk_swing_momentum` | Trunk swings / leans to heave the load |
| 3 | `curl_incomplete_rom` | The elbow never straightens at the bottom, or never closes at the top |

### 1.1 What makes this movement different from the seven already shipped

**Every rule is a single-joint isolation cue, and that inverts the usual observability
problem.** The seven shipped movements fail in ways that need a multi-segment reading — a
knee-to-ankle-to-toe projection, a shoulder-hip-ankle plank line, a scapular retraction the
sensor cannot see at all. The curl's faults are all "did one segment stay where it belongs",
which MediaPipe resolves well. There is no `pushup_scapular_winging`-class sensing failure in
this movement: the one rule that does not ship fails on its *citation*, not its sensor (§3).

**Its rep signal is the cleanest of the eight.** Measured on Fit3D 3D mocap ground truth
(`joints3d_25`, `dumbbell_biceps_curls`, 8 subjects × 5 reps = 40 reps), the elbow angle
excurses 39°→162°, a ~123° swing — larger and better-separated than any shipped movement's
signal. Left and right elbow angles correlate **r = 0.992–0.996** (mean 0.996) across all 8
subjects, so the two arms move as one unit and averaging them is a noise reduction rather
than a distortion (§4.2).

**Its literature base is one paper, and it is a protocol paper, not a fault paper.** Parpa et
al. 2025 (PMC12550948, RAG doc `PMC12550948_bicep_curl_dumbbell_vs_cable_emg.txt`) is an EMG
comparison of dumbbell vs Bayesian cable curls. It backs three of the four rules — but only
through its *proper-execution protocol*, which defines correct form and therefore defines
deviation from it. It never studies a fault. Every citation_support string in this detector
is a protocol quote, and §7 requires that this be said in-code rather than dressed up.

---

## 2. There is no labeled ground truth for Bicep Curl, and `validated` stays `False`

REHAB24-6 ships arm abduction, arm VW, table push-ups, leg abduction, lunge and squats — no
bicep curl. Fit3D **does** ship curls with 3D mocap ground truth and rep boundaries
(`dumbbell_biceps_curls`, plus `drag_curl`, `dumbbell_hammer_curls`,
`dumbbell_curl_trifecta`), but **no binary correct/incorrect label on any rep**. So no
fire-rate/AUC-against-correctness check is possible, and `BICEP_CURL_DETECTOR.validated`
stays at its default `False`. Beta is the factual label.

**What the Fit3D reps CAN do, and what that is worth.** Forty unlabeled reps of 3D ground
truth cannot validate a threshold, but they can answer a strictly weaker question: *does this
threshold fire on ordinary reps performed by people trying to do the movement correctly?*
Measured on the per-rep elbow-angle extremes:

| Spec threshold | Measured over 40 reps | Reps that would fire |
|---|---|---|
| incomplete **extension**: `max(elbow_angle) < 150°` | per-rep max 149.7°–179.9° (mean 170.2°) | **1 / 40** (at 149.7°) |
| incomplete **flexion**: `min(elbow_angle) > 60°` | per-rep min 27.8°–59.0° (mean 43.2°) | **0 / 40** (worst 59.0°) |

**Both thresholds sit within ~1° of the edge of the observed distribution.** That is recorded,
not repaired — §7's no-tuning rule is absolute, and in any case the finding is ambiguous by
construction: Fit3D carries no correctness label, so the 149.7° rep may genuinely have been a
short rep. What it does establish is that these two thresholds are **sensitive**, and a
future validation should expect fire rates driven by the tail rather than the body of the
distribution.

**One caveat that cuts one way only.** The table above is computed on *3D* joint angles;
production reads *projected image-plane* angles. Perfectly collinear 3D points project to
collinear 2D points, so a genuinely straight arm reads ~180° from any view and the extension
threshold is safe at the limit. But a *nearly* straight arm's bend can be **amplified** by
projection when the bend plane is oblique to the image plane — a true 160° can read
noticeably lower. So the extension term's error is biased **toward firing**, on a threshold
already shown to be sensitive. The flexion term is biased the other way (an amplified bend
reads more flexed, i.e. further from firing), so it errs toward missed faults. Both
directions are stated in-code on the rule that owns them.

---

## 3. `wrist_flexion_curl` is WITHDRAWN — and it is a citation failure, not a sensing failure

The parent spec's fourth Bicep Curl rule is **withdrawn and not implemented**. This project
has two treatments for a rule it will not fire, and picking the right one is the whole
decision:

- **Registered-but-permanently-silent** (`pushup.rule_scapular_winging`, row's fifth,
  `band_pull_apart.rule_loss_of_scapular_retraction`) means: *real, well-cited fault; the
  sensor cannot see it.*
- **Withdrawn from the parent spec** (OHP bar-path 2026-07-25, deadlift bar-drift 2026-08-01)
  means: *no citation supports the rule as written.*

`wrist_flexion_curl` presents both symptoms at once — its observability is rated `low` on
every view, *and* its own `citation_support` in the parent spec ends "the injury-risk
magnitude of wrist flexion is UNVERIFIED in this source." The tie is broken by reading the
source rather than the parent spec's paraphrase of it.

**Parpa PMC12550948 was read in full. It never discusses wrist flexion.** Every wrist- and
grip-related statement in the paper is about **forearm rotation** (supination / pronation) or
**grip type**, which is a different degree of freedom from flexion/extension:

- "It primarily involves elbow flexion accompanied by either dynamic or mostly isometric
  shoulder flexion and **wrist supination or pronation**." (line 18)
- "biceps brachii and brachioradialis activation were the highest with the **supinated grip**
  during the ascending phase" (line 33, citing Coratella 2023)
- The protocol prescribes "holding a dumbbell in each hand in a **supinated grip**." (line 75)

Nowhere does the paper state that the wrist bending into flexion is a fault, a cheat, or a
loading risk. The parent spec's rule asserts "Wrist flexion recruits wrist flexors and can
strain the wrist joint" and sets a 30° threshold; **neither the mechanism nor the number
appears in the cited source.** That is the OHP/deadlift pattern exactly: a threshold with no
provenance attached to a citation that does not measure it.

The observability problem is real too — landmarks 19/20 are small, and a dumbbell occludes
the hand for much of the rep — but it is not the *reason*. Had the citation held, this rule
would have shipped silent. It does not, so it is withdrawn, and the parent spec gets a
WITHDRAWN blockquote in the style of the other two rather than a silent deletion.

**Open spec question, recorded not resolved:** does the Bicep Curl rule set want a genuine
wrist rule? It would need (a) a source that measures wrist *flexion* under curl load with a
number, and (b) a hand-landmark reading that survives dumbbell occlusion. Neither exists
today.

**The KG node is not the gap.** `Bicep Curl:Wrist Flexion Under Load` resolves with a
non-empty `corrections` bucket (`Wrists In Line With Forearms`). The node stays; nothing in
this detector points at it.

---

## 4. Detector design

### 4.1 The fault_id namespace: `curl_*`, and the parent spec's ids are renamed

The parent spec gives Bicep Curl **unprefixed** ids: `elbow_drift_forward`,
`trunk_swing_momentum`, `incomplete_rom`, `wrist_flexion_curl`. Every movement after Squat
prefixes (`lunge_*`, `deadlift_*`, `pushup_*`, `ohp_*`, `row_*`, `bpa_*`), and the collision
is not hypothetical: `row_incomplete_rom` and `bpa_incomplete_rom` both already exist, and a
third bare `incomplete_rom` would be indistinguishable from either in
`merge_by_fault`, in the analyses table, and in the frontend's `byFault` map — all of which
key on `fault_id` alone with no movement qualifier.

**Shipped ids are `curl_elbow_drift_forward`, `curl_trunk_swing_momentum`,
`curl_incomplete_rom`.** The rename is annotated in the parent spec (a NOTE, not a
WITHDRAWN — nothing about the rules themselves changes) so a reader who arrives at the
original wording cannot silently re-introduce the bare ids.

`FAULT_LANDMARKS` in `frontend/src/lib/pose.ts:55` covers only the five squat faults and gets
no entry here. That is a **pre-existing** gap shared by all six non-squat detectors, not one
this movement introduces; the skeleton overlay simply highlights nothing for these faults.
Out of scope (§8), recorded in TODO.md.

### 4.2 Rep segmentation: `avg_elbow_angle`, polarity `min`, `rep_start="extended"`

The rep is `arms extended at the sides → flexion → peak → controlled lowering → extended`, so
the signal peaks at its **minimum** and the rep **starts extended**. Both match the majority
of shipped detectors and neither is a novel setting.

**Why `avg_elbow_angle` rather than `min_elbow_angle`** (used by Push-up, Row, Band Pull
Apart). Those three take an extremum across the two arms because their rules need the worse
arm. For *segmentation* the question is different — which series has the cleanest excursion —
and here it is answered by measurement, not preference: L/R elbow angles correlate
**r = 0.992–0.996** across all 8 Fit3D subjects. The arms are in phase, so the mean is the
same excursion with the per-arm landmark noise halved, while `min` would inherit whichever
arm was noisier on each frame. This mirrors `overhead_press`, which uses `avg_elbow_angle` for
the same reason.

**Stated limitation: alternating curls would defeat this signal, and worse than they defeat
`min`.** If the arms alternate, their mean is roughly *constant* — the excursion cancels — and
`segment_reps` would find nothing and fall back to whole-clip analysis. `min_elbow_angle`
would at least oscillate (at twice the true rep rate, which is its own problem). This is not
corrected: Parpa's protocol is bilateral ("holding a dumbbell in each hand"), all 40 Fit3D
reps are bilateral, and choosing the signal for a variant neither the citation nor the data
contains would be trading a measured decision for a guessed one. The failure mode is
degradation to the pre-existing whole-clip path, not a wrong verdict.

**`DEFAULT_MIN_REP_SECONDS = 0.4` is safe here, and this is measured rather than asserted.**
Band Pull Apart's Task 6 recorded that this floor has never been checked against real cadence
— the same gap that forced High Knee's segmentation to ship disabled. Measured for this
movement from Fit3D `rep_ann.json` (`s03/s04/s05/s07/s08/s09/s10/s11`, 40 reps, 50 fps
verified by `ffprobe`): **1.92–3.68 s/rep, mean 2.54 s** — 4.8× to 9.2× the floor. No
`min_rep_seconds` override. The residual risk is the same one Band Pull Apart recorded and is
not removed by a larger n: these are 8 subjects performing deliberately for a mocap capture,
and a real user can be faster and sloppier. There is no citable cadence figure to set an
override against, so none is set.

### 4.3 Phases

`setup → concentric → peak → eccentric`, assigned over the **per-rep** slice
`run_detector` hands to `assign_phases`, following `row_assign_phases` and
`band_pull_apart_assign_phases` exactly:

- `setup` — the first 15% of the rep window. Because `rep_start="extended"`, these are the
  **arms-extended** frames, which is precisely where incomplete extension is visible.
- `peak` — the most-flexed 30% of the rep (the 30th percentile of `avg_elbow_angle` and
  below; note the polarity inversion versus Band Pull Apart's *widest* 30%).
- `concentric` / `eccentric` — before / after the most-flexed frame.
- `unknown` — any invalid frame, checked **before** the setup cutoff, so an occluded frame in
  the opening 15% is not labelled `setup` (which matters because `_setup_baseline` reduces
  over exactly those frames).

**The extension term's `setup` window is narrow enough to silence it, and the boundary was
measured rather than assumed.** Two framework interactions bite:

1. `setup` is 15% of the rep window and `contiguous_true_segments` needs
   `min_frames = max(3, ceil(0.20·fps))` consecutive frames, so the term needs
   `0.15·fps·T ≥ 0.20·fps` — i.e. **T ≥ 1.333 s per rep**, fps-independent above 15 fps. Against
   the measured 1.92–3.68 s/rep, the fastest real rep sits at **1.44× this floor**. That is a
   materially tighter constraint than `DEFAULT_MIN_REP_SECONDS` (0.4 s), which is what §4.2's
   cadence figure was checked against — the 4.8× margin recorded there is *not* the binding one
   for this rule.
2. Worse, `segment_reps` trims each window to the signal's **excursion**. A lifter who pauses
   with the arms extended between reps has that hold cut away, so `setup` covers mid-range
   frames rather than the bottom, and the shorter window can push `setup` back under
   `min_frames` on its own. Measured on a 63-frame-per-rep fixture with a between-reps hold:
   windows came out **37 frames**, `setup` **5 frames** (one short of 6), and the frames it did
   cover read **84–110°** instead of the true 130° bottom — so the term both measured the wrong
   part of the rep and then reported nothing.

**So whether this term fires depends on the shape of the rep, not only its duration.** It is
fragile, not dead: on a smooth excursion it fires at 1.92 s and 2.54 s/rep. Not repaired — the
15% setup fraction and `min_frames` are shared framework constants and neither has a cited basis
to move for one movement. The failure mode is a missed fault, never a false one. Both facts are
pinned by `PhaseWindowWidthTest` and
`EndToEndSegmentationTest::test_rep_trimming_can_silence_the_extension_term`, and recorded in
TODO.md.

**Why incomplete extension is scoped to `setup` and not to the end of the eccentric.** The
lifter who fails to lower all the way finishes rep *N* short — but reps are contiguous, so
that same short position **is** rep *N+1*'s `setup`, and the rule catches it there. The one
rep this misses is the last one in the clip. Stated, not corrected: the alternative is
scoping the extension term across `eccentric` too, which spans the whole mid-range where a
90° elbow is *correct*, and would fire on every rep.

### 4.4 The setup baseline

`_setup_baseline(core, key)` — the median of `key` over this window's valid `setup` frames,
NaN when there are none — is lifted from `row._setup_baseline` / `band_pull_apart`, including
its stated limitation: `setup` is 15% of an already-trimmed rep window, so on a short rep it
can be 1–2 frames and may already overlap loaded frames, biasing the measured change
**smaller** than the true one (a missed fault, never a false one).

Only `curl_trunk_swing_momentum`'s second term consults it. Its first term (within-rep
oscillation range) and all of `curl_incomplete_rom` and `curl_elbow_drift_forward` are
absolute readings, so a NaN baseline silences one term and nothing else.

### 4.5 Metrics — and the metric layer contains no thresholds

`bicep_curl_compute_raw` / `bicep_curl_assign_phases` emit scale-free per-frame quantities and
phase labels only. The only constant either may define is `_DEGENERATE_LENGTH = 1e-6`, a
division-by-zero guard. Every number that decides anything lives in a `rule_*` function.

| Metric | Definition |
|---|---|
| `left_elbow_angle`, `right_elbow_angle` | `angle(shoulder, elbow, wrist)`, 180° = straight |
| `avg_elbow_angle` | mean of the two; NaN-tolerant. **The rep signal.** |
| `min_elbow_angle` | the more-flexed arm — drives the incomplete-**flexion** term |
| `max_elbow_angle` | the less-flexed arm — drives the incomplete-**extension** term |
| `left_upper_arm_lean_deg`, `right_upper_arm_lean_deg` | unsigned angle of `shoulder→elbow` from image-vertical-down |
| `max_upper_arm_lean_deg` | the worse arm's lean — drives rule 1 |
| `trunk_lean_image_signed_deg` | signed pitch of `hip_mid→shoulder_mid` from vertical, **image frame, not facing-corrected** |
| `shoulder_width` | `dist(11,12)`; the normalizer, emitted so a scale question is answerable later |
| `upper_arm_length` | mean `dist(shoulder, elbow)`; emitted as a diagnostic only — see §4.7 |

**Which arm each ROM term reads, and why they differ.** `max_elbow_angle` (the *less*-flexed
arm) drives extension and `min_elbow_angle` (the *more*-flexed arm) drives flexion. Both
choices are the **generous** reading — the rep is called incomplete only if *both* arms fell
short at that end. The parent spec names no side for either term. This is deliberately the
opposite of `row.rule_incomplete_rom`, which takes the conservative reading and says so; the
reason for diverging is that this rule's two terms are already shown (§2) to sit ~1° from the
edge of the real-rep distribution, and pairing a sensitive threshold with a conservative
side-selection would compound two independent pushes toward false firing. Documented in-code
at the constants, since it is a rule-level reading of an under-specified spec line and not
something a reader should have to infer.

**Required landmarks, and the all-or-nothing rule.** `required` is both shoulders, both
elbows, both wrists and both hips. If `visible_point` drops any **one**, the frame is
`valid=False` and carries no metrics at all, so *every* rule goes silent for that frame — not
just the one whose input vanished. This mirrors `pushup_compute_raw`, `ohp_compute_raw`,
`lunge_compute_raw`, `row_compute_raw` and `band_pull_apart_compute_raw`: an unmeasurable
frame is refused wholesale rather than degraded.

### 4.6 Rules

**Rule 1 — `curl_elbow_drift_forward`.** Fires when `max_upper_arm_lean_deg > 25°`
(**FROM THE SPEC**) on any `concentric` frame — the phase scope is the spec's own ("at any
frame during **concentric**"), and `peak` is deliberately **not** added even though drift is
largest there, because widening a phase scope is a rule-level change that would make the rule
fire on reps the spec's wording exempts. Severity ramps 25° → 62.5°
(**RULE-LEVEL**, 2.5× the fire threshold, the `pushup.rule_hip_sag` convention).
KG query `Elbow Drift Forward`.

**Rule 2 — `curl_trunk_swing_momentum`.** A genuine disjunction of two independent terms,
both **FROM THE SPEC**:
- (a) within-rep oscillation `max(trunk_lean) − min(trunk_lean) > 12°`, reduced over the rep's
  valid frames;
- (b) `|trunk_lean − setup_baseline| > 10°` on a `concentric` frame (the spec's own scope:
  "backward lean **during concentric** exceeds the setup baseline by > 10°").

They are not nested: a rep that leans 11° one way and holds fires (b) but not (a); a rep that
oscillates ±7° around its baseline fires (a) but not (b). `evidence["primary_label"]` records
which drove the verdict, compared by severity directly and never by branching on a
categorical `fired_on` string — the trap `row.rule_incomplete_rom`'s docstring documents
hitting. Ramps 12°→30° and 10°→25° (**RULE-LEVEL**, 2.5×). KG query `Using Momentum`.

**Rule 3 — `curl_incomplete_rom`.** A **phase-conditional** disjunction, both terms
**FROM THE SPEC**:
- at `setup`: `max_elbow_angle < 150°` → incomplete **extension**;
- at `peak`: `min_elbow_angle > 60°` → incomplete **flexion**.

This is the first rule in the codebase whose two terms live in *different phases*. Row's and
Band Pull Apart's ROM cues both sit at the peak; a curl's two ROM failures sit at opposite
ends of the rep by definition. One mask with a phase-conditional score function keeps a
single `contiguous_true_segments` pass, and the phases are disjoint and non-adjacent
(`concentric` separates them), so no segment can span both terms. Ramps 150°→110°
(**RULE-LEVEL**, the 40° width taken from `pushup.rule_shallow_depth` so the two elbow ramps
cannot drift) and 60°→100° (**RULE-LEVEL**, the same 40° width). KG query
`Incomplete Range Of Motion`.

### 4.7 View handling — rules 1 and 2 gate, rule 3 downgrades

**Measured production reality, re-measured for this spec rather than inherited.** Running
`estimate_view_for_pose(path, allow_front=False)` over all 49 files under
`data/runtime/pose_json` on 2026-08-09: **`rear_oblique` 37, `rear` 9, `unknown` 3, `side`
never.** (This reproduces the figure Band Pull Apart's design doc recorded; it is re-stated
here because that doc itself warns about a figure inherited from the Lunge doc that no longer
reproduced.) `front` and `front_oblique` are unreachable under `allow_front=False`
(`src/pose/view_estimation.py:14-16`).

**The consequence is uncomfortable and is stated rather than hidden: `side` — the view the
parent spec rates `high` for all three rules — does not occur in production at all.** No rule
here will earn its spec-rated observability on a real clip. What the gate protects is
narrower: that a rule reads the *right plane*, not that it reads it well.

**Rules 1 and 2 gate out pure `front`/`rear`.** Both measure a **sagittal** quantity. From a
pure rear view the sagittal axis is perpendicular to the image plane, so `upper_arm_lean`
computed there reads *lateral elbow flare* and `trunk_lean` reads *frontal-plane sway* —
different faults, or none. That is not a low-confidence reading of the right quantity (what
the ×0.65 discount exists for); it is a **confident reading of the wrong plane**. Row's design
doc argues "downgrade, never gate" on the grounds that gated rules ship silent, and that
objection does not apply here: the gate leaves `rear_oblique` standing, which is 37 of 49
real pose JSONs. Written in the **negative** (`view_type not in {"front", "rear"}`) so that
`unknown` passes and so the form needs no edit if `allow_front` is ever enabled — the same
construction `band_pull_apart.rule_trunk_extension_compensation` uses.

On the oblique views that survive the gate, the sagittal axis is foreshortened, so a real
drift or lean reads **smaller** than it is: the failure mode is a missed fault, never a false
one. Confidence is scaled by `VIEW_UNAVAILABLE_CONFIDENCE_SCALE` there.

**Rule 3 downgrades and does not gate.** An elbow angle is the right quantity from every
view; obliquity makes it noisier, not different. `high` on `side`, `medium` (×0.65) elsewhere
— matching `band_pull_apart.rule_incomplete_rom`. The direction of the projection error is
recorded on the rule per §2.

### 4.8 The spec's two facing-dependent sub-criteria, and what replaces them

The parent spec's rule 1 says the lean must be "toward the anterior (wrist) side", and its
rule 2's second term says the lean must be "backward". **Both directional qualifiers are
dropped, and the metric is taken unsigned instead.** Three reasons, in order of weight:

1. **The citation backs the undirected claim and not the directed one.** Parpa's protocol is
   "the elbows kept close to the torso throughout the whole movement" and "avoiding trunk
   movements and jerky motions." Neither names a direction. An unsigned departure from the
   setup posture is *exactly* what the source prescribes against; a signed one asserts more
   than the source does.
2. **Recovering "anterior" needs a facing proxy, which needs a threshold with no citation.**
   Band Pull Apart solved its own facing problem with a `wrist_depth_offset` sign and a
   measured `0.02` floor — available only because that movement holds the band in front of
   the torso *by definition*, which pins the sign. A curl's wrists travel from the hips to the
   shoulders and their depth offset changes sign within the rep, so the same construction does
   not transfer. Inventing a different one is the move the OHP bar-path (2026-07-25) and
   deadlift bar-drift (2026-08-01) withdrawals both rejected.
3. **The undirected reading loses almost nothing.** The elbow drifting *backward* out of the
   vertical hang (the drag-curl position) is also a loss of elbow fixation, and a trunk that
   pitches *forward* to start the lift is also momentum. Firing on both is a wider net than
   the spec describes, in the direction the citation supports.

Annotated in the parent spec as a NOTE.

### 4.9 The elbow-displacement disjunct is unreachable, and is not implemented

Parent spec rule 1 offers a second cue: "or if elbow x-displacement anterior of the
shoulder–hip vertical line exceeds `0.5 × upper_arm_length`". Beyond needing the same
anterior direction §4.8 just rejected, **it is the first cue restated in different units, and
strictly weaker**: displacement `= upper_arm_length · sin(lean)`, so the `0.5` threshold is
`lean > arcsin(0.5) = 30°` — always satisfied when the angular term's `25°` is. Any frame the
displacement term catches, the angular term has already caught.

This is the same defect `row.rule_momentum_jerk`'s second condition had (a strict subset of
its first, therefore dead code that read as coverage). Implementing it would add a metric,
a threshold and a branch that can never change a verdict. `upper_arm_length` is still emitted
as a diagnostic so the equivalence stays checkable without re-deriving it. Annotated in the
parent spec as a NOTE.

---

## 5. KG queries — resolved before being written, not after

Each string below was checked against `data/kg/sports_kg_v3.graphml` with
`retrieve_graph_context(query, movement="Bicep Curl")` — the function **production** calls,
not `resolve_nodes`. Observed results, not predicted ones:

| Rule | Query | Resolves to | Buckets |
|---|---|---|---|
| 1 | `Elbow Drift Forward` | `Bicep Curl:Elbow Drift Forward` | **THIN** — only the `HAS_FAULT` backlink |
| 2 | `Using Momentum` | `Bicep Curl:Using Momentum` | `quality_impacts: Forward Momentum` |
| 3 | `Incomplete Range Of Motion` | `Bicep Curl:Incomplete Range Of Motion` | `quality_impacts: Range Of Motion` |

**The one gap is recorded, not masked.** `Elbow Drift Forward` has connectivity 0 — the exact
shape of Band Pull Apart's `Bent Elbows` gap, and a one-line fix in
`scripts/knowledge/stub_general_movements_v3.py:71-78` (the node's fault list is `[]` there).
Because the graphml is gitignored, regenerating it is a deploy step, so the fix is logged
against TODO.md's existing "many faults have no KG node" item rather than made here.

**Rejected substitution, for the same reason Band Pull Apart rejected its own.** Pointing rule
1 at the shared `Range Of Motion` QualityDimension returns a rich bucket set — and its
`corrections` bucket is `Wrapping Surface Adjustment`, meaningless for this movement. A
semantically correct thin card beats a semantically wrong full one.

---

## 6. Testing

`tests/test_bicep_curl.py`, following `tests/test_band_pull_apart.py`:

- **Fixture builder** producing synthetic landmark frames with a controllable elbow angle,
  upper-arm lean and trunk lean, so each rule can be driven across its threshold in isolation.
- **Per-rule threshold pins** — fires just past the spec threshold, silent just short of it,
  for all three rules and both terms of rules 2 and 3.
- **Both ROM terms independently**, since they live in different phases: a rep that fails
  extension only, one that fails flexion only, one that fails both.
- **The unreachability of the displacement disjunct (§4.9)** pinned numerically, so a future
  edit that "restores" it has to confront the arithmetic.
- **View gating**: rules 1 and 2 silent on `front`/`rear`, firing on `rear_oblique` and
  `unknown`, with the confidence discount applied on the obliques; rule 3 firing on all views
  with the discount off `side`.
- **`test_metric_keys_match_the_emitted_metrics_exactly`** — the two-way match between
  `BICEP_CURL_METRIC_KEYS` and what `bicep_curl_compute_raw` emits. A key the tuple omits is
  silently dropped by `run_detector` and read back as NaN by every rule.
- **`EndToEndSegmentationTest`** — a synthetic multi-rep clip segmenting on
  `avg_elbow_angle`, verifying rep count and that per-rep phases are assigned (the check that
  actually verifies the `min` / `extended` interface-design inference).
- **`tests/test_movement_registry.py`** — add `"Bicep Curl": ("avg_elbow_angle", "min",
  "extended")` to the shared rep-signal table.
- **`tests/test_analyze_pose_service.py`** — rotate the stale "unimplemented movement"
  example off Bicep Curl if it names it.

Commands: `.venv\Scripts\python.exe -m pytest tests/ -q` and
`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`. Two backend test
flakes are known-unrelated on this machine; check a failure against a baseline on `main`
before attributing it to this change.

---

## 7. Honesty constraints

- **No threshold tuning.** Every cited number stays as the parent spec states it. §2's finding
  that both ROM thresholds sit ~1° from the edge of the real-rep distribution is **written up,
  not repaired**.
- **Every constant is labeled in-code as exactly one of `FROM THE SPEC` or `RULE-LEVEL CHOICE
  MADE HERE`.** Never blurred. All severity ramps are RULE-LEVEL — the parent spec states no
  ramp for any Bicep Curl fault, and the Lunge section states its ramps explicitly, so the
  absence is meaningful.
- **Citations are copied verbatim from the parent spec at implementation time**, never
  recalled from memory.
- **Parpa is a protocol paper and every citation_support here is a protocol quote.** The
  in-code strings say so. The paper defines correct execution and monitors deviation; it does
  not study any of these faults as faults. That is weaker support than Fukunaga gives Band
  Pull Apart or Ford gives the squat, and stating it is the point.
- **`validated=False`**, with §2's evidence stated at the registration site.
- **No metric is substituted under another metric's fault_id.** The withdrawn wrist rule is
  absent, not re-pointed at something measurable.

## 8. Out of scope

- **Any frontend file.** `/api/movements` derives from the registry; `Bicep Curl` already
  exists in `frontend/src/lib/movements.ts:13` with its i18n key and card art. Registering the
  detector flips it from "Soon" to analyzable with no frontend edit.
- **`FAULT_LANDMARKS` entries** (§4.1) — a pre-existing gap across all six non-shipped-squat
  detectors, not this movement's to close.
- **Regenerating the KG** to fix `Elbow Drift Forward`'s connectivity (§5) — a deploy step.
- **The other two Group D movements** (Arm Abduction, Arm VW).
- **Validation against labeled data** — none exists (§2).
