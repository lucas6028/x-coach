# Arm Abduction Rule Detector — Design Spec

**Status:** design spec · **Date:** 2026-08-09
**Movement:** Arm Abduction (standing lateral / shoulder-abduction raise) · **Detectors after
this one:** 9/16
**Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`
§Arm Abduction (lines 966–1010)

---

## 1. Purpose and why Arm Abduction is next

Group D runs Bicep Curl → Arm Abduction → Arm VW. Bicep Curl shipped as the eighth detector on
2026-08-09; this is the ninth, and it ships the same way every one since Push-up has: cited
rules, `validated=False`, Beta in the UI, no frontend edit.

The parent spec gives Arm Abduction four rules. **Two ship, one is registered permanently
silent, and one is withdrawn:**

| # | fault_id (shipped) | Treatment | Cue |
|---|---|---|---|
| 1 | `arm_abd_shoulder_shrug` | **SILENT** (§3) | Shoulders hike toward the ears — upper-trap dominance |
| 2 | — | **WITHDRAWN** (§4) | Raising into/through the 70–120° impingement arc, or past target+15° |
| 3 | `arm_abd_contralateral_trunk_lean` | ships (§5) | Torso side-bends to help hoist the arm |
| 4 | `arm_abd_lr_asymmetry` | ships | One arm lags the other at the top |

That is structurally the same shape as Push-up (4 live + 1 silent) and Band Pull Apart (3 live
+ 1 silent), with one withdrawal on top.

### 1.1 Three things make this movement different from the eight already shipped

**(a) Labeled correct/incorrect ground truth EXISTS for this movement.** The Deadlift, Row, Band
Pull Apart and Bicep Curl specs each open with "no labeled data exists, so `validated` stays
`False`." That sentence is **false here** and must not be inherited: REHAB24-6 `Ex1` **is** arm
abduction, and it carries a per-repetition human `correctness` label — 178 reps, 9 subjects, 90
correct / 88 incorrect.

**Lunge got there first, and this is the second such movement, not the first.** REHAB24-6 `Ex5` is
lunge, and `notes/lunge-rule-validation.md` is the 174-rep validation that was actually run against
it. What is new here is the **gap between the two**: this is the first movement whose labeled data
exists while the check has **not** been run. §2 scopes what that licenses, and why `validated` is
*still* `False`.

**(b) This is the first detector whose spec-rated `high` view is actually reachable in
production.** Every previous movement's best rules wanted `side`, and `side` never occurs
(§6.6). Arm Abduction's rules are rated `high` on **front/rear**, and `rear` and `rear_oblique`
are 46 of the 49 real pose JSONs. A frontal-plane quantity reads the same plane from behind as
from in front — mirrored, and every metric here is unsigned — so for the first time the rules can
earn their rating on a real clip rather than only in principle.

**(c) The silent rule is silenced by a MEASUREMENT, not by an argument.** `pushup
.rule_scapular_winging`, `row`'s fifth rule and `band_pull_apart
.rule_loss_of_scapular_retraction` are all silent because MediaPipe has no scapular landmarks —
argued from the landmark set. §3 measures the failure instead, on two datasets and two variants,
against 3-D ground truth. In passing it converts `band_pull_apart
.rule_loss_of_scapular_retraction`'s asserted premise ("MediaPipe's shoulder landmark … moves
with the humerus") into a measured one.

---

## 2. Labeled ground truth exists — and `validated` stays `False` anyway

### 2.1 What REHAB24-6 Ex1 is

From `Segmentation.csv`, `exercise_id == 1` (`EXERCISE_NAMES["1"] == "arm abduction"`):

| Property | Value |
|---|---|
| Repetitions | **178** |
| Correctness | **90 correct / 88 incorrect** |
| Subjects | **9** (person ids 1–9), every one contributing **both** classes |
| Videos | 13, two orthogonal cameras each |
| `cam17_orientation` | 88 `front` / 90 `half-profile` / 0 `profile` |
| `mocap_erroneous` | 0/178 |
| Rep length | 83–316 frames @ 30 fps = **2.77–10.53 s**, median 4.83 s |
| `exercise_subtype` | **`right arm` on 178/178** |

That last row is load-bearing and gets its own section.

Every Ex1 clip also ships marker-driven 3-D (`data/REHAB24-6/Ex1/{video_id}-30fps.npy`, 26
joints per `joints_names.txt`) **and** cached MediaPipe landmarks
(`data/REHAB24-6/processed/mediapipe_landmarks_cache/{video_id}-Camera1{7,8}-30fps.npz`, all 13
Ex1 videos present). So a pose estimate and a ground truth can be compared frame by frame with
no extraction step. The markers are a rigid skeleton, checked rather than assumed: the right
upper-arm length is constant to `std/mean = 3.1e-16`.

### 2.2 Ex1 performs a variant this rule set does not model, and that bounds everything below

**All 178 Ex1 reps are unilateral (`right arm`).** The parent spec's rule 4 says "during a
bilateral raise" and its rule 3 adds "(For a single-arm raise, sign the lean relative to the
working side)" — so the spec knows the variant exists, but the app does not: `movements.ts`
offers one "Arm Abduction" and the detector has no way to be told which variant it is watching.

The consequence for rule 4 is definitional, and measured: on the marker 3-D, `|elevation_L −
elevation_R|` at the working arm's peak runs **64.3°–132.2°, median 104.2°, and exceeds the
spec's 12° threshold on 178/178 reps**. That is **not a false-positive rate.** It is evidence
that Ex1 is not performing the movement the app calls Arm Abduction: a rule that says "your two
arms did completely different things" is *correct* about a one-armed raise. What it means is
that **Ex1 cannot validate rule 4 at all**, in either direction.

It also means every other Ex1 number in this document is measured on the unilateral variant.
Where that matters, §3 and §5 say so and re-measure on the bilateral one.

### 2.3 Fit3D `side_lateral_raise` supplies the bilateral variant

Fit3D ships `side_lateral_raise` with `joints3d_25` mocap ground truth and `rep_ann.json` rep
boundaries: **8 subjects × 5 reps = 40 reps**, 50 fps (ffprobe: `50/1`). It carries **no
correctness label**, so it is the Bicep Curl §2 instrument — it answers "does this threshold fire
on ordinary reps performed by people trying to do the movement correctly?", not "does it find
faults". Rep length 70–248 frames = **1.40–4.96 s, median 2.30 s**.

### 2.4 Where each parent-spec threshold sits in the real distribution

Recorded, **not repaired** — §9's no-tuning rule is absolute.

| Threshold | REHAB24-6 Ex1 (marker 3-D, unilateral, 178 reps) | Fit3D (mocap 3-D, bilateral, 40 reps) |
|---|---|---|
| shrug: `neck_gap` shrinks > 18% | **fires 96.6%** on MediaPipe / **0.6%** on the marker clavicle (§3) | **fires 34/40** on 3-D truth |
| arc: peak elevation > `target(90°)+15°` | fires 168/178 = **94.4%** | fires 8/40 = 20% |
| arc: peak elevation > 120° | fires 143/178 = 80.3% | fires 3/40 |
| lean: lateral trunk lean > 12° | **fires 0/178** (max observed **7.6°**) | fires 1/40 (max 14.1°) |
| asymmetry: `|L−R|` > 12° at the peak | fires 178/178 — variant artifact, see §2.2 | **fires 2/40** (median 4.4°, max 16.8°) |

Two readings that the table alone does not give:

- **The arc threshold moves with the dataset, by a factor of nearly five (94.4% vs 20%).** That
  is not noise; the two datasets prescribe different heights (Ex1 median peak 130.2°, Fit3D
  median 97.1°). §4 is about exactly that.
- **The lean cue separates even though it never fires.** Ranking Ex1's incorrect reps above its
  correct ones on the same quantity gives **per-subject median AUC 0.800** (pooled 0.647, 9
  subjects). Real cue, threshold in the tail — the `lunge_insufficient_depth` shape, which this
  project's standing treatment is *ship it and record the placement*, not withdraw. §5.

**One projection caveat, stated once and applying to every 3-D row above.** Production reads
image-plane angles. An arm elevation and a trunk lean are both frontal-plane quantities measured
from a frontal-plane view, so obliquity foreshortens them and a real deviation reads **smaller**
than it is — the failure mode is a missed fault, never a false one. Measured directly on Ex1
cam17: MediaPipe's arm-elevation error against the markers is a **mean 20.6° per rep (p90
43.8°)**, and its median peak reads 157.4° against the markers' 130.1°. That is large, and it is
the direct limit on any rule reading an elevation *magnitude* — which, after §4, is none of them.
It does not bound rule 4, which reads a **difference of two like-measured quantities** where the
common-mode error cancels. (Measured on `cam17_orientation == front`, a label
`estimate_view_for_pose(allow_front=False)` can never emit, so it is a statement about the
metric, not about what a user would see.)

### 2.4b How to reproduce every number above

Measured by throwaway scripts, not committed — the same convention
`notes/lunge-rule-validation.md` §3.2 uses for its marker-based passes. Reproduce as follows;
nothing here needs a GPU, a model, or an extraction step.

- **REHAB24-6 Ex1, marker 3-D.** Load `data/REHAB24-6/Ex1/{video_id}-30fps.npy` (26 joints per
  `joints_names.txt`, `[:, :, :3]`, **Y is up**). Take `RightArm`(12)/`RightForeArm`(13) with
  `RightUpLeg`(21) for right-arm elevation (`LeftArm`(7)/`LeftForeArm`(8)/`LeftUpLeg`(16) for the
  left), `RightShoulder`(11)/`LeftShoulder`(6) as the clavicle candidates and `Head`(4) for the
  gap. Window each rep by `Segmentation.csv` (`;`-delimited, `exercise_id == 1`,
  `first_frame`..`last_frame`, clamped to the array). Lateral trunk lean: decompose
  `mid_shoulder − mid_hip` onto the horizontal component of the hip-line axis against vertical.
- **REHAB24-6 Ex1, MediaPipe.** `data/REHAB24-6/processed/mediapipe_landmarks_cache/
  {video_id}-Camera17-30fps.npz`, key `image`, frame-aligned to the same indices. Reference every
  height to the **mid-hip only** — normalising by a shoulder-derived scale makes the two shoulders
  antisymmetric by construction and produced a spurious ±0.932 pair on the first attempt.
- **Fit3D.** `data/fit3d/train/{s}/joints3d_25/side_lateral_raise.json`, H36M-17 order per
  `src/fit3d/dataset.py` (**Z is up** here, unlike REHAB24-6). `rep_ann.json[action]` is a **flat
  list of boundary frame indices**, so k boundaries give k−1 reps.
- **The production corpus.** `estimate_view_for_pose(Path(p), allow_front=False)` over
  `data/runtime/pose_json/*.json` (it takes a `Path`, not a `str`); shoulder widths via
  `landmarks_to_array` + `visible_point(a, 11/12)`, discarding the degenerate zero-width clips
  (3 of 46).

### 2.5 Why `validated` is still `False`

`ARM_ABDUCTION_DETECTOR.validated` drives the Beta badge, and its meaning — "checked against
labeled ground truth" — is a product claim. **Nothing in this task runs the check.** The Lunge
precedent (`notes/lunge-rule-validation.md`, 869 lines, its own phase) is what a validation of
this detector would look like: a replay harness over the production path, per-subject AUC,
structural-silence accounting, a camera-routing decision, and a written verdict. What §2 does is
establish that the check is *possible* — the second time on a non-squat movement, after Lunge —
and scope it: rule 4 is unvalidatable on Ex1 (§2.2), rule 1 is silent so there is nothing to
validate, and rule 3 is the one rule Ex1 could genuinely speak to.

Recorded in TODO.md as a scoped follow-up, not attempted here.

---

## 3. `shoulder_shrug_elevation` is REGISTERED-BUT-PERMANENTLY-SILENT — a measured sensing failure

This project has two treatments for a rule it will not fire, and the choice is the whole
decision:

- **Registered-but-silent** (`pushup.rule_scapular_winging`, `row`'s fifth,
  `band_pull_apart.rule_loss_of_scapular_retraction`): *real, well-cited fault; the sensor cannot
  see it.*
- **Withdrawn** (OHP bar-path 2026-07-25, deadlift bar-drift 2026-08-01, curl wrist-flexion
  2026-08-09): *no citation supports the rule as written.*

### 3.1 The citation holds

`data/rag/docs/PMC12029123_arm_abduction_scapular_emg_angles.txt` (Mun WL, Jung EY, Lei S, Roh
SY, *Medicina* 2025) was read. The injury mechanism is stated verbatim and unambiguously:

> "overactivation of the muscles that elevate the scapula, such as the upper trapezius and
> levator scapulae, and low activation of the muscles that stabilize the scapula … can lead to
> increased shoulder instability, which can increase the risk of musculoskeletal conditions such
> as impingement syndrome and rotator cuff injury"

> "Persistent overactivity of the UT can lead to scapular dysfunction (or dyskinesia), such as
> subacromial impingement or glenohumeral instability … Therefore, when performing AW at higher
> shoulder abduction angles, care should be taken to avoid the excessive activation of the UT."

**Two honesty notes on that citation, neither of which changes the treatment.** (i) Mun studies
a **Pilates Reformer "Arm Work"** movement at four abduction angles (0°, 90°, 135°, 160°), not a
standing lateral raise, and it measures **EMG**, never kinematics — so it supplies **no
threshold** in any landmark unit, and the parent spec's `18%` has no provenance. (ii) The parent
spec's citation_support paraphrases Mun as "UT activation consistently increases as the shoulder
abduction angle surpasses 120°"; in the source, UT was highest at **160°** across all phases, and
the `120°` figure appears there as a *citation to a different study* on elastic-band scapular
retraction. Loose attribution, recorded. Neither matters for a rule that never fires — and both
would matter a great deal if it did.

### 3.2 The sensing fails, and it was measured rather than argued

`neck_gap = ear_y − shoulder_y` is a shrug proxy **only** if the shoulder landmark stays put
while the humerus moves. On this movement it does not, and the failure has two independent
components.

**(a) The gap collapses during abduction as a matter of anatomy — in 3-D ground truth, on the
bilateral variant, with no pose estimator anywhere in the path.** On Fit3D `side_lateral_raise`,
the within-clip Spearman correlation between the head→shoulder vertical gap and the arm's
elevation is **−0.699 to −0.954 across all 8 subjects (median ≈ −0.92)**, and the gap travels
**27%–94% of its own baseline** within a clip. The spec's 18% shrink threshold fires on **34 of
40 reps** performed by people doing the exercise deliberately for a mocap capture. Decomposed,
both endpoints move the same way: the shoulder joint rises (per-subject ρ +0.00 to +0.94) *and*
the head drops (ρ −0.32 to −0.86). The glenohumeral joint rising during abduction is
scapulohumeral rhythm; it is the movement, not a fault.

**(b) MediaPipe reports the glenohumeral joint, not the acromion — so the one reading that could
rescue the metric is unavailable.** On REHAB24-6 Ex1, comparing MediaPipe's shoulder landmark
against the two marker candidates for "the shoulder", each measured as height above the mid-hip
and expressed as a fraction of its own baseline:

| Point | within-rep ρ vs true arm elevation (correct reps only) | travel, % of baseline |
|---|---|---|
| marker **clavicle** (acromion — true scapular elevation) | +0.948 | **1.0%** |
| marker **glenohumeral** | +0.979 | 13.9% |
| **MediaPipe landmark 11/12** | — | **11.2%** |
| MediaPipe `|ear − shoulder|` gap, working side | **−0.957** | 6.1% (ear) |
| MediaPipe gap, **resting** side (the control) | **+0.068** | — |

MediaPipe's shoulder tracks the glenohumeral joint (11.2% vs 13.9%), not the acromion (1.0%) —
an order of magnitude apart. **The resting-side control is what makes this a statement about the
arm rather than about the head or the framing:** on the arm that does not move, the gap is
uncorrelated with the working arm's elevation. And the fire rates split exactly along the same
line: the 18% threshold fires on **1/178** reps read off the marker clavicle and **172/178 =
96.6%** read off MediaPipe.

**The confound is variant-independent, which is what licenses silencing rather than gating.** On
a bilateral raise both shoulders ride their own humerus, so nothing is left to read the shrug
against — component (a) is measured on the bilateral variant precisely to establish this. The
spec's own mitigation was measured too and does not rescue it: restricting to frames below 90° of
elevation (the "early or disproportionate shrug" the parent spec prescribes) takes the MediaPipe
fire rate from 96.6% to **49.4%**, which is half of every rep in a dataset that is half correct.

### 3.3 Not substituted, and the metric is not emitted

Shipping a different metric under this fault_id would attach Mun's citation to a quantity Mun
says nothing about. And unlike Bicep Curl's `upper_arm_length` — emitted as a live diagnostic so
an unreachable disjunct's arithmetic stays checkable — **`shoulder_ear_gap` is not emitted at
all**: no live rule reads it, and emitting it would force landmarks 7/8 into `required`, where
the all-or-nothing gate (§6.4) would let a lost ear silence the two rules that do fire. Measured
on the 49-file production corpus the ears are never lost (0.00% of 9426 frames), so the cost
today is zero — but that corpus is squats filmed from behind, and paying a live cost for a dead
metric is the wrong default regardless.

**The KG is not the gap.** `Shoulder Shrug` → `Arm Abduction:Compensatory Shoulder Shrug`,
`quality_impacts: Shoulder Depression`. The metric is the gap.

**Open, recorded, not resolved:** a shrug rule for this movement would need a shoulder-height
reading taken *at matched arm elevation* (comparing like with like across the rep) rather than
against a setup baseline. That is a novel construction with no citation and no validation, and
inventing it here is the move §9 forbids.

**Cross-reference, flagged and deliberately NOT asserted.** `band_pull_apart.rule_shrugging`
ships live on the same `shoulder_ear_gap` construction. Its excursion is horizontal abduction at
roughly fixed elevation, so the confound measured here plausibly differs in kind, and nothing in
this document measures it. Logged in TODO.md as a check to run, not as a defect found.

---

## 4. `excessive_elevation_impingement_arc` is WITHDRAWN — three independent failures

### 4.1 The citation does not say what the rule says

`StatPearls NBK554518` was fetched and read. It describes the painful arc as a **diagnostic
sign**:

> "Pain is reproduced between approximately 70° and 120° of active shoulder abduction, with
> relative relief beyond 120°, which is supportive of subacromial pathology."

Asked directly for any statement that raising the arm through the arc, or above a specific angle,
is itself a fault, an error, or a thing to avoid during exercise, the source yields **nothing**.
The arc is where a person who *already has* subacromial pathology hurts. The parent spec's
rationale — "repeatedly loading through this arc with inadequate scapular upward rotation risks
impingement" — inverts the inference: it reads a sign of existing pathology as a cause of it. No
cited source read here makes that step.

### 4.2 The first disjunct is vacuous, and is a restatement of rule 1

The heuristic's first cue is "sustained `arm_elevation_angle` in ~70–120° performed with a
concurrent shrug (`shoulder_shrug_elevation` true)". Measured on Ex1's marker 3-D:

> **178 of 178 reps enter the 70–120° band**, spending a median 30% of their frames there
> (range 20%–60%).

Passing through 70–120° *is* what an abduction is. So the arc conjunct is always true and the
cue reduces to "rule 1 fired" — the exact defect `row.rule_momentum_jerk`'s second condition and
Bicep Curl's elbow-displacement disjunct both had: a branch that reads as coverage and can never
change a verdict. After §3 it reduces further, to "a permanently silent rule fired", i.e. to
never.

### 4.3 The second disjunct has no referent, and the pipeline has no target

The remaining cue is "raising `> target + 15°` (e.g. `>105°` when the prescribed target is 90°)".
**There is no prescribed target anywhere in this pipeline** — grepped across `src/pose/` and
`backend/app/`: no target angle, no prescription, no per-user ROM goal. Fixing one would be an
uncited rule-level number, and the two datasets show it cannot be fixed at all:

| Peak elevation, 3-D truth | REHAB24-6 Ex1 (unilateral) | Fit3D `side_lateral_raise` (bilateral) |
|---|---|---|
| median | 130.2° | 97.1° |
| `>105°` (i.e. target 90°) | 168/178 = **94.4%** | 8/40 = **20%** |

A threshold whose fire rate swings from 94% to 20% between two datasets of the same named
movement is not measuring a fault; it is measuring which variant was performed. And on Ex1 the
direction is wrong as well: **correct reps go higher than incorrect ones** (median 132.4° vs
125.2°; AUC that incorrect reps rank high = **0.333 pooled**, an inversion).

Three independent failures — a citation that does not support the claim, a vacuous conjunct, and
a threshold with no referent — so the rule is **withdrawn and absent**, not silent. A silent stub
would assert that this is a real fault the sensor cannot see; the sensor sees elevation angles
perfectly well.

**Open spec question, recorded not resolved.** The KG's own Arm Abduction fault list contains
**`Arm Abduction:Incomplete Elevation`** — the *opposite* fault, with the richest bucket set of
the three (`quality_impacts: Humerus Abduction`, `causes: Limited Shoulder ROM`). Every other
movement in the parent spec got an incomplete-ROM rule; Arm Abduction got "raised too high"
instead and has no ROM rule at all. Whether the rule set wants one is a spec question needing a
source that puts a number on insufficient abduction. **No rule is invented here to fill it.**

---

## 5. `contralateral_trunk_lean` SHIPS — and why this is not the wrist-flexion case

This rule presents the wrist-flexion signature at first glance: the parent spec's own
`citation_support` ends "**UNVERIFIED** in a peer-reviewed source (no read source isolated
frontal-plane trunk lateral flexion during abduction)". Fetching StatPearls NBK554518 confirms
the paraphrase — asked for any mention of trunk lean, lateral trunk flexion, side-bending or
contralateral compensation during abduction, it yields **nothing**.

**It ships anyway, and the discriminator is measurement rather than paraphrase.** Curl
wrist-flexion was withdrawn because the phenomenon was absent from the source **and** the
observability was rated `low` on every view **and** nothing measured suggested the cue carried
information. Here:

- **The cue orders incorrect reps above correct ones.** Ex1, marker 3-D, 178 labeled reps:
  per-subject median **AUC 0.800** (9 subjects, range 0.040–0.942), pooled 0.647, on
  `max |lateral trunk lean|`; 0.760 per-subject on the lean measured against the rep's own setup
  baseline. The quantity carries information about rep correctness.
- **Observability is `high` on front/rear**, and unlike every previous movement those views are
  reachable (§6.6).
- **The injury mechanism is verified** — StatPearls attributes impingement in part to "inadequate
  scapular upward rotation and posterior tilt", and gross trunk compensation during elevation is
  a coarse form of that. It is the *specific* frontal-plane substitution finding that is
  unverified, and the parent spec already says so.

So this is the `lunge_insufficient_depth` shape, not the wrist-flexion shape: a real cue whose
cited cut sits in the tail of the observed distribution. `notes/lunge-rule-validation.md` §5.4
settled the treatment for exactly that case — *"That is still not evidence the rules are wrong …
Neither threshold moves."*

**The threshold does not move, and its placement is recorded at the constant.** 12° fires on
**0/178** Ex1 reps (max observed 7.6°) and **1/40** Fit3D reps (max 14.1°). Read plainly: as
shipped this rule will almost never fire, and when it does the lean is gross. That is written up,
not repaired.

### 5.1 Two sub-criteria are dropped, and both are unimplementable rather than unwanted

- **"away from the raising arm"** — directional, and on a **bilateral** raise there is no raising
  arm, so the qualifier is undefined for the variant the app models. On the unilateral variant it
  would need a working-side determination the detector cannot make. The metric is taken
  **unsigned**, the Bicep Curl §4.8 construction: an unsigned departure is what the verified
  mechanism (compensation during elevation) actually describes, and a signed one asserts more
  than any read source does. Cost: a lean *toward* the working arm also fires — a wider net, in
  the direction the citation supports.
- **"or if it grows with load across a set"** — the pipeline has no load, and analyses default to
  three reps. Not implementable as stated; absent, not approximated.

---

## 6. Detector design

### 6.1 The `fault_id` namespace is `arm_abd_*`, and the parent spec's ids are renamed

The parent spec gives unprefixed ids: `shoulder_shrug_elevation`,
`excessive_elevation_impingement_arc`, `contralateral_trunk_lean`, `lr_abduction_asymmetry`.
Every movement after Squat prefixes (`lunge_*`, `deadlift_*`, `pushup_*`, `ohp_*`, `row_*`,
`bpa_*`, `curl_*`), because `merge_by_fault`, the analyses table and the frontend's `byFault` map
all key on `fault_id` with **no movement qualifier**.

The prefix is **`arm_abd_`, not `abduction_`**, and that is a deliberate collision guard rather
than a style choice: Group E's **Leg Abduction** is also coming, with its own faults, and a bare
`abduction_*` namespace would be ambiguous between the two the moment it ships. Shipped ids:
`arm_abd_shoulder_shrug`, `arm_abd_contralateral_trunk_lean`, `arm_abd_lr_asymmetry`.

`FAULT_LANDMARKS` in `frontend/src/lib/pose.ts:55` covers only the five squat faults and gets no
entry here — a **pre-existing** gap shared by all seven non-squat detectors, out of scope (§10),
already in TODO.md.

### 6.2 Rep segmentation: `avg_arm_elevation_deg`, polarity `max`, `rep_start="extended"`

The rep is `arms at the sides → abduction → top → controlled lowering → sides`, so the signal
peaks at its **maximum** and the rep starts away from the peak — `rep_polarity="max"` (matching
Overhead Press and Band Pull Apart) and `rep_start="extended"` (matching every detector except
Deadlift). Registry table entry: `"Arm Abduction": ("avg_arm_elevation_deg", "max", "extended")`.

**Why the mean of the two arms rather than `max` or `min` of them, decided by measurement.** On
Fit3D `side_lateral_raise`, left and right arm elevation correlate **r = 0.9896–0.9964 across all
8 subjects**. The arms move as one unit, so the mean is the same excursion with the per-arm
landmark noise halved, while an extremum would inherit whichever arm was noisier on each frame.
This is the Bicep Curl §4.2 argument and the Overhead Press precedent, re-measured for this
movement rather than assumed. Per-clip excursion of the mean is **61.7°–109.4°**; the
`max`-of-both alternative gives 64.6°–109.7°, i.e. no meaningful excursion advantage to offset
the noise cost.

**The loser's failure mode, stated: `avg` degrades on the unilateral variant, and `max` would
not.** On a one-armed raise the resting arm contributes a near-constant ~26° (Ex1 marker median
peak of the resting arm: 26.6°), so the mean still excurses — halved in amplitude but
monotonically, and `segment_reps` normalises before thresholding. So the unilateral variant
degrades the signal-to-noise rather than destroying the signal, unlike Bicep Curl's alternating
curls where the mean cancels outright. Not corrected: choosing `max` to serve a variant the app
does not model, at a measured cost on the one it does, trades a measured decision for a guessed
one.

**Cadence clears the floors, both of them, and this is measured rather than asserted.**
`DEFAULT_MIN_REP_SECONDS = 0.4`: Ex1 reps run 2.77–10.53 s and Fit3D 1.40–4.96 s, so the
tightest real rep is **3.5×** the floor. No `min_rep_seconds` override. The tighter constraint
Bicep Curl §4.3 discovered — a phase-scoped rule needs `phase_fraction · T ≥ min_frames/fps`
where `min_frames = max(3, ceil(0.20·fps))` — is checked per rule below rather than globally,
because it binds differently on a 15% `setup` window than on a 30% `peak` one. **Neither shipped
rule masks on `setup`**, which is what keeps this movement clear of the trap that silenced Bicep
Curl's extension term: `peak` is 30% of the rep, needing `T ≥ 0.667 s` against a measured minimum
of 1.40 s (**2.1×**), and `concentric` is wider still.

### 6.3 Phases

`setup → concentric → peak → eccentric`, assigned over the **per-rep** slice `run_detector` hands
to `assign_phases`, following `row_assign_phases` and `band_pull_apart_assign_phases`:

- `setup` — the first 15% of the rep window (arms at the sides).
- `peak` — the most-elevated 30% of the rep (the **70th** percentile of `avg_arm_elevation_deg`
  and above; note the polarity inversion versus Row's *most-flexed* 30%). This is the parent
  spec's "top-hold".
- `concentric` / `eccentric` — before / after the most-elevated frame.
- `unknown` — any invalid frame, checked **before** the setup cutoff, so an occluded frame in the
  opening 15% is not labelled `setup`.

### 6.4 Metrics — and the metric layer contains no thresholds

`arm_abduction_compute_raw` / `arm_abduction_assign_phases` emit scale-free per-frame quantities
and phase labels only. The only constant either may define is `_DEGENERATE_LENGTH = 1e-6`, a
division-by-zero guard.

| Metric | Definition |
|---|---|
| `left_arm_elevation_deg`, `right_arm_elevation_deg` | `angle(hip, shoulder, elbow)` per side; ~0° = arm at the side, 90° = horizontal |
| `avg_arm_elevation_deg` | mean of the two; NaN-tolerant. **The rep signal.** |
| `arm_elevation_asymmetry_deg` | `|left − right|`; NaN if either side is missing. Drives rule 4 |
| `lateral_trunk_lean_deg` | **unsigned** angle of `hip_mid→shoulder_mid` from image-vertical, in the image x–y plane (§5.1). Drives rule 3 |
| `shoulder_width` | `dist(11,12)`; the normalizer, emitted so a scale question is answerable later (§6.7 needs it) |

**Required landmarks, and the all-or-nothing rule.** `required` is both shoulders, both elbows
and both hips — **not** the ears (§3.3), and **not** the wrists (§6.7 drops the only cue that
would have read them). If `visible_point` drops any **one**, the frame is `valid=False` and
carries no metrics at all, so *every* rule goes silent for that frame. This mirrors
`pushup_compute_raw`, `ohp_compute_raw`, `lunge_compute_raw`, `row_compute_raw`,
`band_pull_apart_compute_raw` and `bicep_curl_compute_raw`.

### 6.5 Rules

**Rule 1 — `arm_abd_shoulder_shrug`.** Registered, permanently silent, returns `[]`. §3.
KG query `Shoulder Shrug`.

**Rule 3 — `arm_abd_contralateral_trunk_lean`.** Fires when `lateral_trunk_lean_deg > 12°`
(**FROM THE SPEC**) on a `concentric` frame — the phase scope is the spec's own ("during
concentric"). Severity ramps 12° → 30° (**RULE-LEVEL**, 2.5× the fire threshold, the
`pushup.rule_hip_sag` convention; the parent spec states no ramp for any Arm Abduction fault, and
the Lunge section states its ramps explicitly, so the absence is meaningful). KG query
`Trunk Lean Compensation`.

**Rule 4 — `arm_abd_lr_asymmetry`.** Fires when `arm_elevation_asymmetry_deg > 12°`
(**FROM THE SPEC**) on a `peak` frame — the spec's "at the top-hold". Ramps 12° → 30°
(**RULE-LEVEL**, 2.5×). KG query `Muscle Imbalance`.

> **The 12° in rule 4 needs its provenance stated precisely, in-code, because the citation is a
> different quantity in different units.** Terré & Solana-Tramunt (PMC12110944) was fetched and
> read: it measures **middle- and lower-trapezius EMG symmetry** during bilateral scapular
> retraction at 45° and 90° of shoulder abduction, and every threshold in it is a **percentage**
> — "asymmetries between 10% and 15% are often associated with a higher risk of injury and
> reduced performance", on a limb-symmetry scale of asymmetry 0–79% / limit 80–89% / normal
> 90–100%. **No angular threshold appears anywhere in the paper.** A 12° difference on a 90°
> raise is ~13%, which lands inside the cited band — but that correspondence is a
> **reconstruction, not a provenance**, it silently assumes the 90° target §4.3 showed the
> pipeline does not have, and the parent spec never states it. This is the
> `ohp_asymmetric_press` situation exactly (cited at 7° scapular / 1.5 cm lateral shift, shipped
> as 0.15 normalized wrist height), and it takes the same treatment: **ship the spec's number,
> unchanged, with the mismatch written at the constant.** Re-expressing the rule as a percentage
> was considered and rejected — changing units changes what fires, which the no-tuning rule
> covers, and it would still be transferring an EMG figure to a kinematic quantity.

### 6.6 View handling — all three rules downgrade, none gates

**Measured production reality, re-measured for this spec rather than inherited** (the Bicep Curl
doc warns that an inherited view figure once stopped reproducing). Running
`estimate_view_for_pose(path, allow_front=False)` over all 49 files under `data/runtime/pose_json`
on 2026-08-09: **`rear_oblique` 37, `rear` 9, `unknown` 3, `side` never.** `front` and
`front_oblique` are unreachable under `allow_front=False` (`src/pose/view_estimation.py:14-16`).

**This is the first movement whose spec-rated `high` views are reachable.** Both live rules
measure a **frontal-plane** quantity and are rated `high` on **front/rear**. A frontal-plane
quantity reads the same plane from behind as from in front — mirrored — and both metrics here are
unsigned by construction (§5.1, and `|L−R|` is sign-invariant), so `rear` earns the full rating
with no discount and no facing determination. That is 9 of 49 real pose JSONs at full confidence,
where every previous detector's best rules wanted a `side` view that occurs zero times.

**No rule gates.** Band Pull Apart and Bicep Curl gate out the views where their sagittal metrics
would read the wrong plane. Nothing here needs that: an arm-elevation difference and a lateral
trunk lean are the *right* quantity from every reachable view, and obliquity makes them noisier,
not different. `high` on `front`/`rear`, `medium` (× `VIEW_UNAVAILABLE_CONFIDENCE_SCALE` = 0.65)
elsewhere — the `band_pull_apart.rule_incomplete_rom` treatment. Written in the **positive**
(`view_type in {"front", "rear"}`) because the set of fully-observable views is the small one
here, which is itself the novelty; `unknown` and the obliques take the discount.

On `rear_oblique` — 37 of 49 clips — the frontal axis is foreshortened, so a real lean or
asymmetry reads **smaller** than it is: a missed fault, never a false one.

**State the ceiling alongside the novelty.** Because `front` is unreachable, `rear` is the *only*
production view that ever earns `high` here — 9 of 49 clips at full confidence and the other 40 at
×0.65. That is strictly better than every previous detector, whose `high` view occurs zero times,
and it is still a minority.

### 6.7 Rule 4's wrist-height disjunct is NOT implemented — its threshold is frame-scale dependent

The parent spec offers a second cue: "or if peak wrist heights differ by `> 0.05` normalized
units". Unlike Bicep Curl §4.9's displacement disjunct, this one is not redundant — a wrist-height
difference and an elevation-angle difference genuinely differ when arm lengths or elbow bends
differ. **It is dropped for a different reason: `0.05` in raw MediaPipe normalized units is not a
well-defined criterion**, because normalized image coordinates scale with how much of the frame
the subject occupies. Measured across the 43 production pose JSONs that carry a usable shoulder
width:

> per-clip median `shoulder_width` runs **0.0591 to 0.4923** normalized units — an **8.3× spread**
> (p90/p10 = 1.54×). So `0.05` normalized units is **0.102 shoulder-widths** on the widest-framed
> clip and **0.846** on the narrowest: the same physical asymmetry fires or does not depending on
> how far the phone was from the lifter.

`ohp_asymmetric_press` avoids this by normalizing its asymmetry by shoulder width explicitly;
this spec line does not, and renormalizing it would be inventing a threshold. `shoulder_width` is
still emitted as a diagnostic so the spread stays checkable without re-deriving it, and the
arithmetic is pinned by a test. Annotated in the parent spec as a NOTE.

The spec's trailing "**sustained across reps**" qualifier is also dropped: no rule in this
codebase carries cross-rep state — `run_detector` scores one rep at a time and `merge_by_fault`
reports the rep count afterwards. Stated, not approximated.

---

## 7. KG queries — resolved before being written, not after

Each string below was checked against `data/kg/sports_kg_v3.graphml` with
`retrieve_graph_context(query, movement="Arm Abduction")` — the function **production** calls,
not `resolve_nodes`. Observed results, not predicted ones:

| Rule | Query | Resolves to | Buckets |
|---|---|---|---|
| 1 (silent) | `Shoulder Shrug` | `Arm Abduction:Compensatory Shoulder Shrug` | `quality_impacts: Shoulder Depression` |
| 3 | `Trunk Lean Compensation` | `Arm Abduction:Trunk Lean Compensation` | `quality_impacts: No Compensatory Trunk Movement` |
| 4 | `Muscle Imbalance` | `Muscle Imbalance` (generic Cause node) | **ZERO buckets, 1 edge — DANGLING** |

**Row 4 is the `Row:Compensatory Movements` case, not the "thin card" case, and conflating them
would be the mistake.** Rows 1–3 are thin-but-non-empty — one populated bucket each, which is what
Band Pull Apart and Bicep Curl accepted. Row 4 has **no buckets at all**, which is the
dangling-seed trap the OHP KG fix (PR #48) existed to eliminate and which `row.py`'s Step 0 names
explicitly. It is accepted anyway, and each leg of that decision was checked rather than assumed:

- **There is no better query.** The graph has **no Arm Abduction asymmetry fault node**.
  `Asymmetry` and `Left Right Asymmetry` resolve to the generic `Symmetry` QualityDimension, which
  is *also* zero-bucket (its 7 inbound edges are all Squat and Overhead Press). Under
  `movement="Overhead Press"` the same `Muscle Imbalance` query *additionally* returns the
  OHP-scoped fault that carries the content — which is why `ohp_asymmetric_press` gets a real card
  and there is no Arm-Abduction-scoped counterpart to return.
- **The user-visible failure the OHP fix targeted does not occur**, verified by reading the
  frontend rather than inferring: `frontend/src/components/FaultCard.tsx:55-57` pushes a
  causes/risks/cue rung only `if (...).length` and wraps the whole block in `rungs.length > 0`. A
  zero-bucket seed renders a **thinner** card — fault name, severity, evidence — never an empty
  "Causes:" heading with nothing under it.
- **Substituting a rich node was rejected** for the reason Band Pull Apart and Bicep Curl both
  rejected it: `Range Of Motion` returns full buckets whose `corrections` entry is
  `Wrapping Surface Adjustment`, meaningless for this movement. **A semantically correct empty
  card beats a semantically wrong full one.**

Because the graphml is gitignored, authoring the node is a deploy step; logged against TODO.md's
existing "many faults have no KG node" item.

---

## 8. Testing

`tests/test_arm_abduction.py`, following `tests/test_bicep_curl.py`:

- **Fixture builder** producing synthetic landmark frames with a controllable per-arm elevation
  and lateral trunk lean, so each rule can be driven across its threshold in isolation.
- **Per-rule threshold pins** — fires just past the spec threshold, silent just short of it, for
  both live rules.
- **`rule_shoulder_shrug` returns `[]` for every input**, including one built to shrug hard —
  and the assertion must not be vacuous. Bicep Curl's equivalent test was green for the wrong
  reason (the phase window was structurally too short to fire), so this test asserts the rule is
  silent on a clip where **another** rule does fire, proving the frames reached the rules at all.
- **The wrist-height disjunct's scale dependence (§6.7)** pinned numerically, so a future edit
  that "restores" it has to confront the arithmetic.
- **View handling**: both rules fire on `front`/`rear` at full confidence and on `rear_oblique`
  and `unknown` at ×0.65, and **no** view silences either — the inverse of the gating tests every
  detector since Lunge has carried.
- **`test_metric_keys_match_the_emitted_metrics_exactly`** — the two-way match between
  `ARM_ABDUCTION_METRIC_KEYS` and what `arm_abduction_compute_raw` emits. A key the tuple omits
  is silently dropped by `run_detector` and read back as NaN by every rule.
- **`EndToEndSegmentationTest`** — a synthetic multi-rep clip segmenting on
  `avg_arm_elevation_deg`, verifying rep count, that per-rep phases are assigned (the check that
  actually verifies the `max` / `extended` interface-design inference), and — per §6.2 — that the
  `peak` window of the shortest realistic rep clears `min_frames` after `segment_reps` has
  trimmed the window to the excursion.
- **`tests/test_movement_registry.py`** — add `"Arm Abduction": ("avg_arm_elevation_deg", "max",
  "extended")` to the shared rep-signal table, and a `validated is False` case.
- **`tests/test_analyze_pose_service.py:116-123`** — this file's "unimplemented movement" example
  currently names **Arm Abduction** and asserts it is absent from the registry. It must be
  rotated to a still-unimplemented movement (Arm VW); its own comment says so.

Commands: `.venv\Scripts\python.exe -m pytest tests/ -q` and
`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`. Two backend test
flakes are known-unrelated on this machine; check a failure against a baseline on `main` before
attributing it to this change.

---

## 9. Honesty constraints

- **No threshold tuning.** Every cited number stays as the parent spec states it. §2.4's finding
  that the lean threshold fires 0/178 and the asymmetry threshold 2/40 is **written up, not
  repaired**, and so is §6.5's finding that rule 4's number has no provenance in its citation.
- **Every constant is labeled in-code as exactly one of `FROM THE SPEC` or `RULE-LEVEL CHOICE
  MADE HERE`.** Never blurred. All severity ramps are RULE-LEVEL.
- **Citations are copied verbatim from the parent spec at implementation time**, never recalled
  from memory. Where this spec quotes a source directly (Mun, StatPearls, Terré) the quote comes
  from the RAG doc or the fetched article, not the parent spec's paraphrase — and §3.1 and §6.5
  record where the paraphrase and the source diverge.
- **Every measurement in this document names its dataset, its variant and its instrument**,
  because the two datasets perform different variants (§2.2) and disagree (§4.3). No number is
  quoted as if it were variant-independent unless it was measured on both.
- **`validated=False`**, with §2's evidence stated at the registration site — including the fact
  that labeled data now exists, so a future reader does not re-derive the false "none exists"
  claim.
- **No metric is substituted under another metric's fault_id.** The silent shrug rule reads
  nothing; the withdrawn arc rule is absent, not re-pointed at something measurable.

## 10. Out of scope

- **Any frontend file.** `Arm Abduction` already exists in `frontend/src/lib/movements.ts:15`
  with its i18n key and card art; `/api/movements` derives from the registry, so registering the
  detector flips it from "Soon" to analyzable with no frontend edit.
- **`FAULT_LANDMARKS` entries** (§6.1) — a pre-existing gap across all non-squat detectors.
- **Regenerating the KG** to add an Arm Abduction asymmetry node (§7) — a deploy step.
- **Running the REHAB24-6 Ex1 validation** (§2.5) — its own phase, scoped in TODO.md.
- **Checking `band_pull_apart.rule_shrugging` against the §3.2 confound** — logged, not run.
- **The remaining Group D movement** (Arm VW).
