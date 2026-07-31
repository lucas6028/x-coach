# Lunge rule validation against REHAB24-6 Ex5

Phase 2 of the Lunge detector plan (`.superpowers/sdd/2026-07-30-lunge-detector/`). The four
cited rules in `src/pose/movements/lunge.py` — `lunge_knee_past_toes`, `lunge_knee_valgus`,
`lunge_insufficient_depth`, `lunge_pelvic_drop` — were replayed over **174 human-labeled
repetitions** from REHAB24-6 `Ex5` ("leg lunge"), 8 subjects, filmed simultaneously by two
orthogonal cameras.

This is the **first movement in this repository ever checked against labeled ground truth**.
Squat, Overhead Press and Push-up all shipped carrying the caveat "thresholds are spec-derived
and UNVALIDATED"; §8.4 of the parent spec has been outstanding since 2026-07-18.

**No threshold was changed in response to anything below.** Every cited number is exactly what
the literature and the parent spec state. `LUNGE_DETECTOR.validated` remains `False`.

Reproduce:

```
.venv\Scripts\python.exe scripts/rehab24/validate_lunge_rules.py ^
  --pose-dir data/REHAB24-6/processed/lunge_pose_json ^
  --segmentation data/REHAB24-6/Segmentation.csv ^
  --out data/REHAB24-6/processed/lunge_rule_validation.json
```

`--report-only` re-prints the report from the saved JSON without re-running (~1 s vs ~15 min).
The JSON and the pose corpus are gitignored; this note is the committed record.

---

## 1. What this measures — and what it does not

**REHAB24-6 labels each repetition `correct` or `incorrect` and never says which fault
occurred.** There is no per-fault annotation anywhere in the dataset. So when a rule fires on
an incorrect repetition, that is **not** evidence the rule found that repetition's actual
error — the rep may have been marked incorrect for something else entirely, and the rule may
have fired on an artifact.

Everything below therefore measures one thing: **does a rule's underlying signal carry
information about whether the repetition was performed correctly?** It does **not** measure
per-fault precision, and no number here should be quoted as if it did.

Three further limits on the claim, binding rather than decorative:

- **This is a replay harness on production-path inputs, not the production path.** Each labeled
  window is handed to `run_detector` as its own clip, with the view label and confidence the
  production estimator really produces and every gate unmodified. But real analysis runs the
  whole recording, segments its own reps, and smooths across rep boundaries. Also,
  `lunge_assign_phases` labels the first 15% of its input `setup`, and here that input **is**
  the rep — so the opening 15% of every repetition is discarded as setup.
- **8 subjects is small, and the reps are not independent.** ~22 reps from each of 8 people.
  Every headline below is therefore a **per-subject** statistic; pooled figures are secondary
  and labeled as such. **No p-value is computed on pooled reps** — the independence assumption
  it needs does not hold. One subject (person 3) contributed 21 incorrect reps and **zero**
  correct ones, so that subject yields no AUC at all and drops out of every median: the
  denominators below are 7/8 subjects, or 6/8 within some strata.
- **Per-subject reporting is AUC-only; contingency tables are pooled.** At ~10 reps per subject
  per class, a per-subject 2×2 is four cells of single digits — a sensitivity of 0/2 and one of
  0/9 would both print "0.000" and neither would mean anything. AUC survives that n because it
  is a rank statistic over all pairs. The pooled tables carry the subject caveat instead of
  being split into eight uninterpretable ones.
- **Every AUC carries the number of reps it was computed over, and they differ between
  columns.** `rank_auc` drops a rep whose score is non-finite, which happens whenever the lead
  side is unresolved — and the shipped-lead and labeled-lead columns lose different reps. Where
  the two columns are compared, a **matched-n** figure restricting both to the reps *both*
  scored is given alongside, so the contrast cannot be read as a denominator artifact.
- **Whatever separates here is validated on THIS dataset**: a lab recording with fixed cameras,
  controlled lighting and instructed errors. Nothing here licenses a claim about phone video in
  a gym.

---

## 2. Dataset, and the checks run before trusting any number

Ex5 as loaded from `Segmentation.csv`, asserted before the first rule ran
(`assert_dataset_shape` raises rather than warns if any of these moves):

| Property | Value |
|---|---|
| Repetitions | **174** (all processed; 0 skipped) |
| Correctness | **96 incorrect / 78 correct** |
| Subjects | **8** (person ids 2–9) |
| cam17 orientation | **88 `front` / 86 `half-profile` / 0 `profile`** |
| cam18 orientation | **88 `side` / 86 `half-profile`** (via the documented cam17→cam18 mapping) |
| Lead leg (label) | 91 `front leg right` / 83 `front leg left` |
| Records evaluated | **348** = 174 reps × 2 cameras, each run twice (production + oracle) |

Two assertions were run first because getting either wrong would have silently corrupted every
number downstream:

1. **Frame-index contiguity.** `slice_rep` indexes frames by list position while
   `Segmentation.csv` gives frame numbers. All 18 pose JSON files were checked frame by frame
   for `frames[i]["frame_index"] == i`; **all 18 pass**. One rep (PM_117a) has a `last_frame`
   two frames past the end of its clip and is clamped, not dropped.
2. **Correctness polarity.** `Segment.correctness` is an `int` (1 = correct) and `contingency`
   takes a `bool` where `True` means correct. The conversion is explicit and the totals are
   asserted at **78 correct / 96 incorrect**. Positive = **incorrect**, throughout.

### Camera routing

Each rule is scored on the camera that affords its required view (`RULE_CAMERAS`):

| Rule | Camera | View there |
|---|---|---|
| `lunge_knee_past_toes` | cam18 | `side` on the 88 cam17-`front` reps |
| `lunge_insufficient_depth` | cam18 | `side` on the same 88 |
| `lunge_knee_valgus` | cam17 | `front` on those 88 |
| `lunge_pelvic_drop` | cam17 | `front` on those 88 |

Strata are named by `cam17_orientation` throughout so the 88/86 split stays stable across all
four rules. cam17 rules are additionally reported with and without the 40 reps carrying
level-2/3 extra-person contamination (cam18 is level 0 throughout).

### Two passes

- **Production pass** — the rule receives the view label `estimate_view_for_window` actually
  produces (`allow_front=False`, gates unmodified).
- **Oracle pass** — the rule receives the dataset's orientation instead, at confidence 1.0.
  `front` maps to the literal `"front"` label, which production can **never** emit; any oracle
  result on a `front` rep is a statement about the **rule**, never about what a user would see.

### Measurement conditions that limit everything below

- **Frame validity.** `lunge_compute_raw` marks a frame invalid unless **all ten** of both
  shoulders, hips, knees, ankles and foot indices clear MediaPipe's visibility floor — one
  dropped landmark silences every lunge rule for that frame. Measured: **74.0% of frames valid
  on cam17, 58.4% on cam18** (minimum 0.000 — some rep windows have no usable frame at all).
  The sagittal camera is the worse of the two, exactly as `pushup.py`'s equivalent note
  predicts: a side view is where far-side landmarks are most often occluded.
- **The harness does NOT bypass rep segmentation, contrary to the design spec's §4.2.** That
  section claims "ground-truth rep boundaries mean `segment_reps` is bypassed entirely, which
  isolates rule quality from segmentation quality". It does not: `run_detector` runs its own
  segmentation on whatever it is handed. Measured fallback distribution — **cam17: 152 reps
  segmented (`fallback=None`), 11 `only_partial_reps`, 11 `no_reps_detected`; cam18: 91
  segmented, 53 `only_partial_reps`, 30 `no_reps_detected`.** So for 152 of 174 cam17 reps the
  rules scored a **sub-window** of the labeled repetition, re-cut by `segment_reps`. All
  continuous scores below are computed over exactly the frames the rules saw, so score support
  matches rule support — but the isolation §4.2 promised was not achieved, and the spec sentence
  is wrong as implemented. The headline finding in §3 does not depend on it:
  `full_window_premise` recomputes the lead-leg premise from the **whole labeled window** with
  `segment_reps`, `centered_median` and phase assignment all out of the picture, and §3.1 prints
  both variants side by side. The **per-rule** numbers in §4 do carry the re-cut.

---

## 3. The headline: lead-side resolution fails on this dataset

**All four rules read their metric from `f"{lead}_..."`.** If `resolve_lead_side` picks the
wrong leg, every rule scores the trailing leg's geometry and calls it the lead's. So this is
measured first, and it changes how everything after it must be read.

| | cam17 | cam18 |
|---|---|---|
| Resolved (not `None`) | 154/174 (unresolved **11.5%**) | 152/174 (unresolved **12.6%**) |
| **Accuracy vs `exercise_subtype`** | **96/154 = 0.623** | **72/152 = 0.474** |

cam18 is **below chance**. The cross-tabulation shows it is a bias, not a left/right label
inversion: on cam17, `front leg right` reps resolve `right` 71/79 (90%), but `front leg left`
reps also resolve `right` 50/75 — the heuristic calls the right leg the lead in 121 of 154
resolutions. A swapped anatomical mapping would produce the mirror image in both cells, and
does not.

### 3.1 The premise fails — but only as this pipeline can measure it

`resolve_lead_side` documents itself as a *substitution*: the parent spec defines the lead leg
as "the more flexed / **more anterior** foot", the anterior half is unusable in a frontal view,
so only the more-flexed half is used. Checking that half directly against the label:

> **The labeled lead knee is the more flexed knee at the rep's bottom in only 101/169 = 59.8%
> of cam17 reps and 77/161 = 47.8% of cam18 reps.**

Those are the **full-window** figures — computed straight from the labeled window with
`run_detector`'s own re-segmentation, smoothing and phase assignment out of the picture
(`full_window_premise`), so the finding does not rest on the harness's windowing. Through the
scored windows the rules actually saw, the same numbers are 105/169 = 62.1% and 78/161 = 48.4%.

**But three controls say this is a statement about the measurement, not about lunges.**

1. **The two cameras disagree with each other.** They film the same body at the same instant,
   yet they disagree about which knee is more flexed on **52/156 = 33%** of reps. (An
   independent re-derivation with its own geometry got 58/156 = 37%.) That is measurement error;
   anatomy cannot differ between two simultaneous views of one body.
2. **The two premise rates disagree by 12 points across those same simultaneous views**
   (59.8% vs 47.8%). Two reads of one physical premise cannot both be right.
3. **The number is dominated by MediaPipe's pseudo-depth.** `geometry.angle_degrees` uses all
   three coordinates, so a "knee angle" here is partly a function of a learned `z` channel.
   Recomputing the identical angle in the image plane alone collapses cam17 from **59.8% to
   17.2%** and moves cam18 the other way, from 47.8% to 57.1%. (The independent re-derivation
   got 14.8% for cam17 — same collapse.) A cue whose answer swings that far on whether one
   coordinate is included is not measuring knee flexion reliably.

So the supported claim is narrower than "more flexed doesn't identify the lead leg in a lunge",
and it is deliberately stated as:

> **The more-flexed-knee cue, as this pipeline measures it, does not identify the labeled lead
> leg from either view available here.**

That still fully condemns `resolve_lead_side` as shipped and still fully bounds all four rules —
nothing actionable is lost. What is dropped is the generalization about lunges as a movement,
which this data cannot support. **The distinction matters for the fix, not just for accuracy: if
the premise holds in 3-D and is merely unrecoverable from this projection, the repair is a
depth-robust lead cue, not a different cue.** §5 conclusion 1 is written accordingly.

**The existing guard cannot help either way.** `LEAD_SIDE_MIN_SEPARATION_DEG = 5.0` refuses an
answer when the two knees are within landmark noise of each other. On the reps the cue gets
**wrong**, the median left-right separation is **19.4°** (cam17) and **25.4°** (cam18) — far
outside a 5° band, so the guard passes them straight through. It fires on only ~12% of reps.
(The 23–28° overall median separation quoted in earlier drafts of this note was computed over
all reps and did not establish this; the wrong-reps figure does.)

The design spec's §7 risk table predicted this — *"Lead-leg heuristic inaccurate → Measured
directly against `exercise_subtype`; if poor, every rule inherits it and the writeup says so"*.
It was poor. This is that sentence.

**Consequence for §4.** Every per-rule number below is measured through an input that is wrong
on roughly a third (cam17) to half (cam18) of reps. Those numbers are therefore **lower bounds
under a broken input**, not clean measurements of the rules. To separate the two, each rule is
additionally scored with the lead leg taken from `exercise_subtype` — the **lead-oracle**
column. It is an **AUC-only** diagnostic: the rules resolve the lead side internally, so
substituting it outside them cannot change what fires. No threshold moves in either variant.

---

## 4. Per-rule results

Reading guide. **Per-subject median AUC is the headline**; pooled AUC is secondary. AUC is
threshold-free — it says whether the metric *orders* incorrect reps above correct ones at all,
independently of where the cited cut sits. All four metrics are higher-is-worse, so **an AUC
below 0.5 is a real inversion and is reported signed, never folded to 1−AUC**.

**Every contingency table below carries its structural silence.** A rep can be unable to fire
for three reasons that have nothing to do with the metric: its **view gate** was shut, its
masked phase was shorter than `min_frames` (6 at 30 fps), or its lead side was unresolved.
`contingency` counts every such rep as a true negative or false negative, which **deflates
sensitivity and inflates specificity by an amount a reader cannot see unless it is printed**.
So each table gives the counts *and* a **conditional** row restricted to the reps where the rule
could actually act. Where the two diverge, the conditional row is the one that means something.

### 4.1 `lunge_knee_past_toes` — cam18, spec threshold 0.10

| Cut | Per-subject median AUC (range) | n scored | **Lead-oracle** median (range) | n | Fired | Sens / Spec |
|---|---|---|---|---|---|---|
| All 174 | 0.462 (0.083–0.644), 7/8 subj | 150 | **0.725** (0.567–1.000) | 159 | 53 | 0.229 / 0.603 |
| `front` → cam18 `side` (88) | **0.171** (0.000–0.800), 7/8 subj | 80 | **0.833** (0.444–1.000) | 86 | 53 | 0.449 / 0.205 |
| `half-profile` (86) | 0.850 (0.000–1.000), 6/8 subj | 70 | 0.845 (0.550–0.900) | 73 | 0 | 0.000 / 1.000 |

Structural silence in the all-174 table: **86 view-gated** (every `half-profile` rep) and **32
could-not-fire** (22 of them an unresolved lead side). **Conditional on the 76 reps where the
rule could act: tp 22 / fp 31 / tn 5 / fn 18 → sensitivity 0.550, specificity 0.139.**

**As shipped, on the only stratum where the cue is validly observable — the 88 genuinely
sagittal cam18 reps — the metric orders CORRECT reps above incorrect ones** (per-subject median
0.171, pooled 0.348). That is an inversion, not weak separation, and it is reported as one.

**With the lead leg taken from the label, the same metric on the same frames separates well:
per-subject median 0.833 on that stratum, 0.725 overall.** Restricting **both** columns to the
80 reps *both* score, the contrast is **0.171 vs 0.850** — so it is not a denominator artifact.
The cue carries real information about rep correctness; the shipped rule cannot get at it
because it reads the wrong leg.

**The `half-profile` stratum's 0.850 does not contradict that.** Same metric, same broken lead
resolution, yet it orders strongly the right way there. The reason is visible in the lead-oracle
column: **0.850 shipped vs 0.845 labeled — the lead choice barely matters on that stratum**, so
the wrong-leg penalty that inverts the sagittal reps is simply not levied there. (In a
half-profile projection both legs' anterior travel foreshortens similarly, so the two knees'
forward ratios move together.) The rule fires 0 times there regardless: its hard gate requires a
confident `side` label, which neither pass produces, and the 0.10 threshold sits at percentile
88.6 of that stratum's scores anyway.

**Production and oracle passes are identical here, and that is correct, not a bug.** The gate
needs `view_type == "side"`; the estimator returns `side` on all 88 cam18-`side` reps (0.69–0.99
confidence, matching Phase 0's 88/88), and `rear_oblique` on all 86 `half-profile` ones, while
the oracle maps `half-profile` → `front_oblique`. Neither pass yields `side` off the sagittal
stratum, so the fire decision cannot differ.

### 4.2 `lunge_insufficient_depth` — cam18, spec threshold 100°

| Cut | Per-subject median AUC (range) | n scored | **Lead-oracle** median | n | Fired | Sens / Spec |
|---|---|---|---|---|---|---|
| All 174 | 0.500 (0.392–0.917), 7/8 subj | 148 | **0.390** (0.220–0.800) | 157 | 6 | 0.010 / 0.936 |
| `front` → cam18 `side` (88) | 0.792 (0.600–1.000), 7/8 subj | 80 | **0.320** (0.080–0.889) | 86 | 5 | 0.020 / 0.897 |
| `half-profile` (86) | 0.183 (0.000–0.750), 6/8 subj | 68 | 0.260 (0.024–0.850) | 71 | 1 | 0.000 / 0.974 |

Structural silence in the all-174 table: 0 view-gated (this rule has no hard gate) but **48
could-not-fire** — 22 unresolved lead side, the other 26 windows whose `bottom` phase is shorter
than the 6-frame floor. **Conditional on the 126 reps where the rule could act: tp 1 / fp 5 /
tn 54 / fn 66 → sensitivity 0.015, specificity 0.915.**

**This is the rule where the lead-oracle column reverses the apparent result, and the reversal
is the honest reading.** As shipped, the sagittal stratum looks informative (0.792). But
`resolve_lead_side` selects the *more flexed* knee by construction, so "the maximum angle of the
selected knee" is a biased statistic of the pair rather than a measurement of the lead leg — and
when the leg is taken from the label instead, separation collapses to **0.320 on that stratum
and 0.390 overall, i.e. below chance**. At matched n=80 the same stratum reads **0.792 shipped
vs 0.300 labeled**, so the collapse is not a denominator artifact either. The apparent signal was
an artifact of the selection rule, not evidence that incorrect reps are shallower.

**Conclusion: on this dataset, lead-knee depth carries no usable information about rep
correctness — with the caveat that 48 of 174 reps could not have fired at any knee angle.**
"6 fires in 174" is therefore partly a statement about window length, not only about which
errors the dataset contains; the conditional table (1 fire in 126 actionable reps) is the fair
version, and it is no better. This is a plausible property of the data rather than of the rule
— REHAB24-6's
instructed errors for a lunge need not include "did not go deep enough", and the honest
statement is that the fault is **not exercised here**, not that the rule works or fails.

The threshold is also nowhere near the action: 100° sits at percentile **84.5** of the observed
maximum-lead-knee-angle distribution, so it fires on 6 of 174 reps. That is `rank_auc`'s
documented case — a cited cut sitting in the tail of the distribution. **It does not move.**

### 4.3 `lunge_knee_valgus` — cam17, spec threshold 0.10

| Cut | Per-subject median AUC (range) | n scored | **Lead-oracle** median | n | Fired | Sens / Spec | Threshold pctile |
|---|---|---|---|---|---|---|---|
| All 174 | **0.590** (0.263–0.852), 7/8 subj | 149 | 0.620 (0.370–0.708) | 164 | 99 (57%) | 0.667 / 0.551 | 26.2 |
| `front` (88) | 0.600 (0.067–1.000), 7/8 subj | 70 | 0.650 (0.056–1.000) | 78 | 28 (32%) | 0.429 / 0.821 | 48.6 |
| `half-profile` (86) | 0.600 (0.200–1.000), 7/8 subj | 79 | 0.760 (0.400–1.000) | 86 | **71 (83%)** | 0.915 / 0.282 | **6.3** |
| extra-person-clean (134) | 0.629 (0.139–0.917), 7/8 subj | 112 | 0.639 (0.133–0.810) | 124 | 75 (56%) | 0.676 / 0.583 | 24.1 |

Structural silence in the all-174 table: 0 view-gated, **32 could-not-fire** (20 unresolved lead
side). **Conditional on the 142 reps where the rule could act: tp 64 / fp 35 / tn 25 / fn 18 →
sensitivity 0.780, specificity 0.417** — so the unconditional 0.667 / 0.551 was understating how
freely this rule fires, not overstating it.

This is the only rule whose numbers barely move under the lead-oracle (0.590 → 0.620; at matched
n=149, 0.590 vs 0.614) — and the reason is a projection fact already documented in its docstring:
in a frontal view a knee's in-image flexion and its medial offset are the same degree of freedom,
so the proxy reads similarly whichever leg is selected.

**Weak but non-null separation, at ~0.59–0.63 per-subject median**, stable across the
extra-person split (excluding the 40 level-2/3 reps moves it 0.590 → 0.629, so MediaPipe
person-locking is not driving it).

**The `half-profile` stratum's sensitivity 0.915 must not be read as a result.** The rule fires
on **83% of reps in that stratum** — 43 of 47 incorrect and 28 of 39 correct. The spec threshold
sits at percentile **6.3** there: nearly every rep exceeds it. A rule that fires on almost
everything has high sensitivity trivially. The fire rate, not the sensitivity, is the primary
read.

**Contamination check** (predicted before the run: the proxy sums medial *and* anterior knee
travel in every view production reaches, so a deep well-tracked lunge reads as valgus). Spearman
rank correlation of the valgus proxy against the lead knee's bottom-phase angle, **within the
78 correct reps only**, where by definition there is no valgus to find — the contamination
signature is a negative correlation (deeper = smaller angle = larger proxy):

| Lead leg used | Correct reps | Incorrect reps |
|---|---|---|
| Labeled (clean read) | **ρ = −0.325** (n=72) | ρ = −0.211 (n=89) |
| `resolve_lead_side` | ρ = −0.536 (n=63) | ρ = −0.282 (n=83) |

**Contamination is present but modest, and weaker than predicted.** On the clean read the
correct reps show ρ = −0.325 — the predicted sign, and stronger on correct reps than incorrect
ones (−0.325 vs −0.211), which is the predicted *shape*. But −0.33 is a weak monotone
association, not the "fires on deep reps of both classes equally" picture the prediction
described. The honest verdict: **step depth explains part of this rule's firing, not most of
it.** The threshold does not move, and the docstring's limitation stands as written.

### 4.4 `lunge_pelvic_drop` — cam17, spec threshold 8°

| Pass / cut | Per-subject median AUC | n scored | **Lead-oracle** median | n | Fired | Sens / Spec (all) | **Sens / Spec (actionable)** | Thr. pctile |
|---|---|---|---|---|---|---|---|---|
| **Production**, all 174 | 0.613 (0.093–0.986), 7/8 | 149 | **0.467** (0.000–0.857) | 164 | **10** | 0.042 / 0.923 | **0.103 / 0.750** (n=63) | 54.4 |
| Production, `front` (88) | 0.500 (0.000–1.000), 7/8 | 70 | 0.500 | 78 | 10 | 0.082 / 0.846 | 0.103 / 0.750 (n=63) | 64.3 |
| Production, `half-profile` (86) | 0.500 (0.111–1.000), 7/8 | 79 | 0.333 | 86 | **0** | 0.000 / 1.000 | **n/a (n=0)** | 45.6 |
| Production, extra-person-clean (134) | 0.679 (0.083–1.000), 7/8 | 112 | 0.524 | 124 | 6 | 0.054 / 0.967 | 0.167 / 0.895 (n=43) | 57.1 |
| **Oracle**, all 174 | same metric, same AUC | 149 | 0.467 | 164 | **41** | 0.240 / 0.769 | 0.284 / 0.700 (n=141) | 54.4 |
| Oracle, `half-profile` (86) | 0.500 | 79 | 0.333 | 86 | **31** | 0.404 / 0.692 | 0.452 / 0.667 (n=78) | 45.6 |

**Read specificity first, as the spec's §6.5 requires — but read the RIGHT specificity.** The
predicted failure mode was *false positives on deep, correctly-performed reps* from split-stance
foreshortening.

The unconditional figures do not test that prediction:

- **The half-profile stratum's 1.000 is vacuous.** The rule fired zero times there because the
  view estimator's `side` mislabel gated it off on all 39 correct reps. A specificity for a
  silenced rule is not a measurement of anything.
- **The 0.923 counts 54 of its 78 correct reps as true negatives when 37 were view-gated and 18
  could not fire.** On the **24 correct reps where the rule could actually act, it false-fired on
  6 — specificity 0.750, a 25% false-positive rate.**

**So the evidence is consistent with §6.5's prediction materializing, not with its refutation.**
An earlier draft of this note recorded the opposite; that was the wrong number, and this is the
one place in the document where a number retires a stated risk, so the correction matters. A 25%
false-positive rate on the reps where the rule ran is exactly the shape "false positives on deep
correct reps" predicts. What this dataset cannot do is confirm the *mechanism*: with 6 false
fires there is no power to show they are concentrated on the deep reps specifically, so the risk
stays **open and unretired** rather than confirmed.

**Separately, the rule barely fires: 10 times in 174 reps in production (4 tp / 6 fp), and 1 fire
per ~6 actionable reps.** Per the brief's own instruction, a fire rate near zero on **both**
classes means **"not exercised by this dataset"**, not "the rule works". With the lead leg taken
from the label the metric sits at **0.467 per-subject median — chance**. Nothing here says this
rule detects Trendelenburg; nothing here says it does not. REHAB24-6's instructed lunge errors
evidently do not include a contralateral pelvic drop in any quantity this could measure.

Excluding the 40 level-2/3 extra-person reps moves specificity 0.923 → **0.967** (conditional
0.750 → 0.895) and the per-subject median 0.613 → **0.679**, so MediaPipe person-locking is not
what is driving the false fires.

**This is the one rule where the two passes diverge, and the gap is a GATE failure.**
Production fires 10, oracle fires 41. The cause is measurable and specific: **the view estimator
labels 84 of 86 cam17 `half-profile` reps as `side`**, and `rule_pelvic_drop` returns `[]` on
`side` (a frontal-plane tilt is meaningless in a sagittal projection — correct behaviour given
the label). The label is what is wrong. A half-profile view is not a sagittal one.

This qualifies Phase 0's headline. The `side` gate does open — 88/88 on genuinely sagittal cam18
reps, at 0.69–0.99 confidence, as reported in `notes/lunge-view-reconnaissance.md`. But **the
same estimator also emits `side` on 84 of 86 clearly non-sagittal cam17 reps**, so its `side`
verdict has good *sensitivity* and poor *specificity*. The same mislabeling silently downgrades
`lunge_knee_valgus` on those reps too (`side` ∉ `ALIGNMENT_OBSERVABLE_VIEWS` → observability
`medium`, confidence ×0.65), without changing whether it fires.

---

## 5. Honest conclusions

1. **The lead-side substitution is what failed validation, ahead of any threshold — and the
   failure is one of measurement, not of anatomy.** Accuracy 0.623 (cam17) / 0.474 (cam18)
   against the label. The premise it substitutes for the spec's "more flexed / **more anterior**
   foot" — that the lead knee is the more flexed one at the bottom — holds on only 59.8%/47.8%
   of reps as this pipeline computes knee flexion. But the two simultaneous cameras disagree
   about which knee is more flexed on 33% of reps, the two premise rates disagree by 12 points,
   and dropping MediaPipe's pseudo-depth swings cam17 from 59.8% to 17.2%. So the supported
   statement is that **the more-flexed cue, as measured here, does not identify the labeled lead
   leg from either available view** — not that it fails to identify it in lunges generally.
   **Consequently the indicated fix is a depth-robust lead cue, not necessarily a different
   cue**: this data cannot distinguish "the premise is wrong" from "the premise is unrecoverable
   from this projection". Recorded, not patched.
2. **`lunge_knee_past_toes`'s cue is informative; the rule as shipped cannot reach it.**
   Lead-oracle per-subject median AUC **0.833** on the sagittal stratum, versus **0.171** —
   inverted — with the shipped lead-side resolution; **0.850 vs 0.171 at matched n=80**, so the
   contrast is not a denominator artifact. This is the strongest single result in the run, and it
   is a result about the *metric*, conditional on fixing item 1.
3. **`lunge_knee_valgus` separates weakly but genuinely** (per-subject median 0.590; 0.629
   excluding contaminated reps), and is the least sensitive of the four to the lead-side
   failure. Its firing is partly explained by step depth (ρ = −0.325 on correct reps) — real
   contamination, weaker than predicted. **Validated only in the narrow sense that its signal
   carries some information about rep correctness on this dataset**, which is a lab recording
   with fixed cameras and instructed errors.
4. **`lunge_insufficient_depth` and `lunge_pelvic_drop` are not exercised by this dataset — and
   part of their silence is structural, not evidential.** Six and ten fires respectively across
   174 reps, and chance-level or below separation on the clean read. But **48 of 174 reps could
   not have fired `lunge_insufficient_depth` at any knee angle** (window too short or lead
   unresolved), and **117 of 174 could not have fired `lunge_pelvic_drop`** (84 view-gated, 33
   could-not-fire), so the raw fire counts are partly statements about window length and view
   labeling. The conditional tables — 1 fire in 126 actionable reps, 10 in 63 — do not rescue
   either. That is still not evidence the rules are wrong. Both faults may be entirely real and
   simply absent from, or invisible in, REHAB24-6's instructed lunge errors. **Neither threshold
   moves**; a threshold tuned to make a fire rate look better would no longer be the cited
   number, and its citation would become a false provenance claim.
5. **`lunge_pelvic_drop`'s predicted failure mode is NOT refuted — the evidence leans toward it.**
   On the 24 correct reps where the rule could act it false-fired on 6: **specificity 0.750**.
   The 0.923 and 1.000 figures that look reassuring are artifacts of counting view-gated and
   structurally-silent reps as true negatives. The §6.5 risk stays **open**; with 6 false fires
   there is no power to confirm the foreshortening mechanism specifically.
6. **The estimator's `side` label has poor specificity**, mislabeling 84/86 half-profile reps.
   That is a gate failure with a concrete cost — it silences `lunge_pelvic_drop` on half the
   dataset and downgrades `lunge_knee_valgus` there. No gate or threshold was changed in
   response.
7. **`LUNGE_DETECTOR.validated` stays `False`.** That flag drives the Beta badge and its meaning
   ("checked against labeled ground truth") is a product claim. Nothing measured here supports
   flipping it: one rule's metric is informative only under an oracle lead leg, one separates
   weakly, and two are unexercised.

## 6. What would be needed to go further

- **Per-fault labels.** The single largest limitation. Until a dataset says *which* fault a rep
  contained, no rule's precision can be measured — only whether its signal tracks a binary
  correct/incorrect verdict.
- **A DEPTH-ROBUST lead-leg cue.** Item 1 above bounds all four rules, and §3.1's controls say
  the shipped cue's failure is a projection problem before it is a biomechanical one — dropping
  one coordinate moves its accuracy by 43 points. So the first thing to try is the same
  more-flexed cue computed from a depth estimate that is actually a depth estimate (this repo has
  measured NLF/MeTRAbs recovering true depth where MediaPipe's `z` does not), before abandoning
  the cue itself. Other candidates, none implemented here: foot-index separation along the body
  axis, ankle-velocity asymmetry during the descent, a learned per-rep classifier. Each is a
  change to the detector and belongs to its own task with its own validation.
- **A second dataset, ideally in-the-wild.** REHAB24-6 is a lab recording. Any claim above is
  scoped to it.
- **A harness that genuinely bypasses `segment_reps`** (`replace(LUNGE_DETECTOR,
  rep_signal=None)` forces the `segmentation_disabled` whole-window path), so §4.2's isolation
  claim becomes true for the per-rule numbers too. §3's lead-leg finding already has its
  full-window variant in `full_window_premise`, but the §4 per-rule numbers are measured on
  re-cut sub-windows for 152/174 cam17 reps and 91/174 cam18 reps.
- **Better frame validity.** 26% of cam17 frames and 42% of cam18 frames carry no metrics at
  all under the all-or-nothing landmark gate. That is a ceiling on everything above.

---

## Files

- `src/rehab24/lunge_rule_validation.py` — the harness (pure helpers + orchestration + report).
- `scripts/rehab24/validate_lunge_rules.py` — thin CLI entry point.
- `tests/test_lunge_rule_validation.py` — 64 unit tests, no data dependency.
- `data/REHAB24-6/processed/lunge_rule_validation.json` — per-rep raw records (**gitignored**).
- `notes/lunge-view-reconnaissance.md` — Phase 0, which this note qualifies in §4.4.
