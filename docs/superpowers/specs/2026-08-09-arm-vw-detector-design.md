# Arm VW Rule Detector — Design Spec

**Status:** design spec · **Date:** 2026-08-09
**Movement:** Arm VW (scapular V-to-W protraction/retraction) · **Detectors after this one:**
10/16 · **Group D complete**
**Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`
§Arm VW (lines 1211–1252)
**Immediate predecessor:** `docs/superpowers/specs/2026-08-09-arm-abduction-detector-design.md`

---

## 1. Purpose and what is different this time

Group D runs Bicep Curl → Arm Abduction → Arm VW. Arm Abduction shipped as the ninth detector on
2026-08-09; this is the **tenth**, and the **last of Group D**. It ships the way every one since
Push-up has — cited rules, `validated=False`, Beta in the UI, no frontend edit.

The parent spec gives Arm VW four rules. **Three ship, one is registered permanently silent, and
two sub-criteria inside the shipping three are dropped or withdrawn:**

| # | fault_id (shipped) | Treatment | Cue |
|---|---|---|---|
| 1 | `vw_incomplete_excursion` | ships, one disjunct dropped (§5) | V→W arm-elevation swing under 40° |
| 2 | `vw_shrug_substitution` | **SILENT** (§4) | Shoulders rise when they should be depressing |
| 3 | `vw_loss_of_elevation` | ships, one disjunct **withdrawn** (§6) | The V never gets above 120° |
| 4 | `vw_lr_asymmetry` | ships, **view-gated** (§7) | One arm lags the other at the V or the W |

### 1.1 Four things make this movement different from the nine already shipped

**(a) The labeled data finally matches the variant the app models.** REHAB24-6 `Ex2` **is** arm
VW — 208 repetitions, 94 correct / 114 incorrect, 9 subjects, 12 videos — and it is
**bilateral**, verified by measurement rather than inferred from a blank field (§2.2). Arm
Abduction's spec had to reach for Fit3D `side_lateral_raise` because its labeled set (Ex1) was
unilateral on 178/178 reps and therefore could not speak to a two-arm rule at all. **This spec
does not need a second dataset for that purpose.** At 208 reps Ex2 is also the **largest labeled
set of any non-squat movement so far** (Lunge 174, Arm Abduction 178).

**(b) The neck-gap shrug construction fails a second time, under a reversed excursion, and this
time the failure mode is different.** Arm Abduction §3 measured `neck_gap = ear_y − shoulder_y`
firing on 96.6% of MediaPipe reps because the shoulder rises with the arm. Arm VW's pull-down
runs the arm the *other* way, and the spec's own mitigation ("flag only where depression is
expected") is structurally sounder as a result — so the rule was not silenced by inheritance. It
was re-measured, and it fails for a **new** reason: the rep opens at the V, the most-shrugged
position in the whole movement, so an 18%-shrink-from-baseline test fires on **0/208** reps on
three separate instruments while the underlying metric still tracks arm elevation at
ρ = −0.957. Broken in the opposite direction, and just as unusable. §4.

**(c) The Arm Abduction module's stated reason for *not* gating its asymmetry rule is refuted by
measurement here.** `arm_abd_lr_asymmetry` ships live on every view on the argument that
"obliquity foreshortens both arms together, so a real asymmetry reads smaller — a missed fault,
never a false one." On Ex2, split by camera orientation: from a **true frontal** view MediaPipe's
`|L − R|` sits at 5.9° against the markers' 4.6°, and the argument holds. From a **half-profile**
view it sits at **16.0°** against the markers' **4.1°**, and the 12° threshold fires on **66 of
99** reps that the 3-D truth says are symmetric. Obliquity does not foreshorten the asymmetry; it
**fabricates** it. §7 gates this movement's asymmetry rule accordingly. §7.3 says exactly why
`arm_abduction.py` is **not** edited on this branch.

**(d) It is the first shipped rule scoped to `setup`, and the margin is 1.25×.** Bicep Curl §4.3
found the arithmetic that silences a phase-scoped rule — `phase_fraction · T ≥ min_frames / fps`
— and Arm Abduction avoided it by scoping nothing to the 15% `setup` window. Rule 3 has to read
the V, and the V *is* the opening of the rep. Measured through the real `segment_reps`
(§8.3): it clears on **234/234** segmented reps, but the shortest is 1.67 s against a 1.333 s
requirement. That is a margin, not a comfort, and the end-to-end test pins it.

---

## 2. REHAB24-6 Ex2 — and `validated` stays `False` anyway

### 2.1 What Ex2 is

From `Segmentation.csv`, `exercise_id == 2` (`EXERCISE_NAMES["2"] == "arm VW"`):

| Property | Value |
|---|---|
| Repetitions | **208** |
| Correctness | **94 correct / 114 incorrect** |
| Subjects | **9** (person ids 1–9), every one contributing both classes |
| Videos | 12, two orthogonal cameras each |
| `cam17_orientation` | 109 `front` / 99 `half-profile` / 0 `profile` |
| `mocap_erroneous` | 0/208 |
| Annotated rep length | 90–585 frames @ 30 fps = **3.00–19.50 s**, median 4.83 s |
| `exercise_subtype` | **empty on 208/208** |

Every Ex2 clip ships marker-driven 3-D (`data/REHAB24-6/Ex2/{video_id}-30fps.npy`, 26 joints per
`joints_names.txt`, **Y up**) alongside cached MediaPipe landmarks
(`data/REHAB24-6/processed/mediapipe_landmarks_cache/{video_id}-Camera17-30fps.npz`, all 12 Ex2
videos present, keys `image` (T,33,2) and `world` (T,33,3)). No extraction step is needed to
compare an estimate against a ground truth frame by frame.

**One subject is near-degenerate and every per-subject number in this document is reported with
and without it.** Person 8 contributes **2 correct / 20 incorrect**; the other eight run roughly
even. A per-subject AUC computed on 2 positives is not a measurement, and leaving it in the
median quietly moved two of the numbers below (§6.2, §5.2).

### 2.2 Ex2 is BILATERAL — measured, not read off the blank field

`Segmentation.txt` says `exercise_subtype` marks exercises where "we distinguish between right-
and left-sided execution". Empty could mean bilateral or unrecorded, and Arm Abduction's whole §2
turned on Ex1's `right arm`, so the question was settled from the markers rather than assumed:

- per-rep left/right **excursion ratio** (min/max): median **0.954**, p10 0.858, min 0.791
- **within-rep correlation** r(left, right) elevation: median **0.9977**, min 0.9628
- left excursion 50.5–122.9° (median 87.6), right 43.5–121.5° (median 88.4)

The two arms move as one unit. Ex2 performs the variant `movements.ts` offers, which is what
makes it usable for a two-arm rule — and is the single largest difference between this spec and
Arm Abduction's.

### 2.3 The rep shape, which settles the segmentation interface

Marker 3-D, average of the two arms, over the annotated windows:

| | median | p10 | p90 |
|---|---|---|---|
| elevation at the rep's first frame | 140.4° | 126.7 | 150.0 |
| elevation at the rep's last frame | 141.1° | 124.6 | 150.0 |
| **minimum** (the W) | **54.7°** | 33.4 | 72.8 |
| **maximum** (the V) | **143.8°** | 132.8 | 152.5 |
| position of the minimum within the rep | **0.508** | 0.409 | 0.640 |

V (high) → W (low, at the middle of the rep) → V. The effort peak is the signal's **minimum**, so
`rep_polarity="min"` — the Row / Bicep Curl polarity and the **inverse** of Arm Abduction's — and
the rep opens away from that extremum, so `rep_start="extended"`. §8.1 restates this as the
registry entry, and the end-to-end test is what actually verifies it.

### 2.4 Fit3D supplies a cross-check, and `w_raise` is not it

Fit3D ships both `w_raise` and `overhead_trap_raises` with `joints3d_25` mocap and `rep_ann.json`
boundaries. **The filename is not the movement**, which is why both were measured:

| | `w_raise` (42 reps) | `overhead_trap_raises` (41 reps) | Ex2 markers (208) |
|---|---|---|---|
| peak elevation, median | 75.4° | **147.4°** | 143.8° |
| trough, median | 49.7° | **67.9°** | 54.7° |
| swing, median | 28.0° | **76.9°** | 87.7° |
| duration | 1.90–4.16 s | 2.12–4.50 s | 3.00–19.50 s |
| within-rep r(L, R) | 0.9785 | 0.9989 | 0.9977 |

`w_raise` is a small-amplitude movement around shoulder height — its arms never go overhead, so
it is not a V-to-W cycle and nothing in this document is measured on it.
`overhead_trap_raises` matches Ex2's kinematic envelope closely (its rep boundaries fall at the
**W** rather than the V, i.e. the same cycle phase-shifted). **It is identified by that envelope
and not by a label**, so every Fit3D number quoted below inherits that assumption and is used
only as a second opinion, never as the sole support for a decision.

### 2.5 Why `validated` is still `False`

`ARM_VW_DETECTOR.validated` drives the Beta badge, and its meaning — "checked against labeled
ground truth" — is a product claim. **Nothing in this task runs the check.**
`notes/lunge-rule-validation.md` (869 lines, its own phase) is what a validation looks like: a
replay harness over the production path, per-subject AUC, structural-silence accounting, a
camera-routing decision and a written verdict. What §2 establishes is that the check is not only
possible but, for the first time, **unobstructed** — bilateral data, both classes on every
subject, both instruments cached. Recorded in TODO.md as a scoped follow-up, not attempted here.

Arm Abduction is the second movement with labeled data and the first whose data existed while the
check had not been run; this is the third and the second such gap. Two movements now carry that
debt, which is itself the argument for paying it.

---

## 3. The parent spec's citations, re-read — and what they turn out to study

Every quote below comes from the RAG doc, not from the parent spec's paraphrase.

| Source | What the parent spec uses it for | What it actually studies |
|---|---|---|
| Jung EY, Roh SY, Mun WL, *Life* 2025, **PMC12734928** | rule 1, "greater scapular excursion increases activation" | **quadruped and single-leg push-up-plus / sternum-drop**, EMG, three 2-second phases |
| Abiara S et al., *PeerJ* 2025, **PMC12335237** | rules 2 and 3 | four LT-activation exercises — **prone cobra, wall slide, scapula setting, prone trapezius exercise** — EMG, plus overhead functional tasks |
| Mun WL et al., *Medicina* 2025, **PMC12029123** | rule 3's elevation optimum | **Pilates Reformer "Arm Work"** at 0°/90°/135°/160° abduction, EMG |
| Terré M, Solana-Tramunt M, *Healthcare* 2025, **PMC12110944** | rule 4's 12° | mid/lower-**trapezius EMG symmetry** during bilateral scapular retraction |

**None of the four studies a standing open-chain V-to-W drill, and none reports a kinematic
threshold in any landmark unit.** All four are EMG papers. That does not invalidate the rules —
the mechanisms are real and the quotes are accurate — but it means every number in the parent
spec's Arm VW section is the spec author's, not a source's, and each rule below says so at its
constant. This is the generalised form of the lesson Arm Abduction §4 drew from the impingement
arc: *verifying that a source contains a quoted string is not verifying that it supports the
claim the quote is attached to.* Second movement, four for four.

---

## 4. `shrug_substitution` is REGISTERED-BUT-PERMANENTLY-SILENT — a second measured sensing failure

This project has two treatments for a rule it will not fire:

- **Registered-but-silent** (`pushup.rule_scapular_winging`, `row`'s fifth,
  `band_pull_apart.rule_loss_of_scapular_retraction`, `arm_abduction.rule_shoulder_shrug`):
  *real, well-cited fault; the sensor cannot see it.*
- **Withdrawn** (OHP bar-path, deadlift bar-drift, curl wrist-flexion, arm-abduction
  impingement arc): *no citation supports the rule as written.*

### 4.1 The citation holds

Abiara PMC12335237 states the mechanism directly: shoulder pain is "characterized by increased
activation of the upper trapezius and decreased activation of the lower trapezius and serratus
anterior", and "ratios lower than 1.0 for the UT/LT ratio are preferred (suggesting the LT is more
active than the UT), although lower than 0.6 are ideal … ratios >1.0 are considered non-optimal
for rehabilitation interventions". Jung PMC12734928 supplies the scapular-dyskinesis framing. The
fault is genuine. What fails is the sensing.

**One honesty note that does not change the treatment.** Abiara's `Exercise C` — "Participants
stood against the wall and began with their arm abducted to 90°, their elbows bent to 90°, and
their palms facing forward" — is the closest thing in any cited source to the W position, and the
paper reports its UT/LT ratio as **over 1.0**, i.e. it was *not* successful at favouring the lower
trapezius, and the authors conclude "only the Modified Prone Cobra (Exercise B) can be
recommended." The cited literature is therefore lukewarm about the exercise this rule set is built
around. Recorded, and out of scope: the rule set is the parent spec's.

### 4.2 The sensing fails, and this is the second movement on which it was measured rather than argued

The spec's proxy is `neck_gap = ear_y − shoulder_y` against a setup baseline, flagged during the
pull-down and W hold "where the shoulders should stay depressed". Measured on Ex2's 208 reps, with
each candidate "shoulder" taken as height above the mid-hip and the gap referenced to the rep's
opening frame:

| Point | ρ(gap, arm elevation) over the pull-down | 18% shrink fires | gap travel, % of baseline |
|---|---|---|---|
| marker **clavicle** (acromion — true scapular elevation) | med **−0.305** | **0/208** | 1.2% |
| marker **glenohumeral** | med **−0.998** | **0/208** | 36.3% |
| **MediaPipe** `\|ear − shoulder\|` | med **−0.957** (range −1.000..−0.404) | **0/208** | — |

Shoulder-height travel as a fraction of its own baseline: marker clavicle **0.6%**, marker
glenohumeral **9.8%**. **MediaPipe reports the glenohumeral joint, not the acromion** — the Arm
Abduction §3.2 finding reproduced on a second movement, under a **reversed** elevation direction,
which is what makes it a property of the landmark rather than of that movement.

**Two independent failures, and the second is new.**

1. **The metric is an arm-elevation readout, not a shrug readout.** ρ = −0.957 on MediaPipe
   against the arm's own elevation. Whatever it flags, it flags because the arm moved.
2. **The 18% threshold can never fire on this movement's baseline convention.** The rep opens at
   the **V** — arms overhead, shoulders legitimately at their most elevated. Every subsequent
   frame has a *larger* gap, so "shrink below baseline" is negative throughout the pull-down and
   the W hold. **0/208 on all three instruments.** This is the exact inverse of Arm Abduction,
   where the same construction fired on 96.6% of MediaPipe reps, and it is just as unusable.

And the cue carries no information about the labels: the clavicle-gap shrink scores **pooled AUC
0.484, per-subject median 0.549** at ranking incorrect reps above correct ones.

### 4.3 Not substituted, and the metric is not emitted

Shipping a different quantity under this fault_id would attach Abiara's UT/LT citation to
something Abiara says nothing about. And as in Arm Abduction, `shoulder_ear_gap` is **not emitted
at all**: no live rule reads it, and emitting it would force landmarks 7/8 into `required`, where
the all-or-nothing gate (§8.4) would let one lost ear silence the three rules that do fire.

**The KG is not the gap.** `Shoulder Shrug` → `Arm VW:Compensatory Shoulder Shrug`,
`quality_impacts: Shoulder Depression`. The metric is the gap.

**Open, recorded, not resolved:** a working shrug rule for this movement needs shoulder height read
*at matched arm elevation* — comparing like with like across the rep rather than against a
baseline taken at a different arm position. Arm Abduction recorded the same requirement. It is a
novel construction with no citation and no validation, and inventing it here is what §11 forbids.

### 4.4 What this does and does not say about `band_pull_apart.rule_shrugging`

`bpa_shrugging` ships **live** on this same construction, and Arm Abduction logged a check to run.
This measurement **narrows** that item without discharging it. The gap now measurably tracks arm
elevation on two movements whose elevation runs in **opposite** directions (abduction ρ = −0.957
rising, VW ρ = −0.957 falling), so the confound scales with the *magnitude* of the elevation
excursion rather than its sign. Band Pull Apart's excursion is horizontal abduction at roughly
fixed elevation, so its confound should be small — which is an argument **for** `bpa_shrugging`
being sound, not against it. **It is still not measured on Band Pull Apart's own data**, and no
claim is made here. TODO.md, unchanged in status, better informed.

---

## 5. `incomplete_scapular_rom` SHIPS on one disjunct — and its warrant is weaker than trunk-lean's

Shipped as `vw_incomplete_excursion`, fire threshold **swing < 40°** (FROM THE SPEC), scoped to
the whole rep window.

### 5.1 The second disjunct is dropped for a reason already measured

"or elbow fails to descend to within `0.05` (normalized y) of the shoulder line at the W" is
dropped for the reason Arm Abduction §6.7 established and pinned: **`0.05` in raw MediaPipe
normalized units is not a well-defined criterion**, because normalized image coordinates scale
with how much of the frame the subject occupies. Across the 43 production pose JSONs carrying a
usable shoulder width, the per-clip median `shoulder_width` runs **0.0591 to 0.4923** — an
**8.3× spread** — so `0.05` units is 0.102 shoulder-widths on the widest-framed clip and 0.846 on
the narrowest. `shoulder_width` is emitted here as a diagnostic for the same reason, and the
arithmetic is pinned by a test.

### 5.2 The first disjunct ships, and what it ships on is not what trunk-lean shipped on

| Instrument | `swing < 40°` fires |
|---|---|
| REHAB24-6 Ex2, marker 3-D (208 reps) | **0/208** |
| REHAB24-6 Ex2, MediaPipe image (208 reps) | **0/208** |
| Fit3D `overhead_trap_raises` (41 reps) | **0/41** |

Minimum observed swing: **47.0°** on the markers. The threshold sits below the entire observed
distribution on three instruments.

**It is not logically dominated by rule 3, and the tempting claim that it is would be wrong.**
Rule 3 is silent when V ≥ 120 and W ≥ 75; a rep with V = 120 and W = 85 satisfies both and still
has a 35° swing. So this is not the vacuous-branch defect that killed `row.rule_momentum_jerk`'s
second condition, Bicep Curl's elbow-displacement disjunct and the impingement arc's first
conjunct. It is a live branch that simply never fires on anything measured.

**Say plainly what the warrant is.** `arm_abd_contralateral_trunk_lean` shipped past an UNVERIFIED
citation because the cue scored a per-subject median AUC of **0.800**. This cue scores **0.452
(pooled 0.476)** across 9 subjects and **0.494 (pooled 0.502)** across the eight non-degenerate
ones — *exactly at chance*. It ships anyway, and on a different basis:

1. **Semantic correctness.** A rep whose arms swing less than 40° between the V and the W really
   is an incomplete V-to-W excursion. Firing on one is never wrong, whatever Ex2's error type
   happens to be.
2. **A background-cited mechanism.** Jung PMC12734928 states "greater scapular excursion is known
   to increase muscle activation" and reports that the larger-excursion variation "elicited higher
   trapezius activation, especially during large scapular excursions". Both quotes are real. The
   study is a **quadruped / single-leg push-up-plus**, not this drill, and it supplies no
   kinematic number — so the *mechanism* is cited and the *threshold* is the parent spec's.
3. **The AUC is evidence about Ex2's error type, not about the rule.** REHAB24-6 does not record
   which error each incorrect rep contains. That the excursion magnitude fails to separate its two
   classes says its incorrect reps are wrong some other way; it does not say a truncated rep is
   fine.

### 5.3 Attaching an arm metric to a scapular fault node is not the substitution §11 forbids

`vw_incomplete_excursion` measures **arm** elevation swing and points at
`Arm VW:Insufficient Scapular Retraction`. That looks like the metric substitution §11 forbids and
is not, because **the parent spec itself declares this rule a proxy**: "Use the visible
arm-excursion proxy for the (non-observable) scapular travel … True A-P scapular retraction is not
directly measured (see observability)", and rates it `medium` for the arm excursion, `low` for
true scapular protraction/retraction. The forbidden move is shipping metric B under a fault_id
whose citation is about metric A *without saying so*. This one says so, in the spec, in the
docstring, and in the observability rating the rule emits.

---

## 6. `loss_of_elevation_angle` SHIPS on the V — the W disjunct is WITHDRAWN

Shipped as `vw_loss_of_elevation`, fire threshold **V-phase elevation < 120°** (FROM THE SPEC),
scoped to `setup`.

### 6.1 The 120° is the low end of a cited optimum, never a stated fault threshold

Mun PMC12029123 was re-read. Its own finding: "The LT showed the highest muscle activity at the
shoulder abduction angle of 135° (p < 0.001)", measured at 0°/90°/135°/160° during a Pilates
Reformer arm-work movement. Its discussion cites other work — "Researchers recommend shoulder
abduction near 145°, aligning with the muscle fiber direction, for maximum LT activation" and "In
a previous study, the LT activation was the highest at 120° of shoulder abduction compared to at
30°, 60°, and 90°". Abiara adds that its LT-targeting prone exercise is performed with "arms
abducted above 90°".

So the literature gives an **LT-optimal band of roughly 120–145°**. Using **120° as a floor** is a
defensible reading of "stay in the band", but no source states it as a failure threshold, and the
parent spec never says where it came from. Recorded at the constant; **not moved**.

Measured, this places the threshold just under the observed distribution: Ex2's median V peak on
the markers is **143.8°**, sitting inside the cited optimum. That is the
`lunge_insufficient_depth` shape — a real cue whose cited cut lives in the tail — and this
project's settled treatment for it (`notes/lunge-rule-validation.md` §5.4) is *"Neither threshold
moves."*

### 6.2 It is the best-discriminating cue measured on this movement

Ranking Ex2's incorrect reps above its correct ones on the V peak (marker 3-D, low = fault):

| | pooled | per-subject median |
|---|---|---|
| all 9 subjects | 0.596 | 0.660 (range 0.000–1.000) |
| **8 subjects, person 8 excluded** | **0.713** | **0.735** (n=8, range 0.439–1.000) |

The near-degenerate subject was suppressing it. At 0.735 this is comparable to
`arm_abd_contralateral_trunk_lean`'s 0.800 and is the only cue in this movement that carries
real information about rep correctness.

**Fire rates, and one semantic note.** The parent spec says "V-phase **peak** < 120°", which
strictly means *the maximum over the V window is below 120*. The codebase idiom is a per-frame
mask plus `contiguous_true_segments`, which fires on any **sustained run** below 120 within the
window — a strictly weaker condition, so it fires more. Both are recorded rather than one being
silently chosen:

| Reading | markers | MediaPipe (image) |
|---|---|---|
| max over `setup` < 120 (spec-literal) | 6/208 | 0/208 |
| **sustained run below 120 in `setup` (shipped)** | **31/208** | **9/208** |

The shipped reading is the codebase idiom and is the more sensitive of the two; 31/208 = 15% on
3-D truth and 9/208 = 4.3% through the estimator are both plausible fault rates rather than a
false-positive machine. **`setup` is the *opening* V.** With `rep_start="extended"` the rep runs
V → W → V, so the closing V falls in `eccentric` and is not read; measured, the rep's global
maximum sits near the *end* on most reps (median argmax position 0.918), so reading only the
opening V under-reads the movement's best moment. That is the conservative direction — a missed
fault, never a false one — and it is stated rather than repaired.

### 6.3 The W disjunct is WITHDRAWN, for two reasons that both hold

"or W-phase abduction < 75° (elbows collapsed toward the body)".

**(i) The 75° appears in no cited source.** Mun measures 0°/90°/135°/160°. Abiara's wall slide
begins "abducted to 90°" and its prone exercise is "above 90°". Terré tests 45° and 90°. There is
no 75° anywhere, and no source describes a *floor* on the W position at all.

**(ii) The quantity the detector can compute puts the entire observed distribution below the
cut.** `angle(hip, shoulder, elbow)` is a frontal-plane reading, and in the W the elbow travels
down **and back** — an anterior-posterior component the parent spec itself rates non-observable
from a monocular frontal view. The consequence is measured:

| Instrument | median W elevation | `< 75°` fires |
|---|---|---|
| Ex2 marker 3-D | 58.4° | **187/208** |
| Ex2 MediaPipe image | 24.6° | **206/208** |
| Fit3D `overhead_trap_raises` | 67.9° | **39/41** |

A criterion that fires on 90–99% of reps in a dataset that is 45% correct is not measuring a
fault. Its discrimination confirms it: per-subject AUC **0.360** across 9 subjects, **0.510**
across the eight non-degenerate ones — at chance, and the apparent inversion was a person-8
artifact.

**Withdrawn, not silent, and the distinction is load-bearing.** A silent stub asserts "real fault,
the sensor cannot see it". The sensor sees frontal-plane elevation angles fine — it is the *number*
that has no source and the *quantity* that does not capture what the spec meant by the W. So the
disjunct is **absent**, exactly as the impingement arc is absent from `arm_abduction.py`.

**Recorded and not acted on:** whether a real W-position rule is possible needs either a source
that puts a number on the W, or a metric that captures the A-P component the frontal reading
loses. Neither is invented here.

---

## 7. `lr_vw_asymmetry` SHIPS — and it is the first asymmetry rule in this project to GATE on view

Shipped as `vw_lr_asymmetry`, fire threshold **`|L − R| > 12°`** (FROM THE SPEC), scoped to
`setup` ∪ `peak` (the spec's own "at the V peak and at the W hold"), **gated to
`{front, rear}`**.

### 7.1 The 12° has the same non-provenance it has in Arm Abduction

Terré & Solana-Tramunt PMC12110944 measures **middle- and lower-trapezius EMG symmetry** during
bilateral scapular retraction at 45° and 90° of abduction, and every threshold in it is a
**percentage**: "asymmetries between 10% and 15% are often associated with a higher risk of injury
and reduced performance", on a limb-symmetry scale of 0–79 / 80–89 / 90–100. **No angular
threshold appears anywhere in the paper.** Shipped unchanged anyway, following
`ohp_asymmetric_press` and `arm_abd_lr_asymmetry`, with the mismatch written at the constant.
Re-expressing the rule as a percentage was considered and rejected: changing units changes what
fires, which the no-tuning rule covers.

### 7.2 The gate, and the measurement that forces it

`|L − R|`, taken as the maximum over each window, against the 12° cut, split by `cam17_orientation`:

| cam17 | instrument | V window: median (fires) | W window: median (fires) |
|---|---|---|---|
| **front** (109 reps) | marker 3-D | 4.6° (3/109) | 6.4° (12/109) |
| | **MediaPipe image 2-D** | **5.9° (13/109)** | **7.4° (20/109)** |
| | MediaPipe `world` 3-D | 27.0° (107/109) | 27.8° (104/109) |
| **half-profile** (99 reps) | marker 3-D | 4.1° (0/99) | 5.8° (5/99) |
| | **MediaPipe image 2-D** | **16.0° (66/99)** | **22.2° (88/99)** |
| | MediaPipe `world` 3-D | 28.8° (96/99) | 20.4° (86/99) |

Read the two MediaPipe-image rows against their marker rows. **From a true frontal view the
difference metric behaves** — 5.9° against 4.6°, and the common-mode-cancellation argument holds.
**From an oblique view it is fabricated** — 16.0° against 4.1°, and the shipped threshold fires on
**66 of 99** reps the 3-D truth calls symmetric. The near arm and the far arm foreshorten by
*different* amounts, so obliquity does not shrink the asymmetry; it manufactures one.

**MediaPipe's own 3-D does not rescue it** — `world` is worse on both views, firing on 107/109
frontal reps. `world` is a metric hip-centred output and is **not** the same tensor as the image-z
that `angle_degrees(dims=3)` consumes in production, so this is a proxy in both directions; what
it rules out is the hope that adding depth fixes the metric.

So this rule **gates**, joining `band_pull_apart` and `bicep_curl` (which gate out the views where
their sagittal metrics read the wrong plane) rather than `arm_abduction` (which discounts and
stays live everywhere). The gate is not an invention: the parent spec rates this rule `high` on
**front/rear** specifically, and gating to that set is implementing the rating.

**State the ceiling, because it is severe.** Production is `rear_oblique` 37, `rear` 9,
`unknown` 3, `side` 0 over 49 pose JSONs (§8.6). Gating to `{front, rear}` with `front`
unreachable means **this rule is live on 9 of 49 clips and silent on the other 40.** That is the
price of not firing falsely on two thirds of them, and it is the honest trade — but the doc should
not read as though the rule is broadly available. It is not.

**And two inferential steps sit underneath the gate.** Ex2's cameras are `front` and
`half-profile`, both front-hemisphere. The gate *excludes* the views where fabrication was
**measured** (obliquity) and *admits* one view where it was **not** (`rear`). Geometrically a
frontal-plane difference reads the same from behind, mirrored, and `|L − R|` is sign-invariant —
but MediaPipe's landmark regime on rear views is untested here. The 9 clips that earn `high` earn
it on that geometric argument, not on a measurement.

### 7.3 Why `arm_abduction.py` is NOT edited on this branch

`arm_abd_lr_asymmetry` ships live on every view, on the stated rationale that "obliquity
foreshortens both arms together, so a real asymmetry reads smaller — a missed fault, never a false
one". §7.2 measures that rationale to be false for the metric it names. The fix is one line. It is
**not applied here**, and the reason is evidence rather than scope:

- The measurement is on **Ex2** (arm VW), not Ex1 (arm abduction), whose unilateral variant makes
  its own false-positive rate unmeasurable in this metric.
- It is on the **`image` 2-D cache**, while production runs `angle_degrees(dims=3)`; the cache has
  no image-z, so the production path is not reproduced.
- It is on **front-hemisphere obliquity**, while production is rear-hemisphere.

Changing a shipped rule's firing behaviour across two inferential steps is precisely the move this
project's honesty rules exist to prevent — the same reasoning that stops a threshold from being
"repaired". So: the parent spec's Arm Abduction asymmetry NOTE is annotated, TODO.md carries a
scoped check to run against abduction's own data, and `src/pose/movements/arm_abduction.py` is
untouched.

### 7.4 Two sub-criteria are dropped

- **"or if `|wrist_y_L − wrist_y_R| > 0.05` (normalized)"** — the same frame-scale dependence as
  §5.1. Dropped, `shoulder_width` emitted so it stays checkable.
- **"sustained across reps"** — no rule in this codebase carries cross-rep state. `run_detector`
  scores one rep at a time and `merge_by_fault` reports the rep count afterwards. Absent rather
  than approximated.

### 7.5 What Ex2 says about this rule, which is very little

`|L − R|` on the marker 3-D scores per-subject AUC **0.378** over the V window and **0.536** over
the W window (0.375 / 0.513 without person 8), and exceeds 12° on 3/208 and 17/208 reps. Ex2's
incorrect reps are not asymmetric ones. As with §5.2, that is a statement about the dataset's
error type — and unlike Arm Abduction, it is a statement the dataset is *entitled* to make here,
because Ex2 is bilateral. It just does not support or refute the rule.

---

## 8. Detector design

### 8.1 Namespace `vw_*`, and the registry entry

The parent spec's ids are unprefixed (`incomplete_scapular_rom`, `shrug_substitution`,
`loss_of_elevation_angle`, `lr_vw_asymmetry`). Every movement after Squat prefixes, because
`merge_by_fault`, the analyses table and the frontend's `byFault` map all key on `fault_id` with
**no movement qualifier**. Shipped ids: `vw_incomplete_excursion`, `vw_shrug_substitution`,
`vw_loss_of_elevation`, `vw_lr_asymmetry`. Checked against the ids already in flight — no
collision with `arm_abd_shoulder_shrug`, `bpa_shrugging`, `ohp_asymmetric_press` or
`arm_abd_lr_asymmetry`; three shrug rules under three ids is the correct outcome given
`merge_by_fault` has no movement qualifier.

Registry table entry: `"Arm VW": ("avg_arm_elevation_deg", "min", "extended")` — §2.3.
`min` matches Row and Bicep Curl and is the **inverse** of Arm Abduction's `max`.

`FAULT_LANDMARKS` in `frontend/src/lib/pose.ts:55` covers only the five squat faults and gets no
entry here — a **pre-existing** gap shared by all eight non-squat detectors, out of scope (§12),
already in TODO.md.

### 8.2 Rep signal: the mean of the two arms

`avg_arm_elevation_deg`, for the reason Overhead Press, Bicep Curl §4.2 and Arm Abduction §6.2
each established and this spec re-measures rather than inherits: on Ex2 the two arms correlate
**r = 0.9977 within-rep (min 0.9628)** and on Fit3D `overhead_trap_raises` **0.9989 (min
0.9962)**. The arms move as one unit, so the mean is the same excursion with per-arm landmark
noise halved, while an extremum would inherit whichever arm was noisier on each frame. Per-rep
excursion of the mean on Ex2 markers: median **87.7°** (min 47.0).

Unlike Arm Abduction, there is **no unilateral variant to degrade on** — Ex2 is bilateral on
208/208 and `movements.ts` offers one Arm VW. The stated-limitation paragraph that Arm Abduction
needed does not apply.

### 8.3 Phases, and the `setup` margin

`setup → concentric → peak → eccentric`, assigned over the **per-rep** slice `run_detector` hands
to `assign_phases`, mirroring `row_assign_phases` and `bicep_curl_assign_phases`:

- `setup` — the first 15% of the rep window. **This is the V**, and rules 3 and 4 read it.
- `peak` — the **least**-elevated 30% of the rep (the 30th percentile of `avg_arm_elevation_deg`
  and below). **This is the W hold**, the parent spec's isometric. Row/Bicep Curl polarity; the
  inverse of Arm Abduction's 70th-percentile-and-above.
- `concentric` / `eccentric` — before / after the least-elevated frame (the pull-down and the
  return to V).
- `unknown` — any invalid frame, checked **before** the setup cutoff.

**The `setup` scope is the Bicep Curl trap, and the margin is measured on the REAL segmenter, not
on annotation windows.** A phase-scoped rule needs `phase_fraction · T ≥ min_frames / fps` with
`min_frames = max(3, ceil(0.20 · fps))` (`base.py:197`). Running the actual
`segment_reps(smoothed avg_arm_elevation, fps=30, polarity="min", rep_start="extended",
min_rep_seconds=0.4)` over all 12 Ex2 videos — which trims each window to the excursion, and is
therefore tighter than the annotations — yields **234 reps of 50–752 frames = 1.67–25.07 s**
(median 4.65 s):

| Window | fraction | frames: min / median | requirement | reps failing |
|---|---|---|---|---|
| `setup` | 15% | **7** / 20 | `T ≥ 1.333 s` | **0/234** |
| `peak` | 30% | 15 / 41 | `T ≥ 0.667 s` | 0/234 |

The shortest segmented rep is 1.67 s against `setup`'s 1.333 s requirement: a **1.25× margin**,
against `peak`'s 2.5×. At Fit3D's 50 fps (`min_frames` = 10) the same requirement is 1.333 s
against a 2.12 s shortest rep, a 1.59× margin. This clears — and it is the tightest clearance any
detector in this project has shipped with, so `EndToEndSegmentationTest` pins it explicitly
(§10). `min_rep_seconds` stays at `DEFAULT_MIN_REP_SECONDS` (0.4 s); the shortest segmented rep is
4.2× that.

### 8.4 Metrics — and the metric layer contains no thresholds

`arm_vw_compute_raw` / `arm_vw_assign_phases` emit scale-free per-frame quantities and phase labels
only. The only constant either may define is `_DEGENERATE_LENGTH = 1e-6`, a division-by-zero guard.

| Metric | Definition |
|---|---|
| `left_arm_elevation_deg`, `right_arm_elevation_deg` | `angle(hip, shoulder, elbow)` per side; ~0° = arm at the side, 90° = horizontal, ~180° = overhead |
| `avg_arm_elevation_deg` | mean of the two; NaN-tolerant. **The rep signal**, and rule 3's input |
| `arm_elevation_asymmetry_deg` | `\|left − right\|`; NaN if either side is missing. Drives rule 4 |
| `shoulder_width` | `dist(11,12)`; emitted as a **diagnostic only** so §5.1's and §7.4's scale measurements stay checkable |

**Rule 1 needs a rep-level excursion, and it computes it from `avg_arm_elevation_deg` inside the
rule** rather than as a metric key — a per-frame metric cannot hold a rep-level range, and
`run_detector` hands each rule the whole rep window, so the rule has everything it needs.

**Required landmarks, and the all-or-nothing rule.** `required` is both shoulders, both elbows and
both hips — **not** the ears (§4.3), and **not** the wrists (the only cue that would have read
them is §7.4's, which is not implemented). If `visible_point` drops any **one**, the frame is
`valid=False` and carries no metrics, so *every* rule goes silent for that frame. This mirrors
`pushup_compute_raw`, `ohp_compute_raw`, `lunge_compute_raw`, `row_compute_raw`,
`band_pull_apart_compute_raw`, `bicep_curl_compute_raw` and `arm_abduction_compute_raw`.

### 8.5 Rules

**Rule 1 — `vw_incomplete_excursion`.** Fires when the rep's `avg_arm_elevation_deg` excursion
(max − min over valid frames) is **< 40°** (**FROM THE SPEC**). Severity ramps 40° → 16°
downward (**RULE-LEVEL**, 0.4× the fire threshold — `lower_is_worse=True`, mirroring
`lunge_insufficient_depth`'s convention for a "not enough" quantity). KG query
`Insufficient Scapular Retraction`. §5.

**Rule 2 — `vw_shrug_substitution`.** Registered, permanently silent, returns `[]`. §4.
KG query `Shoulder Shrug`.

**Rule 3 — `vw_loss_of_elevation`.** Fires when `avg_arm_elevation_deg < 120°` (**FROM THE SPEC**,
low end of Mun's cited 120–145° LT-optimal band) on a `setup` frame. Severity ramps 120° → 60°
downward (**RULE-LEVEL**, 0.5×). KG query `Insufficient Scapular Retraction`. §6.

**Rule 4 — `vw_lr_asymmetry`.** Fires when `arm_elevation_asymmetry_deg > 12°` (**FROM THE SPEC**)
on a `setup` or `peak` frame, **and only when `view_type ∈ {front, rear}`**. Ramps 12° → 30°
(**RULE-LEVEL**, 2.5×, the `pushup.rule_hip_sag` convention). KG query `Muscle Imbalance`. §7.

Rules 1 and 3 share a KG query because the graph has one Arm VW ROM-adjacent fault node; §9.

### 8.6 View handling — one rule gates, two discount

**Production view census, re-measured for this spec rather than inherited** (Bicep Curl's doc
warns that an inherited view figure once stopped reproducing). Running
`estimate_view_for_pose(path, allow_front=False).view_type` over all 49 files under
`data/runtime/pose_json` on **2026-08-09**: **`rear_oblique` 37, `rear` 9, `unknown` 3,
`side` 0.** `front` and `front_oblique` are unreachable under `allow_front=False`
(`src/pose/view_estimation.py:14-16`). This reproduces the Arm Abduction census exactly.

- **Rules 1 and 3 discount, they do not gate.** Both read an arm-elevation *magnitude*, which is
  the right quantity from any view; obliquity foreshortens it, so a real shortfall reads as a
  *deeper* shortfall — the rule errs toward firing, in the same direction that a threshold placed
  below the observed distribution errs away from it. `high` on `{front, rear}`, `medium`
  (× `VIEW_UNAVAILABLE_CONFIDENCE_SCALE` = 0.65) elsewhere.
- **Rule 4 gates**, per §7.2. `high` on `{front, rear}`, silent elsewhere.

Written in the **positive** (`view_type in {"front", "rear"}`), as Arm Abduction does, because the
fully-observable set is the small one for frontal-plane quantities. `front` is listed even though
`allow_front=False` can never emit it: it is the spec's observability rating transcribed, and
`run_detector` is called with whatever label its caller supplies — the REHAB24-6 replay harness
(`src/rehab24/lunge_rule_validation.py` `ORACLE_VIEWS`) deliberately feeds the literal `"front"`.

**One magnitude caveat, stated once and applying to rules 1 and 3.** Production reads image-plane
angles, and MediaPipe's arm elevation on Ex2 is off by a **mean 25.4° per rep (p90 33.8°)** against
the markers — 20.5° on `front` clips, 30.9° on `half-profile`. It systematically **over-reads the
excursion**: median peak 166.4° against the markers' 143.8°, median trough 24.6° against 58.4°,
median swing 143.5° against 87.7°. Both directions push rules 1 and 3 **toward silence** (a bigger
apparent swing clears the 40° floor; a higher apparent V clears the 120° floor), which is why
their measured MediaPipe fire rates (0/208 and 9/208) sit below their marker rates (0/208 and
31/208). Missed faults, not false ones. The Ex2 MediaPipe cache stores image x,y only, so these
are pure image-plane projections; production's `angle_degrees(dims=3)` folds in MediaPipe's
estimated z, and under the RTMPose path (`z = 0.0`) the two coincide exactly.

---

## 9. KG queries — resolved before being written, not after

Each string was checked against `data/kg/sports_kg_v3.graphml` with
`retrieve_graph_context(query, movement="Arm VW")` — the function **production** calls, not
`resolve_nodes`. Observed results, not predicted ones:

| Rule | Query | Resolves to | Buckets |
|---|---|---|---|
| 1 | `Insufficient Scapular Retraction` | `Arm VW:Insufficient Scapular Retraction` | `causes: Limited Scapular Retraction` |
| 2 (silent) | `Shoulder Shrug` | `Arm VW:Compensatory Shoulder Shrug` | `quality_impacts: Shoulder Depression` |
| 3 | `Insufficient Scapular Retraction` | as row 1 | as row 1 |
| 4 | `Muscle Imbalance` | `Muscle Imbalance` (generic Cause node) | **ZERO buckets, 1 edge — DANGLING** |

**Rules 1 and 3 share a query, deliberately.** The graph has exactly three Arm VW fault nodes —
`Insufficient Scapular Retraction`, `Compensatory Shoulder Shrug`, `Trunk Lean Compensation` — and
no incomplete-elevation node. Both an under-swung excursion and an under-elevated V are the same
thing in the graph's vocabulary. Substituting the generic `Range Of Motion` node for one of them
was rejected for the reason Band Pull Apart, Bicep Curl and Arm Abduction all rejected it: it
returns full buckets whose `corrections` entry is `Wrapping Surface Adjustment`, meaningless here.
The two rules stay distinguishable by `fault_id`, `fault_name`, citation and evidence; only the
retrieved card is shared.

**Row 4 is the dangling-seed case, accepted, exactly as `arm_abd_lr_asymmetry` accepted it.** The
graph has no Arm VW asymmetry fault node; `Asymmetry` and `Left Right Asymmetry` resolve to the
generic `Symmetry` QualityDimension, which is *also* zero-bucket. The user-visible failure the OHP
KG fix (PR #48) targeted does not occur — `frontend/src/components/FaultCard.tsx:55-57` pushes a
causes/risks/cue rung only `if (...).length` and wraps the block in `rungs.length > 0`, so a
zero-bucket seed renders a **thinner** card, never an empty "Causes:" heading.

**Recorded, not acted on:** the graph carries `Arm VW:Trunk Lean Compensation` and the parent spec
gives Arm VW **no trunk-lean rule** — the exact mirror of Arm Abduction, where the graph carried
`Incomplete Elevation` and the spec had no ROM rule. Two movements, two unused nodes, in opposite
directions. Filling either needs a source; **no rule is invented here.**

Because the graphml is gitignored, authoring an Arm VW asymmetry node is a deploy step; logged
against TODO.md's existing "many faults have no KG node" item.

---

## 10. Testing

`tests/test_arm_vw.py`, following `tests/test_arm_abduction.py`:

- **Fixture builder** producing synthetic landmark frames with controllable per-arm elevation, so
  each rule can be driven across its threshold in isolation.
- **Per-rule threshold pins** — fires just past the spec threshold, silent just short of it, for
  all three live rules.
- **`rule_shrug_substitution` returns `[]` for every input**, including one built to shrug hard —
  and **the assertion must not be vacuous**. Bicep Curl's equivalent test was green for the wrong
  reason (the phase window was structurally too short to fire); this test asserts silence on a
  clip where **another** rule does fire, proving the frames reached the rules at all.
- **Rule 4's view gate**: fires on `front`/`rear` at full confidence, and is **silent** on
  `rear_oblique`, `front_oblique`, `side` and `unknown` — while rules 1 and 3 stay live at ×0.65
  on those same views in the same clip. That contrast is the point of §7 and a single test should
  show both halves.
- **The dropped `0.05`-normalized disjuncts' scale dependence** (§5.1, §7.4) pinned numerically,
  so a future edit that "restores" either has to confront the arithmetic.
- **`test_metric_keys_match_the_emitted_metrics_exactly`** — the two-way match between
  `ARM_VW_METRIC_KEYS` and what `arm_vw_compute_raw` emits. A key the tuple omits is silently
  dropped by `run_detector` and read back as NaN by every rule.
- **`EndToEndSegmentationTest`** — a synthetic multi-rep clip segmenting on
  `avg_arm_elevation_deg`, verifying rep count and per-rep phase assignment (the check that
  actually verifies the `min` / `extended` interface-design inference — and it would *pass* under
  `max` for the wrong reason unless it asserts the W hold is the `peak`, so it asserts that), and
  — per §8.3 — that **the `setup` window of the shortest realistic rep (1.67 s at 30 fps) still
  clears `min_frames` after `segment_reps` has trimmed the window**. This is the 1.25×-margin pin.
- **`tests/test_movement_registry.py`** — add `"Arm VW": ("avg_arm_elevation_deg", "min",
  "extended")` to the shared rep-signal table, and a `validated is False` case.
- **`tests/test_analyze_pose_service.py`** — its "unimplemented movement" example currently names
  **Arm VW** (rotated there by the Arm Abduction change) and asserts it is absent from the
  registry. It must be rotated again, to a Group E/F movement still unimplemented (`Sit-up`).

Commands: `.venv\Scripts\python.exe -m pytest tests/ -q` and
`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`. Two backend test
flakes are known-unrelated on this machine; check a failure against a baseline on `main` before
attributing it to this change.

---

## 11. Honesty constraints

- **No threshold tuning.** Every cited number stays as the parent spec states it. §5.2's finding
  that the swing threshold fires 0/457 across three instruments, §6.1's that the 120° sits under
  the observed distribution, and §7.1's that the 12° has no provenance in its citation are all
  **written up, not repaired**.
- **A dropped criterion is not a moved threshold, and the two are labelled differently.** §6.3
  withdraws the W disjunct, §5.1 and §7.4 drop the `0.05`-normalized ones. In each case the reason
  is that the criterion has no cited referent or no well-defined unit — never that its numbers
  looked wrong.
- **Every constant is labelled in-code as exactly one of `FROM THE SPEC` or `RULE-LEVEL CHOICE
  MADE HERE`.** Never blurred. All severity ramps are RULE-LEVEL.
- **Citations are copied verbatim from the parent spec at implementation time**, never recalled
  from memory; where this spec quotes a source directly the quote comes from the RAG doc, and §3
  records that **all four** sources study a different exercise than this one.
- **Every measurement names its dataset, its instrument and its window.** Marker 3-D, MediaPipe
  `image` 2-D and MediaPipe `world` 3-D disagree, and §7.2 quotes all three rather than the
  flattering one.
- **Per-subject AUCs are reported with and without person 8** (2 correct / 20 incorrect), because
  including it moved two of them (§6.2, §6.3).
- **`validated=False`**, with §2's evidence stated at the registration site — including that
  labeled data exists and is bilateral, so a future reader does not re-derive the false "none
  exists" claim that Deadlift, Row, Band Pull Apart and Bicep Curl carry.
- **No metric is substituted under another metric's fault_id.** The silent shrug rule reads
  nothing; the withdrawn W disjunct is absent, not re-pointed at something measurable; and §5.3
  states why an arm metric under a scapular fault node is the parent spec's declared proxy rather
  than a substitution.

---

## 12. Out of scope

- **Any frontend file.** `Arm VW` already exists in `frontend/src/lib/movements.ts` with its i18n
  key and card art; `/api/movements` derives from the registry, so registering the detector flips
  it from "Soon" to analyzable with no frontend edit.
- **Editing `src/pose/movements/arm_abduction.py`** to gate its asymmetry rule (§7.3) — logged in
  TODO.md and annotated in the parent spec, deliberately not applied.
- **`FAULT_LANDMARKS` entries** (§8.1) — a pre-existing gap across all non-squat detectors.
- **Regenerating the KG** to add an Arm VW asymmetry node (§9) — a deploy step.
- **Running the REHAB24-6 Ex2 validation** (§2.5) — its own phase, scoped in TODO.md.
- **Checking `band_pull_apart.rule_shrugging` against the §4 confound** (§4.4) — narrowed, not run.
- **Group E** (Sit-up, Shoulder Bridge, Leg Abduction), which is where the 11th detector goes.
