# Leg Abduction rule validation against REHAB24-6 Ex4

Design spec: `docs/superpowers/specs/2026-08-09-leg-abduction-detector-design.md`. Module:
`src/pose/movements/leg_abduction.py`. Harness:
`src/rehab24/leg_abduction_rule_validation.py`.

**This is the first movement in the 16-movement programme where the labeled data ran during
DESIGN and changed the rule roster.** Lunge (`notes/lunge-rule-validation.md`) was checked after
the fact and no rule changed. Here the check silenced `rule_insufficient_abduction_rom`, settled a
sub-clause the citation and the knowledge graph disagreed about, and corrected the working-side
resolver's construction. **No threshold was changed in response to anything below** — the shipped
cut is the parent spec's own ratio, re-expressed in a different reference frame by an identity,
and the silenced rule's cut is the parent spec's own number, left where it is.

`LEG_ABDUCTION_DETECTOR.validated` remains `False`. §1 is why.

Reproduce:

```
.venv\Scripts\python.exe scripts/pose/run_pose_extraction.py --dataset unlabeled ^
  --video-dir data/REHAB24-6/Ex4 ^
  --output-dir data/REHAB24-6/processed/leg_abduction_pose_json --no-video

.venv\Scripts\python.exe scripts/rehab24/validate_leg_abduction_rules.py ^
  --pose-dir data/REHAB24-6/processed/leg_abduction_pose_json ^
  --segmentation data/REHAB24-6/Segmentation.csv ^
  --out data/REHAB24-6/processed/leg_abduction_rule_validation.json
```

Pose extraction is the long pole (~1 h for 24 clips on this machine); the replay itself is a few
minutes. `--report-only` re-prints the report from the saved JSON. The JSON and the pose corpus
are gitignored; this note is the committed record.

---

## 1. What this measures — and what it does not

**REHAB24-6 labels each repetition `correct` or `incorrect` and never says which fault
occurred.** There is no per-fault annotation in the dataset. So when a rule fires on an incorrect
repetition, that is **not** evidence the rule found that repetition's actual error.

Everything below measures one thing: **does a rule's underlying signal carry information about
whether the repetition was performed correctly?** It does not measure per-fault precision, and no
number here should be quoted as if it did. That limit is exactly why `validated` stays `False`,
and it is a **fifth** distinct reason in this registry — the first that is not a gap in the
evidence but a ceiling on what the evidence can support.

Four further limits, binding rather than decorative:

- **The incorrect repetitions are performed incorrectly on request.** So the absence of a fault
  from the incorrect class is a fact about this protocol, not about the world. §4 leans on the
  other direction — a cut firing preferentially on the **correct** class — precisely because that
  direction is a statement about the rule rather than about the protocol.
- **This is a replay harness on production-path inputs, not the production path.** Each labeled
  window is handed to `run_detector` as its own clip, with the view label and confidence the
  production estimator really produces and every gate unmodified. Real analysis runs the whole
  recording and segments its own repetitions. `leg_abduction_assign_phases` labels the first 15%
  of its input `setup`, and here that input **is** the repetition.
- **Repetitions that took a fallback path contribute no score.** When `segment_reps` finds
  nothing inside a labeled window, `run_detector` hands the rules the whole window and
  `assign_phases` labels only its first 15% `setup` — so a `max()` over the "active" frames would
  include standing frames the segmented path would never have scored. Those repetitions are
  excluded from every AUC and the count is reported alongside it.
- **9 subjects is small and the repetitions are not independent.** Every AUC is reported per
  subject as well as pooled; the pooled figure is secondary. This project has twice been burned
  by a pooled number that collapsed within subject.

---

## 2. The dataset, and the cost of the eight-landmark gate

**210 repetitions, 120 correct / 90 incorrect, 9 subjects, 12 videos**, every subject
contributing both classes, so all 9 yield a per-subject AUC. `cam17_orientation` is 116 `front`
and 94 `half-profile`; `exercise_subtype` is 106 "left leg" and 104 "right leg", one leg per
video. `assert_dataset_shape` raises unless every one of those counts still holds, so a dataset
change cannot reach this note as a plausible-looking number.

**The eight-landmark validity gate is expensive: median 0.600, p10 0.000.** Half the frames of a
median repetition carry all eight required landmarks, and **at least a tenth of repetitions carry
none at all**. That is the price of requiring both ankles for the support limb — two landmarks
more than any other Group E module needs — and it is the visible cause of the segmentation
outcome below.

**35 of 210 repetitions (17%) took a fallback path** — `no_reps_detected` 17, `only_partial_reps`
18 — meaning `segment_reps` found no complete repetition inside the labeled window. Those
repetitions are excluded from every AUC in §4 and §5, which are therefore computed over
**163/210**. The exclusion is stated at every number rather than buried here.

---

## 3. The working-side resolver — the first in this registry with ground truth

Of the 175 repetitions that reached it:

| outcome | count |
|---|---|
| correct | **163** |
| wrong | **1** |
| refused as ambiguous | 11 |

**Accuracy when it answers: 0.994. Coverage: 0.937.** A further 35 repetitions never reached it,
because `segment_reps` had already declined the window — a segmentation outcome, counted
separately rather than folded into the resolver's refusals (an earlier draft of this harness
conflated them and reported the resolver's coverage as 0.781, which was wrong).

This matters beyond the number. Every side-relative quantity in the module is read off the leg
this function names, so a wrong answer would put the rule on the *stance* leg's landmarks. One
error in 164 answers, with the ambiguous cases declined rather than guessed, is the behaviour the
design assumed and the first time in this registry that assumption has been checkable at all.

**It also caught a design error during development.** The first construction referenced each thigh
to the *other* leg, which makes both quantities approximately the angle *between* the legs —
near-equal by construction. It scored **7 correct / 14 wrong / 30 refused** on the first 51
repetitions: worse than a coin flip. The fix (a trunk-referenced, side-independent pair, see
`leg_abduction._thigh_trunk_angles`) is the reason the table above reads as it does.

---

## 4. `abd_insufficient_rom`: the cut fires on the correct class

**This is why the rule is registered permanently silent.**

Scored the way the rule would score it — lower peak abduction is worse — the cue's AUC is
**0.206 pooled**, and **every one of the 9 subjects is below chance**: 0.000, 0.000, 0.200,
0.200, 0.218, 0.238, 0.306, 0.333, 0.347 (median 0.218, max 0.347). This is not a weak signal.
It is a signal pointing the wrong way, consistently, in every subject.

The fire rates say the same thing in the units that matter:

| cut | fires on CORRECT reps | fires on INCORRECT reps |
|---|---|---|
| 25° | 23/93 (0.25) | 4/70 (0.06) |
| **30° — the parent spec's own cut** | **39/93 (0.42)** | **8/70 (0.11)** |
| 35° | 49/93 (0.53) | 14/70 (0.20) |

At the spec's threshold the rule would fault **42% of the repetitions humans judged correct**, at
nearly four times its rate on the incorrect ones.

**The number was not moved.** There is no cited value to move to — the parent spec says so itself
("the specific degree threshold is a practical target, not a value stated in the source"), and
González-de-la-Flor PMC12372021's only quantities for hip-abduction exercises are EMG amplitudes.
Lowering the cut until the false-alarm rate looked acceptable would be fitting a threshold to
labels.

**What this does and does not say.** REHAB24-6's incorrect repetitions are performed incorrectly
on request, so the *scarcity* of short repetitions in the incorrect class is a fact about the
protocol. The finding used here is the other direction — the cut fires preferentially on the
**correct** class — which is a fact about the rule.

---

## 5. `abd_pelvic_drop_trunk_lean`: the shipped disjunct, and the one that is not

### 5.1 The shipped trunk-lean signal

**AUC 0.840 pooled; per subject 0.690, 0.703, 0.790, 0.793, 0.833, 0.850, 1.000, 1.000, 1.000 —
median 0.833, minimum 0.690, and all 9 subjects above chance.** Computed over 163/210
repetitions.

At the shipped cut (0.15 of trunk length = 8.63° off the support limb):

```
fired 44/210    tp 39   fp 5   fn 51   tn 115
```

Precision 0.886, specificity 0.958, sensitivity 0.433. The rule is **conservative**: it misses
more than half the incorrect repetitions and is right about 39 of the 44 it flags. That is the
right shape for a coaching cue with a cited threshold, and it was not tuned to produce it — 8.63°
is `asin(0.15)`, the conservative end of the parent spec's own 0.10–0.15 band.

**Split by camera geometry**, which the dataset records per repetition:

| orientation | fired | tp | fp | fn | tn |
|---|---|---|---|---|---|
| `front` (116 reps) | 35 | 30 | 5 | 23 | 58 |
| `half-profile` (94 reps) | 9 | 9 | 0 | 28 | 57 |

The oblique camera costs **sensitivity**, not precision: 9 firings, zero false positives, 28
misses. A lateral lean projected obliquely reads smaller than it is, so the rule falls silent
rather than becoming wrong. That is the opposite of Shoulder Bridge's census, where the
unfavourable camera produced near-full-severity **false alarms** — and it is the benign failure
mode of the two.

### 5.2 The unimplemented pelvic-tilt disjunct, and what omitting it costs

The pelvic-hike signal scores **AUC 0.848 pooled, per-subject median 0.800, minimum 0.690** —
statistically indistinguishable from the shipped one. It is not omitted for want of signal (see
the design spec §6: the citation says pelvic *drop*, the knowledge graph has only `Pelvic Hiking`,
and the data separates on *hiking*).

**Rank correlation between the two signals: ρ = 0.713.** That is not high enough to call the
omission free. The two are related but carry genuinely independent information, so **a real
detection opportunity is being declined here**, and the design spec's argument has to carry that
weight rather than lean on redundancy. Recorded as a loss, not explained away.

---

## 6. The view estimator, checked against ground truth on an upright subject

`view_estimation.py`'s module docstring, limit 1, voids the front/rear/oblique labels for
**horizontal** subjects. Sit-up measured them inverted and Shoulder Bridge measured them
unstable, both inside that documented regime. Leg Abduction is the first Group E movement filmed
upright, and Ex4 records the true orientation per repetition, so this is the first chance to ask
whether standing up is enough.

It is not:

| dataset orientation | what production emitted |
|---|---|
| `front` (116) | `rear_oblique` **116/116** |
| `half-profile` (94) | `side` 92, `rear_oblique` 2 |

**Frontal-observable labels emitted across the whole corpus: 0 of 210.** The estimator is
systematically inverted — it calls the frontal camera oblique and the oblique camera sagittal —
on an upright subject, in the plane the exercise is performed in.

Two consequences:

- **The shipped rule's 0.65 confidence discount is a constant on this corpus.** It applies to
  every repetition equally and distinguishes nothing. Nothing in §5 is evidence that view gating
  works, and the production and oracle passes are byte-identical for the same reason (the rule
  discounts, it never gates).
- **Limit 1 understates the problem.** The failure is not confined to horizontal subjects. This
  is logged in `TODO.md` as an unscoped audit: at minimum `squat.rule_knees_inward` and
  `arm_abduction`'s two frontal rules gate or discount on these labels and have never been
  checked against ground truth.

---

## 7. What was NOT done

- No threshold was tuned. Both cuts are the parent spec's own numbers.
- `arm_abduction.rule_contralateral_trunk_lean` measures the same compensation from the IMAGE
  vertical and was not changed, although a support-limb reference now exists. Changing a shipped
  rule on another movement is out of scope for this branch.
- The `RunResult.fallback`-not-threaded-into-`RuleContext` gap (recorded by Deadlift, again by
  Shoulder Bridge) recurs here and is not fixed. This harness works around it by excluding those
  repetitions; a rule in production cannot.

---

## 8. The report, verbatim

```
REHAB24-6 Ex4 (standing leg abduction) -- Leg Abduction rule validation
==============================================================================
repetitions evaluated : 210 across 9 subjects
validity rate         : median 0.600 p10 0.000
                        (fraction of frames carrying all 8 required landmarks; the two
                         ankles are what no other Group E module needs)
rep-segmentation paths: {'None': 175, 'only_partial_reps': 18, 'no_reps_detected': 17}
                        reps on a fallback path contribute NO score to any AUC below --
                        they were handed the whole window, which is a different quantity

WORKING-SIDE RESOLVER vs the dataset's `exercise_subtype` -- the only side resolver in
this registry with ground truth to check against.
  correct 163   wrong 1   refused-ambiguous 11   of 175 reps that reached it
  accuracy when it answers 0.994   coverage 0.937
  a further 35 of 210 repetitions never reached
  the resolver at all: `segment_reps` found no complete rep inside the labeled window,
  so run_detector took a fallback path. That is a SEGMENTATION outcome, not a resolver
  one, and it is counted here rather than folded into the refusals above.

VIEW ESTIMATOR vs the dataset's recorded orientation. Leg Abduction is the first Group E
movement filmed UPRIGHT, so view_estimation.py's limit 1 -- which voids these labels for
HORIZONTAL subjects -- does not apply. This is the test of whether standing up is enough.
  cam17 front          -> rear_oblique x116
  cam17 half-profile   -> side x92, rear_oblique x2
  frontal-observable labels emitted: 0/210. The shipped rule's confidence
  discount therefore applies to every repetition equally -- it is a CONSTANT here, and
  distinguishes nothing. It is not evidence that the gating works.

RULES -- POSITIVE = the repetition is INCORRECT. The dataset never names the fault, so
these say whether the signal tracks correctness, NOT whether the fault was the one.
THE TWO PASSES ARE EXPECTED TO BE IDENTICAL, and that is the finding rather than a
confirmation: the only rule here DISCOUNTS confidence off a frontal view and never gates
firing, so there is no gate effect to separate from a rule effect. A DIFFERENCE between
the passes would mean a view gate had appeared that this writeup does not describe.
  abd_pelvic_drop_trunk_lean
    oracle     fired  44/210   tp  39  fp   5  fn  51  tn 115
    production fired  44/210   tp  39  fp   5  fn  51  tn 115

THE SAME RULE, SPLIT BY THE CAMERA GEOMETRY THE DATASET RECORDS.
  abd_pelvic_drop_trunk_lean
    front          fired  35/116   tp  30  fp   5  fn  23  tn  58
    half-profile   fired   9/ 94   tp   9  fp   0  fn  28  tn  57

HOW MUCH THE UNIMPLEMENTED PELVIC-TILT DISJUNCT COSTS -- rank correlation between the
shipped signal and the one deliberately left out.
  trunk lean vs pelvic hike: rho 0.713

SIGNALS -- threshold-free AUC, pooled AND per subject. The pooled figure is secondary:
210 repetitions come from 9 people and are not independent observations.
  trunk_tilt_deg       pooled 0.840   per-subject median 0.833 [0.690, 1.000] n=9   over 163/210 reps
    {'1': 0.7934, '2': 1.0, '3': 0.703, '4': 1.0, '5': 0.7909, '6': 0.6905, '7': 1.0, '8': 0.85, '9': 0.8333}
  abduction_deg        pooled 0.206   per-subject median 0.218 [0.000, 0.347] n=9   over 163/210 reps
    {'1': 0.3471, '2': 0.0, '3': 0.2182, '4': 0.2, '5': 0.0, '6': 0.2381, '7': 0.2, '8': 0.3333, '9': 0.3056}
  pelvic_hike_ratio    pooled 0.848   per-subject median 0.800 [0.690, 1.000] n=9   over 163/210 reps
    {'1': 0.7934, '2': 1.0, '3': 0.7636, '4': 0.9, '5': 1.0, '6': 0.6905, '7': 0.8, '8': 0.8167, '9': 0.6944}

FIRE RATES -- what each candidate cut would do to repetitions humans judged CORRECT.
This, not AUC, is what decides a 'not enough' rule.
  trunk_tilt_deg
    cut    8.63   correct-reps fire   9/ 93 (0.10)   incorrect-reps fire  41/ 70 (0.59)
    cut   12.00   correct-reps fire   2/ 93 (0.02)   incorrect-reps fire  17/ 70 (0.24)
    cut   15.00   correct-reps fire   1/ 93 (0.01)   incorrect-reps fire  11/ 70 (0.16)
  abduction_deg
    cut   25.00   correct-reps fire  23/ 93 (0.25)   incorrect-reps fire   4/ 70 (0.06)
    cut   30.00   correct-reps fire  39/ 93 (0.42)   incorrect-reps fire   8/ 70 (0.11)
    cut   35.00   correct-reps fire  49/ 93 (0.53)   incorrect-reps fire  14/ 70 (0.20)
```
