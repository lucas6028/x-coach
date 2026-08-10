# Jumping Jacks detector — design

Fifteenth of sixteen, and the second of Group F. Parent spec:
`docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`, Group F. Module:
`src/pose/movements/jumping_jacks.py`. Tests: `tests/test_jumping_jacks.py`. Validation harness:
`src/egoexo/jumping_jacks_validation.py` + `scripts/egoexo/run_jumping_jacks_validation.py`,
pure helpers tested in `tests/test_jumping_jacks_validation.py`. Result note:
`notes/jumping-jacks-rule-validation.md`.

**Outcome: no rule ships live. Two are permanently silent, three are withdrawn, and the detector
is deliberately NOT REGISTERED — the first time in the programme. The labeled data decided all of
it, and it decided against the two rules this document was originally written to ship.**

| parent-spec rule | outcome | decided by |
|---|---|---|
| `jj_incomplete_leg_rom` | **PERMANENTLY SILENT** | it clears every bar this programme sets — a primary sentence, a KG node grounded in *this* exercise, human corroboration that the fault is real — and the spec's 1.3 cut **fires on 79.1% of the repetitions humans judged correct** |
| `jj_incomplete_arm_rom` | **PERMANENTLY SILENT** | needs **no threshold at all** and the metric is clean; silent because its only source attributes an injury association, in the same document, to the range of motion the rule would coach users toward |
| `jj_knee_valgus_landing` | **WITHDRAWN, absent** | a zero-parameter control: replacing both knees with **perfectly straight-limb positions** still trips the 0.82 cut on **68.5%** of open-phase frames against 79.4% for the real knees. The rule reads stance geometry, not knee alignment |
| `jj_stiff_landing` | **WITHDRAWN, absent** | DeVita & Skelly's *stiff* landing is 77° of knee flexion; the rule fires below **20°**, so the paper's own stiff condition sits nowhere near the cut — and the cue carries a measured **+41.2° projection bias in the firing direction** |
| `jj_landing_asymmetry` | **WITHDRAWN, absent** | no KG node; a three-quantity disjunction that would put an arm, a foot or a knee behind one `fault_id`; and cross-rep state this architecture has never had |

And four findings that are not rule outcomes:

| finding | what it is |
|---|---|
| **The variant finally matches, and the labels are about different faults** | EgoExo-Fitness's 121 judged Jumping Jacks actions are the largest judged class in that dataset and the exercise is the right one. Its eight criteria and the parent spec's five rules **overlap in exactly one pair**. `validated=False` for the programme's **sixth** distinct reason. |
| **The KG's Jumping Jacks faults are seeded from two exercises blended, and the blend reproduces to the decimal** | the seeding script's "arm tension 8-27%" is `Jumping Jacks 8.3%` and `Clap Jacks 27.0%` — the two ends of a range spanning two different EgoExo action classes. |
| **`.ab` had never been tried, and it doubles the recoverable corpus** | Sit-up decoded `.aa` alone and reached 3 complete records. `.aa`+`.ab` is also a contiguous gzip prefix and reaches **6 complete records plus a partial 7th** — 11 judged actions, 3 simultaneous exo cameras each. |
| **A silent rule with a concrete upgrade path, which is new** | every earlier silent rule needed a paper nobody has written or a per-user baseline this architecture lacks. This one needs a **download**: EgoExo judges the foot-split criterion on 121 actions, 12 of them FAILED, so a cut could be read off human judgement — but the missing `.ac` part leaves only negatives reachable. |

---

## 1. Purpose, and what makes Jumping Jacks unlike the fourteen before it

Every previous movement in this programme was blocked on **data of the wrong thing**. Sit-up's
EgoExo corpus was a full sit-up where the spec wrote a curl-up; Shoulder Bridge's matching
records were behind the missing archive part; Torso Twist's three corpora contained three
different exercises, none of them the app's.

Here the data is of the right exercise, there is a lot of it, and it is labelled more richly than
anything this programme has met — per-criterion True/False by two or more annotators. The blocker
is new and it is sharper: **the labels judge different faults from the ones the parent spec wrote
rules for.** §2 establishes that with both taxonomies side by side, because it is the fact that
decides `validated` and it should not be paraphrased.

The second thing that is different is the shape of the reasoning that follows. Two of this
section's five rules were decided by a measurement this repository already held
(`notes/fit3d_view_dependence_summary.md`, which read every squat cue twice — once from mocap 3-D
truth, once from the 2-D projection in each of four calibrated cameras); the other three were
decided by arithmetic in a cited abstract, by the graph's negative filter, and — the one this
document changed its own mind on — by a **zero-parameter control run on the corpus itself**.

That last one is worth flagging at the top, because it is the lesson. The Fit3D table ranks
`knee_width_ratio` the **most view-robust cue this project has ever measured** and `knee_angle`
**view-corrupted with a +41.2° bias**, so this document was drafted shipping the valgus rule and
withdrawing the stiff-landing one. The valgus metric is indeed view-robust. It is also, in a
side-straddle, **a measurement of the stance rather than of the knees** — which no amount of
view-robustness fixes, and which only a control could show (§7.0).

---

## 2. The taxonomy mismatch, which is this movement's central fact

### 2.1 The corpus

`data/EgoExo-Fitness` carries **121 judged `Jumping Jacks` actions** (195 annotations) — the
largest judged action class in the dataset, ahead of Kneeling Side Torso Twist (165 annotations),
Sit-ups (142) and Shoulder Bridge (130). Each action carries a per-criterion True/False
verification from 2+ annotators and a 1–5 quality score. Its guidance text is unambiguously this
exercise:

> "tighten your waist and abdomen, and tense your arms. lift your arms with shoulder strength,
> press your arms down with back strength, and use your arms to drive your body to jump. **jump
> with your feet open and close**, relax your calves as much as possible, and do not lower or
> raise your head."

### 2.2 Two taxonomies, one overlapping pair

| EgoExo-Fitness criterion | fault rate (of 121) | parent-spec rule that models it |
|---|---|---|
| Perform the jump by opening and closing your feet | **9.9%** (12) | `jj_incomplete_leg_rom` |
| Keep your arms tense and ready for movement | 8.3% (10) | — |
| Press your arms down using the strength of your back | 8.3% (10) | — |
| Use the movement of your arms to help drive your body to jump | 6.6% (8) | — |
| Lift your arms using shoulder strength | 4.1% (5) | — |
| Relax your calves as much as possible during the jump | 4.1% (5) | — |
| Maintain a steady head position | 2.5% (3) | — |
| Tighten your waist and abdominal muscles for stability | 0.8% (1) | — |
| — | | `jj_knee_valgus_landing` |
| — | | `jj_stiff_landing` |
| — | | `jj_incomplete_arm_rom` |
| — | | `jj_landing_asymmetry` |

**One pair.** The corpus does not judge knee alignment, landing stiffness, how high the hands
travel or left–right symmetry; the parent spec does not model arm tension, back-driven arm
return, calf relaxation or head steadiness.

### 2.3 So `validated` is False for a NEW reason

Sit-up's reason — *the labeled data describes a different variant* — was reused by Shoulder Bridge
and held three times over for Torso Twist. **It does not apply here.** The variant is right. What
is missing is an overlap between what the data judges and what the rules claim, which means no
shipped rule here can be scored against a human judgement even though a large, well-labelled
corpus of the exercise exists.

This programme counts its distinct reasons carefully, and the temptation runs the other way — it
would be tidier to fold this into Sit-up's and keep the count at five. It is a sixth, and the
count is only worth keeping if it is kept honestly.

**What this does not say:** that the un-judged faults are unreal. EgoExo's criteria are that
dataset's *coaching guidance* decomposed into checkable statements, not an exhaustive taxonomy of
what can go wrong in a jumping jack. Knee valgus on landing has a real injury literature (§3.2)
and nobody asked those annotators to look at it.

### 2.4 The reachable subset, and the one question it can answer

`.aa`+`.ab` reaches **11** of the 121 judged actions (§8.1) — 10 with all three exo cameras and
one (`wNsRwL_action_9`) with only `exo_r`, from the record the stream truncates inside. **All
eleven are judged True on the foot-split criterion**; the only faults in the reachable set are
three "Keep your arms tense" failures in `yT4RK3`, which no rule models. So the reachable subset
can answer exactly one question about a rule — *does `jj_incomplete_leg_rom` fire on repetitions
humans judged correct?* — and **every firing is a false positive by the corpus's own judgement**.
There is **no positive class**, so no sensitivity, AUC or fault-level claim is possible and none
appears anywhere in this document. §5.1 is the answer, and it silenced the rule.

**And there is a third inferential step on top of the two this programme usually records, which
limits what any fire rate here means.** EgoExo-Fitness distributes *preprocessed* frames — its own
README says "Preprocessed video frames in 30 fps" and "currently the raw videos are not available"
— and the frames are **456 × 256**. Production input is phone video at 720p or better. Landmark
error is roughly constant in *pixels*, so in the *normalized* units these rules are built from it
is about **2.8× larger here than in production**, and they are ratios of small distances. A fire
rate measured on this corpus therefore bounds the rule's behaviour on
low-resolution footage, not on the footage it ships against. `arm_vw` refused to move a shipped
rule across two inferential steps (a different metric cache, a different obliquity regime); this
is a third, and the same refusal applies: **the number is reported, and no threshold moves.**

---

## 3. The citation audit

The parent spec cites two peer-reviewed papers and one Wikipedia RAG doc across the five rules.
Both papers were re-fetched and read; the RAG doc was read in place.

### 3.1 DeVita & Skelly (1992) — the arithmetic that withdraws `jj_stiff_landing`

*Medicine & Science in Sports & Exercise* 24(1):108–115, PMID 1548984. Re-fetched. Subjects landed
from **a 59 cm vertical fall** in instructed soft and stiff conditions:

> "Soft and stiff landings averaged **117 and 77 degrees of knee flexion**." … "The stiff landing
> had larger GRFs, but only the ankle plantarflexors produced a larger moment." … the hip and knee
> muscles absorbed more energy in the soft landing (hip −0.60 vs −0.39 W·kg⁻¹; knee −0.89 vs
> −0.61), the ankle muscles more in the stiff landing.

The parent spec's heuristic flags a landing whose "knee angle remain[s] > ~160°", i.e. **fewer
than 20° of bend**. The paper's *stiff* condition — the one it measured larger ground reaction
forces on — is **77° of bend, a knee angle of about 103°**, and would not come close to firing the
rule. The number 160 appears in the paper nowhere.

The spec additionally asserts that "the ≥/<90° knee-flexion soft/stiff convention originates
here". The abstract states 117 and 77 and states no convention. That claim is an inference about
the literature attributed to a specific paper — the *inference* failure mode this programme first
recorded on Arm Abduction's impingement arc, arriving in a new place: not in the rationale, but in
a claim about where a number comes from.

**What DeVita & Skelly does support**: that stiffer landings produce larger GRFs and shift energy
absorption from the hip and knee to the ankle. That is a real mechanism and §7.1's withdrawal does
not dispute it.

### 3.2 Tamura et al. (2017) — verified, and about a much harder task

*PLoS ONE* 12(6):e0179810, PMC5478135. Re-fetched. Task: a **single-leg drop vertical jump from a
40 cm box**. Dynamic knee valgus was classified from **3-D knee abduction angle at peak vertical
GRF** measured by an 8-camera motion capture system (valgus group 4.4 ± 3.0°, varus −5.3 ± 4.0°).
Result: knee angular impulse 0.093 vs 0.045 Nms/kg·m and hip 0.019 vs 0.067 (both p<0.01),
concluding valgus "may reduce the capacity to attenuate the impact imposed on the knee joint".

Three qualifiers the parent spec's VERIFIED mark does not carry, all of which the shipped rule's
`citation_support` string now does:

- the task is **single-leg** and from a box; a jumping jack is a bilateral low hop;
- valgus was a **3-D joint angle**, not any knee-to-ankle width ratio — **no width ratio appears
  in the paper**;
- so the 0.82 the rule fires at cannot have come from it: it is `squat.rule_knees_inward`'s
  own number, and §7.0 is why that transfer does not hold.

### 3.3 The RAG doc, and the paragraph that silences the arm rule

`data/rag/docs/jumping_jacks_wiki.txt` (Wikipedia, "Jumping jack", CC BY-SA) is the sole support
for both ROM rules, and the parent spec says so — "descriptive support only". It states the
targets:

> "a physical jumping exercise performed by **jumping to a position with the legs spread wide**.
> The **hands go overhead**, sometimes in a clap, and then return to a position with the feet
> together and the arms at the sides."

And, four sections later:

> "A similar jump exercise is half-jacks. **They were created to prevent rotator cuff injuries,
> which have been linked to the repetitive movements of the exercise.** They are like regular
> jumping jacks, but the arms go halfway above the head instead of all the way above it."

The parent spec noticed this and wrote it into `jj_incomplete_arm_rom`'s own rationale ("extreme
forced ROM is not automatically safer — this rule targets clearly incomplete reps") while still
proposing the rule. §6 treats it as disqualifying for a live rule, and the asymmetry with the leg
rule is the point: **the same document supports both targets, and only one of them has a
counter-indication attached in the same document.**

This is not the Torso Twist inversion (there the paraphrase reversed the source's instruction).
Nothing here is misquoted. It is a **counter-indication in the supporting source** — the eighth
distinct way this programme has found a `citation_support` string to be true while the rule built
on it should not ship.

### 3.4 What no source states

No source states: a stance-width ratio, a knee-to-ankle width ratio, a knee-angle cut for this
movement, an overhead-reach threshold, or an asymmetry percentage. Every number in this section is
the parent spec author's or this codebase's own, and each shipped constant says which in its
block comment.

---

## 4. The measurement layer

Three metric keys, each re-anchored away from the image frame.

| key | construction | invariant under |
|---|---|---|
| `stance_width_ratio` | `‖ankle_L − ankle_R‖ / ‖shoulder_L − shoulder_R‖` | roll, mirroring, scale |
| `knee_width_to_ankle_width` | `‖knee_L − knee_R‖ / ‖ankle_L − ankle_R‖` | roll, mirroring, scale |
| `hands_above_head_ratio` | `dot(wrist_mid − nose, trunk_up) / shoulder_width`; signed | roll, mirroring, scale |

### 4.1 Distances, not image-x differences

The parent spec writes all four of its width quantities as image-x differences — `|x25−x26| /
|x27−x28|`, `|x27−x28| / |x11−x12|`. A difference of x coordinates is not roll-invariant: at 90°
of camera roll the two ankles share an x and the ratio reads zero. The distance form agrees with
it **exactly** when the camera is upright and degrades gracefully when it is not, and
`MetricLayerTest::test_the_stance_ratio_is_a_distance_ratio_not_an_image_x_difference` asserts
both halves so the rewrite is not merely asserted to be equivalent.

**This is a correction to the spec's prose, not to the codebase.** `pose_rule_detector
.raw_frame_metrics` has computed `knee_width_to_ankle_width` from `distance(...)` since the first
detector; the coded squat rule was already right and the spec's `|x|` wording was already the
outlier.

### 4.2 The arm quantity is a comparison, not a number

The parent spec's arm criterion is "both wrists fail to rise above the nose (y15 and y16 > y0,
remembering y increases downward)". Read literally it needs the image vertical to be the world
vertical — the reference Group E spent three movements establishing is not recoverable from a
frame. Projected onto the **trunk axis** (`hip_mid → shoulder_mid`) the identical comparison
becomes a dot product onto a body axis: roll-invariant, mirror-invariant, and still exactly the
spec's criterion. Leg Abduction §1.2's rule — prefer a projection onto a body axis when one exists
— applied to an upright subject whose trunk *is* the axis.

Worth stating plainly because it is unusual in this programme: **this rule needs no threshold.**
The criterion fires at zero. §6 is therefore the first silent registration whose reason is not a
missing number.

### 4.3 Nine landmarks are read and only eight are required

`required` is both shoulders, both hips, both knees and both ankles — what the rep signal and both
live rules need. The wrists and the nose are read for `hands_above_head_ratio` and are **not**
required.

Every earlier module gated the whole frame on every landmark it read. Torso Twist required its
wrists because its **rep signal** was built on them; here the rep signal is the feet, and the
hands are the fastest-moving landmarks in the movement — they sweep a half-circle every repetition
and motion-blur at the top. Requiring them would let a blurred hand mark the frame invalid and
silence the two live rules, which never read it. The rule that does read the arm metric already
tests `np.isfinite`, which is how a per-metric gap is meant to be handled.

The principle is unchanged and only its application moved: **require what the rep signal and the
live rules need.** `ValidityGateTest` pins both directions — dropping any of the eight
invalidates, dropping a wrist or the nose leaves the leg metrics intact and NaNs only the arm one.

### 4.4 `open` is the landing window, and that is a deliberate substitution

Three of the parent spec's five rules are keyed to "the landing frame" — a single-frame impact
event. **An impact instant is not identifiable from landmarks**: there is no ground plane in the
image, no force plate, and Group E established that the image vertical is not the world vertical,
so "the lowest point of the hips" is not available either.

What *is* identifiable, and roll-invariantly, is the **wide-stance plateau**. The feet reach
maximum separation at touchdown and stay there through ground contact until push-off, so the
open-phase frames contain the landing by construction. Both live rules scope to `open` and
aggregate over it — the maximum stance width for the ROM rule, the minimum knee/ankle ratio for
the valgus rule.

**And the phase-fraction trap is avoided by construction rather than by luck.** Bicep Curl
established that a phase-*scoped* rule using `contiguous_true_segments(mask, min_frames)` is
structurally silent whenever `phase_fraction × rep_frames < min_frames`. Here `min_frames` is
`max(3, ⌈fps × 0.20⌉) = 6` at 30 fps, `open` is the top 30% of a repetition, and a 2 Hz jack is 15
frames — about 4 open frames. A mask-and-run rule would be silent on any jack faster than roughly
1.3 Hz. Both rules therefore test `min_frames` against the **whole repetition** and aggregate over
whatever open frames exist, which is `torso_twist.rule_trunk_not_braced`'s shape.
`PhaseFractionTest` pins the arithmetic and then pins that a 2 Hz clip still fires end to end.

### 4.5 The framework knob reserved for this movement is not needed by it

`base.py:55` names this movement by name: "`min_rep_seconds` for fast cyclic movements (jumping
jacks, high knees) … which the default would discard as noise", and the RS-SP1 audit prescribed
lowering it on the basis of "~1–2 Hz".

At 2 Hz a repetition lasts 0.5 s and **clears** the 0.4 s floor. The fastest jumping-jack cadence
with a citable number anywhere in this project's sources is the RAG doc's Guinness record — "the
most jumping jacks performed in one minute is 136" — which is 2.27 Hz, **0.44 s per repetition,
still above the floor**. §8.3 measures the real cadence on the recovered footage.

So the default stays, because lowering it would buy nothing and would admit shorter excursions as
repetitions. The framework comment is left alone: it also names High Knee, which runs at ~3 Hz and
is the sixteenth detector's problem.

---

## 5. `jj_incomplete_leg_rom` is PERMANENTLY SILENT, and the data silenced it

**This rule had more going for it than any other in the section.** It clears all three of the bars
Torso Twist section 5 set out, and one more no rule in this programme had cleared before:

1. **A primary sentence naming this exercise.** The RAG doc's "jumping to a position with the legs
   spread wide ... and then return to a position with the feet together".
2. **A knowledge-graph node grounded in the right exercise.** `Jumping Jacks:Incomplete Foot
   Split`, and the seeding script's own grounding figure -- "foot split 10%" -- reproduces from the
   labels as 9.9% (12/121). It is the one component of the Clap Jacks blend (7.3) that is correct.
   It is **dangling** (its only edge is `HAS_FAULT` back to the movement), a graph-content gap
   recorded and not a reason to withdraw.
3. **Human corroboration that people commit the fault**: it is the *most-failed* of the corpus's
   eight criteria.
4. **A clean metric.** `stance_width_ratio` is a ratio of two frontal-plane distances -- roll-,
   mirror- and scale-invariant, and first-order invariant to azimuthal obliquity because both terms
   foreshorten together (4.1).

### 5.1 And the spec's cut fires on the correct population

Replayed through the real `run_detector` over the 11 reachable judged actions -- 31 (action,
camera) pairs, 91 scored repetitions, every action judged **correct** on this exact criterion:

```
fire rate, per repetition, at the spec's 1.3 cut   79.1%   (of 91 scored repetitions)
fire rate, per (action, camera) pair               90.3%   (of 31 pairs)
median widest stance of a repetition               1.163 shoulder widths
```

**The correct population sits below the cut**, and not marginally: the median performance is 1.163
against a 1.3 threshold. Four of eleven actions clear 1.3 comfortably and seven do not -- subject
variation, not one outlier dragging a median. `notes/jumping-jacks-rule-validation.md` section 4
carries the per-action table.

**Three confounds do not explain it, checked rather than assumed.** Resolution inflates variance,
not a median (2.4). Obliquity cancels to first order, by the construction chosen in 4.1.
Segmentation is clean -- median validity 1.000, no action on the whole-clip fallback (8.2).

**The alternative reading is stated rather than dismissed:** the criterion may simply be laxer than
the rule -- an annotator asked whether the feet "open and close" may be answering *did they open at
all*, not *did they open wide enough*. That would make the 79% a disagreement about strictness
rather than an error. It does not change the outcome, because a rule firing on 79% of what the only
available human judgement accepts cannot be shown to a user.

### 5.2 The 1.3 is left where it is

A cut fitted to the observed distribution could be manufactured at will. Silencing rather than
moving is what this programme requires, and it is what `abd_insufficient_rom` did on the same
finding shape. `SilentLegRomRuleTest::test_the_specs_cut_is_kept_where_it_is_rather_than_moved`
pins the constant so a later quiet retune has to change a test that says why.

### 5.3 The upgrade path is a download, which is new

Every earlier silent rule in this registry needed either a paper nobody has written
(`abd_insufficient_rom`, `tt_insufficient_rotation_rom`) or a per-user baseline this architecture
does not have (`situp_excessive_speed`, `abd_momentum`). This one needs neither.
**EgoExo-Fitness judges this exact criterion on 121 actions, 12 of them FAILED**, so a cut
separating the classes could be read off human judgement rather than authored. What blocks it is
that `frames_open` is missing its `.ac` part, leaving 11 reachable and all 11 negative -- no
positive class. That is a download, not a research programme.


## 6. `jj_incomplete_arm_rom` is REGISTERED PERMANENTLY SILENT, for a new reason

Every previous silent rule in this registry is silent because a **number** or a **sensor** is
missing — `abd_insufficient_rom` and `tt_insufficient_rotation_rom` (no source states a range),
`bridge_lumbar_hyperextension` (the sensor cannot see the quantity). Neither is missing here:

- **No number is needed.** The criterion is a landmark comparison — are the hands above the head —
  and `hands_above_head_ratio` fires at zero (§4.2).
- **The sensor is fine.** A dot product onto the trunk axis; roll-invariant, mirror-invariant,
  scale-free, pinned by `InvarianceTest`.

**It is silent because its only source cautions against the range of motion it would coach users
toward.** §3.3: the same Wikipedia document that defines "the hands go overhead" records that
half-jacks exist because rotator-cuff injuries "have been linked to the repetitive movements of
the exercise". A rule that tells a user their arms did not go high enough is pushing them toward
the range that paragraph attaches an injury association to.

Two further failures, either sufficient alone:

- **No KG node means what this rule means.** The movement's three fault nodes are `Incomplete Foot
  Split`, `Poor Arm-Leg Coordination` and `Insufficient Arm Tension`. The only arm node is about
  **tension**, not **range**, and it is dangling. Seeding an arm-range card from an arm-tension
  node is Torso Twist's wrong-axis mistake in different clothing.
- **The labeled data does not judge it.** Three of the eight criteria concern the arms — tension,
  shoulder-driven lift, back-driven press-down — and none concerns how high the hands travel.

**Silent rather than withdrawn**, on the registry's usual distinction: the fault is real, the
metric works, the sensor can see it. What is missing is corroboration, so the rule is registered
where a future source can wake it up, and
`SilentArmRuleTest::test_the_metric_it_would_have_used_is_computed_and_correct` keeps the quantity
honest in the meantime.

---

## 7. Three rules are WITHDRAWN

### 7.0 `jj_knee_valgus_landing` -- a zero-parameter control withdraws it

**This document was originally written to ship this rule**, on the strength of
`notes/fit3d_view_dependence_summary.md`: over 40 squat repetitions x 8 subjects x 4 calibrated
cameras, `knee_width_ratio` is the **most view-robust cue this project has ever measured** (MAE
0.02 against a 0.82 cut, r=0.90, noise/sig 0.44, against `knee_angle`'s MAE 42.4 deg and noise/sig
1.21). That measurement is not wrong. It is about the wrong thing.

**The metric is confounded by the very stance this movement is defined by.** In a squat -- feet
about shoulder width, shanks near vertical -- `knee_width / ankle_width` is near 1.0 when the knees
track the feet, which is what makes 0.82 meaningful there. In a wide side-straddle the legs splay
from a pelvis that does not widen, so a knee sits partway along the hip-to-ankle line and its
separation is **necessarily** smaller than the ankles' -- with no valgus whatsoever.

**The control is zero-parameter.** Replace both knees with their projections onto the same-side
hip-to-ankle line -- a perfectly straight limb, zero valgus by construction -- and recompute:

```
over 2 353 open-phase frames of the 11 judged actions

  observed knees            median 0.769    below the 0.82 cut on 79.4% of frames
  PERFECTLY ALIGNED knees   median 0.810    below the 0.82 cut on 68.5% of frames
```

**Of the 79.4 points of firing, 68.5 are stance geometry.** About 11 points are attributable to any
inward deviation at all -- on a population every action of which a human judged correct. The rule
reads the movement, not the fault. This is the discipline that refuted the keypoint blind-spot
claims elsewhere in this project: **run the zero-parameter control before believing the cue.**

The mechanism is pinned independently of the corpus by
`tests/test_jumping_jacks.py::StanceGeometryConfoundTest` -- a perfectly aligned knee trips the
0.82 cut at a 1.6 stance, does *not* at a 1.0 stance, and the confound is monotone in stance width.
So the withdrawal does not rest on 456 x 256 footage.

Two further failures, neither of them the deciding one:

- **No KG node scoped to this movement.** `"Knee Valgus"` under `movement="Jumping Jacks"` matches
  two **shared** nodes carried by the Squat flagship (`Knee Valgus Load` -> ACL Injury,
  `Knee Valgus Control` -> Frontal Plane Stability, Joint Stiffness). Both are non-empty and both
  are about valgus, so this is the weakest of the three.
- **The citation measures a different and much harder task and supplies no ratio** (3.2).

**What would work, recorded and not built:** the deviation of each knee from its own hip-to-ankle
line, normalized by limb length -- zero for a straight limb at *any* stance width. No source states
a threshold for it, and inventing one is what this programme forbids, so the parent spec's rule is
withdrawn rather than replaced by an uncited construction.

**Not said by this withdrawal:** that knees collapsing inward on landing is fine. Tamura is a real
result about a real mechanism. What is missing is a quantity that measures it in a movement whose
stance is wide by definition.



### 7.1 `jj_stiff_landing` — three failures, and the first is self-contained

1. **The cited paper's own stiff condition would not fire the rule.** §3.1: soft 117° and stiff
   77° of knee flexion; the rule fires below 20° of flexion, so the very condition DeVita & Skelly
   measured larger ground reaction forces on — a knee angle of about 103° — sits nowhere near the
   cut. The number 160 appears in the paper nowhere, and the spec's claim that the ≥/<90°
   convention "originates here" is not in the abstract. **This failure needs no other measurement
   and is why it leads.**
2. **The cue is measured view-corrupted, and the projection bias runs toward the firing
   direction — with a bound that has to be stated.** The projected 2-D knee angle carries MAE
   42.4° with a **systematic +41.2° bias** and noise/sig 1.21, i.e. which camera you used matters
   more than what the athlete did; at one verified squat bottom the true knee angle is **78°** and
   the four cameras report **108° / 118° / 119° / 133°**.

   The direction is geometric and does transfer: the thigh and shank straddle the long axis, and
   an oblique camera compresses the fore-aft component of both, bringing them toward each other's
   line — so a projected knee angle errs **toward 180°**, which is the direction this rule fires
   in.

   **What does not transfer is the magnitude, and the reason is worth being precise about rather
   than waving at.** The bias is bounded above by `180° − θ_true`, so it necessarily shrinks to
   nothing as the knee approaches full extension. A genuinely stiff landing at 170° cannot be made
   to look 41° stiffer; there is not that much room. What the bias can do is open a **moderately
   absorbed** landing — 140°, i.e. 40° of flexion, a real absorption — past the 160° cut, and 40°
   of room is exactly the size of the measured bias. So the corruption bites **precisely in the
   band the rule has to discriminate in**, and nowhere else. That is a narrower claim than "the
   bias points the wrong way everywhere", and it is the one the geometry supports.
   *(Measured on squats, whose knee range is far larger. Only the direction and the bound are
   claimed here.)*
3. **No KG node.** `"Stiff Landing"` and `"Landing"` both return zero matches under
   `movement="Jumping Jacks"`.

**Not said by this withdrawal:** that landing stiff-legged is fine. DeVita & Skelly is a real
result about a real mechanism. What is missing is a threshold that survives being read from a
monocular camera.

### 7.2 `jj_landing_asymmetry` — three failures

1. **No KG node.** `"Asymmetry"` under `movement="Jumping Jacks"` reaches only the shared
   `Symmetry` quality dimension carried by the Squat flagship.
2. **It is a disjunction of three unrelated quantities** — "wrist peak height, ankle lateral
   excursion from hip-midline, and per-side knee-valgus ratio", firing if any differs by 15–20%.
   One `fault_id` whose evidence might be an arm, a foot or a knee cannot produce a coherent
   explanation card, and `fault_id` is the join key between the spec, the registry and every stored
   analysis. Arm VW kept its id through the loss of **one** branch; keeping this one would mean
   choosing which of three faults it is.
3. **"Consistently across reps" is cross-rep state this architecture does not have.**
   `run_detector` scores one repetition at a time; `arm_vw` recorded the same limit for the same
   spec wording.

**This is the first asymmetry rule this programme has withdrawn, and the reason is NOT the missing
number.** `ohp_asymmetric_press`, `arm_abd_lr_asymmetry` and `arm_vw.rule_lr_asymmetry` all ship on
spec-authored thresholds their citations do not state, and that precedent is not being reversed.

### 7.3 The graph's negative filter holds a third time

Recorded queries, run through `retrieve_graph_context(query, movement="Jumping Jacks")` — the
function production calls:

| query | result |
|---|---|
| `Incomplete Foot Split` | `Jumping Jacks:Incomplete Foot Split` — **dangling** (only `related_actions`) |
| `Insufficient Arm Tension` | `Jumping Jacks:Insufficient Arm Tension` — dangling |
| `Poor Arm-Leg Coordination` | one non-empty bucket (`causes: Poor Neuromuscular Control`) |
| `Knee Valgus` | **two SHARED nodes**, both non-empty — `Knee Valgus Load → ACL Injury`, `Knee Valgus Control → Frontal Plane Stability, Joint Stiffness` — neither scoped to this movement |
| `Stiff Landing` | `[]` |
| `Landing` | `[]` |
| `Asymmetry` | `Symmetry`, a shared dimension reached from Squat |

The two rules with **no node at all** are exactly the two withdrawn on other grounds — Leg
Abduction §7.3's finding, now reproduced on a third movement. The positive signal again predicts
nothing on its own: of the three scoped fault nodes, the one the shipped ROM rule seeds from is
**dangling**, the one with a real bucket corresponds to **no rule in the parent spec**, and the
third is about arm *tension* where the spec's arm rule is about arm *range*.

**And the seeding is a blend of two exercises, reproducible to the decimal.**
`scripts/knowledge/stub_general_movements_v3.py:133` records the grounding as *"EgoExo-Fitness TKV
(Jumping/Clap Jacks: arm tension 8-27%, foot split 10%, arm-leg coordination)"*. `Clap Jacks` is a
separate EgoExo class with 74 judged actions and its own guidance — "clap your hands while jumping
back and forth with **alternating feet**", pectoral-driven, no side-straddle. Recomputing the
seed's own statistic per class:

```
"Keep your arms tense ..."            Jumping Jacks   8.3%  (10/121)
"Keep your arms tense."               Clap Jacks     27.0%  (20/74)
"... opening and closing your feet."  Jumping Jacks   9.9%  (12/121)
```

so "arm tension 8-27%" is the two ends of a range spanning two different exercises, and "foot split
10%" is this exercise alone. Torso Twist found a node that faithfully described a *different*
movement; this is the milder cousin — a node seeded from a **blend**, of which one component is
correct, and the correct component is the one the shipped rule uses.

---

### 7.4 And so the detector is not registered

With every rule silent or withdrawn, `src/pose/movements/jumping_jacks.py` deliberately makes **no
`registry.register` call** -- the first time in the programme.

Registration is what makes a movement analyzable in the web app: `registry.list_detectors()` backs
`GET /api/movements`, and `analyze_pose_payload` routes to a detector when one exists and returns
`analysis_pending` ("coming soon") when one does not. Registering here would offer users an
analysis that **can never report a fault** while wearing the Beta tag that says faults are
possible. "Coming soon" is the truthful state of this movement.

**What works is kept, because none of it is what failed:** the metric layer, the phase assignment
and the `open` landing-window substitution, and the repetition segmentation -- measured on real
footage of this exercise at median validity 1.000, no fallbacks, 255 repetitions found, nothing
lost to the duration floor. Whoever obtains the missing `.ac` part, or any corpus with judged-faulty
jumping jacks, can read a threshold off human judgement, wake `rule_incomplete_leg_rom`, add one
line to `registry.py` and ship. `NotRegisteredTest` pins both halves: absent from the registry, and
complete as an object.


## 8. Measured

### 8.1 `.ab` had never been tried

Sit-up §2.3 recorded that the `frames_open` download is split into 3 GiB parts, that **`.ac` is
missing**, and that `.aa` "is the *prefix* of a single gzip stream and decodes standalone until it
runs out" — recovering `zOfbr6`, `zT0YQO`, `z8RAua` and a partial `yT4RK3`. Shoulder Bridge
inherited the same reachable set and recorded that only 2 of its 77 matching-variant actions were
inside it.

`.ab` is contiguous with `.aa`, so **`cat .aa .ab` is also a valid prefix** and decodes roughly
twice as far. Listing it:

| record | reachable |
|---|---|
| `zT0YQO`, `zOfbr6`, `z8RAua`, `yT4RK3`, `Y1t9Ew`, `xYkvB0` | complete (all 6 views) |
| `wNsRwL` | partial — `exo_r` complete, `exo_m` truncated mid-record |

Six complete records instead of three, and `yT4RK3` upgraded from partial to complete. **Eleven of
the 121 judged Jumping Jacks actions are inside that set** — 10 with all three exo cameras,
`wNsRwL_action_9` with `exo_r` only. Extractor: the scratch harness described in §10's
out-of-scope note streams `.aa`+`.ab` through `tarfile` in `r|gz` mode and writes only the
manifest's frame ranges for `exo_*`.

The single-camera action is carried through the harness rather than dropped, and
`cross_view_agreement` returns `None` for it: reporting one camera as "unanimous" would inflate
the agreement count with an action nothing could disagree about
(`AgreementTest::test_a_single_camera_is_not_agreement`).

*(This is a recipe correction, not a new capability: nothing about the archive changed, and the
earlier passes simply did not try the concatenation of the two contiguous parts.)*

### 8.2 The pipeline works; only the thresholds do not

Replayed through the real `run_detector`, 11 actions x 3 simultaneous exo cameras (one action has a
single camera), 31 (action, camera) pairs, 9 601 frames:

| | |
|---|---|
| median validity rate (the 8-landmark gate) | **1.000** |
| pairs on the whole-clip fallback | **0 of 31** |
| repetitions found | **255** |
| repetitions lost to the 0.4 s duration floor | **0** |
| median cadence | **0.93 Hz** |
| fastest cadence | **1.14 Hz** (0.88 s per repetition) |

This is the strongest pipeline result in Group F, and it matters because it isolates what failed.
The validity gate, the phase assignment, the landing-window substitution and the segmentation all
behave on real footage of the right exercise. What fails is two numbers.

**And it settles §4.5 by measurement.** `base.py:55` names this movement as one that "must lower"
`min_rep_seconds`; re-segmenting all 31 pairs at a 0.15 s floor finds **exactly the same 255
repetitions**. The fastest performer holds 0.88 s per repetition — more than twice the floor. The
measurement is deliberately non-circular: every window `segment_reps` *returns* is already at least
`min_rep_seconds` long, so differencing the counts at two floors is the only way to see the floor
bite (`floor_discarded`).

### 8.3 Three simultaneous cameras — and camera placement is not what is wrong

The exo rig films the same instant three ways, so any disagreement on one action is pure projection
error. Both rules being silent, the verdicts compared are what the parent spec's cuts *would* have
said.

| | cross-camera verdict | median cross-camera spread |
|---|---|---|
| `jj_incomplete_leg_rom` | 8 unanimous, **2 split** of 10 comparable actions | 0.107 shoulder widths |
| `jj_knee_valgus_landing` | 9 unanimous, **1 split** of 10 | 0.067 |

Both spreads are small against the distance between the correct population and the cut (1.163 vs
1.3 = 0.137; 0.769 vs 0.82 = 0.051). **That is the point**: unlike Sit-up (a 20° cut against a
28.2° cross-camera spread) and Torso Twist (a 15° cut against a p90 spread of 15.7°), camera
placement is *not* this movement's problem. Two of ten actions would nonetheless have received a
different verdict depending on which camera filmed them.

`wNsRwL_action_9` has a single camera and is reported as **no** agreement verdict rather than as
unanimous — one camera cannot agree with itself
(`AgreementTest::test_a_single_camera_is_not_agreement`).

### 8.4 What the view estimator says, and why nothing keys on it

Nothing in the shipped module reads `ctx.view_type`, for the fifth consecutive movement. Leg
Abduction measured the estimator's labels **systematically inverted** on an upright subject in the
exercise's own plane (0 of 210 repetitions carried a frontal-observable label when 116 were filmed
frontally); a jumping jack is the same regime. What is different here is that it **costs nothing**,
because §8.3 shows camera placement is not the limiting factor and because both metrics are
obliquity-cancelling by construction (§4.1). The earlier draft of this document applied squat's
confidence discount to two live rules; with no live rule there is nothing to discount, and the
constants were removed rather than left as dead code.

## 9. Testing

`tests/test_jumping_jacks.py` (32 cases) and `tests/test_jumping_jacks_validation.py` (18). The
ones that carry an argument rather than coverage:

| test | what it pins |
|---|---|
| `MetricLayerTest::test_the_stance_ratio_is_a_distance_ratio_not_an_image_x_difference` | 4.1 — the distinguishing property against the parent spec's own `|x27-x28|` wording, asserted in both directions (identical when upright, divergent under roll) |
| `ValidityGateTest::test_dropping_a_wrist_or_the_nose_leaves_the_leg_metric_intact` | 4.3 — the one structural departure from every earlier module |
| `InvarianceTest::test_every_metric_survives_mirroring_including_its_sign` | the contrast with Torso Twist, whose signed metric could not make this claim |
| `InvarianceTest::test_the_metrics_survive_roll_and_mirroring_through_segmentation` | that the invariance reaches the segmenter, not just one frame; tolerance stated as the float rounding it is |
| `StanceGeometryConfoundTest` (3 cases) | **7.0** — the withdrawal's mechanism, on synthetic geometry, so it does not rest on 456x256 footage: aligned knees trip the cut at a 1.6 stance, do not at 1.0, monotone in between |
| `WithdrawnRulesTest::test_no_withdrawn_rule_leaves_a_metric_behind` | that neither the confounded valgus ratio nor the view-corrupted knee angle survives in the metric tuple |
| `WithdrawnRulesTest::test_no_rule_produces_any_detection_at_all` | the whole-detector consequence, non-vacuously (the clip must really have segmented) |
| `SilentLegRomRuleTest::test_the_specs_cut_is_kept_where_it_is_rather_than_moved` | 5.2 — a later quiet retune has to change a test that says why |
| `SilentArmRuleTest::test_the_metric_it_would_have_used_is_computed_and_correct` | 6 — that "the sensor is fine" is a checked claim, which is why this module carries the first metric in the registry emitted solely for a silent rule |
| `SegmentationTest::test_the_default_min_rep_seconds_admits_the_fastest_cited_cadence` | 4.5 / 8.2 — the framework knob reserved for this movement by name, measured not needed |
| `NotRegisteredTest` (3 cases) | 7.4 — absent from the registry AND complete as an object, so "not registered" cannot decay into "not built" |
| `CadenceTest::test_the_floor_probe_measures_what_the_direct_reading_cannot` | 8.2 — why the floor measurement is a difference of two segmentations rather than a min over one |
| `SummarizeTest::test_the_fire_rate_is_over_action_camera_pairs_not_actions` | that three simultaneous cameras are three chances to fire, so a per-action rate would hide the disagreement the rig exists to expose |
| `AgreementTest::test_a_single_camera_is_not_agreement` | 8.3 — the one-camera action is not counted as unanimous |

## 10. Honesty constraints

- **No threshold moved.** The spec's 1.3 stays 1.3 in the module and the rule is silenced instead;
  the withdrawn 0.82 stays where it lives, in `squat.rule_knees_inward`. The measured distribution
  would have made either easy to fit, which is exactly why neither was.
- **This document changed its own conclusion.** It was drafted shipping two rules; the validation
  run silenced one and a zero-parameter control withdrew the other. The earlier reasoning is not
  hidden — 7.0 states what the Fit3D evidence really supported and why it was not enough.
- **The `validated=False` reason is a SIXTH one**, and 2.3 says why folding it into Sit-up's would
  be the tidier and wrong answer.
- **No rule was validated.** The reachable corpus has no positive class (2.4), so no sensitivity,
  AUC or fault-level claim appears anywhere in this document.
- **The resolution caveat is a limit on interpretation, not a footnote** (2.4). The valgus
  withdrawal does not depend on it: its control is a geometric identity that holds at any
  resolution and is reproduced synthetically in the tests.
- **Every number in 5, 7.0 and 8 comes from a script that ships in this repository**
  (`scripts/egoexo/run_jumping_jacks_validation.py`), not from a scratch probe — the Torso Twist
  requirement, which caught five wrong figures there and caught two here (the 66.4/69.4 pair
  quoted from a partial run became 68.5/79.4 on the full corpus).

### Out of scope

Extracting the remaining 110 judged Jumping Jacks actions (blocked on the missing `.ac` part, and
on `.ad` being useless without it) — which is the single thing that would let this detector ship.
Building the knee-deviation-from-limb-line metric that would measure valgus correctly (7.0): no
source states a threshold for it. Repairing `RunResult.fallback` not being threaded into
`RuleContext` (recorded by Deadlift, Shoulder Bridge, Leg Abduction and Torso Twist). Promoting the
scratch extraction and MediaPipe scripts into `scripts/egoexo/` — the harness that consumes their
output ships, and `notes/jumping-jacks-rule-validation.md` records the recipe.
