# Leg Abduction detector — design

Thirteenth of sixteen, and the one that closes Group E. Parent spec:
`docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`, Group E. Module:
`src/pose/movements/leg_abduction.py`. Tests: `tests/test_leg_abduction.py`. Validation harness:
`src/rehab24/leg_abduction_rule_validation.py`, results in
`notes/leg-abduction-rule-validation.md`.

**Outcome: one rule ships, one is registered permanently silent, two are withdrawn, and for the
first time in this programme the labeled data decided the roster rather than commenting on it.**

| parent-spec rule | outcome | decided by |
|---|---|---|
| `abd_pelvic_drop_trunk_lean` | **SHIPS — the trunk-lean disjunct only** | exact KG node; the spec's own ratio transfers to a body-relative frame with no number invented; measured AUC 0.85 pooled and ≥ 0.70 in every subject on 210 labeled repetitions |
| `abd_insufficient_rom` | **REGISTERED, PERMANENTLY SILENT** | best KG seed of the three and a working metric, but no source states a range and the spec's cut fires on ~1 in 3 repetitions humans judged **correct**, three times its rate on incorrect ones |
| `abd_hip_flexion_er_substitution` | **WITHDRAWN, absent** | citation is band placement in a monster walk; needs a sagittal view of a frontal-plane exercise; no KG node |
| `abd_momentum` | **WITHDRAWN, absent** | half the heuristic is a per-user baseline this architecture does not have; the surviving half is uncited; no KG node |

And one sub-decision that is not a rule outcome but is the most interesting thing here:

| parent-spec sub-clause | outcome | decided by |
|---|---|---|
| `abd_pelvic_drop_trunk_lean`'s **pelvic-tilt** disjunct | **NOT IMPLEMENTED** | the citation says pelvic **drop**, the knowledge graph says pelvic **hiking**, and the labeled data separates on **hiking**. The cited direction and the observed direction are opposites. |

---

## 1. Purpose, and what makes Leg Abduction unlike the twelve before it

**This is the first movement in the programme whose evidence is good enough to change the
answer.** Every previous detector was designed from the parent spec and its citations, then
shipped, and then — for two of them — checked. Here the check ran during design, on 210
human-labeled repetitions of the exercise the app actually models, and it silenced one rule,
confirmed another, and settled a sub-clause that the citation and the knowledge graph disagreed
about. Nothing below is a threshold tuned to labels; what the data decided is *which rules exist*,
not *what their numbers are*.

Three things line up here that have never lined up before:

1. **The variant matches.** REHAB24-6 `Ex4` is standing unilateral hip abduction — confirmed by
   looking at the frames, not by reading the exercise name. The subject stands, plants one leg,
   and carries the other out to the side. That is exactly what the app's own card art
   (`frontend/public/movements/leg-abduction.png`) and its icon comment ("Front view: one leg
   carried out to the side, the other under the hip") depict. Sit-up's variant mismatch —
   curl-up in the spec, full sit-up everywhere else — does not recur.
2. **The pixels are all there.** 12 videos × 2 orthogonal cameras, locally extracted. Shoulder
   Bridge's missing `frames_open` part does not recur either.
3. **The subject stands up**, which makes it possible to test something the other two could not.
   Both other Group E movements are performed lying down, and both abandoned the view estimator
   because `view_estimation.py`'s limit 1 voids its front/rear/oblique labels for **horizontal**
   subjects — Sit-up measured them inverted, Shoulder Bridge measured them unstable. Leg
   Abduction is upright, so that limit does not apply and the labels can be checked against
   ground truth the dataset records per repetition. §1.3 is what that check returned, and it is
   not what §1's optimism would predict.

### 1.1 Group E's reference-frame problem recurs, and this movement can solve it

The Group E update block records the finding that closed Sit-up: every quantity Group E defines
"vs the floor/horizontal" is unrecoverable from a frame, because EgoExo-Fitness ships its
near-sagittal views rotated a quarter turn with no EXIF tag. It then names what that leaves for
this movement — `pelvic-tilt vs horizontal` and `trunk lateral-lean` are **both** specified
against the image horizontal and both need re-anchoring, and `abd_insufficient_rom`'s "thigh
vector relative to the pelvis midline / vertical" is a mixed case whose vertical reading does not
survive.

All of that is correct, and this module resolves it with a reference the other two Group E
movements did not have: **the support limb.** In a standing unilateral exercise the stance leg is
planted and load-bearing, so `hip_stance → ankle_stance` is a body-internal stand-in for the world
vertical. Every metric this module emits is measured against it, so every metric is invariant
under camera roll. A supine movement has no such limb, which is why neither Sit-up nor Shoulder
Bridge could take this route.

### 1.2 The sign IS recoverable here, and the reason is that dot products mirror and cross products do not

Shoulder Bridge's central finding was that the arc it needed could not be signed: two body-
relative constructions were built, both were roll-invariant by design, and both were measured to
fail on real footage. Both were **cross products**.

A cross product against a body axis is invariant under camera roll but *anti*-invariant under
mirroring — its sign flips when the subject faces away from the camera instead of toward it, which
no monocular pipeline can tell. A **dot product** against a body axis is invariant under both.
Every signed quantity in this module is a dot product, and `tests/test_leg_abduction.py::
InvarianceTest` pins both invariances, including that the detections are byte-identical under a
90° roll and under mirroring.

That is not a rebuttal of the Shoulder Bridge finding — construction (A) there was invariant under
mirroring too, by taking a product of two cross products, and it still failed empirically. It is a
narrower, transferable claim: **when a signed body-relative quantity is available as a projection
onto a body axis, prefer it, because it needs no argument about mirroring at all.** Here the
pelvic-tilt sign is recovered cleanly enough that this module can tell hiking from dropping — and
that turns out to matter (§6).

### 1.3 Standing up is NOT enough: the view estimator is wrong on essentially every repetition

This is the finding that most needs stating first, because §1's third premise was written
expecting the opposite.

Ex4 records `cam17_orientation` per repetition, so the production estimator's label can be
compared against ground truth directly — 116 `front` and 94 `half-profile` repetitions, subject
upright, in the exercise's own plane. §8 has the table. What it shows is that the estimator does
not merely lose accuracy: it is **systematically inverted**, calling the frontal camera oblique
and the oblique camera sagittal, on essentially every repetition.

Two consequences, both stated rather than repaired:

- **`FRONTAL_OBSERVABLE_VIEWS` is never satisfied on this corpus.** The shipped rule's 0.65
  confidence discount therefore applies to 100% of repetitions. It is a **constant**, not a
  discriminator, and nothing about this run is evidence that view gating works.
- **The regime boundary in `view_estimation.py`'s limit 1 does not explain this.** That limit
  blames the horizontal-subject case, and Sit-up and Shoulder Bridge both measured failures
  inside it. This is the same shape of failure reproduced **outside** it, on an upright subject
  in the plane the exercise is performed in. The limit is understated, and the honest reading is
  that the front/rear/oblique labels are unvalidated more broadly than the module's own docstring
  admits.

The shipped rule survives this because it never **gates** on a view label — following Sit-up and
Shoulder Bridge, and now for a third reason: the discount it does apply is provably inert here,
so the rule's measured behaviour is the rule's own.

---

## 2. The data situation: 210 labeled repetitions, and they decide things

REHAB24-6 `Ex4`, "leg abduction":

- **210 repetitions**, 120 correct / 90 incorrect, **9 subjects**, 12 videos.
- Every subject contributes **both** classes (the thinnest is person 7 at 19 correct / 1
  incorrect), so per-subject AUC is computable for all 9 — unlike Ex5, where one subject
  contributed 21 incorrect and zero correct and dropped out of every median.
- Two **orthogonal, simultaneous** cameras. `cam17_orientation` is recorded per repetition: 116
  `front`, 94 `half-profile`. Segmentation.txt documents the implied cam18 orientation.
- `exercise_subtype` records the working leg per repetition: 106 "left leg", 104 "right leg", one
  leg per video.
- The frames are shipped **upright**, measured rather than assumed. Camera 18's file name says
  `transposed` and it is. Across the frontal camera the **stance limb sits a median 2.3° from the
  image vertical (p90 4.5°, max 14.7°)**, so on this corpus the support-limb reference and the
  image vertical are nearly the same thing. That is the honest accounting of §4's re-anchoring:
  it costs essentially nothing here and buys roll-invariance for production video, which is a
  phone in someone's hand. (On the sagittal camera the same measurement reads median 5.3°, p90
  14.4°, max 68.2° — the limb foreshortens in projection, which is one more reason the shipped
  rule is routed to the frontal camera.)

### 2.1 What this data can decide, and what it cannot

REHAB24-6 labels each repetition `correct` or `incorrect` and **never names the fault**. So:

- It **can** say whether a rule's underlying signal carries information about whether the
  repetition was performed correctly. That is enough to silence a rule whose signal points the
  wrong way, and enough to keep one whose signal points the right way in every subject.
- It **cannot** say that a rule that fired on an incorrect repetition found *that repetition's*
  actual error. No per-fault precision is measurable, from this dataset, at all.

This is why `LEG_ABDUCTION_DETECTOR.validated` stays `False`. That is a **fifth** distinct reason
in this registry, and the first that is not a gap in the evidence: Deadlift, Row, Band Pull Apart
and Bicep Curl are False because no labeled data exists; Arm Abduction and Arm VW because nobody
ran the check; Sit-up because the labeled data describes a different variant; Shoulder Bridge
because the labels match and the pixels are missing. Leg Abduction is False because **the check
was run, it decided the roster, and it still cannot support a fault-level claim.**

### 2.2 One more limit, stated because it is easy to forget

REHAB24-6's incorrect repetitions are *performed* incorrectly on request. So the absence of a
fault from the incorrect class is evidence about **this protocol**, not about the fault's
existence in the world. §5 leans on that distinction: `abd_insufficient_rom` is silenced because
its cut fires preferentially on the **correct** class, which is a statement about the rule, not
because short repetitions are rare in the incorrect class, which would be a statement about the
protocol.

---

## 3. The citation audit: a review of a different exercise, and a study that measured no kinematics

The parent spec marks all four Leg Abduction rules **VERIFIED (read RAG doc)**. Both documents
were read again here, in place, and the verification is true of the *strings* and much weaker than
it looks about the *claims*.

### 3.1 González-de-la-Flor, PMC12372021 — three of four rules rest on it

The title is *"Optimizing Hip Abductor Strengthening for Lower Extremity Rehabilitation: A
Narrative Review on the Role of **Monster Walk and Lateral Band Walk**"*. Monster walks and
lateral band walks are weight-bearing, banded, walking exercises. They are not standing unilateral
hip abduction.

What the review *does* contain, in a section titled "Review of Hip Abductor Strengthening
Exercises", is one primary sentence naming this exercise directly and with no reference marker:

> "For example, **standing hip abduction** with a cable or band attached to the ankle allows
> resisted movement **in the frontal plane**. The gluteus medius activation in standing hip
> abduction is high (60% MVIC) while also engaging core stability."

That is the strongest citation support available for this movement, and it establishes exactly two
things: the exercise exists as a distinct entity, and it loads the abductors through **frontal-
plane** movement. It states no range of motion, no velocity, no compensation, and no threshold.

Everything the parent spec quotes as *fault* support comes from elsewhere in the document and
carries a reference marker:

| parent-spec quote | where it actually sits | grade |
|---|---|---|
| "weakness leads to a characteristic Trendelenburg gait or compensatory trunk lean [ ]" | the anatomy section, about **gait** | secondary |
| "excessive sway or lateral trunk lean may reduce abductor demand by mechanically offloading the stance limb [ ]" | the **band-walk** biomechanics section | secondary |
| "maintaining frontal plane neutrality" | band-walk technique, one sentence after "optimal squat depth" | primary wording, band-walk subject |
| "side-lying hip abduction ... approximately 80% of MVIC [ ]" | exercise selection, about **side-lying** | secondary, wrong variant |
| "distal band placement introduces a slight external rotation torque" | where to loop the band in a **monster walk** | primary wording, not a fault |
| "Proper form (minimal pelvic sway, controlled steps)" | band-walk technique | primary wording, band-walk subject |

### 3.2 Rodrigues, PMC12416692 — the corroborator for the compensation rule

*"Sex as a moderator of the relationship between hip abduction strength and muscle activation
during single-leg stance."* Thirty-six adults, an EMG study of a 10-second **static single-leg
stance** plus a side-lying 1RM strength test.

The sentence the parent spec quotes — hip-abductor weakness compensated "by increasing ipsilateral
trunk lean [ , ]" — is in the **Introduction**, behind two reference markers, framing the study's
motivation. And the study is explicit that it measured nothing of the kind:

> "Without kinematic data or stability-related performance outcomes, such as pelvic drop or trunk
> sway, this explanation remains speculative."

So the corroborator corroborates the *mechanism* and measured *none* of the kinematics the rule
detects. This is Sit-up's fifth failure mode — a source-measured null on the proposed proxy —
arriving in a weaker form: not a null result, but no measurement at all.

### 3.3 What survives, and why one rule still ships

The mechanical situation genuinely matches, and that is the difference from Shoulder Bridge's
`asymmetric_pelvic_drop` withdrawal. There, the citation described **gait** and the exercise was a
two-leg floor bridge — a subject on their back, with no single-leg stance anywhere in it. Here the
subject **is** in single-leg stance for the whole repetition: one leg planted and load-bearing,
the other in the air. Trendelenburg mechanics and "offloading the stance limb" describe the
loading condition the exercise creates, not a different one.

That is enough to ship a rule whose citation support is honestly labeled SECONDARY at the point of
use — the shipped detection's `citation_support` string says so — and not enough to ship the three
rules that fail on other grounds as well.

---

## 4. The measurement layer: the support limb as this module's vertical

Nine metric keys, in two families.

**Support-limb-referenced, per moving-side hypothesis** — `{side}_abduction_deg`,
`{side}_trunk_tilt_deg`, `{side}_pelvic_hike_ratio`. Each is computed with the *other* leg as the
reference, so it is only meaningful for the leg that is actually working. The rules read exactly
one of the two, after `resolve_moving_side`.

| quantity | construction | invariant under |
|---|---|---|
| `abduction_deg` | angle between the working thigh and the downward support direction | roll, mirroring |
| `trunk_tilt_deg` | angle between `hip_mid → shoulder_mid` and the support limb | roll, mirroring |
| `pelvic_hike_ratio` | `dot(hip_moving − hip_stance, support_unit) / hip_width`; **+ = hike, − = drop** | roll, mirroring |

`trunk_tilt_deg` is the parent spec's own quantity with one substitution. The spec defines lateral
lean as "horizontal offset of shoulder midpoint from hip midpoint, normalized by trunk length" —
i.e. the component of the trunk vector perpendicular to a reference, over the trunk length, which
is exactly `sin` of the angle between the trunk and that reference. Swapping the image horizontal
for the support limb changes the reference and nothing else, so **the spec's 0.10–0.15 band
transfers as 5.74–8.63° with no number invented**. `tests/test_leg_abduction.py::TrunkLeanRuleTest
::test_the_threshold_is_the_spec_ratio_rendered_as_an_angle` pins that identity.

**Trunk-referenced, side-independent** — `left_thigh_trunk_deg`, `right_thigh_trunk_deg`,
`max_thigh_trunk_deg`. Both thighs against the same downward trunk axis, so the two are comparable
and the larger names the working leg.

### 4.1 Why the second family exists, and it is a measured correction rather than a design flourish

The first design used the support-limb pair for everything, including the working-side resolver.
That is wrong, and the labeled data said so immediately. `left_abduction_deg` references the left
thigh to the *right* leg and `right_abduction_deg` references the right thigh to the *left* leg,
so when either leg is lifted **both** quantities are approximately the angle *between the two
legs* — near-equal by construction. Comparing them to pick the working leg scored **7 correct / 14
wrong / 30 refused** on the first 51 labeled repetitions: worse than a coin flip, with the errors
concentrated in the reps where the answer mattered most.

Referencing both thighs to the same trunk axis fixes it, at the cost of a quantity contaminated by
trunk lean. That contamination is acceptable for *ranking two legs against each other*, because it
is common to both; it is not acceptable for scoring a rule, which is why the rules read the
support-limb pair.

### 4.2 The validity gate is stricter here than anywhere else in Group E

Eight required landmarks — both shoulders, both hips, both knees and **both ankles**. Sit-up and
Shoulder Bridge require six and neither requires an ankle. The ankles are the price of the support
limb: without them this module has no vertical at all. The measured cost is in §8.

---

## 5. `abd_insufficient_rom` is REGISTERED PERMANENTLY SILENT

This is the rule with the **best** knowledge-graph seed of the three —
`Leg Abduction:Insufficient Abduction Range`, two non-empty buckets (causes: Weak Hip Abductors;
quality_impacts: Hip Abduction) — and a clean, roll- and mirror-invariant metric. Nothing about
the sensing fails. What fails is the threshold, and the labeled data decided it.

### 5.1 No source states a range of motion

The parent spec admits this in its own citation block: "the specific degree threshold is a
practical target, not a value stated in the source". Reading the source confirms it. González-de-
la-Flor's only quantities for hip-abduction exercises are EMG amplitudes — 80% MVIC for side-lying
(secondary, wrong variant), 60% MVIC for standing (primary, right variant) — and the review never
states a target excursion, a normal range, or a failure cut for any of them.

### 5.2 The spec's cut fires on the correct class

§8 has the census. The direction is the finding: scored as the rule would score it — lower peak
abduction is worse — the cue's AUC is **far below chance**, in every subject, and at the spec's
own ~30° cut it fires on roughly **one in three repetitions humans judged correct** against
roughly **one in eight** judged incorrect.

### 5.3 Why the number is not moved

Moving 30° down until the false-alarm rate looks acceptable would be fitting a threshold to
labels, which this project's rules forbid and which every previous movement in this programme has
declined to do. There is no cited number to move *to*. And the honest reading of the census is not
"the cut is 5° too high" — it is that on this protocol, insufficient range is not what
distinguishes a bad repetition from a good one, and a rule that says otherwise will be wrong in
the user's favour about a third of the time.

### 5.4 Silent, not withdrawn

The fault has an exact graph node, a primary sentence naming the exercise and the plane it loads,
and a working metric. That is strictly more than the two withdrawn rules have. What is missing is
one number that nobody has published. Contrast `bridge_lumbar_hyperextension`, silent because the
**sensor** cannot see its quantity at all.

**The upgrade path, recorded and not taken:** a per-user baseline — "this repetition is shorter
than your own usual" — needs no literature threshold. The architecture has no cross-clip state,
which is the same wall `situp_excessive_speed` hit and the same one that removes half of
`abd_momentum` (§7).

---

## 6. The pelvic-tilt disjunct: the citation and the measurement point opposite ways

The parent spec's `abd_pelvic_drop_trunk_lean` is a **disjunction** of two coupled signals — a
frontal-plane pelvic tilt and a trunk lateral lean, flagged on either. The module implements the
second. This section is why, and it is the most interesting result in the movement.

### 6.1 Three sources of truth, and one of them disagrees with the other two

| source | direction it names |
|---|---|
| the parent spec and its citation | pelvic **DROP** — "Trendelenburg-like", the pelvis falling on the unsupported side |
| the knowledge graph | pelvic **HIKING** — `Leg Abduction:Pelvic Hiking` exists; `"Pelvic Drop"` matches **zero** nodes |
| 210 labeled repetitions | pelvic **HIKING** — the moving-side hip rides *higher*, and it is one of the two strongest signals measured here (§8) |

The graph and the data agree with each other. The citation points the other way.

### 6.2 Neither direction can ship

Firing the spec's rule **as written** — flag a drop — would fire on the sign the data associates
with the *correct* execution. Firing the observed direction instead would be a rule with **no
citation at all**: neither source mentions pelvic hiking, in this exercise or any other.

Sit-up withdrew a rule because its knowledge-graph seed was semantically inverted relative to the
fault. This is the mirror case — the graph and the data agree, and the *citation* is the odd one
out — and it is resolved the same way, by not shipping. **A citation/observation sign
disagreement is a new failure mode in this programme**, and it is only discoverable because §1.2's
dot-product construction recovers the sign at all. On Shoulder Bridge the same question was
unanswerable.

### 6.3 What the omission costs — measured, and it is not free

The pelvic-hike signal scores **AUC 0.848 pooled, per-subject median 0.800, minimum 0.690** —
statistically indistinguishable from the shipped trunk-lean signal. And the rank correlation
between the two across 163 scored repetitions is **ρ = 0.713**: related, but not redundant.

So the honest accounting is that **omitting the disjunct declines a real detection opportunity.**
It would have been comfortable if ρ had come back above 0.9 — "the omitted signal says nothing
the shipped one does not" — and it did not. §6.2's argument has to carry the weight on its own:
one direction contradicts the data, the other has no citation, and neither is a rule this project
ships. The cost is recorded rather than explained away.

---

## 7. Two rules are WITHDRAWN

### 7.1 `abd_hip_flexion_er_substitution` — four independent failures

1. **The external-rotation half cites band placement, not a fault.** The parent spec's support is
   "distal band placement introduces a slight external rotation torque". In place, that sentence
   describes a deliberate feature of a monster walk: "The monster walk often specifically uses an
   ankle or forefoot placement, which not only provides lateral resistance but also introduces a
   slight external rotation torque". The parent spec already grades the toes-up cue as "inferred
   clinical description"; reading the source downgrades it further.
2. **The hip-flexion half rests on band-walk posture.** "Maintaining frontal plane neutrality"
   sits in the lateral-band-walk biomechanics section, between "a slight forward trunk lean
   increases gluteus medius and maximus activation" and "optimal squat depth".
3. **It needs a sagittal view of a frontal-plane exercise.** The parent spec rates it low/medium
   and says the forward-drift component "needs a **side** view ... Not reliably separable from
   true abduction on a **front** view alone". The app films one camera, and the shipped rule needs
   the frontal one.
4. **No KG node.** `retrieve_graph_context("Hip Flexion Substitution", movement="Leg Abduction")`
   returns the generic anatomy nodes `Hip` and `Hip Flexion`, not a fault.

**Not said by this withdrawal:** that the leg drifting forward is fine. It is a real clinical
substitution. What is missing is a source that observes it in this exercise and a view that can
see it.

### 7.2 `abd_momentum` — three independent failures

1. **Half the heuristic does not exist in this architecture.** "Flag if peak angular velocity
   greatly exceeds a per-user baseline" — there is no per-user baseline anywhere in this pipeline.
   This is the same wall `situp_excessive_speed` hit, and the **second** disjunct-level defect of
   that kind in the registry.
2. **The surviving disjunct is computable and uncited.** "The eccentric phase is much faster than
   the concentric" is measurable from this module's phase labels — it is the one thing here that
   could have shipped. What backs it is "Proper execution requires control of the trunk and pelvis,
   optimal squat depth, and consistent band tension" and "Proper form (minimal pelvic sway,
   controlled steps)", both about band walks and neither stating a ratio, a velocity or a duration.
3. **No KG node.** `"Momentum"` matches `Anterior Momentum Generation` and `Forward Momentum`,
   both reached from other movements' subgraphs.

### 7.3 The graph and the citation audit selected the same subset

This movement has exactly **three** fault nodes in the knowledge graph, and the two parent-spec
rules with no node are exactly the two withdrawn on citation grounds. That is the first time in
this programme the graph and the citation audit have independently agreed on which rules should
not exist, and it is worth recording precisely because it could easily have been a coincidence —
n = 2.

---

## 8. Measured on 210 labeled repetitions, through the real `run_detector`

Full tables and method caveats: `notes/leg-abduction-rule-validation.md`. Harness:
`scripts/rehab24/validate_leg_abduction_rules.py`. 210 repetitions, 9 subjects, replayed through
the real `run_detector` one labeled window at a time.

**Working-side resolver**, of the 175 repetitions that reached it: **163 correct, 1 wrong, 11
declined as ambiguous** — accuracy when it answers **0.994**, coverage **0.937**. A further 35
never reached it because `segment_reps` declined the window first; that is a segmentation
outcome and is counted separately.

**The shipped rule**, at the parent spec's own cut (0.15 of trunk length = 8.63°):

```
fired 44/210    tp 39   fp 5   fn 51   tn 115      precision 0.886   specificity 0.958
```

Its signal's AUC is **0.840 pooled**, per subject **median 0.833, min 0.690, max 1.000, all 9
above chance**. Split by the camera geometry the dataset records, the oblique camera costs
**sensitivity and not precision** — `front` 30 tp / 5 fp / 23 fn, `half-profile` 9 tp / **0 fp** /
28 fn. A lean projected obliquely reads smaller than it is, so the rule goes quiet rather than
wrong. That is the opposite of Shoulder Bridge's census, where the unfavourable camera produced
near-full-severity false alarms, and it is the benign failure mode of the two.

**The silenced rule.** Scored as it would score — lower peak abduction is worse — AUC **0.206
pooled**, and **every one of the 9 subjects below chance** (max 0.347). At the spec's own 30° cut
it fires on **39/93 (42%) of repetitions humans judged correct** against 8/70 (11%) of the
incorrect ones. That is the measurement §5 rests on.

**The view estimator** emitted a `FRONTAL_OBSERVABLE_VIEWS` label on **0 of 210** repetitions —
`front` → `rear_oblique` 116/116, `half-profile` → `side` 92/94. §1.3.

**The eight-landmark gate** costs a median validity rate of **0.600 and a p10 of 0.000**: at least
a tenth of repetitions carry no fully-landmarked frame at all. **35 of 210 (17%) took a fallback
path** and are excluded from every AUC above, which are computed over **163/210** and say so.

---

## 9. Testing

`tests/test_leg_abduction.py` (35 cases) and `tests/test_leg_abduction_rule_validation.py` (27
cases). The ones that carry an argument rather than coverage:

| test | what it pins |
|---|---|
| `InvarianceTest::test_every_metric_is_invariant_under_camera_roll` | the Group E re-anchoring mandate, executably |
| `InvarianceTest::test_every_metric_is_invariant_under_mirroring` | §1.2 — the property a cross-product sign would not have |
| `InvarianceTest::test_detections_are_byte_identical_under_roll_and_mirroring` | that the invariance survives all the way to a detection, not just a metric |
| `MetricLayerTest::test_pelvic_hike_is_signed_and_hike_is_positive` | that this module CAN tell the two directions apart, which Shoulder Bridge could not |
| `TrunkLeanRuleTest::test_the_threshold_is_the_spec_ratio_rendered_as_an_angle` | §4 — that re-anchoring invented no number |
| `TrunkLeanRuleTest::test_it_reads_the_support_limb_and_not_the_image_vertical` | the distinguishing property against `arm_abduction.rule_contralateral_trunk_lean` |
| `SilentRomRuleTest::test_it_never_fires_even_on_a_repetition_that_trips_the_specs_cut` | that the silence is real, on the exact case the spec says to flag |
| `ResolveMovingSideTest::test_a_rule_is_silent_when_the_side_is_unresolvable` | that a refused side silences rather than defaults to a leg |
| `EndToEndSegmentationTest::test_a_clean_clip_produces_no_detections` | non-vacuously: it asserts the clip really segmented and was scored |
| `harness::MovingSideAccuracyTests::test_refusals_are_counted_apart_from_errors` | that coverage and correctness are not conflated in the writeup |

## 10. Honesty constraints

- **No threshold is tuned to the labels.** The shipped cut is the parent spec's own ratio,
  re-expressed in a different reference frame by an identity. The silenced rule's cut is the
  parent spec's own number, unchanged, and it is silenced rather than moved.
- **Secondary sourcing is stated at the point of use**, in the shipped detection's
  `citation_support` string, not only in this document.
- **Every per-repetition statistic is reported per subject.** 210 repetitions come from 9 people
  and are not independent; this project has twice been burned by a pooled number that collapsed
  within subject.
- **The unimplemented disjunct's cost is measured**, not asserted to be zero.

### Out of scope

Repairing `RunResult.fallback` not being threaded into `RuleContext` (recorded by Deadlift and
again by Shoulder Bridge; it recurs here on the repetitions that take the whole-clip path).
Changing `arm_abduction.rule_contralateral_trunk_lean` to use a support-limb reference, which is
the obvious follow-up now that one exists — it would change a shipped rule's output on a movement
this branch is not about.
