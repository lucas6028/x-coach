# Torso Twist detector — design

Fourteenth of sixteen, and the one that opens Group F. Parent spec:
`docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`, Group F. Module:
`src/pose/movements/torso_twist.py`. Tests: `tests/test_torso_twist.py`. Measurement harness:
`src/fit3d/rotation_proxy_fidelity.py` + `scripts/fit3d/run_rotation_proxy_fidelity.py`, helpers
tested in `tests/test_rotation_proxy_fidelity.py`.

**Outcome: one rule ships, one is registered permanently silent, two are withdrawn — and the two
withdrawals are decided by things that are not opinions: a projection measurement against 3-D
ground truth, and a source that prescribes the behaviour its own rule flags.**

| parent-spec rule | outcome | decided by |
|---|---|---|
| `tt_trunk_not_braced` | **SHIPS** | the only rule here with a PRIMARY sentence naming this exercise (the RAG doc's 45° torso target), a KG node that means what the rule means, and a metric that survives re-anchoring |
| `tt_insufficient_rotation_rom` | **REGISTERED, PERMANENTLY SILENT** | clean roll- and mirror-invariant metric, real fault, and no source states a range; the only ROM-adjacent KG node is about LATERAL FLEXION — a different axis of a different exercise |
| `tt_lumbar_rotation_dominant` | **WITHDRAWN, absent** | four independent failures, one of them measured: the spec's own 2-D proxy disagrees with 3-D truth on **16.7% of repetitions at the spec's own cut, with a perfect detector** |
| `tt_momentum_over_control` | **WITHDRAWN, absent** | the cited source instructs the user to do the thing half the heuristic flags; the other half has no number anywhere |

And two findings that are not rule outcomes but are the most interesting things here:

| finding | what it is |
|---|---|
| **Four artifacts, four exercises** | the parent spec, the RAG doc and the app card art say *seated Russian twist*; the app icon draws *standing*; Fit3D's `standing_ab_twists` is a *standing cross-body knee-to-elbow* twist; EgoExo-Fitness's judged corpus is a *prone/kneeling lateral flexion*. Nothing in this repository films the exercise the app depicts. |
| **The KG's three Torso Twist faults are seeded from the wrong exercise, and the seeding script says so** | `scripts/knowledge/stub_general_movements_v3.py:152` records the grounding as "EgoExo-Fitness TKV (Kneeling Side Torso Twist: pause-at-bottom 23%, lateral-flexion depth 21%, base 13%, abs)". This is PRIMARY provenance, not an inference from node names. |

---

## 1. Purpose, and what makes Torso Twist unlike the thirteen before it

Every previous movement in this programme had one thing that was at least stable: what the
exercise *is*. Sit-up's variant question was between two readings of one movement (curl-up vs full
sit-up) and Shoulder Bridge's data was missing rather than wrong. Here the movement's identity is
contested by the project's own artifacts, and each of the three corpora that contains something
called a torso twist contains a **different** exercise.

That is not a reason to stop; it is a reason to fix the contract explicitly before writing a rule,
and then to be careful about which measurements transfer across the mismatch and which do not.
§2 fixes the contract. §8 is scrupulous about the distinction: the Fit3D pass in this document
measures **projection geometry**, which is about cameras and transfers, and never a **threshold**,
which is about the exercise and does not.

### 1.1 The structural problem: this movement's defining quantity points into the camera

Every Torso Twist rule reads axial rotation — rotation about the body's long axis, which for an
upright or reclined subject is rotation mostly *into and out of the image plane*. The parent spec
concedes the point in its own heuristic and substitutes a proxy: the change in the **projected
horizontal separation** of a paired landmark line, `|x11−x12|` for the shoulders and `|x23−x24|`
for the hips.

That substitution has two analytic defects and, it turns out, one measured one:

- **Zero derivative at the braced centre.** The projected separation is `width · |cos θ|`, whose
  derivative is `width · sin θ` — zero at `θ = 0`, which is precisely where the rule must separate
  "square" from "slightly turned".
- **Even in θ**, so it cannot tell a twist to one side from a twist to the other. The spec's
  remedy is the left–right x-ordering flip, which only happens past 90°.
- And §8.1's measurement, which is what actually settles it.

This is structurally the same shape as Shoulder Bridge's `angle_degrees` finding — an unsigned,
symmetric quantity that makes one rule unfireable and silently inverts another — arriving in Group
F through a different door. The module's shipped and silent metrics both avoid it, and how they do
so is §4.

---

## 2. The variant, decided rather than assumed

### 2.1 The app's own two assets disagree, and that is an asset defect

| artifact | what it depicts |
|---|---|
| parent spec, Group F | rep phases written in seated geometry — "hips fixed on floor, knees bent, torso held ~45° off the ground" — **seated** |
| `data/rag/docs/torso_twist_russian_wiki.txt` | the Russian twist: "one sits on the floor and bends both knees" — **seated** |
| `frontend/public/movements/torso-twist.png` | a subject seated on the floor, knees bent, hands clasped, torso rotated — **seated** |
| `frontend/src/components/movements/MovementIcon.tsx:148` | comment *and* strokes both draw a **standing** figure |

Three of four agree, and one of the three is the card art the user actually sees when they pick
the movement. **The contract is the seated Russian twist.** The icon is the outlier; because its
strokes draw a standing figure too, it is an asset defect rather than a stale comment.

**Recorded, not fixed.** Changing it is a frontend change on a movement this branch is not about —
the line Leg Abduction drew around `arm_abduction.rule_contralateral_trunk_lean`.

### 2.2 Three corpora, three other exercises

- **REHAB24-6** has no twist at all (Ex1 arm abduction, Ex2 arm VW, Ex3 table push-ups, Ex4 leg
  abduction, Ex5 leg lunge, Ex6 squats).
- **Fit3D `standing_ab_twists`** — 8 train subjects × 4 cameras, `joints3d_25` mocap ground truth,
  `rep_ann.json` boundaries, all present locally. **Looked at, not inferred from the name**, across
  four subjects and three cameras: it is a **standing cross-body knee-to-elbow twist** — one knee
  driven up, the opposite elbow driven down to meet it, alternating. Single-leg support, free
  pelvis, large trunk flexion. It is not a Russian twist and it is not even the hips-square
  standing ab twist the app icon draws.
- **EgoExo-Fitness `Kneeling Side Torso Twist`** — **95 judged actions** with per-criterion True/
  False key-point verification, which is *richer* labelling than REHAB24-6's binary correctness.
  And by its own criteria text it is a prone/kneeling **lateral flexion**: "Lie prone (face down)
  on a yoga mat", "Lower your body towards the ground by bending at the right elbow." Rotation is
  not what it measures.

**`validated=False`, and the reason is Sit-up's** — the labeled data describes a different variant
— **not a new one.** This programme counts its distinct reasons carefully and inflating the count
would be the kind of claim it is built to catch. What is new is only that the same reason holds
three times over, against three corpora each modelling a different exercise.

---

## 3. The citation audit: one paper, and it never mentions the exercise

The parent spec cites **McGill, S.M. (1991), *J Orthop Res* 9(1):91–103, PMID 1824571** for **all
four** Torso Twist rules, marked VERIFIED, plus the Wikipedia RAG doc as supplementary. Both were
read again in place.

### 3.1 What McGill actually measured

Ten men and 15 women, isometric exertions plus **dynamic axial twists at 30 and 60 °/s**, EMG of
the trunk musculature with kinematics and measured torque. Headline results:

> normalized EMG during maximal twisting — rectus abdominis 22%, external oblique 52%, internal
> oblique 55%, latissimus dorsi 74%, upper erector spinae 61%, lower erector spinae 33%; the model
> "severely underpredict[ed] measured torques (e.g. 14 Nm predicted for 91 Nm measured)"; and

> "Such dominant coactivity suggests that **stabilization of the joints during twisting is far
> more important to the lumbar spine than production of large levels of axial torque**."

What that supports and what it does not:

| the spec wants | McGill supplies |
|---|---|
| a reason the brace matters | **yes, primarily** — the stabilization conclusion is his own result |
| the obliques as the dominant rotators | **yes, primarily** — the EMG amplitudes |
| a rotation range of motion | **no** — no range is stated anywhere |
| a tempo or velocity threshold | **no** — 30 and 60 °/s are his own protocol's imposed conditions, performed by healthy subjects with no fault attached |
| thoracic-vs-lumbar contribution | **no** — he separates upper (T9) and lower (L3) erector spinae electrodes, and concludes the *upper* pair stabilizes despite having "no mechanical potential to contribute axial torque". That is a claim about muscle roles, not about which spinal segment should produce the twist |
| anything about this exercise | **no** — a laboratory torque protocol; the Russian twist is never named |

**This is a new shading of the exercise-identity failure mode.** Arm VW's four sources were about
adjacent exercises; McGill is not about an exercise at all. He is a mechanism paper, correctly
cited for the mechanism, and the parent spec's VERIFIED marks read as if he had been cited for the
faults.

### 3.2 The RAG doc, and the sentence the parent spec paraphrases backwards

`data/rag/docs/torso_twist_russian_wiki.txt` is short and every sentence in it matters here.

Supporting the shipped rule, primarily and in its own words:

> "the torso is kept straight with the back kept off the ground at a **45-degree angle**"

Defining the repetition, which is why this module's rep is one swing:

> "the arms should be swung from one side to another in a twisting motion, **with each swing to a
> side counting as one repetition**"

And the sentence the parent spec gets wrong:

> "When moving one's arms during the exercise, it is crucial to **not stop between repetitions** or
> else one will lose the effect of working the abdomen."

The parent spec's `tt_momentum_over_control` renders that as a warning "not to rely on between-rep
momentum". It is the opposite: the source instructs **continuous movement**. §7.2.

---

## 4. The measurement layer: two metrics, both re-anchored to the body

Two metric keys, and each is the answer to one of §1.1's defects.

| key | construction | invariant under |
|---|---|---|
| `trunk_thigh_angle_deg` | angle between `hip_mid → shoulder_mid` and `hip_mid → knee_mid` | roll, mirroring, **and axial rotation** |
| `twist_offset_ratio` | `dot(wrist_mid − hip_mid, shoulder_unit) / shoulder_width`; signed | roll; **magnitude** under mirroring |

### 4.1 The brace angle, and the one place this module differs from Sit-up

The parent spec measures the brace as "the trunk vector hip-midpoint → shoulder-midpoint angle
relative to **vertical**". Group E spent three movements establishing that the image vertical is
not the world vertical and is not recoverable from a frame. Referencing the trunk to the **thighs**
instead makes the quantity a pure angle between two body segments — invariant under camera roll and
under mirroring — while still measuring what the source describes, because a seated twister who
sags back toward the floor opens the trunk away from the thighs.

**Shoulder MIDPOINT, not a same-side shoulder, and this is deliberate.** `situp_compute_raw` reads
same-side `angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)` so a rolled subject never blends one side's
shoulder with the other side's knee. That is right for a sit-up and **wrong here**: this movement's
entire content is the shoulder line rotating about the trunk axis, which swings each individual
shoulder across the frame and would inject the rotation straight into the brace angle. The shoulder
midpoint sits **on** the rotation axis. `MetricLayerTest::
test_the_brace_angle_does_not_move_when_the_subject_only_twists` asserts the contrast rather than
asserting it away: with the trunk posture fixed and only the shoulder line foreshortened, the
midpoint construction is unmoved to 1e-6 and Sit-up's construction moves by degrees.

One difference the comparison should not hide: this module's angle is built from `midpoint(...,
dims=2)`, i.e. **2-D**, whereas Sit-up's `angle_degrees` consumes `dims=3` and therefore also
MediaPipe's estimated `z`. The roll- and mirror-invariance argument holds either way, and 2-D is
the better choice here because `z` is the axis this movement's rotation lives on — including it
would feed the estimator's weakest coordinate straight into the quantity the midpoint construction
exists to keep rotation *out* of.

### 4.2 The twist offset: a translation, not a projected width

`twist_offset_ratio` measures how far the clasped hands have travelled across the body **along the
shoulder axis**, in shoulder widths. It is a dot product onto a body axis, following Leg Abduction
§1.2 — roll-invariant, and mirror-invariant in magnitude.

**It is not the parent spec's proxy, and that is the point.** A hand *translating* across the body
is a first-order signal with a non-zero derivative everywhere; a shoulder line *rotating* about the
vertical projects as `width · |cos θ|` and has neither property (§1.1). The two are not
interchangeable readings of the same thing — one works and one does not (§8.1).

**The sign flips under mirroring, and that is stated rather than hidden.** It names which way the
hands went *in the image*, and no monocular pipeline can map that onto the subject's own left and
right. Nothing in this module claims a body side: the rep signal is rectified, and the only rule
that reads the metric reads its magnitude. `InvarianceTest::
test_the_twist_magnitude_survives_mirroring_and_its_sign_does_not` pins both halves.

### 4.3 The validity gate requires the wrists, and the hands are the occlusion risk

Eight required landmarks — both shoulders, both hips, both knees and **both wrists**. Matching Leg
Abduction's eight and two more than Sit-up's six. The wrists are the price of the rep signal: this
movement's repetition is defined by the hands swinging across the body and no other landmark pair
tracks it.

It bites harder here than Leg Abduction's ankle requirement did, because the Russian twist is
performed **with the hands clasped together** — the configuration in which one wrist is most likely
to be swallowed by the other and by the forearms, at exactly the centre of the swing. Stated rather
than relaxed: dropping the requirement would leave the rep signal NaN on those frames anyway, and
would additionally hand the brace rule a window segmentation could not have found.

### 4.4 First user of the rectified rep signal

`base.py:55` documents `rep_rectify` as existing "for bipolar signals (torso twist swings to both
sides)" and has had no user since it was written. This module is it: `rep_signal=
"twist_offset_ratio"`, `rep_polarity="max"`, `rep_rectify=True`, `rep_start="extended"`. Rectifying
makes each swing its own excursion from zero, so a repetition is one swing — which is the source's
own definition of a repetition (§3.2), not a convenience.

---

## 5. `tt_trunk_not_braced` SHIPS

It is the only Torso Twist rule that clears all three bars at once:

1. **A primary sentence naming this exercise.** The RAG doc's "the torso is kept straight with the
   back kept off the ground at a 45-degree angle" is the technique target, stated descriptively and
   about the Russian twist.
2. **A knowledge-graph node that means what the rule means.** `Torso Twist:Poor Abdominal
   Engagement`, two non-empty buckets (`quality_impacts: Core Stability`, `causes: Weak Core
   Stability`).
3. **A mechanism from the peer-reviewed source's own result**, not a borrowed one: McGill's
   stabilization conclusion is why dropping the brace is a fault and not merely untidy.

### 5.1 The threshold's provenance, stated

The **fire threshold is 15° of deviation from the repetition's opening posture** — the parent
spec's own number. **No source states it.** What the RAG doc states is the 45° *target*, and a
target is not a tolerance; it is also measured against the **ground**, the reference this module
deliberately does not use. So **the 45° cannot be transferred and the 15° is not derived from it**.
This is the treatment Sit-up gave its 20°: the spec's number, shipped with its provenance on the
record, not moved and not dressed up as cited.

The **severity ramp 15° → 40° is a rule-level choice**. The parent spec states no ramp for any
Torso Twist fault, and the Lunge section states its ramps explicitly, so the absence is meaningful.

### 5.2 The spinal-rounding disjunct is withdrawn

The parent spec's rule ORs in a second clause: "shoulder-midpoint moving forward of the
hip-midpoint in x on a side view". That is an **image-x offset**, so not roll-invariant, and its
direction ("forward") is unresolvable without knowing which way the subject faces — the mirroring
ambiguity. `ohp_forward_head` withdrew its bar-path sub-criterion for the same reason;
`arm_vw.rule_loss_of_elevation` established that a `fault_id` survives the loss of a branch,
because the id is the join key between the spec, the registry and stored analyses. The id is
unchanged and the user-facing `fault_name` says "Braced Torso Lost" and nothing about rounding.

### 5.3 The deviation is SIGNED, and the unsigned version fired on the opposite of the fault

The parent spec says "deviates from baseline by > ~15°", and the first implementation took that
literally with an `abs()`. That is not merely non-directional but **actively inverted**:
`trunk_thigh_angle_deg` is monotone in sag — larger means the torso has laid further back toward
the floor, smaller means the subject is sitting *up* — so an unsigned deviation flags a twister
who **tightened** their posture.

**Measured on the shipped path before the fix:** a subject setting up loose at 95° and then
tightening to 50° for the swing was reported *"Braced Torso Lost"* at **severity 1.0**, quoting a
45° deviation.

**And the baseline makes that the ordinary case, not an edge one.** `setup` is the window's first
15% — the frames *before* the subject braces. Set up loose → brace → swing is a normal way to
perform the movement and produces exactly a large positive tightening.

This is `pushup_head_drop`'s finding arriving by a third route; the parent spec §8 states it in so
many words ("a baseline on the unsigned angle is not merely non-directional but *actively
inverted*"). There it forced a signed metric into the metric layer. Here the sign already lives in
the comparison, so the fix introduces **no new number** and fires on a strictly smaller set. Roll-
and mirror-invariance are untouched, because the sign comes from the baseline comparison rather
than from the frame.

**It was not caught by the green suite, and the reason is worth naming:** every fixture ramped
`95.0 + deviation`, so the rule tests *and* `EffectiveThresholdTest` were blind to the direction
by construction. The mirror test now exists —
`test_a_twister_who_TIGHTENS_is_not_told_they_lost_the_brace`, paired with
`test_it_still_fires_on_the_sag_direction_so_the_test_above_is_not_vacuous` so the silence is not
itself vacuous.

### 5.4 What it is blind to, and it is the whole verdict for that user

A baseline measures **change**, not **posture**. A twister who sets up already collapsed and holds
that position for the entire repetition is never flagged. Push-up recorded this cost for
`pushup_head_drop`; there it was one rule among four, and here it is the detector's only live rule.
Pinned twice — at rule level by `test_a_brace_lost_before_the_rep_opens_is_invisible` and end-to-end
by `test_a_brace_held_from_setup_is_invisible_end_to_end_too`.

---

## 6. `tt_insufficient_rotation_rom` is REGISTERED PERMANENTLY SILENT

Nothing about the sensing fails. `twist_offset_ratio` is a working, roll-invariant,
mirror-magnitude-invariant metric free of both of §1.1's defects. What fails is the threshold.

- **No source states a rotation range.** McGill reports EMG amplitudes and torque and no ROM; his
  only angular figures are protocol velocities. The RAG doc defines the swing and states no depth.
  The parent spec's cut — the wrist midpoint failing to pass the hip midline by more than "~0.08 of
  shoulder width" — appears in no source, and the spec does not claim otherwise.
- **The graph cannot supply the missing meaning, and its node is about a different axis.**
  `Torso Twist:Insufficient Lateral Flexion Depth` is the only ROM-adjacent node this movement has;
  it has one non-empty bucket; and the seeding script records it as a stub of a **prone lateral
  flexion** exercise. Seeding a rotation-depth card from a lateral-flexion node would put the wrong
  movement's explanation on the user's screen.

**Silent, not withdrawn**, and the distinction is load-bearing: the fault is real, the metric works,
the sensor can see it, and what is missing is one number nobody has published. That is
`abd_insufficient_rom`'s situation exactly, and it is registered the same way. Contrast
`bridge_lumbar_hyperextension`, silent because the sensor cannot see its quantity at all.

**The upgrade path, recorded and not taken:** a per-user baseline — "this swing is shorter than your
own usual" — needs no literature threshold. The architecture has no cross-clip state. That is the
**third** time this wall has been hit, after `situp_excessive_speed` and `abd_momentum`, and it is
now the single most common reason a Group E/F rule cannot ship.

---

## 7. Two rules are WITHDRAWN

### 7.1 `tt_lumbar_rotation_dominant` — four independent failures

1. **The citation says nothing about where the rotation comes from.** The rule's claim is that the
   twist should be driven thoracically and not lumbarly. McGill concluded that the musculature
   *stabilizes* rather than produces torque, and makes no thoracic-versus-lumbar contribution claim
   (§3.1). The parent spec's own heuristic concedes the sensing half — "true thoracic-vs-lumbar
   segmentation is not resolvable from 33 sparse landmarks" — while still proposing a rule.
2. **The 0.6 ratio is invented.** No source states a hip-to-shoulder rotation ratio, in this
   exercise or any other.
3. **No KG node.** `"Lumbar Rotation"` and `"Insufficient Rotation Range"` both return zero matches
   under `movement="Torso Twist"`.
4. **The proxy is measured to be unfit** — §8.1, and this is the part that could not have been
   argued from the sources.

**Not said by this withdrawal:** that rotating through the lumbar spine is fine. It is the
torsional-injury pathway McGill's stabilization finding implies. What is missing is a source that
states the fault, a number for the ratio, a graph node, and a proxy that survives projection.

### 7.2 `tt_momentum_over_control` — three independent failures

1. **The cited source prescribes the behaviour the rule flags.** The heuristic flags repetitions
   "that show no near-zero-velocity dwell at the side-peaks (no control pause)". The RAG doc says
   it is "crucial to **not stop** between repetitions". The parent spec's `citation_support`
   paraphrases that as a warning against between-rep momentum; read in place, it is an instruction
   to keep moving, and the rule would fault a user for obeying it.

   **This is a seventh distinct citation failure mode for the programme**, after inference,
   absence, exercise identity, secondary sourcing, a source-measured null on the proposed proxy,
   and Leg Abduction's citation/observation sign disagreement. It is a sharper case than the sixth:
   there the citation and the data pointed opposite ways, and here **the contradiction is inside
   the quoted document** — the spec's paraphrase inverts the sentence it is paraphrasing.
2. **The other disjunct has no number.** "A tempo threshold" and "a set ceiling" exist nowhere. The
   RAG doc's only tempo statement is directional and unquantified ("The slower one moves the arms
   from side to side, the harder the exercise becomes"). McGill's 30 and 60 °/s are protocol
   conditions; adopting either as a fault cut would convert a healthy-subject condition into a
   fault.
3. **No KG node.** `"Momentum"` under `movement="Torso Twist"` returns `Anterior Momentum
   Generation` and `Forward Momentum`, both reached from other movements' subgraphs and both
   zero-bucket — the identical result Leg Abduction §7.2 recorded for `abd_momentum`.

### 7.3 The graph's negative filter holds a second time; its positive signal still says nothing

Recorded queries, run through `retrieve_graph_context(query, movement="Torso Twist")` — the
function production calls:

| query | result |
|---|---|
| `Poor Abdominal Engagement` | `Torso Twist:Poor Abdominal Engagement`, **two non-empty buckets** |
| `Unstable Base` | matches **two** nodes — a shared zero-bucket `Unstable Base` and `Torso Twist:Unstable Base` with only `related_actions`. Dangling **and** ambiguous |
| `Insufficient Lateral Flexion Depth` | `Torso Twist:Insufficient Lateral Flexion Depth`, one bucket, **wrong axis** |
| `Lumbar Rotation` | `[]` |
| `Insufficient Rotation Range` | `[]` |
| `Momentum` | two cross-movement zero-bucket nodes |

The two rules with **no node** are exactly the two withdrawn on citation grounds — Leg Abduction
§7.3's negative filter, reproduced. Of the two rules that **do** have a node, one ships and one is
permanently silent, so presence again predicted nothing.

**And this movement adds the sharper case Leg Abduction did not have: a present node can be
actively misleading**, because it faithfully describes a *different movement pattern*. Sit-up
refused an **inverted** seed; this module refuses a **wrong-axis** one, on the same reasoning.

---

## 8. Measured

### 8.1 The sensing-fidelity pass — paying the Row note's deferred debt

The Row status note (parent spec §8, 2026-08-01) recorded that Fit3D cannot support
AUC-against-correctness validation but *can* support the 2-D-cue-vs-3-D-truth fidelity comparison
this project has run elsewhere, and called that "future work, not blocked on absent data". This is
where it is paid.

**Harness:** `src/fit3d/rotation_proxy_fidelity.py`, runner
`scripts/fit3d/run_rotation_proxy_fidelity.py --jitter`, pure helpers tested in
`tests/test_rotation_proxy_fidelity.py`. Every number in §8.1 and §8.2 is that script's output;
re-run it before editing either section. It exists because a number in a citation of record whose
script nobody can re-run is a defect this project has already logged once — the Row residual in
the parent spec.

**Corpus and construction.** Fit3D `standing_ab_twists`, 8 train subjects × 4 cameras × 45
repetitions = 180 (subject, camera, repetition) records. The 2-D side is **mocap-2D**: the mocap
ground truth projected through the real per-camera calibration, i.e. a **perfect detector**. Every
error below is projection alone.

**What the proxy does to a single line** (its estimate of `|θ|` via `arccos(width / rest_width)`):

| | shoulder line | hip line |
|---|---|---|
| true peak \|rotation\| per rep, median | 58.0° | 19.7° |
| proxy peak estimate, median | 77.3° | 41.3° |
| per-rep corr(estimate, \|true\|), median | 0.52 | 0.23 |
| fraction of reps with **negative** correlation | 11% | **35%** |
| per-frame MAE | 20.4° | 17.2° |

On the **hip line** — the smaller, noisier and decisive term of the rule's ratio — the proxy moves
the *wrong way* on more than a third of repetitions, and its per-frame error (17.2°) is as large as
the whole signal (19.7°).

**What the proxy does to the decision the rule makes**, the hip/shoulder ratio against the spec's
0.6 cut:

```
TRUE  hip/shoulder ratio : median 0.44
PROXY hip/shoulder ratio : median 0.58
rank corr(true, proxy)   : 0.876

truth fires   64/180 (35.6%)
proxy fires   86/180 (47.8%)
DISAGREE      30/180 (16.7%)   — 26 proxy-fires-truth-does-not, 4 the other way
```

**The honest qualifier**, because 0.876 is high: the proxy is **not noise, it is biased**. It
inflates, and the inflation runs almost entirely toward false positives. With a perfect detector.
`DecisionAgreementTest::test_a_high_rank_correlation_does_not_imply_agreement` pins the reason
both numbers are reported and neither is allowed to stand for the other.

**The small-angle resolution, against a real noise floor.** On the Fit3D projections one degree of
true rotation moves the shoulder width by **0.00016 of the image width** in the 0–15° band and
**0.00109** in the 45–75° band. MediaPipe's own frame-to-frame movement of the shoulder width,
over **all 130** REHAB24-6 cached-landmark videos, is **0.000323 of the image width** (median;
the hip width, the narrower and decisive line, moves 0.000191). One frame of that movement is
therefore worth about **2.0° of rotation near the braced centre and 0.30° near the peak** — a
sevenfold difference in resolution across the range the rule must span, exactly as
`d/dθ (width·cos θ) = width·sin θ` predicts.

**And the sign is never recoverable in this movement.** The spec's remedy for the even symmetry is
the left–right x-ordering flip, which requires more than 90° of rotation. The true relative trunk
twist measured on this corpus peaks at a **median of 44.9° per repetition** (p90 54.1, max 58.8).
The flip never happens.

**What transfers and what does not.** The corpus is `standing_ab_twists` — a different variant with
a **free pelvis**, so the *truth distribution* of the hip/shoulder ratio does not transfer to a
seated twist with the hips pinned, and no threshold is taken from it. What transfers is the
**projection geometry**: `width·|cos θ|` compresses, is even, and loses the hip line the same way
whatever the subject is doing with their legs.

### 8.2 What camera placement alone does to the shipped rule

Same corpus, same mocap-2D construction, and four cameras film **simultaneously**, so any
disagreement between them on the same repetition is pure projection error. The scored quantity is
the **signed sag** — `max(angle − setup baseline)` — matching §5.4, not its absolute value.

| quantity | median | p90 |
|---|---|---|
| **absolute** trunk-thigh angle, cross-camera spread of the per-rep median | **4.5°** | 10.6° |
| the **sag** the rule scores, value | 6.3° | — |
| the **sag**, cross-camera spread | **5.1°** | 15.7° |

**The derived quantity is less camera-robust than the angle it is built from** (5.1° against 4.5°
of spread, on a signal whose own median is only 6.3°), because taking a maximum over a window
picks up the worst projection excursion rather than averaging it away. Against the 15° cut the
typical disagreement is about a third of the threshold, but **the p90 disagreement is 15.7° — the
size of the cut**, so an unlucky camera placement can decide a near-threshold repetition on its
own.

**Two caveats, and the second is the one that limits this measurement most.** (i)
`standing_ab_twists` is a different variant. (ii) More importantly, its trunk motion is
predominantly **forward flexion**, which moves the trunk–thigh angle in the direction this rule
does *not* score; the sag median of 6.3° is small for that reason. So these figures bound the
**sag direction weakly** and should not be read as "the rule barely moves on real footage" — they
say that on a corpus with little sag, camera placement contributes about as much as the sag does.
A corpus of genuine seated Russian twists would be needed to say more, and none exists.

This is the same shape as Sit-up's finding (a 20° cut against a 28.2° median cross-camera spread)
and it is recorded for the same reason: the threshold is not moved to make the number look better,
because there is no cited number to move it to.

### 8.3 No view gate and no view discount — the fourth time, with a new reason

`view_estimation.py`'s limit 1 voids the front/rear/oblique labels for a **horizontal** subject.
Leg Abduction §1.3 then measured the same labels **systematically inverted** on an **upright**
subject in the exercise's own plane — 0 of 210 repetitions carried a frontal-observable label when
116 of them were filmed frontally.

A seated Russian twist is **neither posture**: the trunk is held at roughly 45°. It sits between two
regimes in **both** of which the labels have been measured wrong, and there is no seated-twist
footage anywhere in this repository on which the question could be settled. Gating or discounting
would dress an unmeasured label as evidence. `ctx.view_type` is deliberately unread, pinned by
`BraceRuleTest::test_no_view_gate_and_no_view_discount`.

### 8.4 The setup-baseline defect, measured and attributed

Row derived an **exactly 2×** inflation of `row_torso_rising`'s effective threshold from
`segment_reps` trimming the rep window and leaving a `setup` slice that is already loaded. This
rule has the same shape, so the trap was measured rather than assumed — and the measurement
**separates the two mechanisms that can cause it**.

Through the real `run_detector` on a three-swing clip:

- effective threshold **18.0°** against a nominal 15.0 — an inflation of **1.20×**;
- and **none of it is trimming**: the segmenter returns the windows untrimmed (0–23, 24–47, 48–71
  on 72 frames), so the whole residual is the `setup` slice carrying part of the ramp;
- fed the same window without the framework's median-5 smoothing the cut measures **17.5°**, the
  smallest sweep step past the algebraic value `15 / (1 − f) = 17.36°` for the fraction `f` of the
  ramp the 3-frame `setup` median already contains.

**This is a statement about that fixture**, not a proof the trimming cannot bite: the fixture's
swings begin and end at exactly zero, so there is nothing to trim. A swing that does not begin from
rest would be trimmed, and Row's mechanism would apply on top of this one.

**Re-confirmed after §5.3's sign change**, rather than assumed to have survived it: all three
figures are unchanged. They are invariant to it because `swing_clip` ramps the angle *upward* from
the baseline — monotone in the sag direction — so `max(x − b)` and `max|x − b|` coincide on this
fixture exactly. That is a property of the fixture, not of the rule, which is precisely why it was
re-run rather than reasoned about: §8.2's headline moved under the same change.

---

## 9. Testing

`tests/test_torso_twist.py`, 39 cases; the Fit3D harness's pure helpers add 23 in
`tests/test_rotation_proxy_fidelity.py`. The ones that carry an argument rather than coverage:

| test | what it pins |
|---|---|
| `MetricLayerTest::test_the_brace_angle_does_not_move_when_the_subject_only_twists` | §4.1 — the one construction choice that differs from Sit-up, asserted as a contrast against Sit-up's own construction rather than in isolation |
| `InvarianceTest::test_every_metric_is_invariant_under_camera_roll` | the Group E re-anchoring mandate, carried into Group F |
| `InvarianceTest::test_the_twist_magnitude_survives_mirroring_and_its_sign_does_not` | §4.2 — the honest limit of a monocular pipeline, asserted in both directions |
| `InvarianceTest::test_detections_survive_roll_and_mirroring_all_the_way_to_the_verdict` | that the invariance reaches a detection, through `segment_reps`' rectification. Compared field-by-field rather than byte-identically, because the metrics agree to ~1e-13 while `build_detection`'s 2-decimal evidence rounding can straddle a boundary — the tolerance is the rounding itself, and it is stated |
| `BraceRuleTest::test_it_reads_the_thighs_and_not_the_image_vertical` | §4.1 — the distinguishing property against the parent spec's own "relative to vertical" |
| `BraceRuleTest::test_a_twister_who_TIGHTENS_is_not_told_they_lost_the_brace` | §5.3 — the false-positive direction of the baseline, which the first implementation had and no green test caught, plus its non-vacuity companion |
| `BraceRuleTest::test_a_brace_lost_before_the_rep_opens_is_invisible` | §5.4 — Push-up's blindness, inherited and named |
| `BraceRuleTest::test_the_citation_records_that_mcgill_never_mentions_this_exercise` | §3.1 — that the qualifier reaches the shipped string a reader will meet, not just this document |
| `BraceRuleTest::test_no_view_gate_and_no_view_discount` | §8.3 |
| `SilentRomRuleTest::test_the_rom_rule_never_fires_even_on_a_repetition_that_trips_the_specs_cut` | that the silence is real, on the exact case the spec says to flag |
| `RegistrationTest::test_it_is_the_first_user_of_the_rectified_rep_signal_hook` | §4.4 |
| `EndToEndSegmentationTest::test_a_clean_clip_produces_no_detections_and_really_was_scored` | non-vacuously: it asserts the clip really segmented and was handed to the rules |
| `EffectiveThresholdTest` (3 cases) | §8.4 — the number, the attribution, and the algebra |

## 10. Honesty constraints

- **No threshold is tuned to anything.** The shipped cut is the parent spec's own 15°, and the
  source's 45° is explicitly *not* transferred into it (§5.1). The silenced rule's cut is the
  parent spec's own and it is silenced rather than moved.
- **The Fit3D pass is labelled as sensing fidelity on a mismatched variant** everywhere it appears,
  and no threshold is taken from it.
- **The `validated=False` reason is an existing one**, not a new one, and §2.2 says which.
- **The withdrawn rules' costs are stated, not explained away** (§7.1, §7.2 closing paragraphs).
- **The shipped rule's blind spot is pinned by a test**, not only described — and so is the
  opposite direction, after the first implementation got it wrong (§5.3).
- **Every number quoted in §8 comes from a script that ships in this repository**
  (`scripts/fit3d/run_rotation_proxy_fidelity.py`), not from a scratch probe. Re-running it after
  §5.3's fix corrected five of them, including §8.2's headline: the earlier figures were computed
  on an unsigned deviation, which is not what the rule scores.

### Out of scope

Fixing `MovementIcon.tsx`'s standing glyph (§2.1). Repairing `RunResult.fallback` not being
threaded into `RuleContext` (recorded by Deadlift, Shoulder Bridge and Leg Abduction; it recurs
here on any clip that takes the whole-clip path). Extracting the EgoExo-Fitness `Kneeling Side
Torso Twist` frames — 95 judged actions with per-criterion labels is the richest labelling this
programme has met, and it is a **lateral flexion** movement, so it belongs to whichever future
movement models that, not to this one.
