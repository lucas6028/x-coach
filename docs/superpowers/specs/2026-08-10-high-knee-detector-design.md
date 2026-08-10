# High Knee detector — design

**Sixteenth of sixteen. The last movement in the programme.**

Parent spec: `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`, section
"High Knee" (five rules). Module: `src/pose/movements/high_knee.py`. Harness:
`src/egoexo/high_knee_validation.py`, runner `scripts/egoexo/run_high_knee_validation.py`.
Measurements: `notes/high-knee-rule-validation.md`.

**Outcome: one rule permanently silent, four withdrawn, the detector NOT registered.** That is the
second unregistered detector in the programme, after Jumping Jacks, and it closes the roster at
**16 movements designed, 14 registered.**

Two of the four withdrawals are measurements rather than arguments, which is the thing worth
carrying out of this movement:

- the two trunk rules die on their **reference axis**: a trunk lean is an angle from the vertical,
  this drill has no vertical, and the substitute (the support limb) is 8.6–23.6° off the trunk in
  normal marching against thresholds of 10–15°;
- the pelvic-drop rule dies on a **zero-parameter control** that costs nothing because the corpus
  supplies it: three cameras film the same instant and disagree by more than the threshold.

---

## 1. What is different about this movement

Three things, and all three shaped the outcome.

**It is fast and cyclic.** `base.py:55` has named this movement since RS-SP1 as the one that "must
lower" `min_rep_seconds` ("high knees run ~3Hz, about 10 frames per rep at 30fps, which the default
would discard as noise"). Jumping Jacks measured that it did not need the knob. This one does — the
first and only use of it in sixteen movements (§5.3).

**Its corpus is rich, matching, and about other faults.** EgoExo-Fitness carries 68 judged High
Knee actions of exactly this exercise. Their seven criteria and the parent spec's five rules
overlap in **zero** pairs (§2). Jumping Jacks had one.

**Every parent-spec heuristic for it is written in image y**, and this corpus ships its side
cameras **rolled 90°** (§4.1). The metric layer is built entirely from cosines and ratios between
body vectors, which is the only reason there are numbers to report.

---

## 2. Two taxonomies, zero overlapping pairs

### 2.1 The corpus

| | |
|---|---|
| judged High Knee actions in EgoExo-Fitness | **68** (120 annotations, 1–3 annotators each) |
| reachable from the truncated `frames_open` archive | **6** |
| (action, camera) pairs | **18** (three simultaneous exo cameras each) |
| frames | **4 698** |

### 2.2 The criteria, and what the spec writes rules about

Fraction of the 68 actions on which a strict majority of annotators judged each criterion FAILED:

| EgoExo criterion | majority-failed | which spec rule models it |
|---|---|---|
| Aim to maintain the fastest speed possible while performing the leg lifts. | 44.1% (30/68) | — |
| Swing your arms in rhythm with the leg lifts. | 26.5% (18/68) | — |
| Maintain a stable upper body throughout the exercise. | 14.7% (10/68) | — (near miss, §6.3) |
| Lift your legs alternately and quickly. | 10.3% (7/68) | — |
| Look straight ahead. | 1.5% (1/68) | — |
| Keep your back straight. | **0.0% (0/68)** | — (near miss, §6.3) |
| Keep the balls of your feet in contact with the ground. | 0.0% (0/68) | — |

And the other direction: `hk_insufficient_knee_lift`, `hk_trunk_lean_back`,
`hk_forward_trunk_collapse`, `hk_contralateral_pelvic_drop` and `hk_stride_asymmetry` are judged by
**no criterion at all**. The corpus's two largest faults are cadence and arm rhythm; the spec
models neither.

`validated` stays False for **Jumping Jacks' sixth reason** — the variant matches and the labels
judge different faults — rather than a seventh.

### 2.3 The one label the corpus does supply, and its tier

EgoExo's checklist has no knee-height criterion, but its free-text comments do. Under a rule fixed
**before** any comment other than six disclosed ones was read (`§6.1`), **15 of 68** actions carry a
leg-height complaint, **12 of them in the 62 actions held out** from the rule's own construction.

This is SECONDARY evidence and is reported as a strictly weaker tier than a checklist label —
Sit-up logged secondary sourcing as a citation failure mode. It establishes that the fault is real
and that human judges care about it. It cannot license a threshold: a comment judges a whole
action, not a repetition.

---

## 3. The citation audit

### 3.1 Matijašević et al. (2025) — a graded pair of targets, and the spec uses both

Re-fetched (PMC12591607, DOI 10.70252/LYKE8231). The scoring tables state:

> "The thigh of the swinging leg reaches **45°** relative to the ground" (Table 1, **A-skip**)
> "The thigh of the swinging leg reaches **90°** relative to the ground" (Table 2, **B-skip**)

A graded pair: the easier drill, then the harder one. 90° is the thigh parallel to the ground —
the knee at hip height. 45° is halfway there, with the knee still well below the hip.

**The parent spec cites the first and implements the second.** Its rationale quotes the A-skip's
45°; its detection heuristic is "flag when the knee never rises to near hip height", which is the
B-skip's 90°. And its prose — "thigh at least ~45° above horizontal" — puts the source's number on
the **wrong side of horizontal**, since 45° relative to the ground is 45° *below* horizontal.

**This is the ninth citation failure mode in the programme, and only half of it is new.** The
inverted paraphrase is Torso Twist's mode 7 recurring. What is new sits underneath it: **the source
states a graded family of targets and the spec cites one grade while implementing the other** — an
unannounced upgrade to the harder variant's criterion carrying the easier variant's citation.
Nothing is misquoted; the quote simply does not govern the code.

**Four transfers separate this paper from this rule**, all of them stated in the paper itself:

1. it scores the **A-skip**, a skipping drill, not high knees;
2. performed **travelling on an athletics track**, not in place;
3. by 63 recreational-to-trained males **explicitly excluded** for "prior experience in athletics";
4. and the paper's own finding is that A-skip had "**a trivial correlation** with 5 m sprint
   performance" and likewise with 20 m — the outcome the battery was built to predict.

### 3.2 Bramah et al. (2018) — verified, and about running

Re-fetched (PMID 30193080). A "controlled laboratory study" of "72 injured runners and 36 healthy
controls" in which "the injured runners demonstrated greater contralateral pelvic drop (CPD) and
forward trunk lean at midstance", with "for every 1° increase in pelvic drop, there was an 80%
increase in the odds of being classified as injured".

The citation is genuinely VERIFIED for its own claim. What the parent spec does not flag is the
**task transfer**: midstance of overground running is not an instant of a stationary marching
drill, and the finding is an injured-vs-healthy contrast, not a technique criterion. That is a
qualification on three rules (`hk_trunk_lean_back`, `hk_forward_trunk_collapse`,
`hk_contralateral_pelvic_drop`), and in each case it is a *supporting* reason for withdrawal, never
the deciding one — the deciding ones are measured (§7).

### 3.3 The KG's one overlapping node, and its misattributed grounding

`scripts/knowledge/stub_general_movements_v3.py:142-151` records:

> `"grounding": "EgoExo-Fitness TKV (top-failed: cadence/speed 44%, arm rhythm 26%, upper-body
> stability 15%, knee lift 10%)"`

All four figures reproduce exactly from the labels — 44.1%, 26.5%, 14.7%, 10.3%. **And that is the
problem with the fourth**: 10.3% is the failure rate of "Lift your legs alternately and quickly", a
criterion about alternation and speed. There is no knee-height criterion in the checklist at all.
So `High Knee:Insufficient Knee Lift` is a plausible node whose only stated evidence measures a
different fault.

A third variety of misleading-but-present node, after Torso Twist's (a node faithfully describing a
**different movement**) and Jumping Jacks' (a node seeded from a **blend** of two): **a node seeded
from the wrong criterion of the right movement.**

### 3.4 KG query resolution, and the negative filter's cleanest result yet

Recorded before any rule was written, via `retrieve_graph_context(query, movement="High Knee")`:

| query | resolves to | buckets |
|---|---|---|
| `Insufficient Knee Lift` / `Knee Lift` | `High Knee:Insufficient Knee Lift` | DANGLING |
| `Unstable Upper Body` | `High Knee:Unstable Upper Body` | quality_impacts: Trunk Stability, Core Stability |
| `Slow Cadence` | `High Knee:Slow Cadence` | corrections: Maintain Even Tempo |
| `Poor Arm-Leg Rhythm` | `High Knee:Poor Arm-Leg Rhythm` | causes: Poor Neuromuscular Control |
| `Trunk Lean` / `Forward Trunk Lean` / `Lumbar Hyperextension` | — | NO NODE |
| `Pelvic Drop` / `Contralateral Pelvic Drop` | — | NO NODE |
| `Pelvic Control` | shared `Pelvic Control` | no scoped node, dangling |
| `Asymmetry` / `Stride Asymmetry` | shared `Symmetry` | no scoped node, dangling |

**The negative filter holds for a fourth movement, and here it is perfect in both directions.** The
four rules with no scoped node are *exactly* the four withdrawn; the one rule with a scoped node is
*exactly* the one kept as silent. Leg Abduction §7.3's finding, reproduced with no exceptions for
the first time.

The positive signal still predicts nothing on its own — the one matching node is DANGLING, and the
three nodes with real buckets correspond to no rule in the parent spec.

---

## 4. The measurement layer

### 4.1 Everything is roll-invariant, because this corpus is rolled

EgoExo's two side cameras ship frames **rolled 90°** with no EXIF: a standing subject lies
horizontally across a 456×256 landscape frame (verified visually, and by the trunk reading ~92° and
~98° from image-down on `exo_l`/`exo_r` against ~0.5° on `exo_m`). Sit-up found this for a supine
movement, where it was tempting to read it as a quirk of filming someone on the floor. It is not; it
is how these cameras ship.

Every parent-spec heuristic for this movement is phrased in image coordinates — `y_knee - y_hip`,
"the shoulder-midpoint x moves behind the hip-midpoint x" — and every one is meaningless on those
frames. The metrics are cosines and ratios between **body** vectors, so they are unaffected.

**The corollary is a caveat, not a reassurance:** MediaPipe is not roll-equivariant (this project
measured a median 9.8° landmark shift under rotation), so the side-camera landmarks are degraded
even though the metrics computed from them are well defined.

### 4.2 Thigh elevation is trunk-relative, and that is a deliberate substitution

`thigh_elevation` = cos(angle between the hip→knee vector and trunk-up). −1 is a thigh hanging
straight down, 0 is the knee at hip height. Bounded to [−1, 1] on every frame, so no threshold on it
can be dominated by a scale error.

The source's target is **ground-relative**; hip flexion, the thing the drill trains, is by
anatomical definition an angle between femur and pelvis. The two differ by exactly the trunk's own
lean — which is what makes the trunk-relative form the right one here: **an athlete who throws the
torso backward to hoist the knee gains ground-relative thigh angle and gains no hip flexion**, so a
trunk-relative rule cannot be cheated by the very fault `hk_trunk_lean_back` describes. Stated as
the substitution it is, not as an equivalence.

### 4.3 The view gate is the rule's own, not the view estimator's

`anterior_axis_length` = |heel→toe, averaged over both feet, with the trunk component removed| /
shoulder width. It says how much of the subject's fore-aft direction survives projection into this
image.

Measured, it separates this corpus's cameras **with no overlap**: 0.156–0.318 on the two side
cameras, 0.027–0.044 on the frontal one. That matters because `view_estimation.py` has now been
measured **inverted** on supine subjects (Sit-up) and **outside its stated regime** on standing ones
(Leg Abduction, 0/210 frontal-observable). A rule that gates on the magnitude of its own reference
axis needs neither.

**Only its LENGTH is used, and the distinction is load-bearing.** Projecting the trunk onto an axis
built by removing the trunk direction returns zero on every frame by construction — the first draft
of this module did exactly that and reported a trunk lean of identically 0.000 across all 18 pairs.
A length is still meaningful.

### 4.4 One repetition is one knee drive

The legs alternate, so `thigh_elevation_difference` (left − right) is **bipolar** and its magnitude
peaks once per drive. Rectifying gives one repetition per drive; not rectifying would give one per
left–right cycle, with one side's peak landing on the window **boundary**, where the Bicep Curl
trimming finding says it is least reliable. Torso Twist is the precedent for `rep_rectify`.

Which leg is driving is not encoded in the phase — it is the sign of the difference at the peak,
readable from the metric tuple by any rule.

### 4.5 Six landmarks required, twelve read

Required: shoulders, hips, knees — the trunk axis every metric is expressed in, plus the rep signal.
The heels and toes are read for the view gate and **not** required: they are the landmarks this
drill occludes and motion-blurs most, and a missing foot must cost the view gate, not the whole
frame. The `jumping_jacks` principle, applied to a different pair of landmarks.

---

## 5. Pipeline properties — everything except the thresholds works

Over the 18 (action, camera) pairs, through the real `run_detector`:

| | |
|---|---|
| median validity rate (6-landmark gate) | **1.000** |
| pairs on the whole-clip fallback | **0 of 18** |
| repetitions segmented (shipped 0.15 s floor) | **150** |
| repetitions SCORED (`select_reps` drops partials) | **146** |
| cadence | median **1.31 Hz**, range **0.70–2.20 Hz** |

### 5.3 The framework knob reserved fifteen movements ago is finally needed

Measured the **non-circular** way, because every window `segment_reps` *returns* is at least
`min_rep_seconds` long by construction and so can never show the floor biting. Re-segmenting the
same signals at the framework default:

| floor | repetitions found |
|---|---|
| 0.15 s (shipped) | **150** |
| 0.40 s (framework default) | **52** |

**The default discards 65.3% of this movement's repetitions.** The surviving cadence is 0.45–1.42 s
per repetition — physically ordinary, so the low floor is not manufacturing noise repetitions.
(150 is the SEGMENTED count, partials included, which is the right denominator for a question
about segmentation. Every fire rate below uses the 146 SCORED repetitions instead.)

**The corpus makes the result stronger, not weaker.** 30 of its 68 actions are judged FAILED on
"maintain the fastest speed possible", so this is a population humans considered *too slow* — and
the default floor still throws away two repetitions in three.

**The value is the framework's own arithmetic, not a fitted one.** `base.py:55` states 0.33 s per
repetition for this movement; 0.15 s is half of that. It is not tuned to the 1.31 Hz this corpus
happens to show. Jumping Jacks left the comment alone precisely because it also named this
movement; that loop is now closed.

One bias worth stating in the direction it runs: cadence is computed over the span the repetitions
occupy, not the whole clip, because dividing by idle frames at the ends would report a slower
cadence — the direction that would falsely support leaving the floor alone.

---

## 6. `hk_insufficient_knee_lift` is PERMANENTLY SILENT

### 6.1 The pre-registered comment rule

Fixed before any comment other than six disclosed actions' was read. A comment is a POSITIVE iff one
sentence contains both a leg token (`leg`, `knee`, `thigh`) and an insufficiency token (`too small`,
`higher`, `not enough`, `insufficient`, `inadequate`, …). Comments carrying an insufficiency plus a
range word but **no** leg or arm token are counted separately as `unattributable` and are **not**
scored as positives — the same phrase is used about the arm swing elsewhere in the corpus.

Result: **12 of 62** held-out actions positive (14 of 109 annotations), 3 of the 6 disclosed.

### 6.2 The spec supplies two numbers and they give opposite verdicts

Over 146 scored repetitions, on the two cameras the gate admits:

| cut | provenance | fires on |
|---|---|---|
| thigh at hip height (elevation 0.0) | the spec's **heuristic** | **100.0% of repetitions, every action** |
| 45° from hanging (elevation −0.707) | the spec's **citation** | 0.0–71.1% by action (0.0–83.3% by camera) |

The implemented cut fires on **every repetition of every action**, including both actions in which
every annotator marked every criterion true. Observed peak thigh elevation runs −0.43 to −0.77, i.e.
**40–65° of hip flexion**: real performers land *between* the source's two targets, and the spec
picked the far one.

**The cited cut is not the answer either, and its failure is the more interesting one: it sorts
this corpus BACKWARDS.** It fires on 0.0% of all three actions whose free-text comments complain
about leg height, and on 7.1–71.1% of the three whose comments do not (two `unattributable`, one
negative). The one human signal
available about this fault is *anti*-correlated with the cited threshold.

At n = 6 with a secondary label that is not by itself a refutation — but it removes the only
argument that could have justified shipping the cited number, namely that it happened to sort the
corpus sensibly. What remains is the four transfers of §3.1. **A number that survives four
transfers, and then sorts the only available labels the wrong way, is not a threshold.** Jumping
Jacks silenced a rule that had *more* going for it than this one.

Neither number is moved. Both are pinned, and so is the fact that they differ.

### 6.3 The two near misses, measured rather than argued

**"Maintain a stable upper body"** is the criterion with a real positive class, and the comments say
what it means: "upper body lacks stability", "excessive upper body sway", "slightly shaking",
"unstable center of gravity". That is **variance**; both trunk rules read a signed **mean**. On the
gated cameras of the six reachable actions:

| judged on stability | actions | pairs | median trunk lean | median SD |
|---|---|---|---|---|
| FALSE | 2 | 4 | −10.10° | 5.77° |
| TRUE | 4 | 8 | −14.77° | 6.19° |

The mean separates **the wrong way** and the variance does not separate at all. At n = 6 actions
that is not a powered null — but it is the *reading of the criterion*, not the n, that decides:
neither rule models sway.

**"Keep your back straight"** is the criterion the two trunk rules would model, and **no action in
the corpus fails it by majority** (0/68).

### 6.4 The upgrade path

A corpus that judges knee **height** would settle it, and the comments show human judges care even
though the checklist does not ask. Failing that, `frames_open`'s missing `.ac` part would raise the
reachable set from 6 actions to most of 68, letting the comment-derived labels be tested at usable
n. A download and a label pass, not a research programme.

---

## 7. Four rules are WITHDRAWN

### 7.1 `hk_trunk_lean_back` and `hk_forward_trunk_collapse` — the reference axis is the size of the fault

They are the two signs of **one scalar**, and they fail on that scalar's reference axis.

**A trunk lean is an angle from the world vertical, and this drill has no vertical.** Group E
established across three movements that the image vertical is not the world vertical, and this
corpus proves it twice over by shipping its side cameras rolled 90°. Leg Abduction's answer — take
the vertical from the **support limb** — is the only construction available, and here it does not
hold:

> **the angle between the trunk and the support limb is 8.6–23.6° (median 13.1°) during normal
> marching**, against rule thresholds of 10–15° (backward) and 15–20° (forward).

Part of that is not even marching: **the stance foot sits under the hip *joint* while the axis is
drawn from the pelvis *midpoint***, so the axis is tilted by atan(half-pelvis / leg length) ≈ 6° on
adult proportions before the subject has moved at all. Nothing a performer does removes it.
(`SupportLimbGeometryTest` pins the mechanism.)

**And the error runs toward one rule's firing direction:**

| cut | fires on |
|---|---|
| 10° backward (`hk_trunk_lean_back`) | **69.7% of scored frames** (56–83% on the two faultless actions) |
| 15° forward (`hk_forward_trunk_collapse`) | **0.0% of scored frames** |

An unsigned or unvalidated baseline offset running toward the fault is `pushup_head_drop`'s finding
and Torso Twist's brace finding for the **third** time. What is new is that here it sinks both signs
at once — one by false-firing on judged-correct performances, one by never firing at all.

Supporting, not deciding: neither rule has a scoped KG node (§3.4); Bramah is about running (§3.2);
and the corpus criterion with a positive class measures variance, not mean (§6.3).

**Not said by this withdrawal:** that throwing the torso backward to hoist a knee is fine. What is
missing is a vertical.

### 7.2 `hk_contralateral_pelvic_drop` — three simultaneous cameras refute it

The three exo views film the **same instant of the same performance**, so any disagreement between
them is pure projection, with no performance variation in it. This is a stronger instrument than a
synthetic level-pelvis control and it costs nothing.

Median pelvic obliquity, restricted to the two cameras whose own gate says they can see anything:

| action | exo_l vs exo_r spread | frame-by-frame r |
|---|---|---|
| yT4RK3_action_2 | 0.97° | −0.483 |
| yT4RK3_action_9 | 2.72° | −0.217 |
| yT4RK3_action_14 | 5.49° | +0.116 |
| xYkvB0_action_15 | 7.90° | −0.413 |
| xYkvB0_action_9 | 8.52° | −0.026 |
| zOfbr6_action_14 | 13.68° | −0.114 |

against the parent spec's "> ~5–8°" threshold. **The camera alone moves the quantity past the low
end of that threshold on four of six actions, and past the high end on two.**

**And frame by frame the two cameras are anti-correlated on four of six** (r = −0.48 to +0.12).** They do not merely offset
each other — they largely disagree about which way the pelvis is tilting at any instant. A quantity
that two simultaneous views of one pelvis report in opposite directions is not measuring the pelvis.

The degenerate frontal camera is **excluded** from these spreads rather than pooled in: a reading
from a camera its own gate rejects is not a second opinion, and pooling it here would *understate*
the spread, i.e. bias the evidence toward keeping the rule.

Supporting, not deciding: no scoped KG node. Explicitly **not** the reason: Bramah's
pelvic-drop→injury association is the strongest single result any citation in this section carries.
The withdrawal is about **measurability**.

**Not said by this withdrawal:** that a dropping pelvis is fine. What is missing is a monocular
quantity that survives the camera.

### 7.3 `hk_stride_asymmetry` — `jj_landing_asymmetry`'s three failures

1. **No scoped KG node.** "Asymmetry" and "Stride Asymmetry" reach only the shared, dangling
   `Symmetry` dimension carried by the Squat flagship.
2. **It is a disjunction of two quantities** — "peak knee-lift height AND per-side pelvic-drop
   angle", firing if either differs by 15–20%. One `fault_id` whose evidence might be a knee or a
   pelvis cannot produce a coherent explanation card, and `fault_id` is the join key between spec,
   registry and every stored analysis. One of the two quantities is the one §7.2 just refuted.
3. **"Consistently across reps" is cross-rep state this architecture does not have.**
   `run_detector` scores one repetition at a time. `arm_vw` and `jj_landing_asymmetry` recorded the
   same limit; the rep semantics here (one drive per repetition, so a repetition contains one side's
   drive) make it **structural** rather than incidental.

**Not said by this withdrawal:** that a habitually under-driving side is fine.

### 7.4 And so the detector is not registered

Registration is what makes a movement analyzable in the web app: `registry.list_detectors()` backs
`GET /api/movements`, and `analyze_pose_payload` returns `analysis_pending` ("coming soon") when no
detector exists. With one rule silent and four withdrawn, registering would offer users an analysis
that **cannot ever report a fault** while wearing the Beta tag that says faults are possible.
`registry.py` carries the reason in place of the import.

**What works is kept**, because none of it is what failed: the roll-, mirror- and scale-invariant
metric layer; the view gate that separates this corpus's cameras with no overlap and no view-
estimator call; the phase assignment and the rectified per-drive repetition definition; and
`min_rep_seconds=0.15`, measured to recover 65% of this movement's repetitions.

**The most promising rule this movement could have is one the parent spec never wrote.** The
corpus's largest fault by a wide margin is **cadence** — 30 of 68 actions judged too slow — the KG
carries `High Knee:Slow Cadence` with a real correction bucket, and cadence is the one quantity here
that is fully roll-, view- and scale-invariant, because it is counted in time rather than measured
in space. It is not built, because this programme implements the parent spec's roster and does not
author new rules. It is recorded because the evidence for it is already in the module.

---

## 8. Testing

`tests/test_high_knee.py`, `tests/test_high_knee_validation.py` and `tests/test_frame_extraction.py`.

Worth naming:

- **`test_the_elevation_is_a_cosine_not_an_image_y_difference`** asserts *away from* the parent
  spec's `y_knee - y_hip` form: the two agree on an upright camera (asserted first, so the test is
  not just measuring a rewrite) and only the cosine survives a 90° roll.
- **`SignedTrunkLeanTest` ramps every fixture BOTH WAYS.** An unsigned deviation from a baseline is
  actively inverted, this project has shipped that bug twice, and both times the reason no green
  test caught it was that every fixture ramped the same way. These paired assertions caught an
  inverted sign during development — in the fixture, which is exactly what they are for.
- **`SupportLimbGeometryTest`** pins the anatomical floor under §7.1: a stance foot under the hip
  joint tilts the reference axis by ~9.5° in the fixture's proportions before anyone moves.
- **`SupportLimbSelectionTest` exists because review found a surviving mutant.** Inverting
  `_support_ankle` to pick the AIRBORNE leg left the whole harness suite green, because every
  fixture stood on both feet and the tie resolved the same way regardless of the comparator's
  sign. A drill whose point is that one knee is driven now has fixtures with one knee driven; the
  mutation kills three tests.
- **`test_no_withdrawn_rule_leaves_a_metric_behind`** — the trunk scalar and pelvic obliquity are
  absent from `HIGH_KNEE_METRIC_KEYS` and recomputed in the harness instead.
- **`test_no_rule_produces_any_detection_at_all`** is the claim non-registration rests on.
- **`test_the_all_silent_exemption_is_earned_not_asserted`** (in `test_kg_query_resolution.py`)
  grows a second entry, and it is earned by an assertion that the module really contains no
  `build_detection` call.

---

## 9. Honesty constraints

- **No threshold was moved.** Both of the spec's disagreeing knee-lift cuts stay where they are and
  are pinned, including that they differ. The withdrawn rules' cuts live in the harness so their
  evidence stays re-runnable, and nowhere else.
- **`min_rep_seconds` is the one number that changed**, and it comes from `base.py:55`'s own
  arithmetic (half of the 0.33 s it states), not from the 1.31 Hz this corpus shows.
- **Every figure here is the harness's output**, and that is now true of the criterion table (§2.2)
  and the comment-mining counts (§2.3, §6.1) as well: `criterion_failure_rates` and
  `classify_knee_lift_comment` ship in `src/egoexo/high_knee_validation.py`, so the
  zero-overlapping-pairs finding and the KG grounding misattribution are both re-runnable.
  Figures quoted from scratch probes during development were corrected when the shipped harness
  was run — twice, once before review and once after it found that the withdrawn quantities were
  being measured over whole clips instead of the scored rep windows. The corrected measurement
  moved the back-lean fire rate from 47.0% to **69.7%**, i.e. AGAINST the rule, which is worth
  stating: the error had been running toward keeping it.
- **n = 6 actions is small and is not hidden.** Where a conclusion rests on the corpus rather than on
  a construction (§6.3), that is said. The two withdrawals in §7.1 and §7.2 rest on constructions
  and controls, which is why they are stated as decided.
- **The reachable set is discovered, not predicted.** `extract_action_frames.py` carries the frame
  ranges of all 68 judged actions and writes whatever the truncated stream reaches.
- **Two denominators, named apart.** 150 repetitions are SEGMENTED (partials included) and back the
  `min_rep_seconds` argument; 146 are SCORED (`select_reps` drops partial windows) and back every
  fire rate. The harness emits both rather than letting one number quietly change meaning.
- **A pose file whose `sample_id` does not resolve to a judged action aborts the run** rather than
  defaulting into the judged-correct bucket.

### Out of scope

A cadence rule (§7.4). Building the missing `.ac` download. Registering the detector. Authoring a
knee-lift threshold from this corpus.
