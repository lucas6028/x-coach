# Sit-up Rule Detector — Design Spec

**Status:** design spec · **Date:** 2026-08-09
**Movement:** Sit-up (curl-up) · **Detectors after this one:** 11/16
**Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`
§Group E → Sit-up (curl-up)

---

## 1. Purpose, and the one thing that makes Sit-up unlike the ten before it

Group D closed on 2026-08-09 with Arm VW as the tenth detector. Group E opens here: **Sit-up,
Shoulder Bridge, Leg Abduction**. This is the eleventh detector and it ships the way every one
since Push-up has — cited rules, `validated=False`, Beta in the UI, no frontend edit.

The parent spec gives Sit-up four rules. **One ships, one is registered permanently silent, and
two are withdrawn:**

| # | fault_id | Treatment | Cue |
|---|---|---|---|
| 1 | — | **WITHDRAWN** (§5) | Excessive speed / loss of trunk control |
| 2 | `situp_hip_flexor_dominance` | **SILENT** (§4) | Rigid straight-body pull about a fixed pelvis |
| 3 | — | **WITHDRAWN** (§6) | Excessive ROM — a full sit-up past the curl-up range |
| 4 | `situp_incomplete_rom` | **ships** (§7) | Scapulae never clear the floor |

One live rule is the thinnest detector in the registry, and §10 states plainly why padding it
would be the failure rather than the fix.

### 1.1 THE SUBJECT IS HORIZONTAL, AND THAT BREAKS THE PARENT SPEC'S MEASUREMENT CONVENTION

Every Group E fault is defined against **the floor**. The parent spec's own convention block says
so twice:

> **Trunk-flexion angle** = angle of the shoulder-midpoint→hip-midpoint vector relative to the
> floor/horizontal (0deg = lying flat, 90deg = fully upright seated).
> **Pelvic-tilt (frontal)** = signed angle of the left-hip(23)→right-hip(24) line relative to
> horizontal.

**The image horizontal is not the floor, and on the only sit-up footage that exists it is 90° away
from it.** EgoExo-Fitness stores its `exo_l` and `exo_r` frames rotated a quarter turn, with the
room's ceiling running down the side of the image, and **carries no EXIF orientation tag** (checked
with PIL on `zOfbr6/exo_{m,l,r}`: `getexif()` is empty, `Orientation` is `None`) — so the roll is
baked into the pixels with nothing downstream to signal it. A trunk-flexion angle read against the
image horizontal reads a supine subject as 90° flexed and a fully seated one as 0°.

Every rule shipped in this project so far has been immune to this by accident: they all measure
**joint-relative** angles (`angle_degrees(a, b, c)`), which are invariant under camera roll. Sit-up
is the first movement whose parent-spec rules are written against a world reference, so it is the
first that has to *choose* one. **§7.2 re-anchors the one shipped rule to the body** — hip-angle
excursion, `angle(shoulder, hip, knee)` — and that choice is a Group E-wide convention, because
Shoulder Bridge and Leg Abduction both say "vs horizontal" too.

### 1.2 THE VIEW RATINGS ARE FICTION, AND THE MODULE THAT WOULD SUPPLY THEM SAYS SO ITSELF

Three of the parent spec's four Sit-up rules are rated `medium`/`high` on **side**. `side` is what
this movement needs and `side` is the view the production estimator has never once emitted:
`arm_vw.py`'s re-measured census over all 49 files under `data/runtime/pose_json` gives
`rear_oblique` 37, `rear` 9, `unknown` 3, **`side` 0**.

Worse than unreachable — *undefined*. `src/pose/view_estimation.py`'s module docstring, limit 1,
was written for exactly this case and forbids exactly this use:

> `signed_orientation` is `sign(left.x - right.x)`, an image-space left/right ordering. Its
> front/rear meaning is validated only for UPRIGHT subjects; for a horizontal body the frontal axis
> no longer maps onto image x, so the `front`/`rear`/`*_oblique` labels carry no validated meaning
> there. **Do not gate a horizontal-movement rule on them.**

So the shipped rule **carries no view gate and no view discount at all** (§7.3). That is a first:
every previous module either gated (`vw_lr_asymmetry`) or discounted (`vw_incomplete_excursion`,
`arm_abd_*`). Doing either here would be dressing an unvalidated label as evidence.

§8 reports what the estimator actually returns on six real supine clips, which is the first
measurement anyone in this project has taken of that question.

---

## 2. The data situation: no threshold can be measured, and the reason is not "no labeled data"

The Deadlift, Row, Band Pull Apart and Bicep Curl specs each open with "no labeled data exists, so
`validated` stays `False`". Arm Abduction and Arm VW opened with the opposite. **Sit-up is a third
case and it must not be collapsed into either.**

### 2.1 What does NOT exist

- **REHAB24-6 has no sit-up.** Its six exercises are `Ex1` arm abduction, `Ex2` arm VW, `Ex3` table
  push-ups, `Ex4` leg abduction, `Ex5` leg lunge, `Ex6` squats. The dataset that carried Lunge, Arm
  Abduction and Arm VW is silent here. (`Ex4` is Group E's *Leg Abduction*, 210 reps — that is the
  next spec's asset, not this one's.)
- **Fit3D has no supine action.** All 47 activity types under `data/Fit3D/train/s03/joints3d_25`
  were listed; the nearest are `pushup`, `burpees`, `mule_kick` and `standing_ab_twists`. There is
  no curl-up, no crunch, no sit-up. Arm Abduction reached for Fit3D `side_lateral_raise` when its
  own labeled set fell short; that escape hatch is not available.

### 2.2 What DOES exist, and precisely what it can prove

**EgoExo-Fitness carries 82 human-judged sit-up actions** (`data/EgoExo-Fitness/processed/manifest.csv`,
`action_name == "Sit-ups"`), each with a per-criterion technical-keypoint verification (TKV) record.
Censused over all 82:

| faulted / n | n_false | n_true | criterion |
|---|---|---|---|
| 44 / 82 | 85 | 57 | Put your feet together. |
| 36 / 82 | 67 | 75 | Extend your arms straight above your head. |
| 28 / 82 | 59 | 83 | Reach forward to touch your feet with your hands. |
| 16 / 82 | 45 | 97 | Open your legs slightly. |
|  9 / 82 | 16 | 126 | Ensure your lower back, upper back, shoulders, and head touch the ground sequentially as you lower back down. |
|  7 / 82 | 26 | 116 | Maintain control over your movement throughout the exercise. Aim for a frequency of about one repetition every 2 seconds. |
|  4 / 82 | 12 | 130 | Lift your head, shoulders, upper back, and then lower back off the ground in turn. |
|  3 / 82 | 11 | 131 | Engage your abdominal muscles to initiate rising up. |
|  0 / 82 |  3 | 139 | Lie on the mat with your back flat. |

Two things follow, and they pull in opposite directions.

**(a) The faults human experts actually mark are not the faults the parent spec models.** The top
four criteria — feet position, arm position, forward reach, leg spacing — have no counterpart in
any of the parent spec's four rules. The two criteria that *do* correspond to a spec rule are the
two rarest: segmental lift (4/82) and cadence control (7/82). This is not a defect in either
document; it is what a *setup-and-form* checklist looks like next to an *injury-mechanism* rule set.
It is recorded because it bounds how much a future validation against this data could ever say.

**(b) EgoExo's sit-up is a FULL sit-up, and the parent spec's is a curl-up.** The dataset's own
canonical guidance for the movement reads:

> "lie on the mat, put your arms straight on either side of your head, put your feet together, open
> your legs, and use your abdominal muscles to get up. keep your head, shoulders, upper back, and
> **lower back off the ground** in turn, and **touch your feet with your hands**. … control your
> speed throughout the action, with the frequency of about **once every 2 seconds**."

Lifting the lower back off the ground and touching the feet is precisely what the parent spec's
`excessive_rom` rule exists to flag. **In the only labeled sit-up data available, the parent spec's
third rule would fault every correctly-performed repetition.** §6 turns that into a withdrawal.

### 2.3 Six clips of real supine kinematics — what they are for

Four EgoExo-Fitness records are recoverable from the truncated `frames_open` archive. The download
is split into 3 GiB parts and **`.ac` is missing** (`.aa`, `.ab`, `.ad` are present), so the
concatenation cannot be decompressed — but `.aa` is the *prefix* of a single gzip stream and decodes
standalone until it runs out. It contains complete frame sets for `zOfbr6`, `zT0YQO`, `z8RAua` and a
partial `yT4RK3`. Six of the 82 judged sit-up actions fall in those records:

| sample_id | frames | quality (1–5) | faults |
|---|---|---|---|
| `zOfbr6_action_1`  | 1456–3076   | 1.0 | 2 |
| `zOfbr6_action_5`  | 7456–8431   | 2.0 | 2 |
| `zOfbr6_action_12` | 14941–16006 | 2.0 | 3 |
| `z8RAua_action_7`  | 6091–6826   | 3.0 | 3 |
| `yT4RK3_action_6`  | 3166–3931   | 3.5 | 2 |
| `yT4RK3_action_12` | 6046–6811   | 2.5 | 2 |

17,793 frames across `exo_m`, `exo_l`, `exo_r` were extracted and run through MediaPipe Pose
(`model_complexity=2`), emitting the same JSON schema `src/pose/process_videos.py` writes.

**These six clips answer SENSING questions and cannot answer THRESHOLD questions.** They are full
sit-ups (§2.2b). A fire rate measured on them is a fire rate on a different exercise from the one
the rules are written for, and §8 labels every number accordingly. What they legitimately settle is:
what does the view estimator return for a supine subject, is a hip angle computable at all, does
`segment_reps` find repetitions on this signal, and does the shipped rule's phase scope clear the
`min_frames` cliff. Those are properties of the pipeline, not of the variant.

---

## 3. Three citation failures, and the third is a new kind

This project has now recorded three ways a `citation_support` string can be true while the rule
built on it is not supported — inference (Arm Abduction's impingement arc), absence (Bicep Curl's
wrist flexion), and exercise identity (all four Arm VW sources). Sit-up adds a fourth and a fifth.

**(4) SECONDARY SOURCING — right paper, right exercise, but the paper is quoting someone else.**
Mandroukas PMC9505236 is genuinely a curl-up/sit-up EMG study. Both numbers the parent spec draws
from it are things Mandroukas *reports from other literature*, marked in the RAG text by a trailing
citation bracket:

- `"The stress placed on the lumbar spine decreases by limiting the amount of trunk flexion to
  35–40° [ ]."` — the bracket is a reference marker. The 35–40° also appears twice in the
  Introduction as what "is generally accepted" and "is recommended", i.e. as received practice.
- `"Nachemson [ ] reported increased pressure on the intervertebral disc at the level of L3 during
  the execution of full sit-ups."` — explicitly attributed.

Mandroukas's **own** result is EMG, and it does support the mechanism: *"Rectus abdominis muscle
activity was greatest in the early stages of trunk flexion and decreased as the range of motion
became greater, more than 35–40°."* That is a real finding about diminishing returns. It is not a
statement that exceeding 35–40° is a fault, and it is not a source for the parent spec's 50–60°.

**(5) SOURCE-MEASURED NULL ON THE PROPOSED PROXY — the source measured the exact observable the
spec proposes, and found nothing.** Barbado PMC4519219's design is four metronome cadences (1 rep
per 4 s / 2 s / 1.5 s / 1 s) and its outcome measures are `COP_ML` (force-plate centre of pressure,
medial-lateral) and `SG_ML` (the trunk's own medial-lateral sway). The parent spec's
`excessive_speed` rule proposes as its secondary signal *"increased medial-lateral wobble of the
shoulder midpoint (x-variance about the sagittal path)"* — that is `SG_ML`. Barbado's main finding:

> "Our main finding was that linear variability of SG_ML did not change significantly as speed
> increased."
> "…they were able to constrain their upper trunk motion to the sagittal plane without significant
> changes between cadences."

The quantity that *did* increase, `COP_ML`, comes off a force plate and is not observable from
video under any camera placement. §5 turns this into the withdrawal.

This is worth naming separately from the other four because it is the only one where the source
**tested** the detector's proposed signal. Absence, inference and identity are all failures to check
what a source says; this is a failure that survives checking what the source says and only falls to
checking what the source *found*.

---

## 4. `situp_hip_flexor_dominance` is REGISTERED-BUT-PERMANENTLY-SILENT

Registered-but-silent (`pushup.rule_scapular_winging`, `band_pull_apart
.rule_loss_of_scapular_retraction`, `arm_abduction.rule_shoulder_shrug`, `arm_vw
.rule_shrug_substitution`) asserts: real, well-cited fault; the sensor cannot see it. All four
conditions hold here.

### 4.1 The fault is real, cited, and human-observable

Mandroukas states it directly and it is his own framing, not a borrowed number:

> "support on the feet activates the hip flexors and reduces the activity of the abdominal muscles"
> the curl-up should be performed "with flexed unsupported knees, without holding the knees or
> feet … (1) to avoid the uneven loading on the lumbar spine, and (2) to isolate the activity of
> the hip flexors"
> the movements "are performed by the hip flexors, particularly by the iliopsoas, rectus femoris,
> and sartorius … [which] increases lordosis in the lumbar spine"

And it is observable *by a human*: EgoExo-Fitness's TKV criterion "Lift your head, shoulders, upper
back, and then lower back off the ground in turn" is marked faulted on 4/82 sit-up actions, with 12
individual annotator `false` votes. Annotators watching video can see a rigid trunk. That is the
strongest possible statement that this is a sensing failure and not a fault-realism failure.

### 4.2 The sensing failure is STRUCTURAL, not marginal — the trunk is rigid by construction

MediaPipe Pose has **no landmark between the shoulders (11/12) and the hips (23/24)**. There is no
mid-thoracic point, no lumbar point, no sacrum. The trunk is represented as a single segment joining
two midpoints. "Does the spine curl segmentally or rotate as a rigid bar?" is a question about the
*interior* of that segment, and the interior does not exist in the landmark set.

This is the same class as scapular winging and scapular retraction — argued from the landmark set,
which is sufficient here in a way it was not for the shrug rules. The shrug rules had a *candidate*
metric (`ear_y − shoulder_y`) that had to be measured before it could be rejected; there is no
candidate metric for intra-trunk curvature at all.

### 4.3 The parent spec's heuristic compares one quantity against itself

Recorded because a future reader will otherwise try to implement it. The heuristic reads:

> Flag if the spine stays near-straight (**shoulder–hip–knee remain close to collinear**, trunk_curl
> change < ~10-15deg) while the **trunk-thigh (hip) angle** closes rapidly

The parent spec's own convention block defines *"Hip angle = angle at the hip landmark formed by
shoulder→hip→knee"*. Both clauses therefore name **the same angle**: the rule asks that
shoulder–hip–knee stay near 180° *while* shoulder–hip–knee closes rapidly. Under any single reading
of the sentence the two conjuncts are mutually exclusive, so the rule as written can never fire.
This is the vacuous-branch defect that killed `row.rule_momentum_jerk`'s second condition, Bicep
Curl's elbow-displacement disjunct and the impingement arc's first conjunct — found a fourth time,
and found *before* implementation rather than after.

### 4.4 The heel proxy measures the setup, not the fault

The spec offers a supporting proxy: *"feet/heels (29/30, 31/32) remain fixed (low displacement) and
knees do not lift, indicating anchored feet."* Anchored feet is the **condition under which**
Mandroukas expects hip-flexor dominance, not the dominance itself. A trained lifter with anchored
feet can still curl segmentally; an untrained one with free feet can still pull rigidly. Shipping
displacement-of-the-heel under a fault_id whose citation is about abdominal-versus-iliopsoas
recruitment would be shipping metric B under metric A's citation — the move this project forbids.

### 4.5 The KG is not the gap

`retrieve_graph_context("Abdominal Disengagement", movement="Sit-up")` resolves to
`Sit-up:Abdominal Disengagement` (a `Fault` node) with a non-empty `causes` bucket (`Weak Core
Stability`) and a non-empty `quality_impacts` bucket (`Core Stability`). The graph has a good home
for this rule. The metric is the gap.

---

## 5. `situp_excessive_speed` is WITHDRAWN — the source measured the proxy and found nothing

Withdrawn (OHP bar-path, deadlift bar-drift, curl wrist-flexion, arm-abduction impingement arc, Arm
VW's W-abduction floor) asserts: no citation supports the rule as written. Three independent
failures here, any one sufficient.

**(a) The observable proxy is a measured null in the cited source.** §3(5). Barbado's `SG_ML` *is*
the spec's "medial-lateral wobble of the shoulder midpoint", and Barbado's headline result is that
it does not change with speed. The signal that does change is force-plate `COP_ML`, which no camera
can see.

**(b) The `~1.0 s` threshold is a protocol value.** The parent spec's own parenthetical gives it
away — *"< ~1.0 s (roughly the fastest cadence tested)"*. C1 is the fastest of Barbado's four
metronome settings. Barbado never nominates it, or any cadence, as a fault threshold; the paper's
conclusion is that trunk sagittal control was *maintained* across all four. This is the
impingement-arc failure mode again: an accurate reading of the paper, an inference the paper does
not make.

**(c) The primary signal does not exist in this architecture.** *"peak |d(trunk_angle)/dt| exceeds a
**per-user baseline** by a large margin"*. There is no per-user baseline. No rule in this codebase
carries state across clips; `run_detector` sees one clip, and `merge_by_fault` aggregates within it.

**What Barbado does support, and why it still is not a rule.** The discussion does say *"fast
curl-up exercises should be used with caution in people with motor control deficits or low-back
disorders, as well as in novice, untrained or unfit individuals"* — but the clause that carries the
risk, *"due to the effect of performance speed on the spinal loads and intradiscal pressure ( )"*,
is again a reference marker to other work (§3(4)), and the recommendation is population-conditioned
in a way a per-clip rule cannot honour.

**An absolute-cadence rule is possible and is not built here.** EgoExo-Fitness's guidance prescribes
"about one repetition every 2 seconds" and its annotators fault deviation on 7/82 actions — a real
number attached to this movement. It is **dataset guidance text, not a peer-reviewed threshold**,
and it belongs to the full-sit-up variant (§2.2b). Adopting it would mean shipping a threshold whose
provenance is an annotation instruction, under Barbado's citation. Recorded for whoever finds a
source that states a cadence; not invented here.

**Withdrawn, not registered-silent, and the distinction is load-bearing.** Concentric duration is
perfectly measurable — `segment_reps` produces rep windows and phases carry frame times. Nothing is
wrong with the *sensor*. What is missing is a source that puts a number on it, and a proxy the
source did not already null out.

---

## 6. `situp_excessive_rom` is WITHDRAWN — and the decisive reason is the knowledge graph

This is the rule the parent spec argues hardest for, and it fails three ways.

**(a) No source states 50–60°, and the nearest number is secondary.** §3(4). Mandroukas's 35–40°
is received practice behind a reference marker; the Nachemson L3 disc-pressure line is explicitly
attributed to Nachemson. The parent spec's own threshold — *"peak trunk-flexion angle > ~50-60deg …
OR the hip angle closes below ~110deg"* — appears in neither source in any form.

**(b) The rule contradicts the only labeled data and the only canonical description of the
movement.** §2.2(b). EgoExo-Fitness's sit-up guidance instructs the performer to lift the lower back
off the ground and touch the feet, and its annotators **fault the failure to do so on 28/82
actions**. A rule flagging "past the curl-up range" would fire on the correctly-performed
repetitions of the only dataset in which this movement is labeled at all.

**(c) THE KG SEED WOULD BE SEMANTICALLY INVERTED, AND THAT HAS NO WORKAROUND.** The graph carries
exactly four `Sit-up:` fault nodes, and they are the EgoExo TKV criteria (the general-tier stubs
were EgoExo-TKV-grounded):

| node | buckets |
|---|---|
| `Sit-up:Incomplete Forward Reach` | `quality_impacts` → `Range Of Motion` |
| `Sit-up:Abdominal Disengagement` | `quality_impacts` → `Core Stability`; `causes` → `Weak Core Stability` |
| `Sit-up:Feet Not Together` | none (dangling) |
| `Sit-up:Arms Not Extended Overhead` | none (dangling) |

There is **no excessive-ROM node**, and the only ROM-adjacent one means the opposite: *incomplete*
reach. Seeding an "you went too far" card from a node meaning "you didn't go far enough" would put a
contradiction on the user's screen. The generic `Range Of Motion` QualityDimension is the fallback
four previous modules already rejected, for a reason that reproduces exactly here — its
`corrections` bucket is `Wrapping Surface Adjustment` and its `quality_impacts` are ten scapular and
arm-activation nodes, none of which mean anything for a curl-up.

Band Pull Apart, Bicep Curl, Arm Abduction and Arm VW all accepted **thin** seeds, and Arm VW
accepted a **shared** seed. None accepted an **inverted** one, and this spec does not either.

**One thing this withdrawal does NOT say.** Over-ranging a curl-up into a full sit-up is plausibly a
real fault for the curl-up variant, and Mandroukas's own EMG result (RA activity falls off beyond
35–40°) is a genuine argument that the extra range buys little. What is missing is a source stating
a threshold, a graph node that does not mean the opposite, and — before either — a decision about
**which sit-up the app ships** (§10).

---

## 7. `situp_incomplete_rom` SHIPS

### 7.1 It is the one rule whose citation is primary and whose claim is variant-independent

Barbado's definition of the exercise is his own, in Methods, with no reference marker:

> "Curl-ups consisted of a head, arms and upper trunk lift **to the point where the scapula was
> lifted from the force plate**, then returning to the starting position."

Mandroukas corroborates the endpoint from the other side — a lift "with a rounded back to
approximately 35–40° from the floor". And unlike every other rule in this set, the claim survives
the variant question: **a full sit-up also requires clearing the scapulae.** A repetition that never
lifts the shoulders off the mat fails the curl-up and fails the full sit-up alike. The
`situp_excessive_rom` withdrawal turns on the two variants disagreeing; this rule does not depend on
which one the app means.

Its KG seed is **aligned, not inverted**: `Sit-up:Incomplete Forward Reach` →
`quality_impacts: Range Of Motion`. Thin (one bucket beyond `related_actions`), and thin is
precedented — `FaultCard.tsx:55-57` pushes each rung only `if (…).length` and wraps the block in
`rungs.length > 0`, so a thin seed renders a thinner card, never an empty heading.

### 7.2 The metric is hip-angle EXCURSION, and re-anchoring is forced, not preferred

The parent spec's quantity is *"peak trunk-flexion angle … < ~20deg"* against the floor. §1.1 shows
the floor is not recoverable from the image. The shipped quantity is instead

```
hip_angle_deg = mean of angle(shoulder, hip, knee) over the two sides
```

and the rule reads its **excursion over the repetition** — `max − min` of `hip_angle_deg` across the
rep's valid frames. For a hook-lying sit-up the feet are planted and the thigh is approximately
stationary, so closure of the shoulder–hip–knee angle *is* rotation of the trunk, in the same unit
(degrees) as the parent spec's quantity.

**This is a change of reference frame, not a change of threshold, and the difference matters under
this project's no-tuning rule.** The number 20 is carried over unchanged. What changes is what it is
measured *against*: the body instead of an image axis that is provably rotated 90° on the only
footage available. A threshold re-expressed into different units was rejected in `arm_vw
.rule_lr_asymmetry` because changing units changes what fires; here the unit is identical and the
alternative is not "less accurate", it is "meaningless".

**Provenance of the 20, stated:** it is the parent spec author's. Barbado supplies the endpoint
("scapula lifted") in kind, not in degrees; Mandroukas supplies 35–40° as the *target*, secondarily
sourced. A floor at 20° is a defensible rendering of "did not get anywhere near the target", and no
source states it. Recorded, not moved.

### 7.3 No view gate and no view discount — a first, and deliberate

Every previous module either gated on a view set or scaled confidence by
`VIEW_UNAVAILABLE_CONFIDENCE_SCALE` outside it. This rule does neither, because §1.2 establishes
that for a horizontal subject the labels the estimator emits carry no validated meaning. A discount
keyed on a meaningless label is not conservative — it is arbitrary, and it would encode a false
claim that the label was informative. `observability` is emitted as the parent spec's own rating
(`high`) with the honest qualifier in the rule docstring rather than a fabricated scale factor.

The direction of the residual error is, however, stated: obliquity **foreshortens** an in-plane
angle, so a genuine full-range curl can read as a smaller excursion, pushing this rule toward
**firing**. Unlike `vw_loss_of_elevation`, whose estimator error ran toward silence, this rule's
geometric error runs toward false positives. That is the honest direction and it is why the
threshold sits low (20°) rather than near the 35–40° target.

---

## 8. Measured on six real supine clips, in all three exocentric views

18 pose files (6 judged sit-up actions × `exo_m`/`exo_l`/`exo_r`, 17,793 frames, MediaPipe
`model_complexity=2`), each run through `estimate_view_for_pose(allow_front=False)` and through the
real `run_detector(SITUP_DETECTOR, …)`. **These are full sit-ups (§2.2b); nothing below is a
threshold measurement.**

### 8.1 The view estimator is not silent on a supine subject — it is INVERTED

| camera | what it actually films | `view_type` returned | n |
|---|---|---|---|
| `exo_l` | near-sagittal (subject side-on, image rolled 90°) | **`rear`** | 6/6 |
| `exo_r` | near-sagittal from the other side | **`rear`** | 6/6 |
| `exo_m` | **head-on**, subject curling toward the lens | **`rear_oblique`** | 6/6 |

`side` and `unknown` were never emitted. The mapping is deterministic — every clip of a given
camera got the same label — so this is not noise; it is a systematic misreading. The estimator calls
the sagittal view `rear` and the frontal view `rear_oblique`, i.e. **the opposite of what those
labels mean for an upright subject.**

This is the measurement behind `view_estimation.py`'s docstring limit 1, which until now was
argued. It is also the concrete justification for §7.3: gating on `{"side"}` would silence the rule
on 18/18 real clips, and *discounting* outside `{"front", "rear"}` would apply full confidence to
the head-on camera and discount nothing on the sagittal ones — precisely backwards. Neither is
conservative. `torso_width_ratio_mean` separates the two groups cleanly (`exo_m` 0.166–0.211 vs
`exo_l`/`exo_r` 0.249–0.451), so a horizontal-subject view estimator is *buildable*; building it is
not this spec's scope (§10, out of scope).

### 8.2 The sensing works: the metric is computable and the reps segment

- **Validity**: 76.8–100.0% of frames pass the six-landmark gate (median 98.9%). The predicted
  failure — a sagittal view of a supine subject losing the far-side shoulder/hip/knee — did not
  dominate; the worst clip (`yT4RK3_action_12/exo_r`, 76.8%) still yields 3 analyzed reps.
- **Segmentation**: `fallback` is `None` on **18/18**. `segment_reps` found 3–12 reps per clip on
  `hip_angle_deg` and `select_reps` analyzed 2–3. The registry entry
  `("hip_angle_deg", "min", "extended")` is confirmed against real supine footage, not only
  synthetic fixtures.
- **Signal shape**: hip angle spans roughly 10–179°, median 104–165° per clip — supine open, curled
  closed, exactly the polarity the phase assignment assumes.

### 8.3 The rule is silent on 18/18, which is the correct behaviour

`situp_incomplete_rom` fired on **no clip and no rep**. Every clip here is a complete, deliberately
performed sit-up; the smallest per-rep hip-angle excursion anywhere in the set is **54.6°**, 2.7×
the 20° threshold. A rule that flagged these would be wrong.

That is a *specificity* observation on 18 clips, not a validation: nothing here contains a
known-truncated repetition, so the rule's sensitivity is untested by anything except the synthetic
boundary fixtures in `tests/test_situp.py`.

### 8.4 The two findings that change how the number should be read

**(a) Camera placement moves the measured excursion by more than the threshold.** The three cameras
film the *same repetitions simultaneously*, so any disagreement between them is pure measurement
error. Per-clip median excursion by camera:

| sample | `exo_l` | `exo_m` | `exo_r` | spread |
|---|---|---|---|---|
| `zOfbr6_action_1`  |  85.7 | 119.8 | 115.1 | 34.1 |
| `zOfbr6_action_5`  |  80.9 | 115.3 | 102.9 | 34.4 |
| `zOfbr6_action_12` | 124.6 | 112.4 | 127.5 | 15.1 |
| `z8RAua_action_7`  | 136.5 |  93.3 | 107.4 | 43.2 |
| `yT4RK3_action_6`  | 104.1 | 112.8 |  96.5 | 16.3 |
| `yT4RK3_action_12` |  85.0 |  94.5 |  72.1 | 22.4 |

**Median spread 28.2°, max 43.2° — larger than the 20° fire threshold itself.** And the sign is not
consistent: the head-on camera reads *higher* than the sagittal one on three clips and *lower* on
three. So the tempting claim that obliquity foreshortens the angle and biases the rule toward firing
is **not supported** — `angle_degrees` consumes `dims=3`, i.e. MediaPipe's estimated z as well as
x/y, so the error has no established direction. §7.3 and the rule docstring were corrected to say
so; the first draft asserted a direction, and the data refuted it.

This is what places the threshold at 20° rather than near the sources' 35–40° target: a cut inside a
distribution whose measurement spread is ~28° would fire on camera placement.

**(b) Re-anchoring to the body fixes the REPRESENTATION, not the ESTIMATOR.** `RollInvarianceTest`
proves `hip_angle_deg` is invariant under rotation of a *landmark set*. Whether the whole pipeline is
invariant is a different question, and it was measured: 300 real `zOfbr6/exo_l` frames were rotated
90° and re-run through MediaPipe.

| | as shipped | image rolled 90° |
|---|---|---|
| frames with a detection | 300/300 | 300/300 |
| hip angle range | 16.2–177.5° | 9.5–178.6° |

**Per-frame |difference|: median 9.8°, p90 18.6°, max 32.5°.** MediaPipe is not roll-equivariant.
Re-anchoring removed an error that would otherwise have been 90°; the residue is the estimator's,
and roughly **half the fire threshold**, produced by camera roll alone. No landmark convention can
remove it — it needs either a roll-normalising preprocessing step or an estimator trained
roll-invariantly, and neither is in scope here.

### 8.5 What this section does not license

No fire rate, no AUC, no threshold. The clips are a different variant (§2.2b), there are six of
them, and REHAB24-6-style correctness labels do not exist for any of them at the repetition level —
EgoExo's judgements are per *action* and per *criterion*, not per rep. What is settled is that the
pipeline runs end-to-end on real supine footage, that the view labels are unusable and why, and that
the measurement noise floor is large enough to matter to threshold placement.

---

## 9. Testing

Tests go in `tests/test_situp.py` as `unittest.TestCase` classes, matching
`tests/test_arm_vw.py`. The required set:

1. **Metric-key correspondence** — `SITUP_METRIC_KEYS` is a two-way match with what
   `situp_compute_raw` emits. A key the tuple omits is dropped by `run_detector` and read back as
   NaN by every rule.
2. **The silent rule is silent, non-vacuously** — `rule_hip_flexor_dominance` returns `[]` on a
   fixture where a *live* rule of this module fires, so the test cannot pass merely because the
   fixture is degenerate. This is the correction the Bicep Curl pass had to make after shipping a
   vacuous "asserts silence" test.
3. **The shipped rule fires and is silent in the right places** — a synthetic rep with 8° of hip
   excursion fires; one with 45° does not.
4. **Rotation invariance, pinned** — the same synthetic rep rotated by 90° in the image plane
   produces byte-identical detections. This is the §1.1 claim as an executable assertion, and it is
   the test that would fail if anyone re-introduces an image-horizontal reference.
5. **Phase-fraction cliff, both sides** — `rule_incomplete_rom` is scoped to the whole rep rather
   than a phase, so the Bicep Curl trap (`phase_fraction · T ≥ min_frames / fps`) reduces to
   `T ≥ min_frames / fps`; pin the shortest window that scores and the one that does not.
6. **Registry** — `Sit-up` resolves via `get_detector`, appears in `list_detectors()` in
   registration order, and `validated is False`.
7. **Occlusion refusal** — dropping one required landmark marks the frame invalid and silences the
   rule for that frame, matching every module since Push-up.

Run: `.venv\Scripts\python.exe -m pytest tests/` and the backend coverage gate
`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.

---

## 10. Honesty constraints, and the open decision this spec does not make

- `validated=False`. Not because labeled data is absent (§2.2) but because the labeled data that
  exists describes a **different variant** and no check has been run against anything.
- Every number in the module names its provenance as either FROM THE SPEC or A RULE-LEVEL CHOICE,
  as in `arm_vw.py`. No number here comes from a source.
- Fire rates quoted anywhere must be measured on **segmented** windows through `run_detector`, not
  on annotation windows — the Arm VW pass measured those 3.7× apart.
- **One live rule is the honest outcome.** Two rules fail on citations that do not support them and
  one on a sensor that cannot see the fault. Adding a rule to make the detector look comparable to
  Squat's five would be inventing thresholds, which is the thing this project's rules forbid.

**THE OPEN DECISION, RECORDED AND NOT TAKEN: which sit-up does the app ship?** The parent spec says
curl-up. Everything else says full sit-up — and the count is four independent artefacts, checked by
reading rather than assumed:

- the knowledge graph's four `Sit-up:` fault nodes (§6c),
- EgoExo-Fitness's canonical guidance and its 82 judged actions (§2.2b),
- `frontend/src/lib/i18n.tsx:1292`, which renders `movement.Sit-up` in Traditional Chinese as
  **仰臥起坐** — the full sit-up. The curl-up is 捲腹, a different word,
- `frontend/src/components/movements/MovementArt.tsx` / `MovementIcon.tsx`, whose shipped artwork
  for this card is a full sit-up.

Two of this spec's three non-shipping outcomes turn on that disagreement. Resolving it is a product
decision plus a KG authoring step, not a detector change. Logged against TODO.md's existing "many
faults have no KG node" item.

**No frontend edit is needed and that was verified, not assumed.** `frontend/src/pages/Movements.tsx`
derives its analyzable set from `GET /api/movements` (itself derived from the detector registry) and
falls back to Squat-only on error; the Sit-up card, artwork, icon and both locales' strings already
exist. Registering the detector is the whole change.

### Out of scope

- Shoulder Bridge and Leg Abduction (the rest of Group E) — separate specs, separate branches.
  Leg Abduction has REHAB24-6 `Ex4` (210 reps, 120 correct / 90 incorrect) and will be the
  best-evidenced Group E detector by a wide margin.
- Any frontend edit. Registration alone surfaces the movement, Beta-tagged, via `GET /api/movements`.
- Authoring the missing KG nodes; the graphml is gitignored and that is a deploy step.
- Repairing `view_estimation.py` for horizontal subjects (limits 1 and 3 in its docstring). This
  spec routes around it and does not touch it — the squat oracle comparison in
  `tests/test_movement_registry.py` pins byte-for-byte output.
