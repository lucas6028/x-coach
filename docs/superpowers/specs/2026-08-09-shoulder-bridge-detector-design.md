# Shoulder Bridge detector — design

Twelfth of sixteen. Parent spec: `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`,
Group E. Module: `src/pose/movements/shoulder_bridge.py`. Tests: `tests/test_shoulder_bridge.py`.

**Outcome: one rule ships, one is registered permanently silent, two are withdrawn.**

| parent-spec rule | outcome | decided by |
|---|---|---|
| `bridge_incomplete_hip_extension` | **SHIPS** | endpoint stated in the own words of BOTH sources; exact three-bucket KG seed; the criterion human-judged on 77 matching-variant actions |
| `bridge_lumbar_hyperextension` | **REGISTERED, PERMANENTLY SILENT** | cited for this exercise, but its test cannot fire and the sign it needs is not recoverable — two constructions built, both measured to fail |
| `bridge_asymmetric_pelvic_drop` | **WITHDRAWN, absent** | citation describes gait; metric specified against the image horizontal; single-leg variant; zero KG nodes |
| `bridge_knee_valgus` | **WITHDRAWN, absent** | citation describes landing and patellofemoral pain; the ratio sits at or below its own cut on repetitions judged correct |

---

## 1. Purpose, and what makes Shoulder Bridge unlike the eleven before it

Two things, pointing in opposite directions.

**The good one.** This is the first Group E movement — and one of very few in the whole programme —
whose flagship rule measures a quantity that **both** cited sources define in **their own words,
with no reference marker**, and whose knowledge-graph seed is exact rather than thin, shared, or
inverted. It is also the first movement whose labeled ground truth describes the exercise the spec
models rather than a cousin of it. After Sit-up, which shipped one rule on a secondarily-sourced
threshold against a graph modelling a different exercise, that is a marked improvement in the
evidence available.

**The bad one.** The rule fires on five of the six real clip-views that exist, all of which are
repetitions human annotators marked **correct on this exact criterion**. §8 is the census. The
threshold is not moved, because every available repair is forbidden by this project's own rules.

### 1.1 The parent spec's Group E measurement convention still does not apply — but for once it does not need to

The Group E update block (parent spec, Group E, UPDATE 2026-08-09) records that every quantity
Group E defines "vs the floor/horizontal" is unrecoverable: EgoExo-Fitness ships its near-sagittal
exocentric views rotated a quarter turn with no EXIF orientation tag. Verified again here on a
different record — `z8RAua`'s `exo_l` frames have the mat running vertically down the image, and
`PIL.getexif()` is empty on all three exo views.

Sit-up had to **re-anchor** its metric to the body to escape this. Shoulder Bridge does not: the
flagship rule's quantity is already `angle(shoulder, hip, knee)`, a joint-relative angle that is
invariant under camera roll by construction. The convention problem lands instead on the three
rules that do *not* ship — `lumbar_hyperextension`'s "hip-midpoint y rises above the straight
line", `asymmetric_pelvic_drop`'s "pelvic-tilt vs horizontal", and `knee_valgus`'s
`|x(25)-x(26)|` — which is part of why none of them do.

### 1.2 The view ratings are fiction again, and this time the labels are not merely inverted but unstable

Sit-up measured the view estimator to be *inverted* on supine subjects, and deterministically so:
the two near-sagittal cameras returned `rear` on 6/6 each and the head-on camera `rear_oblique` on
6/6. Re-measured here on a different record and a different movement, it is worse — the same
camera disagrees with itself between two clips of the same person in the same room:

| clip-view | estimator | confidence |
|---|---|---|
| `z8RAua_action_11_exo_l` | `rear` | 0.38 |
| `z8RAua_action_4_exo_l` | `rear_oblique` | 0.02 |
| `z8RAua_action_11_exo_m` | `rear_oblique` | 0.55 |
| `z8RAua_action_4_exo_m` | `rear_oblique` | 0.17 |
| `z8RAua_action_11_exo_r` | `rear` | 0.72 |
| `z8RAua_action_4_exo_r` | `rear` | 0.71 |

`side` and `unknown` never appear. `src/pose/view_estimation.py`'s module docstring, limit 1,
already forbids gating a horizontal-movement rule on these labels; this is the second independent
measurement behind that prohibition. So `rule_incomplete_hip_extension` carries **neither a view
gate nor a view discount** — the second rule in the registry with neither, following
`situp.rule_incomplete_rom`.

The cost is concrete and is stated in §8: the label cannot separate the axial camera, on which the
rule is grossly wrong, from the near-sagittal cameras, on which it is marginal. Both get
`rear_oblique` somewhere in the table above.

---

## 2. The data situation: the labels exist, they match, and the pixels are missing

### 2.1 What exists, and it is better than any Group E movement has had

EgoExo-Fitness carries **77 human-judged `Shoulder Bridge` actions** across **130 annotator
records**. Its canonical action guidance is byte-identical across all 77 and names this detector's
endpoint verbatim:

> "…raise your spine from the tailbone section by section to roll off the mat until your **knees
> and hips are raised in a straight line with the shoulders**."

One of its twelve technical-keypoint criteria is, verbatim, the fault this detector implements:

> "Progressively raise your body until your knees, hips, and shoulders align in a straight line."

Faulted on **16 of 77 actions (21 of 130 votes)** — the second most-faulted criterion overall and
the most-faulted one that concerns the *movement* rather than hand placement or breathing. Full
census:

| faulted actions | false votes | criterion |
|---|---|---|
| 23/77 | 27 | Place hands naturally by your sides with palms facing down |
| **16/77** | **21** | **Progressively raise your body until your knees, hips, and shoulders align in a straight line** |
| 16/77 | 25 | Inhale again while keeping your torso lifted and still |
| 15/77 | 19 | Gently lower your spine back to the mat, segment by segment |
| 12/77 | 13 | Keep legs hip-width apart and feet relaxed |
| 9/77 | 10 | Inhale and keep your body stable |
| 8/77 | 12 | Pull your navel toward your spine, tilt the pelvis backward, lift your pubis |
| 7/77 | 8 | As you exhale, contract your abdominal muscles |
| 7/77 | 11 | Continue exhaling and lift your spine off the mat, starting from the tailbone |
| 6/77 | 9 | Exhale, relax your chest and ribs |
| 5/77 | 6 | Lie on your back with legs bent at about a 90-degree angle |
| 2/77 | 2 | Maintain a neutral spine position |

**No criterion in the list describes over-arching.** That fact does real work in §5.

**There is no variant mismatch.** Sit-up's decisive problem was that the parent spec modelled a
curl-up while the graph, the dataset, the zh-TW string and the card artwork all modelled a full
sit-up. Here the parent spec, Escamilla's Methods, Colonna's description, EgoExo's guidance and the
knowledge graph all describe the same two-leg supine floor bridge.

### 2.2 What is unreachable, and it is only the pixels

`frames_open` downloads in 3 GiB parts and part **`.ac` is missing** (`.aa`, `.ab`, `.ad` present),
so the concatenation cannot be decompressed. `.aa` is the prefix of a single gzip stream and decodes
standalone until it runs out. Exactly **two of the 77 judged actions** fall in a record it contains:

| sample_id | frames | duration | quality (1–5) | annotators | straight-line criterion |
|---|---|---|---|---|---|
| `z8RAua_action_4` | 4786–5371 | 19.5 s | 3.0 | 1 | **True** |
| `z8RAua_action_11` | 10201–10681 | 16.0 s | 3.0 | 2 | **True** (both) |

2 of 77 is **2.6%**. Both are marked correct on the criterion this detector implements, so a
correct detector is silent on both — which makes them a **specificity** check, not merely a sensing
check. That is more than the Sit-up clips could offer.

3,201 frames were recovered (1,067 per view × 3 exocentric views) and run through MediaPipe Pose
(`model_complexity=2`, `static_image_mode=False`), emitting the schema `src/pose/process_videos.py`
writes so the files feed `run_detector` unchanged. Detection rates: 477–586 of 481–586 per
clip-view, i.e. 94.5–100%.

### 2.3 A fourth reason for `validated=False`, and the first that a download fixes

- Deadlift, Row, Band Pull Apart, Bicep Curl → **no labeled data exists**.
- Arm Abduction, Arm VW → **nobody ran the check** against data that does.
- Sit-up → **the labeled data describes a different variant**.
- **Shoulder Bridge → the labels exist and match, and the pixels are missing.**

REHAB24-6 has no bridge (Ex1 arm abduction, Ex2 arm VW, Ex3 table push-ups, Ex4 leg abduction, Ex5
leg lunge, Ex6 squats) and Fit3D has no supine action among its 47 activity types, so there is no
substitute corpus. This is the most actionable finding in the task and it belongs in TODO.md as an
action, not a limitation: **a 77-action validation run against the exact criterion this rule
implements is one completed download away.**

---

## 3. `angle_degrees` is unsigned, and that breaks two of the four rules at once

`src/pose/geometry.py:73` returns `degrees(arccos(...))`. Range **[0, 180]**.

**Consequence 1 — the fifth vacuous-branch defect.** `lumbar_hyperextension`'s test is "flag if
peak hip angle overshoots the straight line into extension (**> ~190 deg**)". It can never fire.
Prior instances: `row.rule_momentum_jerk`'s second condition, Bicep Curl's elbow-displacement
disjunct, the arm-abduction impingement arc's first conjunct, `situp_hip_flexor_dominance`'s
self-comparison. This is the second caught *before* implementation rather than after.

**Consequence 2 — the shipped rule inherits the same defect in the opposite direction, and this
one the parent spec does not anticipate at all.** The function is exactly symmetric about 180°.
Measured on a synthetic fixture (shoulder at the origin, knee on the +x axis, hip displaced from
the midpoint by ±θ):

| hip offset from the straight line | `angle_degrees(shoulder, hip, knee)` |
|---|---|
| +30° (sagging toward the mat) | 120.00° |
| +20° | 140.00° |
| +10° | 160.00° |
| 0° (straight line) | 180.00° |
| −10° (arched above the line) | 160.00° |
| −20° | 140.00° |
| −30° | 120.00° |

So `incomplete_hip_extension`'s "peak < 160°" fires on a bridge arched 20° past neutral and reports
it as a bridge that was not lifted far enough — **the opposite cue**. Not a false alarm (both are
faults) but a **mislabel**. Pinned by `test_the_metric_cannot_distinguish_a_sag_from_an_arch`.

---

## 4. The sign is not recoverable — two constructions, both measured, both refuted

Both are body-relative and therefore roll-invariant *in principle*, which is exactly what the Group
E re-anchoring mandate asks for. The failure is the **estimator**, not the reference frame — the
same shape of result Sit-up reached when re-anchoring fixed its representation but left a median
9.8° residue from MediaPipe's lack of roll-equivariance.

### 4.1 Construction A — hip vs **ankle** about the shoulder→knee line

Compare `sign(cross(knee−shoulder, hip−shoulder))` against `sign(cross(knee−shoulder,
ankle−shoulder))`. Both cross products flip under rotation *and* under mirroring, so their
**product** is invariant under both. In a two-leg floor bridge the ankle is on the mat, so "hip on
the ankle's side" means the pelvis is sagging.

Verified on the synthetic fixture at 0°/17°/90°/180°/−90° × mirrored — byte-identical, and it
recovers a signed angle of 120/160/**180**/200/240 where the unsigned one gives 120/160/180/160/120.

**On real footage it collapses.** It reads "arched" on **57.0%** and **62.3%** of detected frames of
the two `exo_l` clips — repetitions every annotator marked correct, and on which the pelvis is below
the line for most of every repetition by construction.

### 4.2 Construction B — hip vs **knee** about the shoulder→ankle line

Take the mat to be the line joining the two contact points of a two-leg floor bridge (shoulders and
feet) and the knee to define "up" from it. **It disagrees with the subject's own other side on
nearly every frame**: the per-frame mean of the left and right signs is 0.0 — i.e. exactly opposite
— on 21 of 24 sampled frames of `action_4_exo_r`.

### 4.3 Why both fail, and why no third construction is attempted

Near the straight line, where this rule would have to decide, both cross products go to zero and
the sign becomes noise. That is not a tuning problem; it is the geometry of asking which side of a
line a point lies on when the point is *on* the line. Two independent constructions, two different
failure modes, one conclusion: **the sag/arch arc is not measurable from monocular MediaPipe
landmarks on this footage.**

---

## 5. `bridge_incomplete_hip_extension` SHIPS

### 5.1 The endpoint is primary in both sources, in their own words

**Escamilla PMC11048684**, Methods, no reference marker — and it states **both** ends of the
repetition, which no rule in this programme has previously had:

> start: "the supine hook-lying position with the hips flexed approximately **50°**"
> end: the subject "pushed through the feet and hands, lifting the buttocks upwards until the hips
> were in a **neutral position with 0° hip flexion, with the knees, hips, and shoulders
> approximately in a straight line**."

**Colonna PMC11981018**, describing the exercise in his own words, no reference marker:

> "the pelvis is lifted from the floor until it reaches the **neutral angular position of the
> hip**."

So the source states a repetition running from roughly **130° of hip angle to 180°** — an expected
excursion near 50°. The observed central tendency lands on it: median hip angle **128–134°** on the
two `exo_r` views against Escamilla's ~130° start.

### 5.2 What is *not* primary, said plainly because the parent spec marks it "VERIFIED"

The **rationale** sentences the parent spec quotes are Colonna citing other authors:

- "The greatest hip extension torque during the SBE occurs when the hip is nearly fully extended **[ ]**"
- "In this position, the GM is recruited more than at any other angle within the range of motion **[ , , ]**"

Both are in the document verbatim, so the parent spec's verification claim is true as far as it
goes. But this is the **secondary-sourcing failure mode** Sit-up named, recurring on a different
movement. Here it lands on the rule's *why* rather than on its *what*, which is why the rule ships:
the endpoint that defines the threshold is primary twice over.

### 5.3 The 160 is still the parent spec author's number

Neither source states a *failure* threshold. What they state is the **target** (0° hip flexion,
straight line). 160° renders "20° of hip flexion still remaining", a defensible reading of the
spec's own "the hips stay flexed and sagging". **It is not moved.**

The severity ramp 160° → 130° is a rule-level choice, and its severe end is Escamilla's stated
*starting* position: a repetition whose peak never left the start has achieved nothing. This is the
first severity ramp in the registry whose severe end is a source-stated quantity rather than a round
number. It remains a display/ranking curve, not a cited threshold.

### 5.4 The conflation is shipped, stated, and answered empirically

§3 shows the rule cannot tell a 20° sag from a 20° arch. What makes shipping it defensible rather
than negligent is §2.1: in the only labeled data for this movement, the direction this rule assumes
is the direction annotators fault — **16 of 77 actions** — and **the other direction is not among
the twelve criteria at all.**

### 5.5 Scope is the `top` phase, not the whole rep

This follows the parent spec ("peak taken over the top-hold frames") rather than the module next
door. `situp.rule_incomplete_rom` reads the whole rep because an **excursion** is a property of a
rep; a **peak position** is a property of the moment the position is held, and scoping it to `top`
stops a transient overshoot during the concentric from standing in for a hold that never got there.

The cost is the Bicep Curl phase-fraction interaction, which binds here (it did not for Sit-up,
whose shipped rule is not phase-scoped). **It was measured through `run_detector`, not derived —
and deriving it gets both the number and the mechanism wrong.** That is the Arm VW lesson restated:
a phase-scoped fire condition must be measured on SEGMENTED windows, never computed from a clip
length.

The tempting derivation is `min_frames = max(3, ceil(0.20 · fps)) = 6` at 30 fps, `top` covers 30%
of a rep, so `0.30 · T · fps ≥ 6` ⟹ `T ≥ 0.67 s`. Measured:

| clip | reps | rep **window** | `top` frames in window | fires |
|---|---|---|---|---|
| 0.83 s | 1 | 0.57 s (17 fr) | 5 | **no** |
| 0.93 s | 1 | 0.60 s (18 fr) | 6 | yes |
| 1.00 s | 1 | 0.67 s (20 fr) | 6 | yes |
| 1.33 s | 1 | 0.87 s (26 fr) | 8 | yes |
| ≤ 0.67 s | **0** | — | — | whole-clip fallback |

Two corrections fall out. **The boundary is ~0.60 s, not 0.67 s**, because `top` is a *percentile*
cut rather than a fixed 30% and lands on ~33% of a short window. And **the quantity that binds is
the TRIMMED REP WINDOW, not the clip**: `segment_reps` climbs to the plateaus and returns a window
shorter than the clip that produced it, so a 0.83 s clip yields a 0.57 s window this rule cannot
score. Below roughly 0.7 s of clip there is no repetition at all (`no_reps_detected`), so those
take the whole-clip fallback and are scored by a different path entirely.

So the gap is not "the shortest repetition segmentation will emit" — it is a **band of rep windows
between `min_rep_seconds` and ~0.60 s**. Not closed by raising `min_rep_seconds`: that would tune a
framework constant to flatter one rule, and a bridge held for under 0.6 s is not a bridge. No
analyzed repetition on the real footage fell inside the band (window durations 0.40–17.2 s; the one
0.40 s window is correctly unscored).

### 5.6 It does not fail open on a motionless clip — a first for a "not enough" rule here

`situp.rule_incomplete_rom`, `arm_vw.rule_incomplete_excursion` and every other whole-rep excursion
rule fire at severity 1.0 on a subject holding still, because `segment_reps` thresholds on
**percentiles** of the signal and is therefore scale-free: jitter segments into repetitions and a
tiny excursion reads as a tiny range.

This rule reads an **absolute position**, not a range. A motionless subject is judged on where they
actually are: someone lying flat (~130°) is correctly told they never bridged; someone holding a
good bridge (~175°) is correctly left alone. Pinned in both directions by
`test_a_motionless_clip_is_judged_on_position_not_range`. The framework trap is not absent — this
rule simply does not read the quantity that springs it.

---

## 6. `bridge_lumbar_hyperextension` is REGISTERED PERMANENTLY SILENT

Silent, not absent, and the distinction is load-bearing: a silent stub asserts "real fault, the
sensor cannot see it", an absent rule asserts "no citation supports this as written".

### 6.1 The fault is cited for **this exercise**

Colonna PMC11981018:

> "In patients performing **bridging exercises**, excessive and uncontrolled lumbar lordosis and
> anterior pelvic tilt (APT) are frequently observed due to the dominant hyperactivity of the ES"

> "Others recommend maintaining a straight alignment of the shoulders, hips, and thighs during
> bridging to prevent excessive APT caused by dominant ES activity."

Both carry reference markers, so the support is **secondary** — but it is secondary about the *right
exercise*, which is precisely the distinction the two withdrawals in §7 fail.

### 6.2 Three failures, and only the third is decisive

1. **The test cannot fire** (§3). Repairable in principle by signing the angle.
2. **The sign is not recoverable** (§4), measured twice. This is what silences it.
3. **Even with a perfect sign, the threshold would flag a normal position.** Colonna: "The range of
   motion during maximum physiological hip extension is approximately **20° beyond the neutral
   position**." So 190° — 10° beyond neutral — sits squarely *inside* normal hip extension. Same
   class of objection that helped withdraw `situp_excessive_rom`: the source that grounds the fault
   contradicts the number chosen to detect it.

### 6.3 The pelvis-height proxy is not substituted and its metric is not emitted

The spec offers "hip-midpoint y at top rises above the straight line interpolated between
shoulder-midpoint and knee-midpoint". Image `y` is exactly what the Group E update block names as
unrecoverable. Re-expressing it as a perpendicular distance would need a threshold no source
states — and Colonna makes such a threshold **non-transferable in principle**: hip torque and pelvis
height during the bridge depend on the foot-to-pelvis distance and the knee flexion angle, both of
which the performer chooses. One normalized lift height means different things at different setups.

### 6.4 The upgrade path

A working arch rule needs either a lumbar landmark (MediaPipe has none between shoulders and hips)
or a depth-bearing 3-D estimator that can place the pelvis off the shoulder–knee line with a
reliable sign. This project has measured such estimators elsewhere (the NLF and Multi-HMR notes).
Inventing a sign here is the fabrication this project's rules forbid.

---

## 7. Two rules are WITHDRAWN, both on exercise identity

### 7.1 `bridge_asymmetric_pelvic_drop` — four independent failures

1. **The citation is about gait.** The parent spec's `citation_support` quotes Colonna's
   Trendelenburg passage, which announces its own subject in its first words: "**In a Trendelenburg
   gait**, the Gmed is unable to maintain the pelvis on the opposite side during single-leg support…"
   It sits in Colonna's section on Gmed weakness and its consequences for walking, running and
   landing. The other bridge source, Escamilla PMC11048684, studies **unipedal bridging directly**
   and never mentions pelvic drop, pelvic level, or Trendelenburg at all — checked, not assumed.
   This is the **exercise-identity** failure mode that withdrew the arm-abduction impingement arc
   and put all four Arm VW sources on notice.
2. **The metric is specified against the image horizontal** — "angle of the left-hip(23)→right-hip(24)
   line vs horizontal" — which the Group E update block says is not recoverable.
3. **It is a fault of a variant nothing here performs.** The spec scopes it "esp. single-leg bridge".
   The app models the two-leg bridge and EgoExo's guidance is two-leg throughout ("keep legs
   hip-width apart"); its twelve criteria contain nothing about a level pelvis.
4. **The knowledge graph has no home for it.** `retrieve_graph_context("Pelvic Drop", movement=
   "Shoulder Bridge")` matches **zero nodes** — not thin, not inverted, none.

A body-relative re-anchoring is possible (pelvic tilt against the *shoulder line* rather than the
image horizontal) and is **not built here**: it repairs failure 2 and touches none of 1, 3 or 4, so
it would ship a metric with no source, no variant and no graph node behind it.

### 7.2 `bridge_knee_valgus` — three independent failures

1. **The citation is about landing and patellofemoral pain.** "**Powers [ ] theorized** that hip
   abductor and external rotator weakness may lead to excessive hip adduction and internal rotation,
   resulting in increased knee valgus" is explicitly attributed and sits in Colonna's passage on hip
   dysfunction and knee pathology, whose surrounding sentences concern ACL injury during **landing**
   and lateral patellar tracking. Nothing in either bridge source reports knee valgus during a
   bridge.
2. **Measured on correct repetitions, the ratio is already at or below its own cut.** Across the six
   clip-views the median `knee_width/ankle_width` is **0.726, 0.895, 0.911, 0.927, 1.020, 1.027**,
   and the per-clip minimum reaches **0.043**. The spec's fire threshold is **0.85** (squat's shipped
   one is 0.82). Two of six clip-views sit below the cut on their **median** frame — on repetitions
   every annotator judged correct on every alignment criterion. The cut is inside the noise, not
   above it. Viewed near-sagittally a supine subject's two knees and two ankles project onto nearly
   the same points, so the ratio is a quotient of two small, noisy numbers.
3. **The spec's form is not even roll-invariant.** It specifies `knee_width = |x(25)−x(26)|`, an
   image-x projection a rolled camera collapses. The codebase's own shipped precedent
   (`pose_feature_extraction.py:296`, feeding `squat.rule_knees_inward`) uses the full 2-D distance,
   which is invariant. Failure 2 was measured with the **invariant** form, so fixing this does not
   rescue the rule.

Not said by either withdrawal: that these faults are fine. What is missing is a source observing
them **in this exercise**, and — for valgus — a metric whose noise floor is below its own threshold.

---

## 8. Measured on the six real clip-views, through the real `run_detector`

### 8.1 The fire census, stated first

| clip-view | estimator view | valid frames | reps found | analyzed | peak hip angle (°) | fires |
|---|---|---|---|---|---|---|
| `action_11_exo_l` | `rear` (0.38) | 91.7% | 5 | 3 | 160.3, 163.8, 161.5 | **SILENT** |
| `action_4_exo_r` | `rear` (0.71) | 100.0% | 2 | 1 | 159.5 | 0.02 |
| `action_11_exo_r` | `rear` (0.72) | 100.0% | 2 | 0 → fallback | (whole clip) | 0.08 |
| `action_4_exo_l` | `rear_oblique` (0.02) | 89.1% | 5 | 3 | 167.9, 149.8, 155.5 | 0.15 |
| `action_11_exo_m` | `rear_oblique` (0.55) | 99.2% | 10 | 3 | 131.4, 139.3, 143.9 | 0.95 |
| `action_4_exo_m` | `rear_oblique` (0.17) | 78.2% | 11 | 3 | 110.6, 112.3, 124.9 | **1.00** |

**Five of six fire, and all six are repetitions annotators marked correct on this criterion.**

### 8.2 The census splits by camera geometry, not by repetition

On the four **near-sagittal** clip-views the rule is silent once and otherwise fires at 0.02, 0.08
and 0.15 — a low-severity card saying a repetition landed a few degrees short of the geometric
target. Both actions were scored 3/5 by their annotators with the comments "the movement was not
completed according to the instructional text" and "the movement was not performed according to the
instructional text". **A binary technical-keypoint `True` is not ground truth for a continuous
quantity**, and these three fires are not obviously wrong.

On the two **axial** `exo_m` clip-views it fires at 0.95 and 1.00, and those are simply wrong.
Viewed down the body's long axis the sagittal hip angle is foreshortened into meaninglessness:
median hip angle **90°** on `exo_m` against **128–134°** on `exo_r`, for the same repetitions. The
rule has no way to tell those cameras apart, because the view estimator labels the axial views
`rear_oblique` and one near-sagittal view `rear_oblique` too (§1.2).

### 8.3 What this establishes, and what it does not

**Establishes — the magnitude of the measurement error.** The same repetitions read **110.6° to
167.9°** depending only on which of three *simultaneous* cameras is used: a spread of ~50–58°
against the **20° margin** between this threshold and the straight line it renders.

**Does not establish a fire rate.** n = 2 actions, 1 subject.

**Does not establish a bias direction.** The tempting claim — "MediaPipe systematically under-reads
a straight-line bridge by about 20°, which is exactly the margin this threshold relies on" — is
consistent with the data and is **not made here**. One subject whose repetitions the annotators say
were not completed as instructed cannot support it. `situp.rule_incomplete_rom` had a first draft
that claimed a sign for its residual error and the measurement refuted it; the lesson is applied in
advance this time.

### 8.4 Why the threshold is not moved

Every available repair is forbidden:

- **Moving 160** is tuning a cited endpoint to flatter the estimator.
- **Gating on "is this view sagittal enough"** needs a threshold no source states.
- **Gating on the view label** is what §1.2 measures to be unstable.

The rule ships live with the census pinned by
`test_the_axial_view_fires_this_rule_at_near_full_severity_on_a_correct_rep`, so the next reader
meets this instead of rediscovering it. The 77-action validation that would settle it is one
completed download away (§2.3). Precedent: `situp.rule_incomplete_rom` ships live with a known
false-positive mode pinned rather than repaired, for the same reason.

### 8.5 The `only_partial_reps` fallback bites on the best footage — recorded, not fixed

On `action_11_exo_r` — 100% landmark detection, the cleanest footage available — `segment_reps`
found 2 repetitions and marked **both partial**, so `run_detector` took the `only_partial_reps`
fallback and handed the rule the entire 16-second clip as one window. The rule then scored a whole
clip as though it were one repetition.

Same gap the Deadlift setup-baseline defect recorded: `RunResult.fallback` is not threaded into
`RuleContext`, so a rule cannot decline a window handed to it by the whole-clip path. Worth knowing
that it bites hardest on the best footage.

### 8.6 The sensing works

Landmark detection 94.5–100% per clip-view; the six-landmark validity gate passes **78.2%, 89.1%,
91.7%, 99.2%, 100.0%, 100.0%**. `segment_reps` returns repetitions on 6/6 (2–11 found, 0–3
analyzed). The left and right hip angles disagree by a median **5.9–11.1°** (p90 27.6–32.2°), which
is why the metric takes their mean rather than either side.

---

## 9. Testing

`tests/test_shoulder_bridge.py`, `unittest.TestCase` classes per the project convention.

1. **Metric keys round-trip.** `test_metric_keys_match_the_emitted_metrics_exactly` — the tuple and
   what `compute_raw` emits are a two-way match; a key the tuple omits is read back as NaN by every
   rule.
2. **Validity gate.** One dropped landmark of the six silences every rule for that frame; the frame
   carries no metric keys.
3. **Roll invariance.** Byte-identical detections at 0°/17°/90°/180°/−90°, mirroring the Sit-up
   test, so the joint-relative choice cannot be silently reverted.
4. **The conflation.** `test_the_metric_cannot_distinguish_a_sag_from_an_arch` — ±θ from the
   straight line produce the identical reading. Pins §3.
5. **Phase polarity.** `top` is the *most*-extended 30%, the inverse of Sit-up on the same signal.
6. **The phase-scope floor.** A repetition shorter than 0.67 s segments but is not scored.
7. **Motionless clip.** Judged on position, not range: flat → fires, good hold → silent. Pins §5.6.
8. **The axial false positive.** `test_the_axial_view_fires_this_rule_at_near_full_severity_on_a_correct_rep`.
   Pins §8.2.
9. **The silent rule is silent.** `rule_lumbar_hyperextension` returns `[]` on every input,
   including one constructed to be grossly arched.
10. **Payload is NaN-free** end to end, and the KG seed resolves through the production
    `retrieve_graph_context` path.

---

## 10. Honesty constraints

- No threshold in this module was chosen by fitting. The one fire threshold is the parent spec's;
  the one severity ramp is a rule-level display curve whose severe end is a source-stated quantity.
- No rule ships whose citation describes a different exercise. Two were withdrawn on exactly that.
- No proxy is substituted under a fault_id whose citation does not support it — the pelvis-height
  proxy and the pelvic-tilt re-anchoring are both described and both left unbuilt.
- The shipped rule's known defects (the sag/arch conflation, the axial-view false positive, the
  `only_partial_reps` fallback) are stated at the definition site, pinned by tests, and repeated
  here. None is repaired by inventing a number.

### Out of scope

Frontend changes beyond registration; the `RunResult.fallback` → `RuleContext` threading; a 3-D
estimator for the arch rule; downloading `frames_open.tar.gz.ac`.
