# Camera placement — is oblique the best single-camera position?

Tests two hypotheses against the project's existing results plus one new measurement:

* **H1** — for the project's 16 movements, a single camera at the **oblique-front or
  oblique-rear** is the most accurate placement and catches the most faults.
* **H2** — at those oblique placements, **image→3D** beats image→2D, image→2D→lifted-3D,
  and image→pseudo-3D.

**Verdict: H1 is not supported as stated and is contradicted for sagittal cues; it
survives only in a weakened, weighting-dependent form. H2 holds, but only for
sagittal-plane cues under a fixed threshold — the two scope conditions are load-bearing,
not caveats.**

---

## 1. Why H1 could not be answered from what was already on disk

Fit3D's rig is **all-oblique by construction** — 4 cameras at ±45° corners, no sagittal
view ([[fit3d-dataset-facts]]). So experiments 1–4 measured *how bad oblique is*, never
*oblique vs the alternatives*. REHAB24-6 Ex5 has both a front-ish (cam17) and a side-ish
(cam18) stream, but its `front`/`half-profile` strata are **different repetitions** (the
subject reorients mid-recording, `notes/lunge-view-reconnaissance.md`), so its per-view
AUC split is a between-rep comparison wearing a between-view label.

## 2. New measurement — virtual-camera azimuth sweep on mocap GT

Project the Fit3D mocap 3D GT through a **synthetic pinhole camera** swept 0°→180° in 15°
steps and read the same cues with the same formulas (`src/fit3d/biomech`-style helpers),
against the view-invariant 3D truth. Azimuth **0 = camera in front of the athlete,
90 = pure sagittal, 180 = directly behind**; the rig's real placements land at ~20–25°
and ~153–157° for standing movements. Metric is the **debiased swept-threshold verdict
flip** used in `notes/fit3d_decision_fidelity_summary.md`, so cues in degrees and cues in
ratios are comparable.

Scripts are throwaway (session scratchpad, not committed): `azimuth_sweep2.py`,
`summarise_azimuth.py`, `sensitivity.py`.

**Coverage: 13 Fit3D actions / ~10 of the 16 canonical movements**, 40–45 reps × 8
subjects each — squat, lunge, push-up (diamond + wide), overhead press, row (one-arm +
barbell), deadlift, bicep curl, band pull apart, lateral raise, torso twist, thruster.
Not covered (absent from Fit3D): arm abduction, leg abduction, sit-up, shoulder bridge,
jumping jacks, high knee, knee raise.

### Checks run before trusting any number

| Check | Result |
|---|---|
| Look-at construction vs the rig's real `R` | **4.0–4.3°** mean angular difference |
| Facing from feet vs from the hip line | agree to **2.3–7.7°** |
| Within-rep facing stability (median sd) | **0.1–0.5°** standing, 4.5° lunge, 8.5° torso twist |
| Virtual sweep vs exp 2's *real* per-camera knee MAE | real 153°→30.9, 157°→34.9, 25°→45.6, 20°→58.1; virtual 150°→30.1, 165°→46.8, 30°→41.4, 15°→63.3 |

One convention bug worth recording: the first pass computed facing from the hip
cross-product and came out **175° off**. Fit3D index 1 is anatomically **left**, not
right (already documented in `depth_eval.resolve_lr`), so the hip-derived "forward" was
backwards. Two independent estimators disagreeing by 175° is what caught it. Push-ups
need their own definition again — prone athletes have no meaningful toe heading, so
facing is taken along the body's long axis (ankle→shoulder), which drops the feet-vs-hips
disagreement from 52° to 4.9°.

## 3. H1 — the result

**Per cue, oblique is almost never optimal.** Over 104 movement × cue curves the argmin
sits near-sagittal (75–105°) in **54**, at the frontal/rear extremes (≤15° or ≥165°) in
**31**, and oblique (30–60° / 120–150°) in only **19**.

The split is by which plane the cue lives in:

| cue family | n | median best azimuth | IQR |
|---|---|---|---|
| sagittal (knee, hip, elbow, torso lean, depth ratio) | 65 | **90°** | [75, 105] |
| frontal-plane (knee width, elbow width) | 26 | bimodal at the extremes | [15, 176] |

**Where the oblique does win is as a compromise, and only there.** Recomputing each
movement's single best placement under four cue weightings:

| weighting | oblique wins | sagittal wins | frontal/rear wins |
|---|---|---|---|
| sagittal cues only | **0 / 13** | **13 / 13** | 0 |
| frontal cues only | 3 / 13 | 0 | **10 / 13** |
| both planes equal (50/50) | **11 / 13** | 1 | 1 |
| all 8 cues equal | 9 / 13 | 4 | 0 |

So H1's conclusion appears **only** when one camera is forced to serve two cue families
with opposite optima. It is not that the oblique reads anything best; it is that it is
the least bad average of a view that reads depth well and a view that reads
mediolateral separation well. The squat makes the shape explicit:

| azimuth | sagittal cues | frontal cues | balanced |
|---|---|---|---|
| 0 (frontal) | 0.239 | **0.053** | 0.146 |
| 30 | 0.094 | 0.108 | **0.101** |
| 90 (sagittal) | **0.078** | 0.525 | 0.302 |
| 150 | 0.113 | 0.115 | 0.114 |
| 180 (rear) | 0.234 | **0.032** | 0.133 |

**The 90° frontal-cue numbers are degenerate, and stating that strengthens the row rather
than weakening it.** `_horizontal_span` in image mode uses image-x only, so at a sagittal
view the mediolateral axis lies along the optical axis and both width ratios collapse
toward 0/0 (elbow-width MAE reaches 225.8 on squat, 260.8 on lunge). The correct reading
of that column is not "0.525" but **"mediolateral separation is unreadable at 90°, flip
→ chance."** It is not what manufactures the sagittal penalty: squat `knee_width` alone
flips **0.581** at 90°, *higher* than the two-cue mean, so dropping the degenerate
elbow-width cue makes the sagittal view look worse, not better.

Two things follow. First, the balanced optimum is **shallow** — 0.101 at 30° against
0.107 at 15°, 0.114 at 150° and 0.133 at 180°; the argmin wanders between 15° and 60°
across movements, so "oblique is the best placement" overstates a flat region that spans
most of the non-sagittal range. What the curve says sharply is the *negative*: 90° costs
0.302 on a balanced criterion. Second, the compromise is **not free** — against routing
each cue family to its own best azimuth, the best single camera gives up **+0.03 to
+0.10** verdict flip (squat +0.048, band pull apart +0.100, torso twist +0.014).

**Scope note on the two per-movement tables** (single best placement, and the routing
penalty): both average a generic cue set over each movement, including cues that are not
faults for that movement — `knee_angle` and `depth_ratio` for a bicep curl,
`knee_width_ratio` for a band pull apart. A fault-relevance weighting would move those
numbers. The headline does **not** depend on it: the sagittal-only 13/13 and frontal-only
10/13 rows are per-plane statements computed within a single cue family, and they carry
the argument on their own.

**Front-oblique vs rear-oblique is movement-dependent**, so H1's "or" hides a real split
(sagittal cues, 45° vs 135°): rear better for squat (0.083 vs 0.102), lunge, row, barbell
row, deadlift, lateral raise, thruster — **7**; front better for both push-ups, overhead
press, bicep curl, band pull apart, torso twist — **6**. The pattern is that the camera
should sit on the side the working segment opens toward.

### What this measurement cannot see — and the sign of that term is not established

Projection only — **no detector error and no occlusion**. Every azimuth gets an equally
optimistic reading, which flatters whichever view suffers most from self-occlusion, and
side-on is usually assumed to be that view. The obvious move is to charge the sagittal
view an occlusion penalty and conclude the true optimum shifts toward the oblique. The
project's own lunge measurement does not support that move cleanly, and it is worth
stating why, because the naive reading of it is wrong
(`notes/lunge-rule-validation.md` §2):

* On **window-average** validity the sagittal camera is much worse — **58.4%** valid
  frames against the front camera's **74.0%**, with one-sided losses (`R_knee` 33.4%,
  `R_ankle` 19.8%) and far worse rep segmentation (91/174 reps against 152/174).
* On the **frame that decides the verdict** the gap reverses to a near-tie: cam18 is
  *marginally better* (0.644 vs 0.632), and the true deepest frame fails the landmark
  gate on 36% of reps on **both** cameras.
* The **front** camera carries the worse verdict-relevant pathology: its gate rejects
  frames monotonically with depth (pass rate 0.572 at 0–60° of knee flexion rising to
  0.941 at 120–140°), so it judges a median **34° short of the bottom** against cam18's
  ~8° — a bias in the direction that makes `lunge_insufficient_depth` false-positive.
  (cam18 has its own artefact: 31% of its rejected "deeper" frames read below 40°, which
  a lunge cannot reach — hallucinated landmarks on the occluded leg.)

So the occlusion term is real but its **sign at the judgment frame is not established**:
the sagittal view loses more frames overall, the front-oblique view loses the *right*
frames. This remains the one missing term that could rescue H1 — but it is an open
measurement, not an argument that currently favours either side.

### H1 verdict

Not supported as stated. Three separate reasons:

1. For the cue family the project's flagship faults actually live in (depth / flexion),
   **sagittal wins 13/13 movements and oblique wins 0/13**. Exp 2's flagship result —
   +41° knee bias, noise/signal 1.21, 82% false-fail — is a measurement of how badly the
   *oblique* reads squat depth, which reads against H1 rather than for it.
2. The oblique's win is an artefact of a **weighting choice** (both planes served by one
   camera) and disappears under either single-plane weighting. Since the codebase already
   gates faults on view and `notes/fit3d_decision_fidelity_summary.md` already recommends
   routing by fault type, the weighting that produces H1 is the one the system does not
   use.
3. "For the 16 movements" is unevidenced for ~6 of them (no mocap GT anywhere), and the
   one upper-body correctness signal that exists points the other way — REHAB24 Ex1 arm
   abduction is where monocular falls furthest behind Vicon (−0.14 MediaPipe / −0.27
   RTMPose).

The defensible version: **an oblique camera is a reasonable default when a single camera
must serve both sagittal and frontal faults, and the penalty for that choice is +0.03 to
+0.10 verdict flip against routing.**

## 4. H2 — the result

Holds, inside two scope conditions that are both already measured.

**By cue family** — direct image→3D wins on sagittal cues and *loses* on frontal ones
(`notes/fit3d_decision_fidelity_summary.md`, debiased swept flip, 2D → NLF):

| cue | squat | deadlift | thruster |
|---|---|---|---|
| knee (depth) | 21 → **11** | 17 → **10** | 22 → **9** |
| hip flexion | 12 → **6** | 18 → 13 | 14 → **4** |
| torso lean | 11 / 19 | 9 / 10 | 17 / 14 (tie) |
| knee width (valgus) | 15 / **31** | 9 / **17** | 9 / **20** (2D better) |

#### How much does the valgus row actually support? (audited 2026-08-01)

That row is quoted often enough — including in the H2 verdict below — to be worth
separating into the part that replicates and the part that rests on two people.

**Strong: no 3D model beats 2D on valgus, in any cell.** Pulling `knee_width_ratio`
debiased flip out of the model-comparison JSONs, against the projected-2D baseline:

| action | 2D | NLF | HMR2 | MultiHMR | MeTRAbs | MediaPipe |
|---|---|---|---|---|---|---|
| squat | **0.147** | 0.310 | 0.311 | 0.309 | 0.272 | 0.276 |
| deadlift | **0.089** | 0.174 | 0.182 | 0.251 | 0.218 | 0.305 |
| thruster | **0.087** | 0.200 | 0.220 | 0.303 | 0.257 | 0.369 |

**15/15 model × action cells are worse than 2D**, across five architectures including a
sparse metric regressor (MeTRAbs) and a pseudo-3D one (MediaPipe). This is not an NLF
quirk, and it is mechanistically consistent with `notes/fit3d_2d_vs_3d_summary.md`:
valgus is *detector*-limited, not projection-limited (perfect mocap-2D 0.04 vs real 2D
0.07), so 3D has nothing to fix there and contributes its own lateral-localisation noise.

**Weak, and it bounds how the row may be stated:**

1. **98% of the pooled gap comes from 2 of 8 subjects.** Summing (3D − 2D) per-subject
   flip over the three actions: s05 **+2.26**, s03 **+1.50**, then +0.40, +0.21, 0, 0,
   −0.15, −0.40 — total +3.82, of which s05 + s03 are +3.76. Per subject × action cell
   the split is **2D better 10, 3D better 4, tie 10** (n=24). So the honest reading is
   "3D fails catastrophically on two people and is a wash on the rest", not "3D is worse
   for everyone".
2. **The valgus threshold is not a coaching cut.** `threshold_is_canonical=False`; it is
   a median split sitting **0.07 sd** from the GT mean on squat (0.19 / 0.31 on
   deadlift / thruster), against `knee_angle`'s canonical 90° at **1.61 sd**. A cut
   through the middle of a tight distribution (squat GT 0.850 ± 0.056) maximises flip
   sensitivity — every rep sits on the line, so small errors flip verdicts. The swept
   variant sweeps the *central* GT range and inherits the same property. So this row
   measures noise sensitivity at a synthetic split, not fault-catching at a citable
   threshold — which is [[paper-scope-angle-a]]'s thesis applied to itself.
3. **n is 40 reps × 4 non-independent cameras**, all-competent population (real valgus
   faults are near-absent), no significance test — as exp 3's own caveats state.
4. **It is contradicted on other data.** Fitness-AQA consolidated squat — in the wild,
   rear/rear-oblique, *learned* classifier — is the one place a depth channel helped:
   `knees_inward` **ΔAUC +0.080**, with coefficients confirming `knee_width_ratio` is
   `nlf_3d`'s top feature ([[fitness-aqa-depth-finding]]).

#### Why does 2D win? It does not — the 2D arm was given a perfect detector

Asked for the mechanism, the answer turned out to invalidate the row rather than explain
it. **Experiment 3's `view2d` arm is the mocap GT *projected*** (`decision_eval.py:144`,
`proj = ds.project_world_to_image(j3d_m, cp)`) — a detector with **zero** error — while
its 3D arm is a real model. Valgus is the one cue experiment 4 measured as
**detector**-limited rather than projection-limited (perfect mocap-2D 0.04 vs real
RTMPose 0.07). So the published pairing removes precisely the error term that dominates
2D on this cue. Two mechanisms were separable with what is already on disk; all arms
masked to RTMPose's every-15th-frame grid so the per-rep extreme is sampled identically.

| arm | squat MAE / r / flip | deadlift | thruster |
|---|---|---|---|
| perfect 2D (mocap projected) | 0.025 / 0.91 / **0.134** | 0.029 / 0.93 / **0.094** | 0.028 / 0.94 / **0.082** |
| **REAL 2D (RTMPose)** | 0.065 / 0.76 / **0.299** | 0.081 / 0.56 / **0.255** | 0.081 / 0.56 / **0.278** |
| **NLF 3D** | 0.033 / 0.84 / **0.294** | 0.063 / 0.85 / **0.174** | 0.043 / 0.80 / **0.192** |
| NLF projected to image | 0.043 / 0.81 / 0.290 | 0.078 / 0.78 / 0.227 | 0.055 / 0.73 / 0.214 |
| NLF, depth axis dropped | 0.041 / 0.80 / 0.304 | 0.068 / 0.77 / 0.228 | 0.059 / 0.70 / 0.217 |
| GT, depth axis dropped (control) | 0.020 / 0.92 / 0.138 | 0.025 / 0.93 / 0.090 | 0.022 / 0.96 / 0.087 |

**M1 — detector parity: this is the whole effect.** Against a *real* 2D detector, NLF
ties on squat (0.294 vs 0.299) and **wins on deadlift (0.174 vs 0.255) and thruster
(0.192 vs 0.278)**, and beats it on MAE and ranking r in all three (0.033/0.84 vs
0.065/0.76, etc.). Per-subject the direction survives: NLF better in **6/8, 4/8, 4/8**
against real-2D's 1/8, 1/8, 2/8 (14 vs 4 cells, 6 ties). The replication that looked
strongest — 15/15 model × action cells beating 2D — was 15 real models measured against
one impossible one.

**M2 — depth-axis contamination: refuted.** `knee_width_ratio` takes its horizontal span
over the (x, y) ground plane in 3D, which includes the depth axis, but over image-x alone
in 2D — so the obvious story is that NLF's depth error leaks into the span. It does not:
dropping the depth axis moves NLF from 0.294 to 0.304 (squat), 0.174→0.228, 0.192→0.217 —
no improvement, slightly worse. The control confirms the construction is sound (GT with
the depth axis dropped, 0.138, reproduces perfect 2D's 0.134). NLF's valgus error is
ordinary joint-localisation error, and it is *smaller* than a real detector's.

**Caveats on the reversal.** (a) Per-subject flip sits at **0.34–0.58 for both arms** —
near chance — so within a person neither reading tracks valgus; the pooled numbers look
better only because they absorb between-subject anatomy, the same person-offset pathology
`notes/lunge-rule-validation.md` found on `knee_forward_ratio`. (b) RTMPose is COCO
convention mapped into H36M slots and NLF is SMPL-24 mapped the same way, so **both real
arms carry mapping error while the perfect-2D arm is convention-exact** — a third way that
arm is privileged. (c) n = 40–44 reps × 4 non-independent cameras, all-competent
population, no significance test.

**This contradicts the routing recommendation in two committed notes** —
`fit3d_decision_fidelity_summary.md` ("keep calibrated 2D for **valgus**") and
`fit3d_2d_vs_3d_summary.md` ("3D is **not** the answer" for valgus). Experiment 4's own
finding that valgus is detector-dominated is what predicts this: a cue whose error is
dominated by the detector cannot be evaluated with an arm that has no detector. Left for
the user to decide how to fold back into those notes rather than edited in unilaterally.

#### Detector parity across every cue — and how far it generalises over angle

Running the same parity comparison (real RTMPose 2D vs NLF 3D, same frame grid, debiased
swept flip) on **all five cues**, not just valgus:

| action | knee | hip | torso lean | depth ratio | valgus |
|---|---|---|---|---|---|
| squat | 0.200 / **0.115** | 0.207 / **0.061** | 0.248 / **0.194** | 0.188 / **0.126** | 0.299 / 0.294 (tie) |
| deadlift | 0.191 / **0.101** | 0.219 / **0.110** | 0.134 / **0.100** | 0.246 / **0.124** | 0.255 / **0.174** |
| thruster | 0.241 / **0.089** | 0.171 / **0.043** | 0.202 / **0.131** | 0.166 / **0.094** | 0.278 / **0.192** |

(2D / 3D; lower is better.) **14 wins for 3D, 1 tie, 0 wins for 2D.** MAE and ranking r
tell the same story — e.g. squat knee 37.90→12.84 and r 0.60→0.94. Note that
**torso lean also flips**: exp 3 called it a tie, but that was against the perfect-2D
arm; against a real detector 3D wins all three actions and cuts MAE ~4× (30.76→7.09).
So under detector parity **the "route by fault type" recommendation loses its 2D branch
entirely** — there is no cue left where a real 2D pipeline is the better choice.

**Split by rig azimuth** (the only angle contrast the data supports — front-oblique
~20–25° vs rear-oblique ~153–157°): 3D wins or ties in essentially every cell of both
groups; the single exception is squat valgus at front-oblique (0.33 vs 0.35, inside
noise). So within the oblique band the conclusion is angle-stable.

**But "at every angle" is not established, and the untested cell is where 2D should win.**
No 3D model in this project has ever been evaluated at a sagittal or frontal view — Fit3D
has neither, and §2's azimuth sweep contains no 3D model at all. The geometry says that
matters: NLF's cue error is a *model* error (squat knee MAE 12.84, roughly angle-flat),
while 2D's is dominated by a *projection* error that collapses toward the sagittal view.
From §2's sweep, projected-2D squat knee MAE runs 41.4° at 30° azimuth → 8.2° at 60° →
**1.9° at 90°**, and the real detector adds only ~2° at the oblique angles where both were
measured (real RTMPose 37.90 vs sweep-interpolated ~35.8, reconfirming exp 4's
detector≈0). Extrapolating that ~2° detector term, **a real 2D pipeline at a true side
view would read squat knee flexion around 4° against NLF's 12.8°** — i.e. 2D would win,
and the crossover sits somewhere near 50–55° of azimuth. The term that could overturn
that extrapolation is exactly the one nobody has measured: **self-occlusion, which grows
precisely as you approach the sagittal view.** Until that is measured, the honest scope is
"3D ≥ 2D throughout the oblique band", not "at every angle".

**Supportable claim:** *3D is not worse than 2D on valgus. Under detector parity it ties
or wins on all three actions and on MAE/r in all three; the published "2D better" is an
artifact of pairing a real 3D model against a perfect 2D detector on the one cue that is
detector-limited. Neither arm resolves valgus within a subject.* The Fitness-AQA result
(`knees_inward` ΔAUC **+0.080** for `nlf_3d`, [[fitness-aqa-depth-finding]]) stops being
an anomaly under this reading and becomes the corroborating case.

**Against the other two 3D routes, at oblique views** — the ranking is
direct-3D > pseudo-3D ≈ 2D, and the deciding property is depth *quality*, not skeleton
density (`notes/fit3d_sparse_depth_summary.md`): squat knee cue 2D-view 18.4° → MediaPipe
pseudo-3D 14.2° → MeTRAbs 6.4° / NLF 7.1°; squat knee verdict flip deb-2D 21% → MediaPipe
**24%** → NLF 11%. Pseudo-3D buys nothing over a calibrated single view.

Lifting is the cleanest kill, with a scope note: on REHAB24 LOSO, `lifted3d_vicon` 0.566
≈ `vicon2d` 0.583 against real Vicon 0.702, and a SOTA pretrained lifter gained +0.004
([[lifting-deadend-depth-bottleneck]]). But that was measured on REHAB24's front/side
cameras — **lifting has never been tested on oblique data**, so H2's lifting arm is
inferred at the oblique regime, not measured there.

**The condition that actually bounds H2: the judgment rule.** Fit3D's win is measured
under a **fixed** threshold. On Fitness-AQA — in-the-wild data that is ~1485
rear/rear-oblique against 138 side, i.e. *exactly* H2's camera regime — under a **trained
classifier** the depth channel is redundant: `nlf_3d − nlf_2d = +0.003` (p=0.813), while
detector quality gives `nlf_2d − mediapipe_2d = +0.122` (p<0.001), and the error-overlap
shows genuine redundancy (3D fixes 18 / breaks 18, r=0.943). [[paper-scope-angle-a]]
already reconciles these as two quadrants of one theory rather than a conflict; **E5 is
the unrun experiment that settles it.**

One tension worth naming rather than smoothing: on that same rear-oblique in-the-wild
data, NLF 3D *does* help valgus (`knees_inward` ΔAUC +0.080) — the opposite of Fit3D's
routing — because in a rear view mediolateral separation foreshortens into the camera's
depth axis. The general rule is not "sagittal→3D, frontal→2D" but **whichever axis
foreshortens into depth in the view you actually have needs 3D**, which makes the routing
rule view-dependent and ties H2 back to H1.

### H2 verdict

Supported for **sagittal depth/flexion cues under a fixed threshold**, which is the
project's explainability-constrained regime, and the depth advantage collapses to +0.003
when the judgment rule is a learned classifier.

The **cue-family** scope condition, however, does **not** survive the audit in §4: under
detector parity 3D ties or beats a real 2D detector on valgus too, so "3D wins on
sagittal cues and loses on frontal ones" should be restated as "3D's *measured* margin is
largest on sagittal cues, and its apparent frontal-plane deficit was an artifact of the
2D arm's perfect detector." The **judgment-rule** scope condition is the one that still
bounds H2.

## 5. What would settle what is still open

### The sagittal/frontal gap can be closed with data already on disk — REHAB24-6

Both open items above are blocked on the same thing: no dataset in the 3D-model
evaluations has a sagittal or frontal camera. **REHAB24-6 does, and it is already
downloaded and already inferred.** Checked against the files, not assumed:

| requirement | REHAB24-6 status |
|---|---|
| 3D ground truth | `Ex*/PM_*-30fps.npy` → **(F, 26, 4)** homogeneous, metres, **Y up** (0.02–1.85 m) — 16-camera optical mocap |
| a sagittal view | **yes** — `cam17_orientation == front` ⇒ cam18 sees `side` (`dataset.py:CAMERA18_ORIENTATION`) |
| a frontal view | **yes** — cam17 on those same reps; and Ex3's `profile` on cam17 ⇒ `front` on cam18 |
| **paired**, same rep both views | **yes** — the two cameras are simultaneous, so this is a *within-rep* front-vs-side contrast, not the between-rep strata that limited `lunge-view-reconnaissance.md` |
| perfect-2D arm | shipped — `PM_*-c17/c18-30fps.npy` **(F, 26, 2)**, the mocap projected into each camera |
| real-2D arm | `mp2d`, `hrnet_w32`, `hrnet_w48`, `mmpose` skeleton features already built |
| pseudo-3D arm | `mediapipe_{lite,full}` already built |
| lifted-3D arm | `lifted3d`, `lifted3d_vicon`, `vp3d_lifted` already built |
| direct-3D arm | **`nlf_raw3d/` — 130 npz = 65 recordings × BOTH cameras, all 6 exercises** |

Rep counts with a genuine side-or-profile view, paired against the same rep's front view:

| exercise | reps | exercise | reps |
|---|---|---|---|
| Ex1 arm abduction | 88 | Ex4 leg abduction | 116 |
| Ex2 arm VW | 109 | Ex5 leg lunge | 88 |
| Ex3 table push-up | 107 (profile on cam17) | Ex6 squat | 98 |

**606 paired reps across 6 movements**, of which squat / lunge / push-up overlap the
project's flagships and arm abduction / arm VW / leg abduction add three general-tier
movements Fit3D does not contain. So Fit3D (10 movements, oblique only) and REHAB24-6
(6 movements, front + sagittal) are complementary rather than redundant, and together
they cover 13 of the 16 canonical movements.

Two things to check before trusting numbers from it, both known from prior work on this
dataset: (a) the view labels are **per rep and mixed within a recording** — subjects
reorient mid-take (`lunge-view-reconnaissance.md`), so slice per rep window, never per
file; (b) the sagittal camera's landmark validity is materially worse
(`lunge-rule-validation.md`), which is not a nuisance here but **exactly the unmeasured
occlusion term** — on this dataset it can finally be measured rather than assumed.
No camera calibration ships, but the 3D↔2D correspondences (26 joints × ~5k frames per
recording) make the camera pose recoverable by PnP if a true azimuth is wanted rather
than the shipped labels.

### RESULT — the sagittal measurement, run (2026-08-01)

Ran it. 1,072 reps × 2 cameras = **2,144 rep × camera records**, 6 exercises.

**Camera recovery is exact.** DLT on the shipped 3D↔2D correspondences: median and max
reprojection **0.000 px** over all 130 (recording, camera) pairs, max decomposition
residual 1.7e-15. Recovered K is clean — fx=fy=2181.8, principal point (960, 540) on a
1920×1080 frame — so the shipped 2D is an undistorted pinhole projection.

**The shipped view labels are real**, now in degrees rather than words:

| shipped label | camera | true azimuth, median [p10, p90] |
|---|---|---|
| `front` | cam17 | **4–11°** |
| `half-profile` | cam18 | 34–37° |
| `half-profile` | cam17 | 51–57° |
| `side` | cam18 | **85–94°** |
| `profile` (Ex3) | cam17 | 109° |

So REHAB24-6 supplies a genuine azimuth ladder at ~5° / ~35° / ~55° / ~90°.

**Arms.** `perfect2D` = the mocap projected (zero-error detector). `NLF2D` vs `NLF3D` =
**the same model, same frames, same detection — only the readout differs**, which is the
cleanest parity test available and is immune to the confound that invalidated exp 3's
valgus row.

> **Amended by §6 (2026-08-01).** Two things below need reading through §6. (i) `NLF2D` is
> not merely "the same model": it is `NLF3D` **pushed through a camera** — best-projective-
> camera residual **0.000 px**. So this pair isolates the *projection penalty* exactly, but
> it is not a 2D pipeline. (ii) The tally is **NLF-specific**. A second, independent 3D
> model (HMR2.0) does not reproduce it beyond the frontal bin.

**Headline (NLF only — see §6) — 3D's advantage collapses monotonically toward the sagittal
view, but never reverses:**

| azimuth | 3D better | 2D better | tie |
|---|---|---|---|
| frontal ~5° | **14** | 2 | 5 |
| oblique ~35° | **11** | 4 | 3 |
| oblique ~55° | **9** | 3 | 6 |
| **sagittal ~90°** | **7** | **1** | **13** |

(cells = exercise × cue.) That is the predicted shape — 3D is buying back a *projection*
penalty that camera placement can also remove — and it settles the question the previous
section could only extrapolate: **3D is better or tied at every azimuth from 5° to 90°;
it is never systematically worse.** The strong form of the extrapolation ("2D would win at
a side view") is **refuted for a real detector**.

**But it is confirmed for a perfect one, and that is where the headroom is.** At the
sagittal view the zero-error 2D arm beats NLF 3D on every sagittal cue:

| cue (squat @ ~90°) | perfect2D | NLF2D | NLF3D |
|---|---|---|---|
| knee_angle | 0.042 | **0.037** | 0.043 |
| hip_angle | **0.024** | 0.085 | 0.081 |
| torso_lean_deg | **0.011** | 0.111 | 0.115 |
| depth_ratio | **0.056** | 0.148 | 0.131 |

Compare the same rows at ~5°: knee 0.241 / 0.211 / **0.062**. So at a side view the whole
gap becomes *detector* error, exactly as the projection/detector decomposition predicts —
push-up torso at 90° is 0.029 (perfect) vs 0.116 (both real arms), arm-VW torso 0.071 vs
0.254/0.265. **At a sagittal camera the lever is detector quality, not modality.**

Two other things the run confirms independently: `knee_width_ratio` at 90° is bad for
everything (squat 0.358 / 0.412 / 0.425 ≈ chance), matching §2's geometric prediction that
mediolateral separation is unreadable side-on; and **arm abduction is the one exercise
where 3D underperforms at several azimuths** — consistent with REHAB24-6 LOSO, where arm
abduction was where monocular fell furthest behind Vicon (−0.14 / −0.27).

**Caveats.** (a) **The 3D arm gets an oracle**: NLF's camera-frame output is rotated to
gravity-aligned using the DLT-recovered camera rotation, so the gravity-dependent cues
(torso lean, depth ratio, knee width) benefit from knowing true vertical. Rotation-invariant
angles (knee, hip, elbow) are unaffected. Fit3D's experiments share this property
(`decision_eval.nlf_world_points` uses the GT rotation), so it is consistent with prior
work — but it is an oracle in both, and a deployed system would have to estimate gravity.
(b) Lab recording, instructed errors, rehab movements. (c) Swept flip on this population
measures ranking fidelity, per `_swept_flip`'s own docstring. (d) **No per-subject
breakdown was run here** — given that two earlier claims in this note turned out to rest
on 2 of 8 subjects, the tally above should be treated as pooled evidence whose
concentration is unmeasured. The monotone trend across four independent azimuth bins is
the main reason to believe it.

## 6. Does §5 survive changing the 3D model? (run 2026-08-01)

§5 rests on one 3D model. Since the mechanism it asserts is a statement about *curve
shapes* — 3D error flat in azimuth, 2D projection error collapsing toward sagittal —
"3D is flat" could easily be an NLF property. Re-ran the whole thing with every model that
has raw joints on disk for REHAB24-6.

**Arms (9).** `gt` mocap · `perfect2d` mocap projected · **NLF** parametric 3D + its 2D +
the non-parametric head · **HMR2.0** 3D + its 2D · **MediaPipe** `world` (pseudo-3D) +
`image` (2D). So **two independent image→3D models plus one pseudo-3D family.**
`image→2D→lifted-3D` still cannot be tested: `lifted3d_skeleton_features/` stores
aggregated 1188-d feature vectors, not joints, so the lifting arm of H2 remains inferred.

**Provenance check first, and it reframes §5.** Fitting the best 3×4 projective camera from
each model's own 3D to its own 2D:

| pairing | residual | reading |
|---|---|---|
| NLF 3D → NLF 2D | **0.000 px** | derived |
| HMR 3D → HMR 2D | **0.000 px** | derived |
| MediaPipe world → image | 10.93 px | **independent head** |
| NLF 3D → 2D + 1 px noise | 1.014 px | floor (fit slack) |
| NLF / HMR 3D → *mocap's* 2D | 9.92 / 11.58 px | ceiling (different source) |

So both mesh models' `smpl2d` **is** their `smpl3d` projected. §5's "2D arm" was never a 2D
pipeline — it is the same 3D estimate minus the depth axis, which isolates the projection
penalty exactly but cannot represent a real 2D detector. MediaPipe's `image` head sits at
the different-source ceiling, so **MP-2D is the only genuine 2D detector available here at
raw-joint level**. (Note a weak-perspective null gives HMR 0.20 px but NLF 5.31 px, and
would have been read as "NLF is independent" — wrong, because weak perspective is HMR's
camera model and not NLF's.)

**Fairness rules.** One common frame mask (NLF-valid ∧ HMR-valid ∧ MediaPipe-finite) for
all arms — MediaPipe reports 100% coverage because it emits plausible landmarks when
tracking fails, and those failures concentrate toward sagittal, mimicking the effect under
test. NLF's and HMR's valid masks are element-wise equal on **99.99%** of 363k frames: the
two extractions evidently share an upstream person detector and differ in the pose/mesh
head, so they are compared frame-for-frame but are not independent end to end. Coverage
inside rep windows is 100% in every azimuth bin, so the trend is not a missing-data trend.
Debias grouping identical to §5. All CIs below are **subject-clustered** (10 subjects
resampled with replacement, one draw shared across cells) — a rep-level bootstrap treats
~230 reps per person as independent and gives CIs roughly 2× too tight.

### 6a. The mechanism replicates — this part is not NLF-specific

Debiased MAE across the azimuth ladder (squat / knee angle, degrees):

| arm | ~5° | ~35° | ~55° | ~90° | 5→90 |
|---|---|---|---|---|---|
| perfect2d | 13.49 | 8.68 | 3.68 | 1.87 | ×0.14 |
| nlf2d | 14.90 | 11.65 | 3.77 | 2.24 | ×0.15 |
| hmr2d | 14.41 | 10.06 | 3.76 | 2.14 | ×0.15 |
| mp2d | 14.45 | 9.15 | 3.30 | 2.38 | ×0.16 |
| **nlf3d** | **3.52** | 2.92 | 2.51 | 2.32 | ×0.66 |
| **hmr3d** | 7.51 | 3.85 | 2.99 | **1.96** | ×0.26 |
| mp3d | 8.93 | 5.49 | 3.86 | 3.00 | ×0.34 |

All four 2D arms collapse by ~6–7× and land within 0.5° of each other at 90° — projection
dominates and detector identity barely matters.

That is one cell, so here it is aggregated: **median 5→90 MAE ratio over all 21 exercise ×
cue cells**, which is the honest version of the claim.

| arm | median ratio | [p25, p75] |
|---|---|---|
| perfect2d | **0.31×** | [0.14, 1.91] |
| hmr2d | 0.60× | [0.50, 1.89] |
| nlf2d | 0.67× | [0.46, 1.90] |
| mp2d | 0.71× | [0.54, 2.28] |
| hmr3d | **0.87×** | [0.52, 1.48] |
| nlf3d_np | 0.88× | [0.62, 1.27] |
| nlf3d | 0.90× | [0.69, 1.39] |
| mp3d | 0.95× | [0.68, 1.23] |
| *all 2D pooled* | **0.61×** | |
| *all 3D pooled* | **0.88×** | |

Every 2D arm falls further toward the sagittal view than every 3D arm, and the zero-error
2D arm falls furthest (0.31×) because it has no detector floor to hit — that ordering is
the projection mechanism, and it is the same for **both** independent 3D models. (The
p75 > 1 tail is the frontal-plane cues, which correctly get *worse* side-on.) HMR2.0 is
the worse model overall (mean 3D flip 0.220 vs NLF's 0.186, r = 0.86 between their per-cell
flips) and it is *much* worse at frontal (squat knee 7.51 vs 3.52) — which is exactly why
the next part breaks.

### 6b. The headline does NOT survive — it was carrying NLF-specific margin

Within-model 2D-vs-3D tally (3D better / 2D better / tie), same procedure as §5:

| azimuth | NLF | HMR2.0 | MediaPipe |
|---|---|---|---|
| frontal ~5° | 14 / 2 / 5 | 11 / 5 / 5 | 8 / 8 / 5 |
| oblique ~35° | 11 / 4 / 3 | **6 / 7 / 5** | 3 / 12 / 3 |
| oblique ~55° | 9 / 2 / 7 | 6 / 4 / 8 | 7 / 7 / 4 |
| sagittal ~90° | 7 / 1 / 13 | **6 / 7 / 8** | 6 / 9 / 6 |

(NLF reproduces §5 exactly except one 55° cell moving from "2D better" to "tie" under the
common mask.) Pooled (flip3D − flip2D), scale-free cues, **subject-clustered** 95% CI,
300 draws — **negative = 3D better**, ★ = CI excludes zero:

| model | ~5° | ~35° | ~55° | ~90° |
|---|---|---|---|---|
| NLF | −0.054 [−0.080, −0.034] ★ | −0.029 [−0.052, −0.007] ★ | −0.021 [−0.045, **+0.001**] | −0.020 [−0.032, −0.008] ★ |
| HMR2.0 | −0.035 [−0.063, −0.008] ★ | +0.006 [−0.013, +0.033] | −0.004 [−0.026, +0.013] | −0.007 [−0.029, +0.011] |
| MediaPipe | +0.011 [−0.016, +0.035] | **+0.042** [+0.009, +0.070] ★ | **+0.027** [+0.004, +0.048] ★ | **+0.051** [+0.028, +0.070] ★ |

**The only azimuth where both independent 3D models beat their own 2D readout is frontal
(~5°).** For HMR2.0 the advantage is indistinguishable from zero at 35°, 55° and 90°. NLF
keeps a small edge at 35° and 90° (its 55° bin now straddles zero) but at −0.02 it is a
third of its frontal margin. So §5's "3D ≥ 2D at every azimuth 5–90°" is **true of NLF and
not of image→3D as a class**; what replicates across models is the *frontal* claim.

**Pseudo-3D loses to the 2D detector it ships with — at oblique and sagittal views, not at
frontal** (MediaPipe: +0.042 / +0.027 / +0.051 with CIs excluding zero at 35/55/90; at
frontal +0.011 straddles zero and the tally is a dead-even 8/8/5). This is the one pair
where the 2D and 3D readouts really are independent. Direct evidence on the pseudo-3D arm
of H2, and consistent with [[fit3d-sparse-depth-finding]]: the recovering ingredient is
depth *quality*, and MediaPipe's `z` does not have it.

### 6c. Per-subject — the pooled advantage concentrates at frontal only

The check §5 flagged as missing. Fraction of (subject × cell) comparisons where 3D beats
its own 2D, scale-free cues:

| azimuth | NLF | HMR2.0 | MediaPipe |
|---|---|---|---|
| frontal ~5° | **67%**, 9/10 subjects | 59%, 6/10 | 47%, 4/10 |
| oblique ~35° | 54%, 5/9 | 43%, 3/9 | 35%, 0/9 |
| oblique ~55° | 46%, 3/9 | 41%, 2/9 | 38%, 0/9 |
| sagittal ~90° | 49%, 2/10 | 49%, 2/10 | 33%, 1/10 |

At 55° and 90° the pooled 3D advantage **is not present within individual people** — 46–49%
is chance. This is not merely a power artifact: the frontal bin reaches 67% and 9/10
subjects at the same per-subject sample sizes, which is the internal control.

This coexists with NLF's ★ at 90° in §6b without contradiction, but only in one reading:
the 90° effect is a **small mean shift (−0.02 flip) that is stable across subjects in
aggregate yet too small to surface in any individual's cells**. Per-subject-per-cell
estimates are individually noisy, so a real 0.02 shift can look like chance in every one
of them while the average holds. The practical consequence is the same either way —
at 55–90° the modality choice is not what decides a verdict for a given person.

### 6d. Contamination ruled out

REHAB24-6 flags `mocap_erroneous` reps and reps with extra people in frame (which could
make the two detectors track different bodies). Dropping both (2144 → 1697 records) moves
nothing material: NLF 90° 7/1/13 → 6/2/13, HMR 90° 6/7/8 → 5/6/10, MediaPipe unchanged.

### 6e. What §5 and §6 jointly support

1. **Model-independent:** 2D cue error is projection-limited and shrinks toward the
   sagittal view (median 5→90 ratio 0.61× pooled over 2D arms, 0.31× for the zero-error
   one); direct image→3D error is much flatter (0.88×). Two 3D models, one pseudo-3D.
2. **Model-independent:** at a **frontal** camera, direct image→3D is a real win — the one
   bin where both 3D models clear zero under subject clustering *and* hold within subjects
   (NLF 9/10, HMR 6/10).
3. **Model-dependent, therefore not a claim:** "3D ≥ 2D at oblique and sagittal views."
   NLF keeps a −0.02 edge at 35° and 90°; HMR2.0 has none anywhere past frontal, and
   neither model's edge is visible within an individual. Whether 3D helps there is a
   question about *which* 3D model, not about the modality.
4. **Model-independent:** pseudo-3D is worse than the 2D detector it ships with at 35°,
   55° and 90° — but is a **tie at frontal**, so the advantage of real depth over
   heuristic depth appears exactly where projection starts to hurt.
5. **Unchanged from §5:** a zero-error 2D detector beats every 3D model at 90° on the
   sagittal cues (squat hip 0.024 vs 0.081/0.088, torso 0.011 vs 0.115/0.111) — so at a
   side view the lever is **detector/model quality, not modality**. Note the angle rows
   carry this; `depth_ratio` and `knee_width_ratio` are ratio cues whose cross-skeleton
   (mocap-26 vs SMPL-24 vs MediaPipe-33) *multiplicative* factor an offset-debias cannot
   remove, so quote the angles.

Caveats from §5 that still apply: the gravity oracle on every 3D arm, lab recording with
instructed errors, and swept flip measuring ranking fidelity.

## 7. Why is frontal so much worse than sagittal? (derivation + test, 2026-08-01)

§5 and §6 measure a large frontal-vs-sagittal gap without explaining it. There is a
closed-form reason, it is testable, and the naive version of it turns out to be **wrong in
an instructive way**.

### 7a. The geometry: sin θ for sagittal cues, cos θ for frontal cues

Put the athlete at the origin: forward **f**=(1,0,0), left=(0,1,0), up **z**=(0,0,1); the
camera sits along **c**=(cos θ, sin θ, 0), so θ is exactly the azimuth used throughout this
note. Orthographic projection maps a segment **u** to **u** − (**u**·**c**)**c**:

- a **sagittal** segment **u**=(a, 0, b) → (a sin²θ, −a sinθ cosθ, b), whose horizontal
  extent is **a·|sin θ|**. The vertical part *b* is untouched.
- a **mediolateral** segment **u**=(0, w, 0) → magnitude **w·|cos θ|**.

So projection multiplies sagittal extent by **sin θ** and mediolateral extent by **cos θ**,
while detector noise and skeleton noise shrink with neither. A 3D readout carries no such
factor. That single line predicts the whole §6a table, the opposite trend of
`knee_width_ratio` (perfect2D MAE 0.008 → 0.273 from 5°→90°, ×36; swept flip 0.36 ≈ chance
side-on), and the p75>1 tail.

**Test it on the cue where the law is exact.** The derivation is about projected *extents*;
most cues here are interior angles, which are a nonlinear function of those extents, so
they cannot test it cleanly. `torso_lean_deg` can: for a segment with sagittal component *a*
and vertical *b*, tan(read) = sin θ · tan(true) exactly. Slope of reading-on-truth (truth is
noiseless, so no attenuation bias), `torso_lean_deg` only, perfect2D:

| | ~7° | ~35° | ~55° | ~92° |
|---|---|---|---|---|
| **predicted, sin θ** | 0.113 | 0.579 | 0.815 | 1.000 |
| **measured** | **0.160** | **0.670** | **0.850** | **1.017** |

Four bins, agreement to 0.02–0.09 with no fitted parameter. **The law holds.**

Across *all* sagittal cues pooled the measured slope is much higher (0.595 at frontal), and
that gap is cue composition, not a failure of the law: projection preserves **vertical**
coordinates outright, so `depth_ratio`'s numerator (hip height − knee height) is exact at
every azimuth and only its femur denominator shrinks; a squat's femur points forward *and
outward*, so stance width leaves a mediolateral component that survives at frontal; and an
interior joint angle degenerates far more slowly than an angle-from-vertical. **A frontal
camera is not blind to sagittal cues — it is distorted, and different cues by different
amounts.**

### 7b. Two wrong explanations, both killed by their own test

**Wrong guess 1 — "it is a gain error, so calibrate it."** Refit each camera with an oracle
*affine* map (slope AND offset) instead of the offset-only debias used everywhere else.
On MAE it recovers only **7–17%** (perfect2D at frontal 0.676 → 0.626, −7%). On the verdict
flip it recovers **~2%** (perfect2D frontal 0.235 → 0.241, i.e. nothing). In hindsight the
flip half was never a real question: swept flip depends only on **ranking**, and a positive
affine map cannot reorder anything. So gain is not the mechanism, and calibrating it cannot
be the fix.

**Wrong guess 2 — "the compression factor varies rep to rep."** The tempting version:
d(ln sin θ)/dθ = **cot θ** is maximal at θ→0 and exactly zero at θ=90°, and the azimuth
spread within the frontal bin (p10–p90 = 1.1°–16.3°) does imply a sin θ spread of
0.020–0.281 versus 0.947–0.999 at sagittal. That is a fact about the azimuth *distribution*,
not about the errors — so I tested it: head-to-head of two 2-parameter fits, `reading ≈
β·truth + c` versus `reading ≈ β·(sin(az_rep)·truth) + c`. **Giving the fit each rep's own
azimuth makes it 25–39% WORSE** at frontal and oblique (and −4% at sagittal). Refuted. Part
of the reason is itself instructive: the facing estimate carries **4.4° median** (p90 10.7°)
disagreement between its two independent estimators, which near θ=0 is the same size as the
azimuth being measured — at a frontal camera you cannot even *establish* each rep's azimuth
well enough to use it.

### 7c. What it actually is: the projection is many-to-one

The quantity the verdict rides on is invariant to both offset and gain — the correlation
between reading and truth. Median over sagittal cells:

| arm | ~5° | ~35° | ~55° | ~90° |
|---|---|---|---|---|
| perfect2d | **0.649** | 0.874 | 0.912 | **0.977** |
| nlf2d / hmr2d / mp2d | 0.66 / 0.62 / 0.61 | 0.83 / 0.80 / 0.79 | 0.81 / 0.79 / 0.82 | 0.85 / 0.86 / 0.81 |
| nlf3d / hmr3d | **0.87** / 0.78 | 0.89 / 0.85 | 0.89 / 0.84 | 0.91 / 0.89 |

And the reason it falls — projection is **many-to-one**. Among reps whose *reading* is
nearly identical (within 10% of its range), how much does the *truth* still vary, as a
fraction of its own spread:

| arm | ~5° | ~35° | ~55° | ~90° |
|---|---|---|---|---|
| perfect2d | **0.672** | 0.468 | 0.301 | **0.203** |
| nlf3d | 0.417 | 0.416 | 0.370 | 0.384 |
| hmr3d | 0.545 | 0.484 | 0.495 | 0.428 |

**At a frontal camera, reps that look identical to a zero-error 2D detector still differ by
67% of the cue's full range; side-on, by 20%.** That is the whole gap, and it is the one
thing no calibration can touch: an affine map cannot invert a many-to-one function. The
sin θ law is the *cause* — it compresses the true range into a band comparable to the
residual spread contributed by everything projection discards (stance width, facing, the
out-of-plane configuration) — and the ambiguity table is the *consequence*. Note the 3D
arms are flat at 0.37–0.55 in every bin: they carry a model-error floor that a perfect 2D
detector beats side-on, and never carry the projection term at all.

### 7d. The crossover, and which half of it needs the gravity oracle

3D carries no sin θ factor, so its advantage should be largest where the geometric term is
largest and vanish at sagittal. It does — but the two halves of the crossover have
different standing, because the 3D arms get true-vertical as an oracle on the
gravity-dependent cues and *not* on the rotation-invariant ones. Split r(reading, truth):

| cue group | arm | ~5° | ~90° |
|---|---|---|---|
| **rotation-invariant** (knee/hip/elbow) — *no oracle* | perfect2d | 0.705 | 0.932 |
| | nlf3d | **0.890** | 0.920 |
| | hmr3d | **0.810** | 0.914 |
| **gravity-dependent** (torso lean, depth ratio) — *3D holds an oracle* | perfect2d | 0.494 | **0.979** |
| | nlf3d | **0.774** | 0.891 |
| | hmr3d | **0.735** | 0.766 |

**The frontal crossover is oracle-free**: on rotation-invariant cues alone both 3D models
beat a *perfect* 2D detector at a frontal camera (0.890 / 0.810 vs 0.705). No amount of 2D
accuracy fixes that, because the deficit is geometric. **The sagittal reversal is carried by
the gravity-dependent cues**, where a perfect 2D detector side-on is nearly exact
(r = 0.979) — on rotation-invariant cues at 90° the three arms are within 0.02 of each
other and the modality question simply stops mattering. Same picture in swept flip: frontal
perfect2D 0.235 vs nlf3d 0.146 / hmr3d 0.161; sagittal perfect2D 0.070 vs 0.123 / 0.120.

### The remaining items

- **Occlusion-aware version of §2.** REHAB24-6 supplies the real term directly; §2's
  synthetic sweep can then be corrected rather than caveated.
- **Lifting at oblique azimuths**, to close H2's inferred arm — this one still needs
  Fit3D, since REHAB24-6 has no oblique camera. `half-profile` (466 reps) is the closest
  thing and is worth checking as a third azimuth bin.
- **E5** (already specified in [[paper-scope-angle-a]]): fixed vs learned threshold on the
  same four arms, which decides whether H2's scope condition is the real story.

Nothing else in the project's dataset survey (`notes/dataset-summary.md`) can substitute:
UI-PRMD, KIMORE, IntelliRehabDS, ExeCheck and UTD-MHAD all use **Kinect body tracking as
their "3D"** (a sensor estimate, not mocap) and are effectively single-view; PHYTMO is
IMU + optical reference with no usable RGB pose stream.
