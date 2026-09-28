# Fit3D experiment 3 — does the perception layer flip the *coaching verdict*?

**Question.** Experiments 1 and 2 are stated in cue units (a knee angle is off by N
degrees). A coaching app does not output degrees; it outputs a **pass/fault verdict**
("you didn't reach parallel"). This experiment translates the cue findings into that
verdict and asks the decision-level question the whole depth-bottleneck thread is for:
reading a fault from a single 2D camera, **how often is the verdict wrong vs the mocap
truth — and does direct image->3D (NLF) fix it?**

**Method.** For every rep × camera we read each cue at the **rep extreme** (the
bottom-of-squat instant a coach judges, via `biomech.rep_summary`) three ways with
identical formulas, threshold it into pass/fault, and compare to the verdict from the
view-invariant 3D ground truth:

* `gt` — mocap 3D truth (defines the correct verdict),
* `view2d` — the mocap 3D projected into the camera, i.e. a 2D detector with **zero
  error** (`decision_eval.py:144`). It stands in for a 2D pipeline; a real detector adds
  its own error, which decides the valgus row below,
* `nlf` — NLF monocular 3D (the proposed fix; same npz as experiment 1).

The 2D arm is **three-valued** so the headline is not inflated by *correctable* bias.
Experiment 2 found constant per-view offsets (knee +41°, torso −22°) that one calibration
constant removes without any 3D. So we also report each reading after subtracting **each
camera's oracle mean offset vs GT** — the upper bound on what per-view calibration can buy
— and apply the same debiasing to NLF (it has its own residual bias). The only fair
comparisons are **raw-2D vs raw-NLF** and **debiased-2D vs debiased-NLF**. The deciding
number: *after oracle calibration, does single-view 2D still flip the depth verdict more
than NLF?* `src/fit3d/decision_eval.py`, run via `scripts/fit3d/run_decision_eval.py`.

**Framing.** This is **verdict fidelity vs mocap truth**, not accuracy vs human
correctness labels (that is the n-limited REHAB24 thread). The Fit3D population is all
competent reps, so on a real threshold the meaningful error is the **false-alarm rate**
(falsely failing a good rep). The 4 cameras of one rep are not independent, so we report
descriptive rates + per-subject spread, **no p-value** over the pooled 160 readings.

## Headline — the squat depth verdict (knee angle at parallel, 90°)

40 reps × 4 cameras × 8 subjects = 160 readings; true-fault prevalence 8% (competent reps).

| readout | verdict flip | false-alarm | miss |
|---|---|---|---|
| raw zero-error 2D (uncalibrated) | 76% | **82%** | 0% |
| oracle-calibrated zero-error 2D | 16% | 14% | **42%** |
| raw NLF | 7% | 0% | 92% |
| oracle-calibrated NLF | **7%** | 7% | **0%** |

The same verdict with a real 2D detector, RTMPose, on the frames it was run on (every
15th frame; truth on the same grid, which raises true-fault prevalence to 10%;
`fit3d_detector_parity_significance.md`):

| readout | verdict flip | false-alarm | miss | per-subject flip, mean ± SD |
|---|---|---|---|---|
| raw zero-error 2D | 74% | 83% | 0% | 74 ± 21% |
| raw RTMPose | 64% | **71%** | 0% | 64 ± 15% |
| raw NLF | 9% | 0% | 88% | 9 ± 20% |
| calibrated zero-error 2D | 16% | 12% | 44% | 16 ± 19% |
| calibrated RTMPose | 21% | 19% | **38%** | 21 ± 18% |
| calibrated NLF | **6%** | 6% | **0%** | 6 ± 10% |

* **Uncalibrated, a single 2D camera false-fails most good squats**: 82% with the
  zero-error arm and 71% with RTMPose. The +41° per-view knee bias pushes nearly every
  deep squat above the 90° "didn't reach parallel" line.
* **Calibration is not enough.** One per-view offset removes most false alarms (14%) but
  then *misses 42% of the genuinely shallow reps* — it cannot fix both error types at once,
  because the residual cross-view scatter is rep-dependent, not a constant. Net flip 16%;
  21% (19% FA + 38% miss) with RTMPose.
* **NLF gets both low** (7% FA, 0% miss; 7% flip; 6% on RTMPose's frames).
* Against RTMPose, NLF's knee swept flip is lower by 0.109, 8/8 subjects, p = 2/256
  (combined over three actions, `fit3d_detector_parity_significance.md`).

**Does the calibrated gap survive per subject?** (The *raw* finding clearly does — see below.
This is the question for the novel *calibrated* claim, whose edge is only ~10 FA + ~5 miss
reps.) Per-subject calibrated knee-flip — **NLF ≤ calibrated-2D in 7/8 subjects** (strictly
lower in 5, tie in 2, higher in 1 — s08, by 5 pts). So the *direction* "NLF beats even
oracle-calibrated 2D" is robust, not a 1–2-subject artifact. But the *magnitude* of the mean
gap (deb-2D 16% vs NLF 7%) is **concentrated in 2 hard subjects** (s05 45→0, s07 50→40);
for the other ~half, per-view calibration alone already gets 2D to 0–10%. So the honest claim:
**calibration sharply reduces the depth-verdict gap but does not close it** — it cannot for
the subjects whose form sits near the threshold, where the residual scatter (not a constant)
straddles the line. Direct 3D removes that residual.

(The knee@90 headline box is squat-appropriate. For the hip-hinge **deadlift** knee@90 is
degenerate — 93% prevalence, knee depth isn't the deadlift cue — so read deadlift off the
needs-3D map below, not this box. **Thruster** knee@90 prevalence is 30%: raw-2D false-fails
100%, calibrated-2D flips 19% (16% FA + 25% miss), calibrated-NLF 12%. With RTMPose on its
15-frame grid: raw false alarm 99%, calibrated flip 22% (21% FA + 25% miss), calibrated NLF
10% on the same frames.)

## Cross-cue needs-3D map (fair: debiased swept-threshold flip, both arms)

Verdict-flip averaged over thresholds swept across the central GT range, oracle-calibrated
on both arms. This isolates *which cues need 3D* from *which a calibrated 2D camera handles*.

| cue | squat 2D→NLF | deadlift 2D→NLF | thruster 2D→NLF | verdict |
|---|---|---|---|---|
| **knee_angle** (depth) | 21→**11** | 17→**10** | 22→**9** | **needs 3D** (all 3) |
| **depth_ratio** (hips-below-knee) | 13→13 | 21→**11** | 16→**10** | needs 3D (2/3) |
| **hip_angle** (flexion) | 12→**6** | 18→13 | 14→**4** | needs 3D / borderline |
| torso_lean_deg | 11 / 19 | 9 / 10 | 17 / 14 | tie vs zero-error 2D |
| knee_width_ratio (valgus) | 15 / **31** | 9 / **17** | 9 / **20** | zero-error 2D better; **not** a real detector (below) |

1. **Direct 3D rescues the sagittal flexion/depth verdicts** (knee flexion, hip flexion,
   hip-below-knee) that single-view projection geometrically corrupts — across squat,
   deadlift, and thruster, and *even after oracle per-view calibration*.
2. **The torso and valgus rows compare NLF with a 2D arm that has no detector error.**
   Valgus is the cue whose 2D error is mostly detector error: zero-error 2D 0.04 vs RTMPose
   0.07 (`fit3d_2d_vs_3d_summary.md`). Against RTMPose on the same 15-frame grid
   (`fit3d_detector_parity_significance.md`, combined over the three actions, n = 8
   subjects), NLF's flip is lower on valgus by 0.057 (6/8, p = 0.195) and on torso lean by
   0.053 (7/8, p = 0.141). Both are undetermined. The 2D-better valgus gap above is real
   (+0.120, p = 2/256), but RTMPose adds +0.174 over the same zero-error arm. Experiment 2
   called torso lean view-robust (r=0.87 plus a constant offset), which is why the
   zero-error arm ties NLF there.

So the claim is: *direct image→3D fixes the geometrically corrupted cue family (sagittal
depth/flexion), and on torso lean and valgus it is not worse than a real 2D detector.*
Pooled over cameras, there is no measured cue on which a real 2D detector beats NLF at
Fit3D's oblique views.
Routing valgus to calibrated 2D is not supported by these data.

## Per-subject spread (honest n)

The 4 cameras of one rep are not independent. Per-subject knee-flip (mean ± sd over 8
subjects): squat **2D 76±17% vs NLF 7±13%**; thruster **2D 69±31% vs NLF 22±24%**. The
depth-verdict gap holds within every subject, not just pooled. (Deadlift knee is degenerate
as above; its hip cue: 2D 30±23% vs NLF 13±19%.)

## Caveats

- **Oracle debiasing is an upper bound**, not a deployable calibration — it uses the GT to
  compute each camera's offset. A real system has no per-view GT; the *raw RTMPose* rows
  are the closest to a deployed 2D pipeline (71% squat false alarm), and they are far worse
  for 2D. NLF's arm also uses the ground-truth camera rotation, which a deployed system
  lacks; the knee angle itself does not depend on it.
- **Calibration is modeled as a per-view offset.** An affine/scale per-view calibration would
  face the same *rep-dependent* residual: the 14%-false-alarm + 42%-miss tradeoff is the
  signature of scatter that no per-camera *constant or linear* map can remove (it depends on
  the athlete's pose, not the camera). That is exactly what direct 3D removes.
- **All-competent population**: false-alarm is the measurable error; true-fault prevalence
  is low (squat 8%) so `miss` rests on few reps. Reported descriptively, no significance
  test over the non-independent 160.
- **No side view** in Fit3D's all-oblique rig (4 cameras at ±45° corners,
  `fit3d_view_dependence_summary.md`); a sagittal
  camera would read flexion better, narrowing the raw-2D gap (not the calibrated one).
- `depth_ratio` uses the hip *joint centre* (above the knee even at parallel), so it has no
  universal threshold here — reported on the median/swept split only, not the 90° box.

## Implications

This closes the perception→decision link the depth-bottleneck thread was missing: it is no
longer "NLF recovers cues vs GT" (exp 1) but "**NLF recovers the coaching *verdict* the cue
drives, where 2D — even calibrated — cannot**." Concretely for x-coach: gate
`pose_rule_detector`'s **depth/flexion** faults on a 3D source (NLF); the raw single-view 2D
depth verdict is unusable (82–100% false-fail with the zero-error arm, 71–99% with RTMPose). For **valgus** and torso lean neither source
is shown to be better than the other against a real detector
(`fit3d_detector_parity_significance.md`). NLF's torso-lean and valgus readings here also
use the ground-truth camera rotation, which a deployed system does not have.

Reproduce: `python scripts/fit3d/run_decision_eval.py --pred-root data/Fit3D/derived/preds/nlf
--action squat` (JSON: `data/Fit3D/derived/decision_eval_<action>_nlf.json`; NLF npz under
`data/Fit3D/derived/preds/nlf/`).
