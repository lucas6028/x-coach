# Against a real 2D detector, NLF's verdict advantage is determined for knee and hip angle (−0.109 / −0.128, 8/8 subjects, p = 2/256) and undetermined for torso lean and valgus

*Fit3D · post hoc subject-level significance test of detector parity · run 2026-09-26 ·
tests the parity audit in [`camera-placement-hypothesis.md`](camera-placement-hypothesis.md)
and corrects [`fit3d_decision_fidelity_summary.md`](fit3d_decision_fidelity_summary.md)*

**Result.** RTMPose 2D and NLF 3D were compared on the same frames for five verdict cues
across squat, deadlift and thruster, with the subject as the unit (n = 8). NLF lowers the
debiased swept verdict-flip rate on knee angle by 0.109 (8/8 subjects, p = 2/256) and on
hip angle by 0.128 (8/8, p = 2/256). Depth ratio also clears (−0.085, 8/8, p = 2/256), but
its 3D reading uses the ground-truth vertical. Torso lean (−0.053, 7/8, p = 0.141) and
valgus (−0.057, 6/8, p = 0.195) are undetermined. The earlier "2D is better on valgus"
compared NLF with a zero-error 2D arm. That gap is real (+0.120, p = 2/256), but a real
detector adds +0.174 of flip on the same cue.

**Caveat on reading it.** Post hoc, no pre-registered plan, about 20 tests in this note.
Fit3D has only oblique cameras. Three of the five cues read the 3D arm against the
ground-truth camera rotation. Swept flip measures ranking fidelity, not fault detection
at a coaching threshold. With 8 subjects the smallest attainable p is 2/256 = 0.0078.

## Background

Experiment 3 (`fit3d_decision_fidelity_summary.md`) scored NLF against a 2D arm that is the
mocap ground truth projected into the image. It concluded that valgus is better read in 2D
and torso lean is a tie. The audit in `camera-placement-hypothesis.md` replaced that arm
with RTMPose, a real 2D detector, and found NLF lower in 14 of 15 cue × action cells with
one tie. That audit reported pooled rates without a significance test. This note adds the
test.

## Method

| term | meaning |
| --- | --- |
| arm | one source of cue readings: mocap truth, RTMPose, NLF, or the mocap projection |
| perfect 2D | the mocap 3D projected into each camera; a 2D detector with zero error |
| 15-frame grid | every 15th frame, the frames RTMPose was run on |
| debiased | each camera's mean offset from truth removed per arm, using the truth (an oracle calibration upper bound) |
| swept flip | share of reps whose pass/fault verdict disagrees with truth, averaged over 13 thresholds at the 20th–80th percentiles of the pooled truth |
| GT rotation | the ground-truth camera rotation used to turn NLF's camera-frame joints into world coordinates |

Each cue is read at the rep extreme (`biomech.rep_summary`) from each arm, as in
`src/fit3d/decision_eval.py`. The parity comparison puts every arm on the 15-frame grid,
including the truth. The rep extreme is sample-count biased, since fewer frames can only
make an extreme less extreme, so arms on different grids are not comparable.

The 3D cue formulas split by what they need from the GT rotation. Knee and hip angle are
interior joint angles and are rotation-invariant. Torso lean and depth ratio are measured
against the vertical. Valgus (`knee_width_ratio`, knee span over ankle span) is measured
in the ground plane, which is defined by the vertical.

Per-subject flips are computed on the pooled thresholds. The pooled flip is then exactly
the n-weighted mean of the per-subject flips, so per-subject contributions sum to the
pooled Δ. The test is an exact two-sided sign-flip over all 2^8 = 256 sign patterns of
those contributions. One combined test per cue averages each subject's contribution over
the three actions, because the three actions share the same 8 subjects. Holm correction
is applied over the five combined tests. Per-action p-values are uncorrected and secondary.

Two quantities are reported and differ in the third decimal. Δ is the pooled, n-weighted
difference, and the p-value refers to it. The per-subject mean ± SD is unweighted.

Two choices a reader would otherwise question:

- **Why not permute arm labels.** 2D and 3D readings carry different per-camera offsets.
  Swapping raw readings between arms and then debiasing mis-calibrates both arms and
  inflates the null. The test therefore sign-flips per-subject flip differences.
- **Why no interval.** A subject-cluster bootstrap with 8 clusters under-covers. Dispersion
  is given as per-subject SD, range and count instead.

## Reproduction checks

| check | result |
| --- | --- |
| experiment 3 valgus row, perfect 2D vs NLF at full rate | 0.147 / 0.310, 0.089 / 0.174, 0.087 / 0.200; identical to `decision_eval_<action>_nlf.json` |
| audit parity table, RTMPose vs NLF, 5 cues × 3 actions | all 15 cells identical to `camera-placement-hypothesis.md` to three decimals |
| audit perfect-2D valgus row on the 15-frame grid | 0.134 / 0.094 / 0.082; identical |
| per-subject contributions sum to pooled Δ | exact in every cell |

## Results

Combined over squat, deadlift and thruster. Δ = flip(NLF) − flip(RTMPose); negative means
NLF is better.

| cue | 3D reading uses GT rotation | Δ | per-subject Δ, mean ± SD [min, max] | subjects NLF better | exact p | Holm (5 tests) |
| --- | --- | ---: | --- | ---: | ---: | --- |
| knee angle | no | −0.109 | −0.110 ± 0.063 [−0.236, −0.018] | 8/8 | 0.0078 | holds |
| hip angle | no | −0.128 | −0.127 ± 0.059 [−0.247, −0.042] | 8/8 | 0.0078 | holds |
| depth ratio | vertical | −0.085 | −0.084 ± 0.057 [−0.170, −0.012] | 8/8 | 0.0078 | holds |
| torso lean | vertical | −0.053 | −0.056 ± 0.089 [−0.197, +0.112] | 7/8 | 0.141 | undetermined |
| valgus | ground plane | −0.057 | −0.058 ± 0.119 [−0.250, +0.107] | 6/8 | 0.195 | undetermined |

Per action, RTMPose → NLF flip, subjects NLF better / worse, uncorrected exact p:

| cue | squat | deadlift | thruster |
| --- | --- | --- | --- |
| knee angle | 0.200 → 0.115, 6/1, 0.047 | 0.191 → 0.101, 8/0, 0.008 | 0.241 → 0.089, 8/0, 0.008 |
| hip angle | 0.207 → 0.061, 7/0, 0.016 | 0.219 → 0.110, 7/1, 0.023 | 0.171 → 0.043, 8/0, 0.008 |
| depth ratio | 0.188 → 0.126, 5/2, 0.078 | 0.246 → 0.124, 8/0, 0.008 | 0.166 → 0.094, 6/2, 0.031 |
| torso lean | 0.248 → 0.194, 5/3, 0.367 | 0.134 → 0.100, 5/3, 0.414 | 0.202 → 0.131, 6/2, 0.117 |
| valgus | 0.299 → 0.294, 3/3, 0.969 | 0.255 → 0.174, 7/1, 0.117 | 0.278 → 0.192, 6/2, 0.086 |

NLF's pooled flip is lower in all 15 cells. 8 of 15 per-action p-values are below 0.05,
all on knee, hip and depth ratio. None of the torso lean or valgus cells is.

Per camera (4 cameras × 15 cells = 60, about 5 reps per subject per camera, descriptive
only), 3 cells favour RTMPose: squat valgus on camera 60457274 (0.31 vs 0.42) and on
50591643 (0.26 vs 0.27), and deadlift torso lean on 60457274 (0.11 vs 0.14).

## Secondary: where experiment 3's valgus gap came from

Valgus only, each pairing tested the same way. Δ = flip(second arm) − flip(first arm).

| pairing | squat | deadlift | thruster | combined Δ, p |
| --- | --- | --- | --- | --- |
| perfect 2D → NLF, full rate (experiment 3) | 0.147 → 0.310, p = 0.031 | 0.089 → 0.174, p = 0.219 | 0.087 → 0.200, p = 0.117 | +0.120, 0.0078 |
| perfect 2D → RTMPose, 15-frame grid (detector error) | 0.134 → 0.299, p = 0.062 | 0.094 → 0.255, p = 0.047 | 0.082 → 0.278, p = 0.039 | +0.174, 0.047 |
| RTMPose → NLF, 15-frame grid (parity) | 0.299 → 0.294, p = 0.969 | 0.255 → 0.174, p = 0.117 | 0.278 → 0.192, p = 0.086 | −0.057, 0.195 |

NLF is worse than a zero-error 2D arm, and that difference is determined. A real detector
adds 0.174 of flip over the same zero-error arm, more than the 0.120 NLF adds. The
detector-error p of 0.047 would not survive correction across this note's tests. Read it
as a size comparison, not as a separate finding.

## Secondary: the squat depth verdict at the 90° threshold

Knee angle > 90° at the rep bottom means "did not reach parallel". This is the one cue here
with a coaching threshold. All arms and the truth are on the 15-frame grid, where
true-fault prevalence is 10% (7.5% at full rate). Descriptive, no test.

| arm | raw flip / false alarm / miss | calibrated flip / false alarm / miss |
| --- | --- | --- |
| zero-error 2D | 74% / 83% / 0% | 16% / 12% / 44% |
| RTMPose | 64% / 71% / 0% | 21% / 19% / 38% |
| NLF | 9% / 0% / 88% | 6% / 6% / 0% |

Per-subject calibrated flip: RTMPose 21 ± 18%, NLF 6 ± 10%. At full rate the zero-error
arm and NLF reproduce experiment 3's box exactly (raw 76% / 82% / 0%, calibrated 16% /
14% / 42%; NLF calibrated 7% / 7% / 0%). Thruster (prevalence 30%): RTMPose raw false
alarm 99%, calibrated flip 22% (21% false alarm, 25% miss), NLF calibrated 10%.

## Not supported

- **Front or side cameras.** Every Fit3D camera is oblique. On REHAB24 at a side view, a
  zero-error 2D arm beats NLF on the vertical-referenced cues (squat hip 0.024 vs 0.081,
  torso 0.011 vs 0.115, multi-model REHAB24 section of `camera-placement-hypothesis.md`).
  No real 2D detector has
  been compared with NLF at that view.
- **A deployed NLF.** The 3D arm is rotated by the GT camera rotation. Knee and hip angle
  do not depend on it. Depth ratio, torso lean and valgus do, and the size of that
  dependence was not measured.
- **MediaPipe 2D.** The Fit3D MediaPipe predictions hold world landmarks only
  (`joints_cam` in `data/Fit3D/derived/preds/mediapipe/`), so no MediaPipe 2D arm exists
  here.
- **Equivalence on torso lean or valgus.** p = 0.141 and 0.195 are undetermined. With
  n = 8, p < 0.05 needs about 7 of 8 subjects agreeing with no large opposite case.
- **Fault detection at a coaching threshold.** Swept flip measures ranking over the
  20th–80th percentile band of a competent population. The valgus median split sits 0.07
  SD from the squat truth mean (`camera-placement-hypothesis.md`).
- **Within-person valgus reading.** The pooled flip absorbs between-subject anatomy. With
  per-subject thresholds both arms flip 0.34–0.58 on valgus (`camera-placement-hypothesis.md`).
- **Correctness labels.** Verdicts are scored against mocap truth, not human labels.
- **Other 3D models.** Only NLF was run against RTMPose.

## Reproduce

Script `notes/exploratory/fit3d_parity_significance_20260926/parity_significance.py`,
run from the repo root. It imports the cue formulas and debiasing from
`src/fit3d/decision_eval.py` and `src/fit3d/biomech.py`.

```powershell
.venv\Scripts\python.exe notes\exploratory\fit3d_parity_significance_20260926\parity_significance.py collect
.venv\Scripts\python.exe notes\exploratory\fit3d_parity_significance_20260926\parity_significance.py analyse
```

CPU only; `collect` takes about 20 s, `analyse` a few seconds. Inputs are earlier
extractions on disk: NLF in `data/Fit3D/derived/preds/nlf/`
(`fit3d_depth_recovery_summary.md`) and RTMPose in `data/Fit3D/derived/preds/rtmpose/`
(`fit3d_2d_vs_3d_summary.md`), over the Fit3D train split. Outputs are
`data/Fit3D/derived/parity_significance_records.json` (per rep × camera readings) and
`data/Fit3D/derived/parity_significance.json` (all statistics, per subject and per
camera). `data/` is gitignored.
