# Late fusion of NLF pose and VideoMAE scores 0.696 vs 0.675 LOSO balanced accuracy on REHAB24-6 (undetermined), and ranks repetitions within a session 0.053 AUC higher

*REHAB24-6 · pre-registered late-fusion test · run 2026-09-20 ·
[plan](rehab24_nlf_videomae_late_fusion_validation_plan.md)*

**Result.** Calibrated late fusion of the NLF pose classifier and the frozen VideoMAE
classifier scored 0.6960 ± 0.0622 held-out-subject balanced accuracy against 0.6751 ± 0.0497
for pose alone. The paired delta is +0.0209 ± 0.0549 with 5 of 9 subjects positive,
bootstrap 95% CI [−0.0128, +0.0565], exact sign-flip p = 0.3242. By the registered rule
this is **undetermined**, no fusion gain is claimed, and the stop rule applies: no further
fusion variant is run on REHAB24-6 from this line. The fused arm scores below the
one-parameter position rule (0.7309 ± 0.0481), as both branches do.

**Caveat on reading it.** Two secondaries pass Holm correction: fusion beats the video
branch (+0.0342, 8/9, Holm p = 0.0352) and ranks repetitions inside a session better than
pose (within-session AUC 0.9029 vs 0.8498, 8/9, Holm p = 0.0312). Neither replaces the
primary. The position-balanced contrast, the one that excludes repetition order, is
+0.0652 with 8/9 positive but Holm p = 0.0625, which is also undetermined.

## Background

No fusion arm registered before its result had beaten its stronger branch in this project,
and late fusion had never been run on REHAB24-6. The
[identity/appearance plan](rehab24_videomae_identity_appearance_validation_plan.md) allowed
one fusion pre-registration once its two conditions held (within-session AUC
0.8741 ± 0.0408 against a floor of 0.55; appearance-only control 0.5241 ± 0.0185 against a
ceiling of 0.55) and the [position control](rehab24_videomae_position_control_results.en.md)
closed the linear position path. This is that experiment.

## Terminology

| Term | Meaning |
| --- | --- |
| Sample | One repetition seen by one camera; 2,144 samples, 1,072 repetitions |
| LOSO | Leave-one-subject-out; the held-out person never fits a readout, calibrator or threshold |
| `nlf` | Pose branch: MLP readout on NLF parametric 3D plus projected 2D skeleton features, 2160 dimensions |
| `vm16` | Video branch: the same MLP readout on frozen VideoMAE features, whole-frame letterbox, 4 clips × 16 frames, 768 dimensions |
| `fused` | Unweighted mean of the two branches' Platt-calibrated probabilities |
| `nlf_cal` | The fusion rule with all weight on `nlf`: calibrated and re-thresholded, no video |
| Within-session AUC | ROC-AUC inside one exercise recording, so differences between people cannot contribute |
| Interleaved session | A session where a correct repetition precedes an incorrect one and the reverse also occurs; 34 of the 61 mixed-label sessions |
| Position-balanced AUC | Within-session AUC with correct-first and incorrect-first pairs weighted 0.5 each; repetition order alone scores exactly 0.5 |

## Method

Both branches were refit on the frozen folds of the
[V-JEPA 2 comparison](rehab24_videomae_vjepa2_results.md): ten LOSO folds, validation
subject the next eligible subject in cyclic order, seeds 42, 7 and 1234, the shared MLP
recipe, subject P10 (16 samples) kept in training and excluded from every primary
statistic. The trainer was extended only to export the validation subject's probabilities
from the best checkpoint.

Per fold and seed, each branch was Platt-calibrated on that fold's validation subject, the
two calibrated probabilities were averaged with equal weight, and the decision threshold
was chosen on the validation subject's fused probabilities. Seeds were paired one to one.
The held-out subject was scored once. The design decision a reader could get wrong: the
validation subject is a third person per fold, never pooled across folds, because the
fold-to-validation assignment is not a partition (P1 validates two folds).

The endpoint is balanced accuracy per subject, averaged over the three seeds, then over
P1–P9 with equal weight. The sole primary contrast is `fused − nlf`, tested with an exact
paired sign-flip test over 512 assignments and a subject bootstrap (10,000 resamples). A
gain needed all of: p < 0.05, delta ≥ 0.02, at least 7/9 subjects positive, a CI excluding
zero, and a fused mean no lower than the `vm16` and `nlf_cal` means. Everything above was
fixed and committed before any fit.

## Gates

All eight blocking gates passed before the report was opened.

| Gate | Result |
| --- | --- |
| Frozen protocol | pass; plan and code at commit `adb1cfd0`, clean tree for every fit, CPU, fold and manifest hashes equal the registered values |
| Feature integrity | pass; 2,144 of 2,144 sample IDs in both feature directories by ID diff, dimensions 2160 and 768 |
| Inert trainer change | pass; 60 of 60 cells (2 branches × 10 folds × 3 seeds) bit-identical between the trainer at commit `6a9338e2` and the exporting trainer, probabilities and thresholds. Non-blocking check: all 30 `vm16` cells bit-identical to the stored 2026-09-18 run |
| Validation export integrity | pass; thresholds recomputed from exported validation probabilities equal the stored ones |
| Fold purity | pass; the rows passed to each Platt fit and threshold search belong to the fold's validation subject only |
| Degenerate-share identity | pass; bitwise equality at shares 1.0 and 0.0, zero rank inversions in 120 rankings, 15 rows in 11 cells outside the calibrator's clip range, no saturation ties, no non-positive slope |
| Fused OOF integrity | pass; one fused prediction per sample and seed, branches aligned by sample ID |
| Position reproduction | pass; position rule 0.7309, 61 mixed-label and 34 interleaved sessions, zero unmatched rows |

The uninformative-branch control did not reach its audit trigger (mean −0.0009 against
+0.01), so the leak audit did not run.

## Results

LOSO balanced accuracy, P1–P9, three seeds averaged within subject.

| Arm | Mean ± SD | Delta vs `nlf` | Positive | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| `nlf` | 0.6751 ± 0.0497 | — | — | — | — |
| `vm16` | 0.6618 ± 0.0506 | — | — | — | — |
| `nlf_cal` | 0.6751 ± 0.0507 | 0.0000 ± 0.0014 | 2/9 | — | not tested |
| **`fused`** | **0.6960 ± 0.0622** | **+0.0209 ± 0.0549** | **5/9** | **[−0.0128, +0.0565]** | **0.3242** |
| Position rule (reference) | 0.7309 ± 0.0481 | — | — | — | never paired |

| Subject | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nlf` | 0.6975 | 0.5908 | 0.6434 | 0.6899 | 0.6250 | 0.7190 | 0.6635 | 0.7527 | 0.6937 |
| `vm16` | 0.5870 | 0.7060 | 0.6232 | 0.6173 | 0.6340 | 0.6770 | 0.6727 | 0.7486 | 0.6899 |
| `fused` | 0.6441 | 0.7161 | 0.6420 | 0.6431 | 0.6215 | 0.7727 | 0.7041 | 0.7951 | 0.7253 |
| `fused − nlf` | −0.0534 | +0.1254 | −0.0014 | −0.0468 | −0.0035 | +0.0537 | +0.0406 | +0.0423 | +0.0316 |

Verdict: row 4 of the registered table, undetermined. Of the claim conditions, the margin
and both mean guards held; p < 0.05, 7/9 positive and a CI excluding zero did not. The mean
delta sits on the 0.02 margin but is carried by P2 (+0.1254), where pose alone is weakest;
P1 and P4 lose about 0.05. Re-thresholding contributed nothing (`nlf_cal − nlf` = 0.0000 ± 0.0014),
so the primary and S2 deltas coincide.

## Secondary

One Holm family of four paired sign-flip tests over P1–P9.

| ID | Contrast | Baseline | Fused | Delta ± SD | Positive | 95% CI | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | `fused − vm16`, balanced accuracy | 0.6618 ± 0.0506 | 0.6960 ± 0.0622 | +0.0342 ± 0.0307 | 8/9 | [+0.0163, +0.0540] | 0.0117 | **0.0352** |
| S2 | `fused − nlf_cal`, balanced accuracy | 0.6751 ± 0.0507 | 0.6960 ± 0.0622 | +0.0209 ± 0.0554 | 5/9 | [−0.0128, +0.0569] | 0.3203 | 0.3203, undetermined |
| S3 | `fused − nlf`, within-session AUC, 61 sessions | 0.8498 ± 0.0595 | 0.9029 ± 0.0366 | +0.0531 ± 0.0454 | 8/9 | [+0.0274, +0.0848] | 0.0078 | **0.0312** |
| S4 | `fused − nlf`, position-balanced AUC, 34 sessions | 0.7978 ± 0.0729 | 0.8631 ± 0.0762 | +0.0652 ± 0.0730 | 8/9 | [+0.0192, +0.1107] | 0.0312 | 0.0625, undetermined |

The fused arm orders repetitions inside a session better than pose alone, and the
difference does not show up as a thresholded cross-subject gain. The two statistics
measure different things: within-session AUC needs no threshold and no comparability
between people; balanced accuracy needs one threshold, chosen on a third person, to
transfer to the held-out person.

Descriptive, not tested:

| Quantity | Value |
| --- | --- |
| `nlf` within-session AUC (first measurement for a pose branch) | 0.8498 ± 0.0595 |
| `vm16` within-session AUC | 0.8750 ± 0.0482; `vm16 − nlf` +0.0252 ± 0.0596, 7/9 positive |
| `nlf` / `vm16` position-balanced AUC | 0.7978 ± 0.0729 / 0.8379 ± 0.0967; `vm16 − nlf` +0.0401 ± 0.0764, 8/9 positive |
| Recall / specificity | `nlf` 0.6789 / 0.6712; `vm16` 0.7077 / 0.6158; `fused` 0.7448 / 0.6472 |
| Macro-F1 / pooled ROC-AUC | `nlf` 0.6625 / 0.7264; `vm16` 0.6465 / 0.7228; `fused` 0.6828 / 0.7411 |
| Platt slopes | 0 negative of 60 per branch |
| Held-out Brier, raw → calibrated | `nlf` 0.2201 → 0.2185; `vm16` 0.2174 → 0.2246 |
| Validation-tuned weight | 0.6779 ± 0.0606; delta vs `nlf` +0.0029 ± 0.0625, 4/9 positive. Chosen pose shares spread over the whole grid (0.0: 1, 0.1: 3, 0.2: 1, 0.3: 3, 0.4: 2, 0.5: 5, 0.6: 5, 0.7: 1, 0.8: 1, 0.9: 3, 1.0: 2 of 30) |
| P10 removed from training | `nlf` 0.6812, `vm16` 0.6703, `fused` 0.6948 |
| P10 as held-out subject (16 samples) | `nlf` 0.8167, `vm16` 0.7611, `fused` 0.6167 |

Balanced accuracy by exercise and camera (pooled over subjects and seeds):

| Arm | Ex1 | Ex2 | Ex3 | Ex4 | Ex5 | Ex6 | cam17 | cam18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nlf` | 0.675 | 0.656 | 0.718 | 0.727 | 0.522 | 0.699 | 0.676 | 0.667 |
| `vm16` | 0.706 | 0.603 | 0.664 | 0.654 | 0.676 | 0.633 | 0.634 | 0.690 |
| `fused` | 0.725 | 0.664 | 0.712 | 0.731 | 0.611 | 0.656 | 0.682 | 0.700 |

The weight tuned on the validation subject scored below the fixed equal weight (0.6779
against 0.6960), and the chosen shares covered the whole grid.

## Position control

| Reading | Value |
| --- | --- |
| Position rule, LOSO balanced accuracy (one fitted threshold on the repetition ordinal) | 0.7309 ± 0.0481, 9/9; `fused` at 0.6960 is below it |
| S4, `fused − nlf` position-balanced AUC on the 34 interleaved sessions | +0.0652 ± 0.0730, 8/9, Holm p = 0.0625, undetermined |
| Median Spearman of calibrated probability against within-class position (120 session × class cells with at least 3 repetitions; median over subjects) | `nlf` −0.1905, `vm16` +0.0242, `fused` −0.1076 |

The plan's position table applies only when a gain is claimed, so S4 changes nothing here.
All nine subjects have interleaved sessions (4, 2, 3, 4, 5, 3, 5, 4, 4). Inside a label
class the `nlf` probability correlates negatively with position in 8 of 9 subjects (P5
−0.50, P7 −0.66; P4 +0.40), so a within-class drift is present in the pose branch too.

## Uninformative-branch control

Replacing `vm16` with a repetition-level permutation of itself and rerunning the rule:

| Permutation seed / model seed | `fused_perm − nlf_cal` | Positive |
| --- | --- | --- |
| 101 / 42 | −0.0084 ± 0.0250 | 4/9 |
| 202 / 7 | +0.0039 ± 0.0127 | 6/9 |
| 303 / 1234 | +0.0019 ± 0.0163 | 5/9 |
| Mean | −0.0009 | trigger +0.01 not reached |

Averaging with a branch that carries no information left the pose score within 0.01 in
all three runs. The control gives no sign that the procedure alone raises the score.

## Deviations from the plan

| Item | What the plan said | What happened | What it changes |
| --- | --- | --- | --- |
| Four amendments dated 2026-09-20 (clip-range handling in the degenerate-share gate, A/B against the trainer at `6a9338e2`, fixed leak-audit procedure, implementation smoke) | Registered in the plan's own deviations table | Made after the plan commit and before `init`; no fused score or full-cohort branch score existed | Nothing; the audit was never triggered and the clip rule touched 15 rows with zero inversions |
| Historical-baseline check | Never blocks; the stock seed-42 rerun is compared with the stored NLF summary | Balanced accuracy identical in all folds; thresholds differ in the seventh decimal in folds 4, 6, 7 and 9, so the check reports "not reproduced" | Nothing; logged in `deviations.json` |
| Unregistered report fields | Not mentioned | `summary.json` carries `null_check` and `padding_strata` placeholders marked pending, inherited from the reused reporter | Nothing; they are not part of this plan |

## Not supported

- That pose and video are redundant, or that fusion does not help. p = 0.3242 on nine
  subjects is undetermined; the CI runs from −0.013 to +0.057.
- That fusion helps cross-subject classification. The mean delta reaches the margin on one
  subject's +0.1254; 5/9 is not a consistent gain.
- That the S1 and S3 results show movement information added by video. S3 includes the 27
  sessions where label and position are collinear, and S4, which removes repetition order,
  is undetermined after Holm correction.
- Any revision of the identity, appearance or position-control results.
- Deployable accuracy. Every arm is below a one-parameter position rule that uses no
  pixels and no pose.
- Anything about other fusion rules, pose backbones, checkpoints or fine-tuned encoders,
  which the stop rule now leaves unrun on this dataset.
- A larger sample size from counting seeds, cameras or repetitions; n = 9.

## Reproduce

Code: `src/rehab24/pose_video_fusion.py`, `src/rehab24/loso_cross_validation.py`
(`train_one_fold(return_validation=True)`), CLI `scripts/rehab24/run_pose_video_fusion.py`,
tests `tests/test_rehab24_pose_video_fusion.py`. Run at commit `adb1cfd0` on the local CPU;
1–3 s per MLP fit, 120 A/B fits and 120 registered fits. Total wall time was not recorded.

```powershell
$RUN = "20260920_nlf_vm16_late_fusion"
$D = "data/REHAB24-6/processed/pose_video_fusion/$RUN"
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py init --run-id $RUN
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py features --run-id $RUN
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py ab-check --run-id $RUN
.venv\Scripts\python.exe -m src.rehab24.loso_cross_validation --feature-dir data/REHAB24-6/processed/nlf_parametric_3d2d_skeleton_features --seed 42 --device cpu --summary-output $D/historical_nlf_seed42.json
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py baseline-check --run-id $RUN --rerun-summary $D/historical_nlf_seed42.json
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py fit --run-id $RUN --arm nlf
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py fit --run-id $RUN --arm vm16
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py fit --run-id $RUN --arm nlf --no-p10-training
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py fit --run-id $RUN --arm vm16 --no-p10-training
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py fuse --run-id $RUN
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py fuse --run-id $RUN --no-p10-training
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py gates --run-id $RUN
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py report --run-id $RUN
```

Artifacts under `data/REHAB24-6/processed/pose_video_fusion/20260920_nlf_vm16_late_fusion/`:
`run_config.json`, `folds.json`, `feature_integrity.json`, `ab_check.json`,
`oof/<arm>/seed<seed>.csv`, `oof_val/<arm>/seed<seed>.csv`, `calibration.json`,
`leak_audit.json`, `gates.json`, `summary.json`, `deviations.json`,
`historical_baseline_check.json`. A rerun needs a fresh run id: the CLI refuses when HEAD
differs from the commit `init` recorded.
