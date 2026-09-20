# Plan: test whether calibrated late fusion of NLF pose and VideoMAE beats NLF alone on REHAB24-6

*REHAB24-6 · prospective pre-registration · written 2026-09-20 · status: plan only ·
opened under the fusion gate of the [identity/appearance plan](rehab24_videomae_identity_appearance_validation_plan.md)*

**Goal.** Test whether one fixed late-fusion rule, combining a pose classifier built on
NLF skeleton features with a frozen VideoMAE classifier, improves held-out-subject
exercise-correctness classification over the pose classifier alone. One fusion rule, one
primary contrast, and a registered stop rule. Because REHAB24-6 labels follow recording
position, the plan also registers a position control that any fusion gain is read against.

**Status.** No fused prediction, validation probability or fusion score has been produced.
Both feature sets already exist on disk, so no GPU extraction is needed. The core run is
60 small CPU readout fits plus offline fusion; the trainer A/B check and the P10-removed
sensitivity add refits of the same size. Commit this plan and the frozen constants before
any fit. All gates below are pending.

## Background

No fusion arm that was registered before its result has beaten its stronger branch in this
project. On REHAB24-6, early concatenation of NLF and VideoMAE scored 0.657 against NLF
alone at 0.668 ([Stage A results](videomae_stage_a_results.md), seed 42 only). On
Fitness-AQA, the registered calibrated late fusion scored 0.648 ± 0.008 against pose-only
0.650 ([Stage B results](videomae_stage_b_results.md)). Two unregistered late fusions on
Fitness-AQA did exceed both branches on one fixed split: pose with a person-cropped
VideoMAE arm at 0.682 ± 0.016 (delta +0.032, CI [−0.018, +0.085]), and pose with a
Something-Something-v2 checkpoint at 0.702 (delta +0.052, CI [−0.002, +0.108],
[checkpoint results](videomae_ssv2_checkpoint_results.md)). Neither interval excludes
zero. Late fusion has never been run on REHAB24-6.

The [identity/appearance plan](rehab24_videomae_identity_appearance_validation_plan.md)
blocked further REHAB24-6 fusion until two conditions held, and both did
([results](rehab24_videomae_identity_appearance_results.md)): within-session ROC-AUC
0.8741 ± 0.0408 (9/9 subjects above 0.5, permutation p = 1/10001) against a floor of 0.55,
and an appearance-only control at 0.5241 ± 0.0185 against a shortcut ceiling of 0.55. The
[position control](rehab24_videomae_position_control_results.en.md) then closed the linear
position path: projecting out 16 position-predictive directions moved the AUC to
0.8556 ± 0.0435 (share 4.9%, a lower bound because subjects P4 and P8 were not cleaned).
That gate was read from a secondary (k = 16) after the k = 1 primary failed its positive
control. The non-linear position path is untested.

The same plan fixed what a fusion pre-registration may contain: calibrated late fusion of
NLF and the `full_frame_letterbox` VideoMAE arm only, validation-only calibration, a single
fusion rule, the subject as the pairing unit, and a stop if NLF is not improved on. It also
ruled that the identity and appearance results may not be overwritten by a higher fusion
score; that rule carries over here and extends to the position-control results. The
identity results note added a recommendation, not a rule: lock a position control before
the run. This plan adopts it.

## Terminology

| Term | Meaning |
| --- | --- |
| Repetition | One annotated execution with a binary correct/incorrect label; correct is positive |
| Sample | One repetition seen by one camera (cam17 or cam18); 2,144 samples, 1,072 repetitions |
| Session | One exercise in one recording (`exercise_id` plus `video_id`) |
| LOSO | Leave-one-subject-out; the held-out person never fits the readout, calibrator or threshold |
| OOF | Out-of-fold prediction made while that sample's subject is held out |
| Branch | One single-modality classifier whose probabilities enter the fusion |
| Late fusion | Combining branch probabilities after each branch is trained separately |
| Platt scaling | Two-parameter logistic map `sigmoid(a·logit(p) + b)` fit on validation data |
| Within-session AUC | ROC-AUC computed inside one session, so between-person differences cannot contribute |
| Interleaved session | A session where a correct repetition precedes an incorrect one and the reverse also occurs; 34 of the 61 mixed-label sessions |
| Position-balanced AUC | Within-session AUC with correct-first and incorrect-first pairs weighted 0.5 each; repetition order alone scores exactly 0.5 on it |

## Branches and arms

| Arm | Input | Dimension | Role |
| --- | --- | --- | --- |
| `nlf` | `data/REHAB24-6/processed/nlf_parametric_3d2d_skeleton_features` (NLF parametric body model, 3D plus projected 2D joints) | 2160 | Baseline; primary comparator |
| `vm16` | `data/REHAB24-6/processed/video_backbone_comparison/20260918_vjepa2_vs_videomae/features/vm16` (VideoMAE Kinetics checkpoint, whole-frame letterbox, 4 clips × 16 frames) | 768 | Video branch |
| `fused` | Platt-calibrated `nlf` and `vm16` probabilities, unweighted mean | — | Primary challenger |
| `nlf_cal` | The fusion rule at `nlf` share 1.0: `nlf` calibrated and re-thresholded, no video | — | Isolates what calibration and re-thresholding do without a second branch |

Known single-branch scores, for context only: `nlf` 0.668 ± 0.044 LOSO balanced accuracy
on seed 42 alone ([correctness summary](rehab24_correctness_experiment_summary.md));
`vm16` 0.6618 ± 0.0506 over three seeds, within-session AUC 0.8750 ± 0.0482
([V-JEPA 2 comparison](rehab24_videomae_vjepa2_results.md)). The two were never run on the
same seeds or sample order, so which branch is stronger is not yet known.

`vm16` replaces the historical `full_frame_letterbox` features named by the identity plan.
It has the same checkpoint, geometry and pooling, re-extracted with a sampler that keeps
every frame inside the annotated repetition; the historical sampler could read past the
repetition end. The V-JEPA 2 comparison found the two byte-identical on 2,140 of the 2,142
samples that need no padding. This substitution is registered here, before any fusion
number exists. The historical features are not run.

`nlf_cal` exists because the threshold search is not invariant under a monotone map: it
mixes a fixed 0.1–0.9 grid with sample-derived candidates, casts them to float32 and keeps
the 0.5 default on ties. Re-thresholding calibrated probabilities therefore moves
held-out balanced accuracy by itself. A stand-in simulation on stored `vm16` OOF rows
changed held-out predictions in 10 of 27 folds, by up to 0.0204 balanced accuracy in one
fold. `fused − nlf_cal` is the contrast in which only the video branch differs.

No other NLF variant (`nlf_parametric_3d`, `nlf_nonparam_3d2d`), pose backbone, VideoMAE
checkpoint, V-JEPA 2 arm, early concatenation, gating or stacked meta-classifier is part
of this experiment.

## Dataset and folds

Reuse `folds.json` from the `20260918_vjepa2_vs_videomae` run verbatim and assert before
any fit:

| File | Hash |
| --- | --- |
| `folds.json` | SHA-256 `01ef289c38d680633f67180639d0e566d3bcdaeacd261b2c482d275d6de3a730` |
| `folds_no_p10.json` | SHA-256 `8a8bc9ebc5b8cb411e8554b9cb4843c418aaf53e06f91fa8fd42ab8fe36e16d9` |
| manifest | MD5 `b27e9069b3cc9a2906d57e446630abb1` |

Ten LOSO folds; the validation subject is the next subject in numeric cyclic order with at
least 100 samples. The frozen assignment is test→validation 1→2, 2→3, 3→4, 4→5, 5→6, 6→7,
7→8, 8→9, 9→1, 10→1. Subject P10 (16 samples) stays in training sets, is never a
validation subject, and is excluded from every primary statistic. Sample IDs inside each
fold are used in the stored (sorted) order for both branches.

The validation assignment is not a partition: P1 validates two folds and P10 none. Every
fitted quantity below is therefore fit per fold. Validation rows are never pooled across
folds.

No exclusions. Both feature directories must cover exactly the manifest's sample IDs. Any
missing sample blocks the run; it is not resolved by dropping the sample from one branch.

## Branch readouts

Both branches use the existing `loso_cross_validation.train_one_fold` with the shared
`FoldConfig`: MLP hidden width 128, dropout 0.4, AdamW, learning rate 0.0003, weight decay
0.01, batch size 32, at most 20 epochs, patience 5, positive-class weight from training
labels, per-dimension standardization from training subjects, checkpoint and threshold
selected on validation balanced accuracy. Seeds **42, 7, 1234**. No hyperparameter is
changed for either branch.

The one code change is an export: `train_one_fold` currently discards validation
probabilities. It will additionally return the validation subject's per-sample
probabilities from the restored best checkpoint, not from the last epoch. The change must
leave test probabilities untouched; the inert-change gate below checks this for both arms.

## Fusion rule

Fixed before any result, identical to the rule Stage B registered:

1. Per fold and seed, fit Platt scaling for each branch on that fold's validation subject
   with `src.video.late_fusion.fit_platt` (Platt target smoothing, Newton solver). The
   slope is not constrained or clipped.
2. Fused probability is the unweighted mean of the two calibrated probabilities
   (`fuse_probabilities`, share 0.5).
3. The fused decision threshold maximises balanced accuracy on the same validation
   subject's fused probabilities, using `find_best_threshold` unchanged.
4. Pair seeds one to one: `nlf` seed s with `vm16` seed s. Three fused runs, not a 3 × 3 grid.
5. Score the held-out subject once.

The validation subject is used four times per fold (checkpoint, branch threshold, Platt
fit, fused threshold). Both branches share this, and the held-out subject is never
involved, but fused validation metrics are optimistic and are not reported as evidence.

## Primary endpoint and inference

For each subject and seed, balanced accuracy (mean of correct-class and incorrect-class
recall) over that subject's samples at the fold's validation-selected threshold. Average
the three seeds within each subject. The endpoint is the equally weighted mean over P1–P9.
The sole primary contrast is **`fused − nlf`**, where `nlf` is the branch as the trainer
scores it, with its own threshold.

Report each arm's mean ± between-subject SD, the paired delta mean ± SD, all nine subject
deltas, the number of positive deltas, a paired subject-bootstrap 95% CI and an exact
two-sided paired sign-flip test over all 512 sign assignments. Both come from
`video_backbone_comparison.contrast_summary` unchanged, which fixes 10,000 resamples and
bootstrap seed 20260918 and resamples subjects with all seeds and cameras intact. Samples,
cameras, repetitions and seeds do not increase the inferential sample size of nine.

The practical margin is **0.02 balanced accuracy**, the same project decision margin used
in Stage B and the V-JEPA 2 comparison. It is not a clinically derived effect size.

The first matching row applies:

| # | Condition | Interpretation | Next step |
| --- | --- | --- | --- |
| 0 | Any gate fails | No valid comparison | Repair and document before reading outcomes |
| 1 | p < 0.05, delta > 0, **and** delta ≥ 0.02, at least 7/9 subjects positive, the CI excludes zero, fused mean ≥ `vm16` mean, fused mean ≥ `nlf_cal` mean | **Gain claimed**: the fusion beats NLF and is not below either single-branch arm under this rule | Read the position control before describing what the video branch contributes |
| 2 | p < 0.05, delta > 0, any other condition of row 1 fails | A detectable positive difference that does not meet the claim conditions; name the failed conditions; no gain claimed | Stop rule applies |
| 3 | p < 0.05, delta < 0 | Adding the video branch degrades the pose classifier under this rule | Stop rule applies |
| 4 | p ≥ 0.05 | **Undetermined.** No claim that the branches are redundant | Stop rule applies |

The two mean conditions in row 1 are point-estimate guards, not tests. They keep the
primary comparator fixed at NLF, as the identity plan requires, while preventing a gain
over NLF from being reported as fusion when the video branch alone, or re-thresholding
alone, scores as high.

**Stop rule.** Unless row 1 applies, no further fusion variant is run on REHAB24-6 from
this line: no tuned weights, gating, stacking, early concatenation, other pose branch or
other video checkpoint. A later fusion experiment needs a new pre-registration whose
rationale does not come from this run's held-out scores.

**Precision.** With nine subjects the smallest attainable two-sided sign-flip p is 2/512.
In the last paired comparison on these folds a mean delta of −0.0395 (CI [−0.0904,
−0.0037]) gave p = 0.0664 and was read as undetermined. A true gain near the 0.02 margin
is likely to be undetermined here unless the subject deltas are unusually consistent.

## Position control

In 27 of the 61 mixed-label sessions all correct repetitions precede all incorrect ones or
the reverse. Repetition order alone reaches within-session AUC 0.8239, and a rule with one
fitted threshold on the repetition ordinal scores LOSO balanced accuracy 0.7309 ± 0.0481
(9/9). A video branch can carry recording position through scene or lighting drift; a
fusion gain could therefore be position, not movement. Three registered readings:

1. **Position-balanced AUC on the 34 interleaved sessions** (secondary S4 below), for
   `nlf`, `vm16` and `fused`. All nine primary subjects have interleaved sessions (P1–P9:
   4, 2, 3, 4, 5, 3, 5, 4, 4). Inputs are locked: camera probabilities averaged per
   repetition and seed by `_camera_averaged_repetition_scores`; sessions built by
   `videomae_position_control.build_pair_sessions` with P10 dropped; pair concordances
   averaged over the three seeds; `position_balanced_auc` with `max_distance=None`;
   `interleaved_mask` selects the sessions; session values averaged within subject, then
   subjects equally. OOF rows are joined to repetitions by `sample_id`. The session key
   format differs between modules and the position helpers skip unmatched keys silently,
   so the join must end with exactly 34 sessions per arm and zero unmatched rows.
2. **Position rule as a reference row.** `position_rule_loso` (0.7309 ± 0.0481) is printed
   beside every arm in the main table. It has one fitted parameter, pools repetitions
   rather than samples and was never paired against a model, so it is a reference, not a
   contrast. If `fused` scores below it, the results note must say so in its Result block.
3. **Probability-versus-position diagnostic** (descriptive). The cell is one label class
   inside one session, with at least 3 repetitions, as in `probe_cells`. In each cell,
   Spearman correlation between the arm's Platt-calibrated probability, averaged over
   cameras and then seeds, and within-class position from `build_repetitions`: the rank of
   a repetition among same-label repetitions of its session, scaled to [0, 1]. The label
   is constant inside a cell, so a correlation there is not explained by correctness.
   Report the median over cells per subject, and its median over subjects, for `nlf`,
   `vm16` and `fused`.

| Primary | S4 (`fused − nlf`, position-balanced) | Reading |
| --- | --- | --- |
| Row 1, gain claimed | Positive, Holm p < 0.05 | The gain is present where repetition order carries no information |
| Row 1, gain claimed | Anything else | The gain is not separated from recording position; it may not be described as movement information added by video |
| Rows 2–4 | Any | S4 is reported but changes nothing |

## Secondary analyses

One Holm-corrected family of four paired subject sign-flip tests, all over P1–P9:

| ID | Contrast | Statistic |
| --- | --- | --- |
| S1 | `fused − vm16` | LOSO balanced accuracy, as the primary |
| S2 | `fused − nlf_cal` | LOSO balanced accuracy; both arms pass through the same calibration and threshold code |
| S3 | `fused − nlf` | Within-session ROC-AUC over the 61 mixed-label sessions: camera probabilities averaged per repetition, ranks averaged over seeds, sessions within subject, subjects equally (`within_session_auc_statistic`) |
| S4 | `fused − nlf` | Position-balanced AUC on the 34 interleaved sessions, inputs as locked above |

No secondary result replaces the primary.

Descriptive only, never tested and never quoted as a headline:

- `nlf_cal − nlf`, the re-thresholding component of the primary delta, with all nine
  subject values.
- `nlf` within-session AUC and position-balanced AUC, which have not been measured for any
  pose branch, and `vm16 − nlf` on both.
- Recall, specificity, macro-F1, pooled ROC-AUC, per-exercise and per-camera balanced
  accuracy for strata with at least 20 samples and both labels; undefined cells are marked,
  not scored zero.
- Platt slope and intercept per branch, fold and seed, with the count of negative slopes;
  Brier score before and after calibration on the held-out subject.
- **Validation-tuned weight.** The `nlf` share chosen per fold and seed from
  {0, 0.1, …, 1.0} by validation balanced accuracy; ties go to the share nearest 0.5, then
  to the larger `nlf` share. It shows whether equal weighting was the binding constraint.
  It does not reopen the stop rule.
- **Uninformative-branch control.** Three runs, pairing permutation seed 101 with model
  seed 42, 202 with 7 and 303 with 1234. In each, `vm16` raw probabilities are permuted
  across repetitions within fold and split (validation and held-out separately), both
  camera rows of a repetition moving together, with the generator seeded from
  (permutation seed, test subject, split); then the fusion rule is rerun. A permuted
  branch calibrates to a near-constant, so this measures what averaging with a branch
  that carries no information, plus re-thresholding, does to the pose score. Report
  `fused_perm − nlf_cal` and `fused_perm − nlf`. If the mean `fused_perm − nlf_cal` over
  the three runs exceeds +0.01, the pipeline is audited for a leak before the primary is
  read.
- Both P10 sensitivities: P10 scored as a held-out subject, and P10 removed from training
  using `folds_no_p10.json`.
- **Historical-baseline check** (never blocks). Rerun the stock
  `loso_cross_validation` command on the `nlf` features at seed 42 in manifest order and
  compare fold balanced accuracy and thresholds with
  `correctness_loso_nlf_parametric_3d2d.json` (mean 0.66836). The frozen-fold `nlf` arm
  uses sorted sample order, which changes the training shuffle, so its per-subject scores
  are not expected to match that file; in the closest precedent the same change moved 7–8
  of 9 subject scores by more than 0.01. The `nlf` arm of this experiment is defined by
  the frozen-fold refit. The 0.668 figure stays context.

## Gates

Every row blocks: if it fails, row 0 of the interpretation table applies until it is
repaired and the repair is logged.

| Gate | Required evidence | Status |
| --- | --- | --- |
| Frozen protocol | This plan committed; seeds, margin, arm paths, fold and manifest hashes committed as constants in `src/rehab24/pose_video_fusion.py`; `run_config.json` written by `init` before any fit, recording the git commit (it sits under the ignored `data/` tree and is not itself committed); the three hashes above match | Pending |
| Feature integrity | Sample-by-sample ID diff of both feature directories against the manifest (not a file count): 2,144 IDs, no duplicates, complete camera pairs, finite vectors, key `video_feature`, dimensions 2160 and 768 | Pending |
| Inert trainer change | A/B at one commit: exporting trainer against the unmodified trainer, both arms, all folds and seeds, bit-identical held-out probabilities and thresholds | Pending |
| Validation export integrity | One validation row per validation-subject sample per arm, fold and seed; `person_id` equals the fold's validation subject; `find_best_threshold` on the exported validation probabilities returns the stored threshold | Pending |
| Fold purity | Platt parameters and fused thresholds use only the fold's validation subject; the held-out subject's rows never reach a fit; asserted in code and unit-tested | Pending |
| Degenerate-share identity | At `nlf` share 1.0 the fused probabilities are bitwise equal to the calibrated `nlf` probabilities, and at share 0.0 to calibrated `vm16`; wherever the Platt slope is positive, the calibrated ranking equals the raw ranking on validation and held-out rows | Pending |
| Fused OOF integrity | One fused prediction per sample and seed; branches aligned by `sample_id`, never by row order; branch labels agree; held-out subject equals `test_subject` | Pending |
| Position reproduction | `position_rule_loso` returns 0.7309; 61 mixed and 34 interleaved sessions; every arm's position-balanced statistic covers exactly 34 sessions with zero unmatched rows | Pending |

One non-blocking check accompanies the inert-change gate: the refit `vm16` held-out
probabilities are compared with the stored `oof/vm16/seed{42,7,1234}.csv` of the 20260918
run. That run used the same CPU environment and fold order, so they are expected to be
bit-identical. If the A/B gate passes and this check does not, the environment has
drifted; the refit is the registered arm and the difference is logged as a deviation.

Validation rows are written to a sibling directory `oof_val/<arm>/`, not into
`oof/<arm>/seed<seed>.csv`: the existing OOF integrity gate flags duplicate sample IDs and
any row whose `person_id` differs from `test_subject`, and every validation row is both.

## Implementation deliverables

Logic under `src/`, thin command wrappers under `scripts/`. The new files do not exist yet.

| Area | Planned change |
| --- | --- |
| `src/rehab24/loso_cross_validation.py` | `train_one_fold` optionally returns validation probabilities and IDs from the best checkpoint; default return value unchanged for existing callers |
| `src/rehab24/pose_video_fusion.py` | Frozen constants, branch refit with validation export, per-fold Platt and fusion via `src.video.late_fusion`, fused OOF in the `video_backbone_comparison` column schema, position controls, tuned-weight and uninformative-branch diagnostics, gates, report |
| `scripts/rehab24/run_pose_video_fusion.py` | Thin CLI: `init`, `fit`, `fuse`, `gates`, `report` |
| Reused without change | `late_fusion.fit_platt`, `fuse_probabilities`; `classification_metrics.find_best_threshold`; `video_backbone_comparison.contrast_summary`, `sign_flip_p_value`, `within_session_auc_statistic`, `_camera_averaged_repetition_scores`; `videomae_identity_control.holm_correct`; `videomae_position_control.build_pair_sessions`, `position_balanced_auc`, `interleaved_mask`, `position_rule_loso`, `build_repetitions`, `probe_cells` |
| `tests/test_rehab24_pose_video_fusion.py` | Best-checkpoint validation export, per-fold calibration isolation, held-out-subject leakage, ID alignment and duplicate rejection, degenerate shares, camera-paired permutation and its seeding, tuned-weight tie-breaking, session-key join completeness |

`late_fusion.read_predictions` and `fuse_run` are not reused: they expect one global
validation split, which LOSO does not have. Run the relevant ML tests before any fit and
update `scripts/rehab24/README.md` with the verified commands.

## Deviations from the plan

None at drafting time. Three differences from the identity plan's fusion clause are
registered here, not deviations of this plan: `vm16` stands in for the historical
`full_frame_letterbox` features, the `nlf_cal` guard is added, and the position control is
added. Log every later change with its date, reason, whether held-out scores were already
visible, and its effect on interpretation.

## Not supported

- That pose and video are redundant, from a non-significant result. `p ≥ 0.05` is
  undetermined, and nine subjects cannot support an equivalence claim.
- That a fusion gain reflects movement quality seen by the video branch, unless S4 passes.
  Even then the non-linear position path stays untested at the feature level.
- Anything about other fusion rules, pose backbones, VideoMAE checkpoints or fine-tuned
  encoders. Only one rule on one pair of branches is tested.
- Any revision of the identity, appearance or position-control results. A fusion score
  does not overwrite them.
- Deployable accuracy. The arms are read against a one-parameter position rule at 0.7309
  that uses no pixels and no pose.
- Generalization beyond this laboratory, its two cameras and its ten subjects.
- Named-fault recognition or coaching quality; the labels are binary.
- A larger sample size from counting seeds, cameras or repetitions.

## Reproduce

Execution order: commit plan and constants → `init` and hash checks → implement and test →
feature integrity → trainer A/B → refit both branches with validation export → validation
export and purity gates → fuse → remaining gates → report. Held-out fused scores are not
opened before the gates that precede them pass.

Proposed commands, to be confirmed once implemented:

```powershell
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py init --run-id <run_id>
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py fit --run-id <run_id> --arm nlf
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py fit --run-id <run_id> --arm vm16
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py fuse --run-id <run_id>
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py gates --run-id <run_id>
.venv\Scripts\python.exe scripts/rehab24/run_pose_video_fusion.py report --run-id <run_id>
```

Artifacts under `data/REHAB24-6/processed/pose_video_fusion/<run_id>/`: `run_config.json`,
`folds.json`, `oof/<arm>/seed<seed>.csv`, `oof_val/<arm>/seed<seed>.csv`,
`calibration.json`, `gates.json`, `summary.json`, `deviations.json`. Hardware is the local
CPU; runtime is unmeasured and will be reported, not estimated.

The results note goes to `notes/rehab24_nlf_videomae_late_fusion_results.md` and must
carry the main table with the position-rule reference row, all nine subject deltas, every
gate outcome, the position-control table, the diagnostics above, deviations and runnable
commands. Validate this plan with:

```powershell
.venv\Scripts\python.exe .claude/skills/write-experiment-note/check_note.py notes/rehab24_nlf_videomae_late_fusion_validation_plan.md
```
