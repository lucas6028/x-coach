# The SSv2 checkpoint does not improve REHAB24-6: ΔBA −0.027 (undetermined), within-session AUC 0.78 against 0.87 (p = 0.02)

*REHAB24-6 · pre-registered checkpoint comparison · run 2026-09-08 · plan
[`rehab24_videomae_ssv2_checkpoint_validation_plan.md`](rehab24_videomae_ssv2_checkpoint_validation_plan.md)
· siblings [`rehab24_videomae_framing_results.md`](rehab24_videomae_framing_results.md),
[`rehab24_videomae_identity_appearance_results.md`](rehab24_videomae_identity_appearance_results.md),
[`rehab24_videomae_position_control_results.en.md`](rehab24_videomae_position_control_results.en.md),
[`videomae_ssv2_checkpoint_results.md`](videomae_ssv2_checkpoint_results.md)*

**Result.** Swapping the Kinetics-400 fine-tuned VideoMAE for the Something-Something-v2
(SSv2) fine-tune on REHAB24-6 `full_frame_letterbox` gives leave-one-subject-out balanced
accuracy 0.6341 ± 0.0341 against 0.6612 ± 0.0567; the pre-registered primary ΔBA is
−0.0271 ± 0.0566 over nine subjects, 2/9 positive, exact two-sided Wilcoxon p = 0.164,
bootstrap 95% CI [−0.061, +0.007]. Per the plan's rule the reading is **undetermined**, and
Kinetics remains the quoted REHAB24-6 checkpoint. The secondary within-session ROC-AUC is
0.7805 ± 0.0917 against 0.8741 ± 0.0408, ΔAUC −0.0936, 2/9 positive, p = 0.0195, CI
[−0.151, −0.043]: the SSv2 features rank repetitions inside a recording worse. The
Fitness-AQA column shift of +0.02–0.03 does not transfer.

**Caveat on reading it.** Undetermined is not "no difference" and not "SSv2 is worse
cross-subject": the primary's interval spans −0.06 to +0.01 on a statistic whose SD is
0.057, exactly the resolution the plan said it had. The within-session loss is the
plan's "ΔAUC < 0" row and is reported as a ranking result, not as a checkpoint verdict.
The position-readability and background readouts moved in the direction the
motion-centric hypothesis predicted, but that hypothesis also required a within-session
gain, so no hypothesis row is hit.

## Terminology

| term | meaning |
| --- | --- |
| arm | one feature set run through the unchanged REHAB24-6 LOSO recipe (`videomae_stage_a.run_arm`, frozen since the framing plan) |
| `full_frame_letterbox` | whole frame padded to square with neutral grey; the framing experiment's best arm and the arm every later control was run on |
| `background_only` | the person's fixed per-video box (mocap union, +15%) filled from the pixels either side; scene kept, person gone |
| LOSO BA | leave-one-subject-out balanced accuracy, one fold per subject P1–P9, seeds 42 / 7 / 1234 averaged within subject before comparison; P10 (16 samples) only in sensitivity |
| within-session AUC | inside one recording, the probability that a correct repetition outranks an incorrect one; cameras averaged per repetition, seeds averaged per session, subject-macro, n = 9 |
| probe ρ | subject-macro within-class Spearman correlation between a LOSO ridge prediction and the repetition's rank among same-label repetitions of its recording |
| share | (AUC_k0 − AUC_k16) / (AUC_k0 − 0.5) after projecting out 16 fold-fitted position directions |
| kin / ssv2 | `MCG-NJU/videomae-base-finetuned-kinetics` / `MCG-NJU/videomae-base-finetuned-ssv2` |

## Background

Every REHAB24-6 VideoMAE number in this project came from the Kinetics fine-tune. On
Fitness-AQA the SSv2 fine-tune raised the whole column by 0.02–0.03 regardless of which
pixels were deleted and its exploratory late fusion with pose reached 0.702. REHAB24-6
has a different confound structure (scene at chance, identity closed, position closed
along 16 linear directions with a 4.9% share), and the identity plan's fusion gate needs
its VideoMAE checkpoint named by a rule. The plan fixed one primary (cross-subject ΔBA)
and three hypotheses: H1 checkpoint-general, H2 motion-centric (ΔBA ≥ +0.02, probe ρ
down, share down, background at chance), H3 drift-reading (ΔAUC > 0, probe ρ up, share
up, background above the Kinetics 0.5357).

## Method

Two SSv2 arms were extracted with the REHAB24-6 extractor unchanged except
`--model-name`: `full_frame_letterbox` and `background_only`, same mocap boxes, 16 frames
at stride 2, 4 clips, tokens mean-pooled through the checkpoint's own `fc_norm`, clips
averaged. The Kinetics arms were not re-extracted; their feature dirs went through the
same report invocations, so both checkpoints share folds, seeds, validation subjects and
threshold selection. The classifier recipe, the within-session statistic and the
position-control procedure (ridge probe, λ from the inner leave-one-training-subject-out
sweep, k = 16 projection) are those of the earlier notes, unchanged.

The design decision a reader would otherwise get wrong: the checkpoint choice is made by
the cross-subject primary only. The within-session AUC, the probe and the background arm
are readouts that qualify the number; the plan's power statement says a shift the size
of the Fitness-AQA one (+0.03) was expected to be undetermined here, and it was.

Statistics: paired subject deltas, exact two-sided Wilcoxon (2 × min of the two exact
one-sided tests), 9-subject bootstrap percentile interval (10,000 resamples, seed
20260908); each within-session arm carries its own within-session label permutation null
(10,000 draws, seed 20260908). Practical band 0.02.

## Gates

| gate | result |
| --- | --- |
| G1 checkpoint | pass; every worker log: `fc_norm loaded from checkpoint (weight mean=0.7032, bias mean=-0.0077)` |
| G2 completeness | pass; 2,144 raw and 2,144 materialized bundles per arm, sample-id set identical to the Kinetics arm, one provenance record per arm with `model_name` = ssv2 |
| G3 features changed, variant applied | pass; letterbox: 0 of 2,144 identical to Kinetics (cosine median 0.025, range −0.09 to +0.09); background: 0 identical to Kinetics background, 0 identical to SSv2 letterbox (cosine median 0.448, range 0.161–0.852); framing feature audit PASS on both dirs |
| G4 reproduction | pass; the same invocations on the Kinetics dirs give BA 0.6612, within-session AUC 0.8741, probe median 0.4324 (all to 4 decimals); the position module's k = 0 run on SSv2 features returns the identity-control 0.7805 and the framing report's −0.0271 |
| G5 frozen analysis | pass; plan `869a3f00`, report script `e71e5246`, both before any SSv2 classifier run |
| identity-control 6.1/6.2 on the SSv2 arms | pass; 2,144 OOF rows per seed, 61 mixed-label sessions, camera pairs consistent |
| position-control 6.2 fold purity | pass; 10 pairwise-distinct β digests at k = 16; λ selected per fold from {10, 100, 10000} |
| position-control 6.3 probe after k = 16 | pass on the median (+0.0885, null [−0.0886, +0.0895], p = 0.051); **fail on the mean** (+0.0776, p = 0.033), the same pattern Kinetics showed (median pass, mean p = 0.0016) |

## Results

**Primary.**

| arm | LOSO BA (mean ± SD, 9 subjects) | range | per seed 42 / 7 / 1234 | macro-F1 | recall | specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kin `full_frame_letterbox` | 0.6612 ± 0.0567 | 0.580–0.754 | 0.666 / 0.665 / 0.653 | 0.647 | 0.710 | 0.612 |
| ssv2 `full_frame_letterbox` | 0.6341 ± 0.0341 | 0.543–0.660 | 0.629 / 0.639 / 0.634 | 0.609 | 0.609 | 0.659 |

ΔBA (ssv2 − kin) = −0.0271 ± 0.0566, 2/9 subjects positive, exact two-sided p = 0.164,
bootstrap 95% CI [−0.061, +0.007]. Per subject P1–P9: −0.069, −0.043, +0.069, +0.029,
−0.008, −0.100, −0.025, −0.097, −0.001. Rule row: undetermined. The framing report's own
practical label is "practical loss, undetermined".

The pattern to see: the SSv2 arm has a lower mean and about half the between-subject
spread, and its loss is concentrated in P1, P6 and P8. Nothing in the primary resolves a
0.03 shift in either direction.

## Secondary

**S1, within-session AUC.**

| arm | AUC (mean ± SD) | > 0.5 | bootstrap 95% CI | permutation p | cam17 / cam18 |
| --- | ---: | ---: | --- | ---: | ---: |
| kin | 0.8741 ± 0.0408 | 9/9 | [0.850, 0.899] | 1/10001 | 0.821 / 0.867 |
| ssv2 | 0.7805 ± 0.0917 | 9/9 | [0.720, 0.827] | 1/10001 | 0.695 / 0.781 |

ΔAUC = −0.0936, 2/9 positive, exact two-sided p = 0.0195, CI [−0.151, −0.043]. Per
subject, ssv2: 0.571, 0.799, 0.828, 0.822, 0.837, 0.834, 0.686, 0.817, 0.833. The loss is
carried by P1 (0.837 → 0.571) and P7 (0.885 → 0.686); the other seven subjects sit
0.03–0.10 below their Kinetics value. By exercise, ssv2 / kin: Ex1 0.759 / 0.971, Ex2
0.658 / 0.719, Ex3 0.836 / 0.940, Ex4 0.799 / 0.909, Ex5 0.816 / 0.885, Ex6 0.848 / 0.833.
The SSv2 arm still clears the identity-control rule on its own (mean ≥ 0.55, 9/9, p <
0.05).

**S2, position readability.**

| | kin | ssv2 |
| --- | ---: | ---: |
| probe ρ median, k = 0 (9/9 positive, p = 1/10001 both) | 0.4324 | 0.3843 |
| probe ρ mean, k = 0 | 0.4029 | 0.4008 |
| probe median after k = 16 (inside null) | 0.0525 (yes, p = 0.243) | 0.0885 (yes, p = 0.051) |
| probe mean after k = 16 | 0.1183 (p = 0.0016) | 0.0776 (p = 0.033) |
| within-session AUC k = 0 → k = 16 | 0.8741 → 0.8556 | 0.7805 → 0.7772 |
| share | 4.97% | 1.2% |
| position-balanced AUC, 34 interleaved sessions, k = 0 → 16 | 0.8497 → 0.8320 | 0.7408 → 0.7336 |
| incorrect-first pairs (position rule = 0), k = 0 → 16 | 0.8223 → 0.7952 | 0.6871 → 0.6859 |

SSv2 probe per subject at k = 0: 0.384, 0.300, 0.374, 0.334, 0.591, 0.500, 0.168, 0.526,
0.431 (null [−0.089, +0.089]). The k = 16 arm keeps 9/9 subjects above chance (per subject
0.694, 0.798, 0.875, 0.788, 0.835, 0.829, 0.661, 0.777, 0.738; p = 1/10001). Directions:
probe median −0.048 relative to Kinetics, share −3.8 points. Both are floors, since the
k = 16 probe mean is outside its null for both checkpoints. No test is attached to the
cross-checkpoint difference.

**S3, `background_only` for SSv2.**

| readout | ssv2 | kin reference | informative threshold | informative |
| --- | ---: | ---: | --- | --- |
| LOSO BA (9 subjects) | 0.5303 ± 0.0287, 9/9 > 0.5, exact Wilcoxon vs 0.5 p = 0.0020, CI [0.513, 0.548], range 0.502–0.566, seeds 0.527 / 0.529 / 0.536 | 0.5074 | ≥ 0.55 with p < 0.05 | no (0.53) |
| within-session AUC | 0.5405 ± 0.0666, 6/9 > 0.5, p = 0.0203, CI [0.501, 0.583], range 0.463–0.654 | 0.5357 (7/9, p = 0.032) | AUC − 0.5 > 0.15 with p < 0.05 | no (0.04) |

The non-person path is above chance on both readouts, with every subject above 0.5 on
balanced accuracy, and below both informative margins. The framing report's registered secondary,
SSv2 background against the Kinetics letterbox arm, is −0.131 (0/9, Holm p = 0.0039); the
SSv2 background arm sits 0.023 above the Kinetics background arm's 0.5074, untested. No checkpoint-specific
non-person shortcut is named; the plan's margin, not the p-value, decides that.

**S4, strata of ΔBA (reported only).** cam17 −0.025 (4/9, p = 0.65), cam18 −0.029 (3/9,
p = 0.20). By exercise: arm VW −0.015 (6/9, p = 1.0), arm abduction −0.093 (3/9, p =
0.11), leg abduction −0.019 (4/9, p = 0.82), leg lunge −0.011 (3/7, p = 0.94), squats
+0.023 (4/9, p = 0.73), table push-ups −0.107 (2/7, p = 0.078). All undetermined; the
squat stratum is the only positive point estimate.

**S5, P10-inclusive sensitivity.** LOSO BA with P10: kin 0.6901, ssv2 0.6401, Δ −0.050
(means only). Within-session with P10 (n = 10): 0.8867 → 0.8025, Δ −0.0843, 2/10
positive, p = 0.0195, CI [−0.141, −0.035].

**Holm over the secondary test family** (S1 ΔAUC, S3 BA vs chance, S3 within-session
permutation): raw 0.0195 / 0.0020 / 0.0203, Holm 0.039 / 0.0059 / 0.039; all three remain
below 0.05. Significance is not the informative criterion for S3 (see above).

## Reading table

| row | hit |
| --- | --- |
| H2 motion-centric (S1 up, probe down, share down, S3 flat) | no; S1 is down |
| H3 drift-reading (S1 up with probe up, share up, or S3 informative) | no; S1 is down and the probe went down |
| H1 checkpoint-general (S1 and primary undetermined, S3 flat) | no; S1 is determined |
| S3 checkpoint-specific non-person path | no |
| S1 ΔAUC < 0 with p < 0.05 | **yes** |

The identity plan's fusion gate is not closed for SSv2 by this table, but the checkpoint
was not adopted, so the question does not arise.

## Deviations from the plan

| item | plan | what happened | changes |
| --- | --- | --- | --- |
| framing report | one run with all three arms | run twice: kin + ssv2 first (05:52–06:16) so the primary was readable before the background arm existed, then all three arms; the primary block is identical across the two runs (deterministic CPU training) | nothing; the two-arm summary is kept as `videomae_checkpoint_summary_two_arm.json` |
| CLI flag order in the reproduce block | `--output-dir` before the subcommand for the identity-control CLI | both CLIs take it after the subcommand; the first identity launch failed on argparse and was relaunched; plan text fixed in `d153e695` before any number was read | nothing |
| `analyze --framing-summary` | the plan's first draft pointed the position-control `analyze` at the checkpoint summary | that helper reads only the `full_frame_letterbox` arm and would have reported "unavailable"; removed in `e71e5246` before any SSv2 classifier run | nothing; the per-subject 0.6612 baseline is the same file either way |
| probe permutation seed | not specified for `probe` | module default 20260906 used for both probes; `analyze` used 20260908 as planned | nothing |
| probe after removal, mean | plan requires the median to reach the null | median passes, mean does not (p = 0.033) | the 1.2% share is a floor, as the 4.9% Kinetics share already was |
| S1 timing | gates before any SSv2 score is looked at | the S1 `analyze` process was started before the framing report finished, its output was read only after the report confirmed G4 | nothing |

## Not supported

- **No equivalence.** ΔBA's interval is ±0.03 around −0.027; the design cannot show the
  checkpoints score the same cross-subject, and it was pre-stated that a 0.03 shift would
  be undetermined.
- **Not "SSv2 is worse cross-subject."** p = 0.164, 2/9. The strata are all undetermined.
- **Not a movement-reading claim for either checkpoint.** `repetition_number` alone still
  reaches LOSO BA 0.7309, above both arms; the probe and share went down for SSv2 but the
  k = 16 probe mean is outside its null, so the position path is only partly closed, and
  only linearly.
- **The within-session loss is not a mechanism.** Two subjects carry most of it; the
  design has no readout that says why SSv2 ranks P1 and P7 worse.
- **Pre-training data and fine-tune labels are confounded**; the SSv2 checkpoint is also
  SSv2-pretrained.
- **Nothing about fusion.** Out of scope by the plan; no fusion score was computed.
- **Nothing about Fitness-AQA.** The +0.03 column shift there stands as reported in its
  own note; this note says it does not transfer to REHAB24-6, not that it was wrong.
- **Boxes are mocap-derived; one lab, two fixed cameras.** As in every REHAB24-6 note.

## Reproduce

Code: `src/rehab24/videomae_features.py` (extractor, `--model-name`),
`videomae_materialize.py`, `videomae_audit.py`, `videomae_framing_report.py`,
`videomae_identity_control.py`, `videomae_position_control.py`, and the new
`src/rehab24/videomae_checkpoint_report.py` + `scripts/rehab24/videomae_checkpoint_report.py`
(19 tests in `tests/test_rehab24_videomae_checkpoint_report.py`).

```powershell
# 1. extraction, three GPU workers per arm (.venv-cuda, GTX 1660 Ti)
.venv-cuda\Scripts\python.exe scripts\rehab24\extract_videomae_features.py --model-name MCG-NJU/videomae-base-finetuned-ssv2 --variant full_frame_letterbox --output-dir data\REHAB24-6\processed\videomae_raw_ssv2_full_frame_letterbox --device cuda --num-chunks 3 --chunk-index <0|1|2>
.venv-cuda\Scripts\python.exe scripts\rehab24\extract_videomae_features.py --model-name MCG-NJU/videomae-base-finetuned-ssv2 --variant background_only --output-dir data\REHAB24-6\processed\videomae_raw_ssv2_background_only --device cuda --num-chunks 3 --chunk-index <0|1|2>
# 2. materialize (checkpoint-scoped parent; the default parent would overwrite the Kinetics Stage A dir)
.venv\Scripts\python.exe scripts\rehab24\materialize_videomae_features.py --raw-dir data\REHAB24-6\processed\videomae_raw_ssv2_<arm> --output-parent data\REHAB24-6\processed\videomae_ssv2\<arm> --token-pooling mean_pool_fc_norm --aggregation mean
# 3. gates
.venv\Scripts\python.exe scripts\rehab24\audit_videomae_features.py data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean data\REHAB24-6\processed\videomae_ssv2\background_only\videomae_mean_pool_fc_norm_mean --report-output data\REHAB24-6\processed\videomae_ssv2_audit.json
# 4. primary, S4, S5, S3 balanced accuracy
.venv\Scripts\python.exe scripts\rehab24\videomae_framing_report.py --arm kin=data\REHAB24-6\processed\videomae_framing\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --arm ssv2=data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --arm ssv2_background_only=data\REHAB24-6\processed\videomae_ssv2\background_only\videomae_mean_pool_fc_norm_mean --primary ssv2:kin --secondary ssv2_background_only:kin --device cpu --output-prefix data\REHAB24-6\processed\videomae_checkpoint
# 5. S1 and S3 within-session (flags after the subcommand)
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py predict --output-dir data\REHAB24-6\processed\videomae_identity_control_ssv2 --arm-dir data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py analyze --output-dir data\REHAB24-6\processed\videomae_identity_control_ssv2 --permutations 10000 --permutation-seed 20260908
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py predict --output-dir data\REHAB24-6\processed\videomae_identity_control_ssv2_background_only --arm-dir data\REHAB24-6\processed\videomae_ssv2\background_only\videomae_mean_pool_fc_norm_mean --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py analyze --output-dir data\REHAB24-6\processed\videomae_identity_control_ssv2_background_only --permutations 10000 --permutation-seed 20260908
# 6. S2
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 0 16 --output-dir data\REHAB24-6\processed\videomae_position_control_ssv2 --arm-dir data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --seeds 42 7 1234 --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 0 --output-dir data\REHAB24-6\processed\videomae_position_control_ssv2 --arm-dir data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 16 --output-dir data\REHAB24-6\processed\videomae_position_control_ssv2 --arm-dir data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 0 --output-dir data\REHAB24-6\processed\videomae_position_control_ssv2 --permutations 10000 --permutation-seed 20260908
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 16 --output-dir data\REHAB24-6\processed\videomae_position_control_ssv2 --permutations 10000 --permutation-seed 20260908
# 7. verdict rows
.venv\Scripts\python.exe scripts\rehab24\videomae_checkpoint_report.py --output data\REHAB24-6\processed\videomae_checkpoint_report.json
```

Runtime: extraction 4 h 35 min for the letterbox arm (01:12–05:47) and 5 h 10 min for
the background arm (05:50–11:00), three GPU workers each; materialize under 2 min per arm;
framing report 24 min for two arms and 17 min for three (after the GPU was idle), on CPU while the GPU
workers competed for cores; identity predict 18 min + analyze 2 min per arm; position
predict (k = 0 and 16, three seeds) 56 min, probes and analyses about 10 min; checkpoint
report under 1 min. Artifacts under `data/REHAB24-6/processed/`:
`videomae_raw_ssv2_{full_frame_letterbox,background_only}/`, `videomae_ssv2/<arm>/`,
`videomae_ssv2_audit.json`, `videomae_ssv2_gates.json`,
`videomae_checkpoint_{seed42,seed7,seed1234,summary}.json` (+ `_summary_two_arm.json`),
`videomae_identity_control_ssv2{,_background_only}/`, `videomae_position_control_ssv2/`,
`videomae_checkpoint_report.json`; logs `videomae_*_ssv2*.log`.
