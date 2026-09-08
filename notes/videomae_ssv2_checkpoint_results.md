# The athlete-deletion shortcut reproduces with an SSv2 checkpoint: retention 0.92 vs 0.91, ΔA = −0.0003 (undetermined)

*Fitness-AQA · checkpoint-robustness arm for Stage B · run 2026-09-07 · plan
[`videomae_ssv2_checkpoint_validation_plan.md`](videomae_ssv2_checkpoint_validation_plan.md) ·
parent note [`videomae_stage_b_results.md`](videomae_stage_b_results.md)*

**Result.** Swapping the Kinetics-400 fine-tuned VideoMAE for the Something-Something-v2
(SSv2) fine-tune and re-running the Stage B athlete-deletion control does not change how
much of the signal survives deleting the athlete. Athlete contribution A (full frame minus
background only, balanced accuracy, seed means) is +0.0127 for SSv2 against +0.0130 for
Kinetics; the pre-registered primary ΔA = A_ssv2 − A_kin is −0.0003, 95% paired bootstrap
CI [−0.075, +0.072], 10,000 resamples, n = 244 test videos. Retention R (share of
above-chance signal that needs no athlete pixels) is 0.92 [0.63, 1.30] for SSv2 and
0.91 [0.61, 1.38] for Kinetics. Per the plan's rule the reading is **undetermined**: the
Stage B caveat stands as written, with the sentence that it was replicated on a second
checkpoint.

**Caveat on reading it.** Undetermined is not "no difference". The interval is ±0.07 on a
statistic whose observed value is 0.013, so the design cannot distinguish a checkpoint that
needs the athlete twice as much from one that does not need it at all. The SSv2 full-frame
arm is also 0.029 higher than the Kinetics one in absolute balanced accuracy, and its late
fusion with pose reaches 0.702; both are secondary or exploratory here and are reported
below without a decision.

## Terminology

| term | meaning |
| --- | --- |
| arm | one feature set fed to the unchanged Stage B classifier grid |
| `full_frame` / `background_only` / `person_crop` | untouched video / athlete's box filled with the scene value / tight crop to the athlete's box, letterboxed |
| BA | balanced accuracy on the fixed test split (244 videos, 172 positive / 72 negative) at the threshold selected on validation, seeds 1–5 |
| A | BA_full_frame − BA_background_only, the part of the signal that needs athlete pixels |
| R | (BA_background_only − 0.5) / (BA_full_frame − 0.5), the retained share |
| kin / ssv2 | `MCG-NJU/videomae-base-finetuned-kinetics` / `MCG-NJU/videomae-base-finetuned-ssv2` |

## Background

Stage B found that with the athlete painted out of every frame the Kinetics-finetuned
VideoMAE still reaches 0.627 against 0.640 on the full frame, so 91% of the above-chance
signal survives deleting the person; a second control showed the survivor is coarse
athlete geometry (box height, clip length), not recording context. Kinetics-400 is a
scene-biased fine-tune and SSv2 a motion-centric one, so the one mechanism that would
predict a different retention is the checkpoint. This run tests that and nothing else.

## Method

Three SSv2 arms were extracted with the Stage B code path and the Stage B variant boxes:
same frame sampling (16 frames, stride 2, 4 clips), tokens mean-pooled through the
checkpoint's own `fc_norm`, clips averaged. The checkpoint passed the extractor's two
loading gates (mean-pooling config, `fc_norm` loaded rather than at default init) and has
the same image preprocessing as the Kinetics checkpoint. The Kinetics arms were not
re-extracted; their existing prediction files were re-read.

Classifier grid unchanged from Stage B: `combined` label mode, seeds 1–5, 20 epochs,
hidden 128, dropout 0.4, lr 3e-4, weight decay 0.01, patience 5, threshold selected on
validation for balanced accuracy.

The design decision a reader would otherwise get wrong: the decision statistic is ΔA, the
difference of two athlete contributions, not R. R is a ratio of two noisy numbers and its
bootstrap interval spans 0.6–1.4 for both checkpoints, which is why the plan fixed ΔA as
primary before any SSv2 number existed. Bootstrap: one shared draw of test videos per
resample scores all four arms at their per-seed selected thresholds, seed-averaged, then
the statistic is taken; 95% percentile interval; seed 20260907.

## Gates

| gate | result |
| --- | --- |
| G1 checkpoint loads through both `fc_norm` gates | pass (weight mean 0.703, bias mean −0.008; logged again by every extractor chunk) |
| G2 completeness | pass; 1623 raw bundles in each of the 3 SSv2 arms, video-id set identical to the Kinetics arm, provenance model = ssv2 in every file |
| G3 controls changed the features | pass; `background_only` 0 of 1623 identical to `full_frame` (cosine median 0.726, min 0.165); `person_crop` 1 of 1623 identical (video `47341_1`, the whole-frame box already seen in Stage B; cosine median 0.768, min 0.293) |
| G4 reproduction | pass; the new report reads the Kinetics prediction files back to 0.6401 / 0.6271 / 0.6662 and R = 0.9073 against the plan's 0.907 |
| G5 frozen analysis | pass; plan `10b7b42e` and report script `9409533d` committed before any SSv2 classifier run |

## Results

| arm | BA (mean ± SD, 5 seeds) | min–max | recall | specificity |
| --- | ---: | ---: | ---: | ---: |
| kin `full_frame` | 0.640 ± 0.021 | 0.612–0.671 | 0.630 | 0.650 |
| kin `background_only` | 0.627 ± 0.019 | 0.593–0.640 | 0.671 | 0.583 |
| kin `person_crop` | 0.666 ± 0.012 | 0.654–0.682 | 0.655 | 0.678 |
| ssv2 `full_frame` | 0.669 ± 0.009 | 0.658–0.679 | 0.723 | 0.614 |
| ssv2 `background_only` | 0.656 ± 0.011 | 0.644–0.672 | 0.690 | 0.622 |
| ssv2 `person_crop` | 0.655 ± 0.004 | 0.651–0.661 | 0.729 | 0.581 |

| checkpoint | A | R [95% CI] |
| --- | ---: | ---: |
| kin | +0.0130 | 0.907 [0.609, 1.377] |
| ssv2 | +0.0127 | 0.925 [0.632, 1.295] |

**Primary.** ΔA = −0.0003, 95% CI [−0.0746, +0.0719], fraction of resamples > 0 = 0.489,
0 dropped resamples. Reading per the plan: undetermined.

Per seed, ssv2 full_frame − background_only: +0.021, +0.008, −0.001, +0.018, +0.018
(4/5 positive). Kinetics: +0.000, +0.035, −0.019, −0.002, +0.051 (3/5 positive).

The pattern to see: deleting the athlete costs both checkpoints about 0.013 of balanced
accuracy, and the whole SSv2 column sits about 0.02–0.03 above the Kinetics column
regardless of what was deleted. The shift is checkpoint-wide, not athlete-specific.

## Secondary

**Person crop minus full frame.** Kinetics +0.026 [−0.025, +0.077] (Stage B's dilution
reading, already known to be fixed-split noise); SSv2 −0.014 [−0.067, +0.039]. Both
intervals contain 0. The sign flips between checkpoints, and neither is determined.

**Each SSv2 arm against pose-only** (normalized, 0.650 ± 0.012), through the Stage B
report script; its denominator gate fails as it did in Stage B (0.650 is outside
0.635 ± 0.010) and every delta below is against that moved denominator. 2,000 resamples.

| arm | Δ vs pose | 95% CI | seeds positive |
| --- | ---: | --- | ---: |
| ssv2 `full_frame` | +0.018 | [−0.053, +0.098] | 5/5 |
| ssv2 `background_only` | +0.005 | [−0.073, +0.087] | 3/5 |
| ssv2 `person_crop` | +0.004 | [−0.072, +0.083] | 4/5 |
| kin `full_frame` | −0.010 | [−0.078, +0.062] | 2/5 |

All undetermined. No SSv2 arm meets the Stage B retention conditions on its own.

**Exploratory, not pre-registered as a test: late fusion of pose and ssv2 `full_frame`**
at weight 0.5, calibrated as in Stage B. BA 0.702 ± 0.025 (per seed 0.738, 0.702, 0.705,
0.700, 0.667), Δ vs pose +0.052, 95% CI [−0.002, +0.108], 4/5 seeds positive, fraction of
resamples > 0 = 0.97. Of the four Stage B retention conditions it passes three (delta
≥ 0.02, no guardrail drop over 0.03, consistent across seeds) and fails the CI lower bound
by 0.002. Stage B's Kinetics late fusion was −0.003 on the same protocol. This is one
fixed split and one checkpoint, was not registered, and is not a retention verdict; the
plan's item 2 named it a reported quantity only.

## Bounds

The box-geometry control (12 numbers, no pixels) stays where Stage B put it: 0.578 ± 0.023,
Δ vs pose −0.073 [−0.140, −0.001]. It contains no checkpoint, so it is the floor for any
retention figure: with the SSv2 full frame at 0.669, geometry alone is (0.578 − 0.5) /
(0.669 − 0.5) = 46% of the SSv2 signal, against 56% of the Kinetics signal.

## Deviations from the plan

| item | plan | what happened | changes |
| --- | --- | --- | --- |
| R_kin target | 0.907 to 4 decimals | script computes 0.9073 | nothing; the plan's figure was 3-decimal |
| G3 identical pairs | at most 1 | exactly 1 (`person_crop`, `47341_1`) | nothing; the allowed whole-frame box |
| Secondary 2 exit code | script exits non-zero on the denominator gate | it did; JSON written first | nothing |
| Extraction | three chunks per arm on the local GPU | as written | nothing |

## Not supported

- **No equivalence.** ΔA's interval is ±0.07 around an effect of 0.013; the run cannot
  show that the two checkpoints use the athlete equally. It shows only that the Stage B
  pattern reproduced on a second checkpoint at the same point estimate.
- **Not "SSv2 is better".** The 0.029 gain of ssv2 `full_frame` over kin `full_frame`
  is within one Kinetics SD of seeds and was not a pre-registered comparison; against
  pose-only it is undetermined.
- **Not a fusion result.** The 0.702 late fusion is exploratory on the same fixed split
  Stage B found noisy for person_crop; its CI contains 0. It is a candidate for a
  pre-registered follow-up on repeated splits, nothing more.
- **Not a statement about fine-tune labels alone.** The SSv2 checkpoint is also
  self-supervised pre-trained on SSv2, so pre-training data and fine-tune labels move
  together.
- **Nothing about REHAB24-6**, where the background arm sits at chance and the confound
  is identity and recording position. Resolved 2026-09-08 in
  [`rehab24_videomae_ssv2_checkpoint_results.md`](rehab24_videomae_ssv2_checkpoint_results.md):
  the +0.02–0.03 column shift seen here does not transfer (ΔBA −0.027, undetermined;
  within-session AUC 0.78 vs 0.87, p = 0.02).
- **The 46–56% that needs no pixels** is a property of the annotation (box height,
  clip length) and no checkpoint can move it.

## Reproduce

Code: `src/video/videomae_feature_extraction.py` (extractor, `--model-name`),
`scripts/video/materialize_videomae_features.py`, `scripts/video/run_videomae_experiment_grid.py`,
`src/video/checkpoint_robustness_report.py` + `scripts/video/run_checkpoint_robustness_report.py`
(primary), `scripts/video/run_stage_b_report.py` (secondary).

```
# 1. extraction, per arm, three chunks in parallel on the GTX 1660 Ti (.venv-cuda, torch cu126)
.venv-cuda/Scripts/python.exe scripts/video/run_videomae_feature_extraction.py \
  --model-name MCG-NJU/videomae-base-finetuned-ssv2 --variant <arm> --device cuda \
  [--variant-manifest data/Fitness-AQA/Squat/Labeled_Dataset/videos_<arm>/manifest.json] \
  --output-dir data/Fitness-AQA/Squat/Labeled_Dataset/videomae_raw_ssv2_<arm> \
  --num-chunks 3 --chunk-index <0|1|2>
# 2. materialize into a checkpoint-scoped parent (the default parent would overwrite the Kinetics dirs)
.venv/Scripts/python.exe scripts/video/materialize_videomae_features.py \
  --raw-dir data/Fitness-AQA/Squat/Labeled_Dataset/videomae_raw_ssv2_<arm> \
  --output-parent data/Fitness-AQA/Squat/Labeled_Dataset/ssv2/<arm> \
  --token-pooling mean_pool_fc_norm --aggregation mean
# 3. grid, per arm (same flags as Stage B step 6, no --normalize-features)
.venv/Scripts/python.exe scripts/video/run_videomae_experiment_grid.py \
  --feature-dir data/Fitness-AQA/Squat/Labeled_Dataset/ssv2/<arm>/videomae_mean_pool_fc_norm_mean \
  --train-keys .../Splits/train_keys.json --val-keys .../Splits/val_keys.json --test-keys .../Splits/test_keys.json \
  --forward-labels .../Labels/error_knees_forward.json --inward-labels .../Labels/error_knees_inward.json \
  --output-root data/Fitness-AQA/Squat/experiments/ssv2_<arm> --label-modes combined
# 4. primary
.venv/Scripts/python.exe scripts/video/run_checkpoint_robustness_report.py \
  --kin-full-frame data/Fitness-AQA/Squat/experiments/videomae_corrected/predictions \
  --kin-background-only data/Fitness-AQA/Squat/experiments/videomae_background_only/predictions \
  --kin-person-crop data/Fitness-AQA/Squat/experiments/videomae_person_crop/predictions \
  --ssv2-full-frame data/Fitness-AQA/Squat/experiments/ssv2_full_frame/predictions \
  --ssv2-background-only data/Fitness-AQA/Squat/experiments/ssv2_background_only/predictions \
  --ssv2-person-crop data/Fitness-AQA/Squat/experiments/ssv2_person_crop/predictions \
  --output data/Fitness-AQA/Squat/experiments/checkpoint_robustness_report.json
# 5. secondary (exits 1 on the denominator gate after writing the JSON)
.venv/Scripts/python.exe scripts/video/run_stage_b_report.py \
  --pose-predictions data/Fitness-AQA/Squat/experiments/pose_only/predictions \
  --videomae-predictions data/Fitness-AQA/Squat/experiments/ssv2_full_frame/predictions \
  --arm ssv2_background_only=... --arm ssv2_person_crop=... --arm kin_full_frame=... --arm box_geometry=... \
  --output data/Fitness-AQA/Squat/experiments/stage_b_report_ssv2.json
```

Runtime: extraction 1 h 40 min – 2 h per arm with three GPU processes (5.5 h for all
three, 11:10–16:40); materialize + grid about 5 min per arm on CPU; primary report about
2 min. Artifacts: `data/Fitness-AQA/Squat/Labeled_Dataset/videomae_raw_ssv2_*/`,
`.../Labeled_Dataset/ssv2/<arm>/videomae_mean_pool_fc_norm_mean/`,
`data/Fitness-AQA/Squat/experiments/ssv2_*/`, `.../checkpoint_robustness_report.json`,
`.../stage_b_report_ssv2.json`.
