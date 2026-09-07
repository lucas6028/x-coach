# Plan: does the Stage B athlete-deletion shortcut survive an SSv2-finetuned VideoMAE checkpoint?

*Fitness-AQA · pre-registered checkpoint-robustness arm for Stage B · written 2026-09-07,
before any SSv2 feature was evaluated · sibling of
[`videomae_stage_b_results.md`](videomae_stage_b_results.md)*

**Question.** Stage B found that with the athlete painted out of every frame the
Kinetics-finetuned VideoMAE still reaches balanced accuracy 0.627 against 0.640 on the
full frame, i.e. 91% of the above-chance signal survives deleting the person. Every
Fitness-AQA VideoMAE number carries that caveat. This plan asks one narrow question:
is that retention a property of the Kinetics-400 fine-tune, which is known to be
scene-biased, or does it reproduce with the Something-Something-v2 (SSv2) fine-tune,
whose labels require temporal reasoning inside a fixed scene? Nothing else about
Stage B is re-opened.

## Terminology

| term | meaning |
| --- | --- |
| arm | one feature set fed to the unchanged Stage B classifier grid |
| `full_frame` | the untouched video |
| `background_only` | the athlete's union box filled with the scene value; the box itself still carries coarse geometry |
| `person_crop` | tight crop to the athlete's box, letterboxed |
| BA | balanced accuracy on the fixed test split at the threshold selected on the validation split, seeds 1–5 |
| retention R | (BA_background_only − 0.5) / (BA_full_frame − 0.5); the share of above-chance signal that needs no athlete pixels |
| athlete contribution A | BA_full_frame − BA_background_only |
| kin / ssv2 | `MCG-NJU/videomae-base-finetuned-kinetics` / `MCG-NJU/videomae-base-finetuned-ssv2` |

## Hypotheses

- **H1, data-side shortcut.** The retained signal is coarse athlete geometry and scene
  content that any video backbone picks up. Prediction: A_ssv2 is not larger than
  A_kin, and R_ssv2 stays near R_kin = 0.91.
- **H2, checkpoint-side shortcut.** The Kinetics fine-tune reads the scene where a
  motion-centric fine-tune would read the athlete. Prediction: deleting the athlete
  costs SSv2 more than it costs Kinetics, A_ssv2 > A_kin, and R_ssv2 falls well below
  0.91.

## Design

Three SSv2 arms, extracted with the same code path, the same frame sampling (16 frames,
stride 2, 4 clips, mean-pooled tokens through the checkpoint's own `fc_norm`, clips
averaged) and the same variant boxes as Stage B: `full_frame`, `person_crop`,
`background_only`. The two zero-parameter controls (box geometry, frame format) contain
no checkpoint and are reused as they stand. The Kinetics arms are not re-extracted; their
existing prediction files are re-read by the new report script, which must reproduce the
published numbers (gate G4) before any SSv2 number is read.

Classifier grid unchanged from Stage B: `label_modes combined`, seeds 1–5, 20 epochs,
hidden 128, dropout 0.4, lr 3e-4, weight decay 0.01, patience 5, threshold selected on
validation for balanced accuracy. n_test = 244 videos.

Checkpoint facts checked before writing this plan: SSv2 config `use_mean_pooling=True`,
`num_frames=16`, 174 labels; `fc_norm` loaded from the checkpoint (weight mean 0.703,
bias mean −0.008); image processor mean/std/size identical to the Kinetics checkpoint.
The SSv2 fine-tune's official sampling rate is 2, which matches the stride already used.

## Primary comparison

**Statistic.** ΔA = A_ssv2 − A_kin, where each A is the seed-mean BA difference between
`full_frame` and `background_only` on the same 244 test videos. Paired bootstrap over
test videos, 10,000 resamples, seed 20260907; both checkpoints and both arms are
resampled on the same video draw, so the pairing is by video.

**Rule, fixed now.**

| outcome | reading |
| --- | --- |
| ΔA 95% CI lower bound > 0 | H2 supported: the SSv2 fine-tune needs the athlete more than the Kinetics fine-tune does. The Stage B caveat is rewritten as partly checkpoint-specific and R_ssv2 replaces R_kin as the number to quote. |
| ΔA 95% CI contains 0 | undetermined at n = 244. The Stage B caveat stands as written and gains one sentence: the athlete-deletion arm was replicated with an SSv2 checkpoint at R = R_ssv2 (with its CI). No claim of equivalence is made. |
| ΔA 95% CI upper bound < 0 | SSv2 needs the athlete less than Kinetics. Reported as such; H1 is not the explanation either, and the note says the shortcut is not reducible to the fine-tune label set. |

R_ssv2 and its bootstrap CI are reported alongside in every branch. R is a ratio of two
noisy numbers and is not used as a decision statistic.

## Secondary (reported, never decisive)

1. `person_crop_ssv2` − `full_frame_ssv2`, paired bootstrap CI. Stage B's dilution
   reading (person crop 0.666 > full frame 0.640, +0.026) is known to be fixed-split
   noise on repeated splits; this is reported for symmetry only.
2. Each SSv2 arm against pose-only (normalized, 0.650) through the existing Stage B
   report script, including late fusion at weight 0.5. The script's denominator gate
   (0.635 ± 0.010) already fails on the re-derived pose baseline and will exit non-zero;
   the JSON is still written and is what gets read.
3. Each SSv2 arm against the box-geometry control (0.578).

## Gates (all must pass before any SSv2 BA is looked at)

| gate | check |
| --- | --- |
| G1 checkpoint | `load_backbone` accepts the SSv2 checkpoint; `fc_norm` not at default init (done at planning time, re-logged by the extractor) |
| G2 completeness | 1623 raw bundles per SSv2 arm, all three arms; same video ids as the Kinetics arms |
| G3 controls changed something | per video, `person_crop` and `background_only` features differ from `full_frame`; at most 1 identical pair (the whole-frame box seen in Stage B) |
| G4 reproduction | the new report script, run on the existing Kinetics prediction dirs, returns BA 0.6401 / 0.6271 / 0.6662 for full_frame / background_only / person_crop and R_kin = 0.907, to 4 decimals |
| G5 frozen analysis | this plan and the report script are committed before any SSv2 classifier run |

## Exclusions and stopping

- No video is excluded; the split files are Stage B's.
- If G2 or G3 fails for one arm, that arm is re-extracted once. A second failure stops
  the run and is reported as a failed gate, not patched around.
- No hyperparameter of the grid is changed for SSv2. If an SSv2 arm's validation curve
  looks degenerate (every seed at the same BA), it is reported as such, not re-tuned.
- Repeated splits are not run. The comparison is against the fixed-split Stage B
  numbers, which is where the 91% figure lives.

## What this plan cannot show

- It cannot separate pre-training data from fine-tuning labels: the SSv2 checkpoint is
  also self-supervised pre-trained on SSv2. A positive result says "checkpoint", not
  "fine-tune labels".
- It says nothing about REHAB24-6, where the background arm sits at chance and the
  confound is identity and recording position.
- The 56% of signal that needs no pixels at all (box geometry, clip length) is a
  property of the annotation and cannot move with any checkpoint.

## Reproduce

```
# extraction (local GPU, .venv-cuda, three chunks per arm)
.venv-cuda/Scripts/python.exe scripts/video/run_videomae_feature_extraction.py \
  --model-name MCG-NJU/videomae-base-finetuned-ssv2 --variant <arm> \
  [--variant-manifest data/Fitness-AQA/Squat/Labeled_Dataset/videos_<arm>/manifest.json] \
  --output-dir data/Fitness-AQA/Squat/Labeled_Dataset/videomae_raw_ssv2_<arm> \
  --device cuda --num-chunks 3 --chunk-index <0|1|2>
# materialize into a checkpoint-scoped parent (never the default parent: it would overwrite the Kinetics dirs)
.venv/Scripts/python.exe scripts/video/materialize_videomae_features.py \
  --raw-dir data/Fitness-AQA/Squat/Labeled_Dataset/videomae_raw_ssv2_<arm> \
  --output-parent data/Fitness-AQA/Squat/Labeled_Dataset/ssv2/<arm> \
  --token-pooling mean_pool_fc_norm --aggregation mean
# grid, one call per arm, same flags as Stage B step 6 without --normalize-features
# report: scripts/video/run_checkpoint_robustness_report.py (to be added under this plan)
```
