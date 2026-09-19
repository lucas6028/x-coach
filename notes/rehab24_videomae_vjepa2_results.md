# Frozen V-JEPA 2 scores 0.622 vs VideoMAE 0.662 LOSO balanced accuracy on REHAB24-6 (undetermined) and 0.070 lower within-session AUC (Holm p = 0.031)

*REHAB24-6 · frozen video-backbone comparison · run 2026-09-18 to 2026-09-19 ·
plan [`rehab24_videomae_vjepa2_validation_plan.md`](rehab24_videomae_vjepa2_validation_plan.md)*

**Result.** With the same repetition-bounded sampler, the same MLP readout and the same
nine held-out subjects, frozen V-JEPA 2 ViT-L features at their native 64-frame clip
(`vj64`) give leave-one-subject-out balanced accuracy 0.6222 ± 0.0595 against
0.6618 ± 0.0506 for the corrected VideoMAE features (`vm16`): paired delta
−0.0395 ± 0.0739, 1/9 subjects improve, bootstrap 95% CI [−0.090, −0.004], exact
sign-flip p = 0.066. Under the pre-registered rule that is **undetermined**, not a
demonstrated loss. The registered secondary that ranks repetitions inside one recording
is clearer: within-session ROC-AUC 0.8049 ± 0.0574 vs 0.8750 ± 0.0482, delta
−0.0701 ± 0.0607, 1/9, CI [−0.109, −0.035], p = 0.0078, Holm-corrected 0.031. V-JEPA 2
is not a promising frozen-feature replacement under this recipe, and it costs about
eight times the encoder time per sample.

**Caveat on reading it.** The 16-frame V-JEPA control (`vj16`, identical frame indices
to `vm16`) sits between the two arms at 0.6395 ± 0.0516; both temporal-budget contrasts
are undetermined (p = 0.42 and 0.41). Nine subjects cannot resolve deltas of 0.02–0.04.
One gate (model identity) failed on its first run because of a key-name lookup bug in
the gate itself and was fixed after the primary delta had been printed; the fix touched
no data, feature, threshold or statistic and is logged under Deviations.

## Terminology

| term | meaning |
| --- | --- |
| `vm16` | VideoMAE-base fine-tuned on Kinetics, 4 clips × 16 frames (stride 2) per repetition, mean over patch tokens then the checkpoint's `fc_norm`, 768 dimensions; the repository's corrected baseline |
| `vj64` | V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), 4 clips × 64 frames (stride 2), mean over the encoder's final tokens, 1024 dimensions; the primary challenger |
| `vj16` | the same V-JEPA 2 encoder fed exactly the `vm16` frame indices; the temporal-budget control |
| repetition-bounded sampler | every sampled frame index is clamped inside the annotated repetition; short repetitions repeat their last frame (registered fix for the historical extractor, which only stopped at the end of the video) |
| LOSO | leave-one-subject-out: the readout never sees the tested person; validation is the next subject in numeric order with ≥ 100 samples |
| BA | balanced accuracy, the mean of correct-class and incorrect-class recall at the fold's validation-selected threshold |
| within-session AUC | ROC-AUC of repetition scores inside one recording (exercise × video), only in recordings with both labels; the statistic the identity and temporal controls report |
| padding fraction | share of a clip's 64 (or 16) frames that are repeats because the repetition is shorter than the clip's 127-frame span |

## Background

VideoMAE's corrected pooling gives 0.657–0.661 LOSO balanced accuracy on REHAB24-6
([framing note](rehab24_videomae_framing_letterbox_results.md)); the temporal-shuffle
control ([note](rehab24_videomae_temporal_shuffle_results.md)) found no order effect
that nine subjects can resolve. The open question was whether a newer, larger,
self-supervised video encoder with a four-times longer clip would classify correctness
better as a frozen feature, and at what inference cost. The plan also registered a
sampler defect: historical clips could run past the end of a short repetition, so every
arm here was re-extracted with the bounded sampler and no historical score is paired.

## Method

Three arms, one frozen readout. Every arm decodes the same 2,144 repetition-camera
samples (1,072 repetitions × 2 cameras, 65 recordings, subjects P1–P10) with the same
cv2 decoder, letterboxes the whole 1080p frame to a grey square and resizes it to the
model's input size (224 for VideoMAE through its own processor, 256 for V-JEPA 2 with the
processor's centre crop disabled), and takes four evenly spaced clips per repetition.
`vm16` and `vj16` draw byte-identical frame indices (verified on all 2,144 samples);
`vj64` spans 127 source frames per clip, so 1,410 of its 2,144 samples contain some
padding (mean padding fraction 0.166, 72 samples above 0.5) while `vm16`/`vj16` pad only
2 samples.

The design decision a reader would otherwise get wrong: all arms ran in float32. On the
GTX 1660 Ti used here fp16 is slower than fp32 (no tensor cores: 14.8 vs 7.9 s per
64-frame clip), and the fp16/fp32 pooled features agree to cosine 1.0, so the plan's
"mixed precision if validated" branch was not needed. V-JEPA 2 is loaded at snapshot
`b3c1679b`, `skip_predictor=True`, encoder `last_hidden_state` mean-pooled over tokens;
VideoMAE at snapshot `488eb9a0`.

Readout per fold and seed (42, 7, 1234): features standardised on training subjects,
MLP with one hidden layer of 128 (dropout 0.4, AdamW, lr 3e-4, weight decay 0.01,
batch 32, ≤ 20 epochs, patience 5), positive class weight from training labels,
threshold chosen on validation BA only. Parameter counts differ with input width:
98,561 (768-dim) vs 131,329 (1024-dim). A torch L-BFGS logistic regression (C = 1,
balanced class weights, tolerance 1e-6) is the registered secondary readout. Subjects
P1–P9 form the primary set; P10 (16 samples) trains but is never validated or counted.

Statistics fixed before results: per-subject BA averaged over seeds, equal-weight mean
over nine subjects, paired subject bootstrap (10,000 resamples, seed 20260918), exact
two-sided sign-flip test over all 512 sign assignments, practical margin 0.02 BA, and a
Holm family of exactly four secondaries (`vj16 − vm16` BA, `vj64 − vj16` BA,
`vj64 − vm16` within-session AUC, `vj64 − vm16` logistic BA). Plan commit `37e21915`,
readout `4c89d6ae`, extractor `9b0d8fad`; `run_config.json` stamps `9b0d8fad` and the
manifest md5 `b27e9069…`; folds were hashed before any feature existed.

## Gates

| gate | result |
| --- | --- |
| frozen protocol | pass; plan, code and fold hashes committed before extraction, run config re-stamped with the final extractor commit |
| model identity | pass after the lookup fix (Deviations, row 4): every bundle carries the pinned revision, clip length, resolution and pooling field; measured dimension 768 / 1024 / 1024 on all 2,144 bundles per arm |
| sampling and geometry | pass; every frame index inside the manifest-derived repetition bounds for all three arms, `vm16` and `vj16` indices identical on 2,144 / 2,144, example frames show the grey letterbox band with the subject uncropped |
| feature integrity | pass; exactly the 2,144 expected ids per arm, finite, no duplicates, complete camera pairs, one provenance per arm |
| fold purity | pass; disjoint subjects in every fold, validation never the test subject and never P10 |
| reproducibility, vm16 vs historical | pass with one documented exception: 2,140 / 2,142 zero-padding samples byte-identical to `videomae_raw_full_frame_letterbox`; the two exceptions are one repetition (Ex5_PM_117a_rep9, both cameras) whose annotated last frame equals the container's frame count, i.e. the registered video-end defect; last clip only, cosine ≥ 0.9994 |
| reproducibility, repeated extraction | pass; the 24 pilot samples extracted a day earlier in a separate process are byte-identical to the full-run bundles for both `vj16` and `vj64` |
| OOF integrity | pass; one prediction per sample, seed and arm, `person_id` equals the test subject, labels equal the frozen labels, camera pairs complete |
| feasibility | pass; all three arms extracted locally, timings below |

## Results

Primary set P1–P9, three seeds averaged within subject.

| arm | BA (mean ± SD) | delta vs `vm16` | improve | bootstrap 95% CI | sign-flip p | Holm |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `vm16` | 0.6618 ± 0.0506 | — | — | — | — | — |
| **`vj64`** (primary) | 0.6222 ± 0.0595 | −0.0395 ± 0.0739 | 1/9 | [−0.090, −0.004] | 0.066 | primary, uncorrected |
| `vj16` (control) | 0.6395 ± 0.0516 | −0.0222 ± 0.0723 | 3/9 | [−0.070, +0.020] | 0.418 | 1.00 |
| `vj64 − vj16` | — | −0.0173 ± 0.0578 | 2/9 | [−0.047, +0.023] | 0.406 | 1.00 |

Per subject, `vm16` → `vj64`: P1 0.587→0.579, P2 0.706→0.662, P3 0.623→0.596,
P4 0.617→0.609, P5 0.634→0.583, P6 0.677→0.716, P7 0.673→0.646, P8 0.749→0.525,
P9 0.690→0.683. Deltas: −0.008, −0.044, −0.027, −0.008, −0.051, +0.039, −0.027,
−0.224, −0.007.

The pre-registered verdict table gives **undetermined**: the delta is negative and the
bootstrap interval excludes zero, but the exact test does not reach p < 0.05
(34 of 512 sign assignments are at least as extreme), so neither "promising improvement"
nor "VideoMAE performs better" is claimed. The sign is carried by eight of nine subjects,
with P8 alone contributing −0.224 of the mean; without P8 the mean delta would be
−0.016, which is descriptive only.

The `vm16` baseline reproduces the framing note's 0.6612 (here 0.6618, bounded sampler,
2,140 / 2,142 identical features), so the comparison sits on the established number.

## Secondary

Within-session ROC-AUC, 61 mixed-label recordings over P1–P9 (three single-class
recordings excluded: Ex1_PM_000 of P1, Ex5_PM_028 of P3, Ex3_PM_044 of P5; P10's
recording excluded by the primary set).

| arm | AUC (mean ± SD) | delta | improve | bootstrap 95% CI | p | Holm |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `vm16` | 0.8750 ± 0.0482 | — | — | — | — | — |
| `vj64` | 0.8049 ± 0.0574 | −0.0701 ± 0.0607 | 1/9 | [−0.109, −0.035] | 0.0078 | **0.031** |

Per subject: P1 0.836→0.753, P2 0.896→0.777, P3 0.893→0.830, P4 0.809→0.815,
P5 0.823→0.741, P6 0.954→0.901, P7 0.860→0.843, P8 0.928→0.732, P9 0.877→0.852.
This is the one contrast that survives Holm correction; it ranks repetitions inside a
recording, so it does not remove recording-position shortcuts (see the position-control
note), and it is a secondary, not the primary.

Logistic readout (L-BFGS, converged on all 10 folds per arm): `vm16` 0.6543 ± 0.0432,
`vj64` 0.6579 ± 0.0572, delta +0.0036 ± 0.0794, 4/9, CI [−0.044, +0.054], p = 0.89,
Holm 1.00. The MLP and the linear readout disagree in sign, which is consistent with
both deltas being inside the noise of nine subjects.

Descriptive summaries (pooled over subjects and seeds, thresholds as selected):

| arm | pooled ROC-AUC | macro-F1 | recall | specificity | cam17 BA | cam18 BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vm16` | 0.7228 | 0.6465 | 0.708 | 0.616 | 0.634 | 0.690 |
| `vj16` | 0.6831 | 0.6257 | 0.617 | 0.662 | 0.629 | 0.652 |
| `vj64` | 0.6492 | 0.6051 | 0.590 | 0.655 | 0.591 | 0.638 |

Per exercise BA (`vm16` / `vj16` / `vj64`; n rows per stratum 594–1,260): Ex1
0.706 / 0.571 / 0.579, Ex2 0.603 / 0.633 / 0.660, Ex3 0.664 / 0.533 / 0.510, Ex4
0.654 / 0.763 / 0.686, Ex5 0.676 / 0.633 / 0.626, Ex6 0.633 / 0.626 / 0.553. Strata are
pooled rates and not paired statistics.

Padding strata for `vj64` (mean padding fraction per sample; rows = samples × 3 seeds):
0 → 0.586 (2,202 rows), (0, 0.5] → 0.632 (3,966), > 0.5 → 0.564 (216). The zero-padding
stratum is not the best one, so the shortfall is not explained by repeated frames alone.
`vm16` and `vj16` have 2 padded samples and no defined non-zero stratum.

P10 sensitivities: the 16-sample P10 test fold scores 0.761 (`vm16`), 0.639 (`vj16`),
0.533 (`vj64`); removing P10 from training moves the nine-subject means to 0.6703,
0.6381 and 0.6241 (delta `vj64 − vm16` −0.046), the same direction as the primary.

## Null checks

Label-shuffled readouts (repetition-level permutation within subject, camera pairs kept,
seeds 101 / 202 / 303), nine-subject mean BA:

| arm | seed 101 | seed 202 | seed 303 |
| --- | ---: | ---: | ---: |
| `vm16` | 0.517 | 0.501 | 0.489 |
| `vj16` | 0.504 | 0.505 | 0.506 |
| `vj64` | 0.513 | 0.515 | 0.497 |

All nine nulls sit at chance (registered audit threshold 0.55 never reached), so no
leakage path is indicated for any arm. These are leakage diagnostics, not a calibrated
test.

## Cost

Uncontended 24-sample pilot on the GTX 1660 Ti (6 GB), float32, batch 1, medians per
repetition-camera sample (four clips):

| arm | encoder s | decode s | end-to-end s | peak allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| `vj16` | 3.69 (p95 3.75) | 5.04 | 8.70 (p95 9.79) | 1,361 |
| `vj64` | 30.2 (p95 32.6) | 20.5 | 51.8 (p95 56.3) | 1,685 |
| `vm16` | not timed (no timer in the VideoMAE CLI); the historical rate is 2.2–9 s per sample, CPU-bound | | | |

The `vj64` encoder is 8.2× the `vj16` encoder per sample for a `vj64 − vj16` BA delta of
−0.017 (undetermined). Full-run figures are contended (two to four extraction processes
shared one GPU and a six-core CPU): encoder medians 13.7 s (`vj16`, 1,460 timed) and
41.9 s (`vj64`, 1,913 timed), summed encoder time 5.7 and 23.0 GPU-hours, wall clock
first-to-last bundle 11.1 h (`vm16`), 12.7 h (`vj16`), 34.4 h (`vj64`). Raw caches are
61 / 45 / 30 MB. The `vm16` arm was CPU-bound in the HuggingFace processor's resize of
the 1920 × 1920 letterboxed frames (~27 s per sample under contention, profiled); a
GPU-side resize was tested and rejected because it changes the features (cosine 0.9992
against the CPU path that reproduces the historical bundles exactly).

## Deviations from the plan

| item | plan | actual | effect |
| --- | --- | --- | --- |
| extraction host | Kaggle intended, local allowed if practical | local GTX 1660 Ti for all three arms in one venv | none on numbers; same environment for every arm |
| precision | mixed precision if validated against fp32 | float32 throughout (faster on this GPU; fp16 cosine 1.0) | none |
| worker scheduling | two resumable workers | resume-safe helper workers added twice (vm16 chunks, then a vj16 chunk) after profiling showed the CPU at 27–40%; one worker crashed on a Windows file-in-use error at a crossover and was relaunched | none; duplicates at crossovers are byte-identical atomic rewrites, completeness checked by bundle count |
| model-identity gate | pass before outcomes | failed on all arms with `actual: None` on the first report because the gate read bare keys while bundles stamp `provenance_<key>`; fixed the lookup and added two tests after the primary delta had been printed | none on any number; recorded because the fix post-dates a visible outcome |
| vm16 vs historical tolerance | rel. L2 ≤ 1e-3 on all zero-padding samples | met on 2,140 / 2,142; the two exceptions are the registered video-end defect | none; documented exception |
| vm16 timing | pilot timing for all arms | VideoMAE CLI has no timer; wall clock and a profile reported instead | cost table incomplete for `vm16` |
| per-sample timing coverage | all samples | `vj64` 1,913 / 2,144 and `vj16` 1,460 / 2,144 timed (the crashed worker's and one helper's timers were not written) | contended medians only |

## Not supported

- Any claim that the JEPA objective, model size or pretraining data is the cause: the
  checkpoints differ in all of these and `vj16` controls only the temporal input.
- "VideoMAE is better": the primary p = 0.066 is undetermined under the registered
  rule; only the within-session AUC secondary reaches Holm-corrected significance.
- Equivalence of any two arms, or a larger sample size from counting seeds, cameras or
  clips: the inferential unit is nine subjects.
- Better or worse temporal reasoning: classification performance says nothing about
  mechanism (the temporal-shuffle note covers what does).
- Fine-tuned, attentive-probe or fused versions of either encoder: only the registered
  frozen readouts were run.
- Clinical validity, named-fault recognition, or transfer to other laboratories, camera
  setups or patient groups.
- The cost table as a general benchmark: one consumer GPU, float32, batch 1, cv2 decode
  of 4:4:4 1080p video, mostly under contention.

## Reproduce

Code: `src/rehab24/clip_sampling.py` (bounded sampler), `src/video/vjepa2_backbone.py`,
`src/rehab24/vjepa2_features.py`, `src/rehab24/videomae_features.py` (`--sampler bounded`),
`src/rehab24/video_backbone_comparison.py`; CLIs under `scripts/rehab24/`. Tests:
`tests/test_rehab24_clip_sampling.py`, `tests/test_vjepa2_backbone.py`,
`tests/test_rehab24_vjepa2_features.py`, `tests/test_rehab24_video_backbone_comparison.py`.

```powershell
$RUN = "data/REHAB24-6/processed/video_backbone_comparison/20260918_vjepa2_vs_videomae"
.venv\Scripts\python.exe scripts/rehab24/compare_video_backbones.py init --run-dir $RUN
.venv-cuda\Scripts\python.exe scripts/rehab24/extract_videomae_features.py --sampler bounded --arm vm16 --revision 488eb9a0565f257b32866000305c8178965eb9f6 --device cuda --run-dir $RUN
.venv-cuda\Scripts\python.exe scripts/rehab24/extract_vjepa2_features.py --arm vj16 --device cuda --run-dir $RUN
.venv-cuda\Scripts\python.exe scripts/rehab24/extract_vjepa2_features.py --arm vj64 --device cuda --run-dir $RUN
foreach ($arm in "vm16","vj16","vj64") {
  .venv\Scripts\python.exe scripts/rehab24/compare_video_backbones.py materialize --run-dir $RUN --arm $arm
  .venv\Scripts\python.exe scripts/rehab24/compare_video_backbones.py evaluate --run-dir $RUN --arm $arm --device cpu --nulls --no-p10-training
}
.venv\Scripts\python.exe scripts/rehab24/compare_video_backbones.py report --run-dir $RUN
```

Extraction may be split with `--num-chunks N --chunk-index k`; relaunching any command
resumes (completed bundles are skipped after a provenance check). Hardware: Intel
i7-10750H, NVIDIA GTX 1660 Ti 6 GB, Windows 11, torch 2.13.0+cu126, transformers 5.5.0,
OpenCV 4.13.0; total wall clock about 35 h for extraction with two to four concurrent
workers, plus about 20 min of CPU per arm for the readouts. Artifacts under `$RUN`:
`run_config.json`, `manifest_audit.json`, `folds.json`, `folds_no_p10.json`,
`environment/`, `pilot/`, `raw/<arm>/`, `features/<arm>/`, `oof/<arm>*/`, `gates.json`,
`gates_vm16_history.json`, `gates_repeatability_vj16.json`, `gates_repeatability_vj64.json`,
`padding_strata.json`, `timing.json`, `budget.json`, `deviations.json`, `summary.json`.
