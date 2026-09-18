# Plan: compare frozen VideoMAE and V-JEPA 2 ViT-L for REHAB24-6 correctness

*REHAB24-6 · prospective comparison · written 2026-09-18 · status: plan only*

**Goal.** Test whether frozen V-JEPA 2 ViT-L features improve held-out-subject
exercise-correctness classification over the repository's corrected VideoMAE
features, and measure the additional inference cost. The primary comparison uses
each checkpoint's configured clip length. A mandatory shorter-clip control tests
whether the conclusion survives matching the sampled frames.

**Status.** No features, predictions or performance measurements have been produced
for this comparison. Commit this plan and the resolved run configuration before
full extraction or evaluation. Kaggle GPU is the intended extraction environment;
readout training can use the local CPU. All gates below are pending.

## Background

The [framing experiment](rehab24_videomae_framing_results.md) established the
whole-frame letterbox preprocessing and corrected VideoMAE pooling used here.
The [temporal experiment](rehab24_videomae_temporal_shuffle_results.md) found an
undetermined frame-shuffle effect on within-session ranking. It does not establish
that VideoMAE ignores temporal order. This comparison asks about predictive
performance; a stronger V-JEPA result would not by itself resolve that mechanism.

An implementation issue must be addressed before comparison: in
`src/rehab24/videomae_features.py`, clip starts respect repetition boundaries,
but `read_clip_frames` stops at the source video's end, not the repetition's end.
Short repetitions can therefore include later frames. Extending that sampler to
64 frames without a boundary guard would amplify the problem. Re-extract both
models with the bounded sampler below; historical scores are context only.

## Terminology

| Term | Meaning |
| --- | --- |
| Repetition | One annotated execution with a binary correct/incorrect label |
| Sample | One repetition observed by one camera |
| Session | One exercise and recording, keyed by `exercise_id + video_id` |
| LOSO | Leave-one-subject-out evaluation; the held-out person is never used to fit the readout |
| OOF | Out-of-fold prediction made while that subject is held out |
| Frozen readout | A small classifier learns from cached embeddings; encoder weights stay fixed |

## Models and comparisons

| Arm | Checkpoint | Input per sample | Representation | Role |
| --- | --- | --- | --- | --- |
| `vm16` | `MCG-NJU/videomae-base-finetuned-kinetics` | 4 clips × 16 frames, stride 2, 224 square | Mean over patch tokens, checkpoint `fc_norm`, then mean over clips; 768 dimensions | Baseline |
| `vj64` | `facebook/vjepa2-vitl-fpc64-256` | 4 clips × 64 frames, stride 2, 256 square | Mean over final encoder tokens, then mean over clips; 1024 dimensions | Primary challenger |
| `vj16` | Same V-JEPA 2 checkpoint | Exact `vm16` source-frame indices, 256 square | Same pooling as `vj64` | Mandatory temporal-budget control |

Use the original V-JEPA 2, not V-JEPA 2.1, an action-conditioned model, or an
action-classification probe. The official [model configuration](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/main/config.json)
specifies 64 frames, resolution 256 and hidden size 1024. Load the encoder through
the [official model interface](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256),
verify which output is the encoder representation, and do not pool predictor
outputs. Record the exact output field and normalization in the run configuration.
Check 16-frame support with a smoke test; do not silently pad it to 64 and call it
a matched-frame arm. If unsupported by the selected implementation, report the
control as blocked and revise the protocol before outcome inspection.

The [VideoMAE checkpoint](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)
is already fine-tuned on Kinetics. This is a practical checkpoint comparison:
capacity, pretraining data, supervision and spatial resolution differ. Even
`vj16` controls only temporal input, not all these factors. No full encoder
fine-tuning, pose fusion or checkpoint search is included in this first experiment.

## Dataset, exclusions and sampling

Use the existing processed manifest and binary labels from `Segmentation.csv`.
Freeze their hashes and a table of counts by subject, exercise, camera and label
before extraction. The existing dataset builder includes each segment/camera row
and does not implement uncertain-label or annotation-quality filtering. Preserve
that cohort and assert labels are exactly binary; do not silently add exclusions.
Correct is the positive class. Expected historical coverage is 2,144 camera
samples / 1,072 repetitions; confirm these counts against the frozen inputs.

Primary reporting covers P1–P9. Preserve the historical fold membership including
P10 in eligible training sets; P10 is a separate small-sample sensitivity, never
an eligible validation subject. Report a second sensitivity that removes P10
from training as well. Cameras of a repetition and every recording from one
person must remain together under subject splitting.

For each arm, calculate four evenly spaced starts within the annotated interval
using the existing start-selection rule with that arm's clip length. Convert the
annotation's frame numbering to decoder indices once, using the dataset's verified
convention. Generate stride-2 indices and clamp every index to that repetition's
inclusive bounds, repeating the final in-repetition frame when necessary. `vm16`
spans 31 source frames and `vj64` spans 127 before clipping. Store all actual
indices, unique-frame counts and padding fractions. Duplicate clips in short
repetitions are retained and counted, not discarded.

Use the existing `full_frame_letterbox` geometry: preserve the whole frame and
pad to square before resizing to each model's resolution. Use each checkpoint's
required RGB scaling and normalization. Disable any subsequent crop that would
remove the padding or subject. Save example tensors/images for both cameras and
all six exercises. Do not pass labels, recording position, subject identity,
exercise ID or camera ID as classifier inputs. Do not apply random augmentation.

Any decode error or missing feature blocks evaluation. Repair it, or freeze one
common exclusion list with reasons before outcomes are inspected; never let the
feature loader silently choose a different subset for each arm.

## Readout and folds

Reuse `videomae_stage_a.run_arm`, `loso_cross_validation.train_one_fold` and
`FoldConfig`. Freeze the explicit train/validation/test sample lists once for all
arms and seeds. Subjects are ordered numerically; validation is the next cyclic
subject with at least 100 samples, excluding the test subject. This skips P10 for
validation. All remaining subjects train the readout.

Use seeds **42, 7, 1234**. Fit feature mean and standard deviation on the training
subjects only. Use the existing MLP with hidden width 128, dropout 0.4, AdamW,
learning rate 0.0003, weight decay 0.01, batch size 32, at most 20 epochs and
early-stopping patience 5. Compute the binary loss's positive weight from training
labels only. Select checkpoint and decision threshold using validation balanced
accuracy only. No test-based threshold adjustment or hyperparameter sweep.

The readout recipe is shared, but its input-layer parameter count differs with
embedding dimension. Report both parameter counts. As a secondary check, run an
L2 logistic regression on each cached arm with `C=1`, training-only standardization,
training-derived balanced class weights and validation-selected threshold. Use
a fixed solver and convergence tolerance recorded before evaluation; convergence
failures must be resolved without consulting test results.

## Primary endpoint and inference

For each subject and seed compute balanced accuracy (BA), the mean of correct-class
and incorrect-class recall, over its repetition-camera samples at the fold's
validation-selected threshold. Average the three seed scores within each subject.
The primary endpoint is the equally weighted mean over the nine subjects.
The sole primary contrast is `vj64 − vm16`.

Report each arm's mean ± between-subject SD, paired delta mean ± SD, all nine
subject deltas, number of positive deltas and a paired subject-bootstrap 95% CI
(10,000 resamples, seed 20260918). Resample subjects with all their seeds and
camera observations intact. Use an exact two-sided paired sign-flip test of the
mean delta, enumerating all 512 sign assignments; zeros remain zeros. State its
exchangeability assumption and the limited precision of nine subjects. Clips,
cameras, repetitions and seeds do not increase the inferential sample size.

Pre-specify **0.02 BA** as a practical improvement threshold, a project decision
margin rather than a clinically established effect size. Claim a promising
improvement only if the mean delta is at least 0.02, at least 7/9 subjects improve,
the paired CI excludes zero and primary p < 0.05. Also require successful gates.
The CI need not exceed 0.02, so this rule does not establish a gain of at least
0.02 with statistical confidence.

| Outcome | Interpretation |
| --- | --- |
| All improvement conditions pass | V-JEPA 2 is a promising frozen-feature replacement under this recipe; assess cost before adoption |
| Significant positive delta below the practical threshold | Detectable gain of limited practical size |
| Significant negative delta | VideoMAE performs better under this recipe |
| Other outcomes, including p ≥ 0.05 | Undetermined; no equivalence or absence-of-effect claim |
| Any required data/model gate fails | No valid primary comparison until repaired and documented |

## Secondary analyses and controls

Report `vj16 − vm16` and `vj64 − vj16` BA contrasts to separate the shorter-input
comparison from the benefit of the longer sampling window. Report within-session
ROC-AUC for the primary arms: average camera probabilities for each repetition,
compute AUC only in sessions with both labels, average seed AUCs within session,
then sessions within subject and subjects equally. List excluded single-class
sessions. Ranking within sessions does not eliminate recording-position shortcuts.

The secondary testing family contains the two temporal-budget BA contrasts, the
`vj64 − vm16` within-session AUC contrast, and the `vj64 − vm16` logistic-readout
BA contrast. Apply Holm correction across these four tests, using the same paired
subject sign-flip procedure. Other summaries are descriptive: macro-F1, recall,
specificity, pooled ROC-AUC, per-exercise and per-camera metrics, padding strata,
and both P10 sensitivities. Camera/exercise strata require at least 20 samples
and both labels, matching the existing reporter. Include counts and mark undefined metrics rather than
assigning them a score of zero. No secondary result replaces the primary.

Run three label-shuffled readout checks per arm with seeds 101, 202, 303,
permuting at repetition level within subject and preserving camera-pair labels.
Refit normalization/readout and select thresholds anew on each permuted dataset.
These are leakage diagnostics, not a calibrated significance test. Audit any
unexpectedly strong null result before interpreting the comparison.

Temporal-shuffle and static-frame interventions on V-JEPA are a separate follow-up
if mechanism becomes the research question. Better classification alone does not
show better motion understanding.

## Kaggle execution and resource budget

1. Inventory the existing local CUDA environment and Kaggle input artifacts.
   Use Kaggle for extraction if local capacity is insufficient or less practical.
   Keep the local CPU-only environment intact. Use the repository-prescribed CLI:
   `uv run --with kaggle kaggle ...`, not the Kaggle MCP tools.
2. Pin source commit, checkpoint revisions, processor settings, Python, PyTorch,
   Transformers, decoder and CUDA versions. Smoke-test both encoders in the same
   Kaggle environment where possible. Never install an unpinned moving branch in
   the final run. Stage only required data/code, keep Kaggle artifacts private,
   and use existing authorized dataset inputs where available.
3. On the GPU actually allocated, pilot 24 repetition-camera samples covering
   both cameras, all exercises and short/long repetitions. Run inference mode,
   one encoder at a time, starting at batch size 1 with supported mixed precision
   and memory-efficient attention. Validate pooled features against FP32 on a
   small subset (finite values, cosine similarity at least 0.999); otherwise use
   FP32 or resolve numerical differences before full extraction.
4. Record GPU model and VRAM, peak allocated memory, decoder time, encoder-only
   latency and end-to-end seconds per repetition-camera sample. Warm up first,
   synchronize CUDA for timing, and report median and p95. Benchmark all arms on
   the same device and batch size; report throughput at any larger safe batch
   separately. Do not assume two GPUs share one memory pool.
5. Forecast each arm's extraction time as sample count × pilot mean time per
   sample, plus measured setup time and a 30% operational reserve. Compare with
   the account's actual available quota and session limits. No fixed runtime or
   Kaggle GPU availability is assumed here. Split jobs into resumable shards
   comfortably below the observed session limit; distribute independent shards
   if multiple devices are available.
6. Save each completed sample atomically and export completed shard bundles.
   Resume only when manifest, weights, sampler, processor and code fingerprints
   match. Use the same feasible batch/precision policy across timing comparisons.
   If `vj64` cannot fit batch size 1, report it as blocked; do not reduce frame
   count or resolution and still label the run `vj64`.
7. Download and checksum features and logs, audit coverage, then run the shared
   CPU readout. Core evaluation is 3 arms × 10 folds × 3 seeds = 90 small MLP
   fits; null checks and the declared sensitivities add readout fits but no GPU
   extraction. Publish total GPU-hours and cache sizes with the accuracy table.

No training-quality scores are inspected during the pilot. GPU work is not
launched by writing this plan; quota and measured feasibility are resolved at
execution time.

## Gates

| Gate | Required evidence | Current status |
| --- | --- | --- |
| Frozen protocol | Committed plan, code/config revisions, sample and fold hashes before evaluation | Pending |
| Model identity | Correct revisions, encoder field, pooling, feature shapes, frozen weights, valid 16-frame control | Pending |
| Sampling and geometry | All indices inside the annotated rep; matched `vm16`/`vj16` indices; crop/normalization checks | Pending |
| Feature integrity | Exact expected sample IDs in every arm, finite vectors, no duplicates, complete camera pairs | Pending |
| Fold purity | Disjoint subjects; normalization and class weights training-only; threshold validation-only | Pending |
| Reproducibility | Repeated pilot produces stable features within registered tolerance; bounded VM sampler matches old sampler on non-padding clips | Pending |
| OOF integrity | One prediction per expected sample/seed/arm; subject equals test subject; correct labels and camera pairing | Pending |
| Feasibility | Successful pilot, recorded memory/time, complete resumable extraction within available resources | Pending |

## Implementation deliverables

Keep logic under `src/` and command wrappers under `scripts/`. Proposed new files
are not existing commands:

| Area | Planned change |
| --- | --- |
| Shared sampling | Add a repetition-bounded index helper and use it in both extractors; preserve old artifact provenance |
| `src/video/vjepa2_backbone.py` | Pinned encoder loading, processor adapter and documented token pooling |
| `src/rehab24/vjepa2_features.py` | Chunked extraction with full indices, padding counts, hashes and resume validation |
| `scripts/rehab24/extract_vjepa2_features.py` | Thin extraction CLI supporting the registered 16/64-frame arms |
| `src/rehab24/video_backbone_comparison.py` | Frozen fold manifest, existing MLP reuse, logistic control, OOF export and paired statistics |
| `scripts/rehab24/compare_video_backbones.py` | Thin evaluation/report CLI |
| Kaggle bundle | Private kernel metadata, dependency lock, checkpoint revisions and resumable shard configuration |

Tests must cover frame-boundary conversion and short-rep padding, exact matched
indices, preprocessing layouts, pooling axes, stale-cache rejection, completeness,
fold leakage, camera-paired null permutations and subject-level statistics. Use
small tensors and synthetic manifests for unit tests; a real GPU smoke test
verifies the actual checkpoint interface. Run the relevant ML tests and required
repository checks before extraction. Update `scripts/rehab24/README.md` with
verified CLI commands once implemented.

## Deviations from the plan

None at drafting time. Log every later change with its date, reason, whether
outcomes were already visible and its effect on interpretation. The bounded
sampler is an intentional difference from historical runs, registered here.
Do not overwrite historical feature folders or use their scores as paired data.

## Not supported

- Causal superiority of the JEPA objective: model size, data and supervision differ.
- Clinical validity, named fault recognition or coaching quality from binary labels.
- Generalization to new laboratories, camera setups or patient populations.
- Better temporal reasoning solely from better correctness classification.
- Equivalence from a non-significant result, or a larger sample size by counting seeds.
- A verdict on optimal fine-tuning or attentive probing; only the registered frozen readouts are tested.

## Reproduce

Execution order: commit protocol/config → validate manifest and folds → implement
and test adapters → Kaggle pilot → freeze resolved environment and shard budget →
extract all three arms → audit → fit readouts → export OOF → paired report.

Existing reference code: `src/rehab24/videomae_features.py`,
`src/video/videomae_backbone.py`, `src/rehab24/videomae_stage_a.py`,
`src/rehab24/loso_cross_validation.py`, and
`src/rehab24/videomae_identity_control.py`. The V-JEPA reference implementation is
the [official repository](https://github.com/facebookresearch/vjepa2).

Store this experiment under
`data/REHAB24-6/processed/video_backbone_comparison/<run_id>/`, containing
`run_config.json`, `manifest_audit.json`, `folds.json`, `environment/`,
`raw/<arm>/`, `features/<arm>/`, `oof/<arm>/seed<seed>.csv`, `gates.json`,
`timing.json` and `summary.json`. Raw bundles retain per-clip pooled features;
materialized bundles expose `video_feature` for the existing readout. OOF rows
include sample, repetition, session, subject, exercise, camera, seed, label,
probability, threshold and fold IDs.

The results note must include the measured accuracy/cost table, all subject
deltas, gate outcomes, exclusions, null diagnostics, deviations, actual hardware,
runtime and exact runnable commands. No runtime estimate becomes a measured
result. Validate this plan locally with:

```powershell
.venv\Scripts\python.exe .claude/skills/write-experiment-note/check_note.py notes/rehab24_videomae_vjepa2_validation_plan.md
```
