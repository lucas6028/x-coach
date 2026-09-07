# Plan: does an SSv2-finetuned VideoMAE change the REHAB24-6 result, cross-subject or within-session?

*REHAB24-6 · pre-registered checkpoint comparison · written 2026-09-08, before any SSv2
feature was extracted on REHAB24-6 · siblings:
[`rehab24_videomae_framing_results.md`](rehab24_videomae_framing_results.md),
[`rehab24_videomae_identity_appearance_results.md`](rehab24_videomae_identity_appearance_results.md),
[`rehab24_videomae_position_control_results.en.md`](rehab24_videomae_position_control_results.en.md),
[`videomae_ssv2_checkpoint_results.md`](videomae_ssv2_checkpoint_results.md)*

**Question.** Every REHAB24-6 VideoMAE number in this project comes from the Kinetics-400
fine-tune: leave-one-subject-out balanced accuracy 0.6612 ± 0.0567 on `full_frame_letterbox`,
within-session ROC-AUC 0.8741 ± 0.0408, and a linearly readable position share of 4.9%. On
Fitness-AQA the Something-Something-v2 (SSv2) fine-tune moved the whole column up by
0.02–0.03 regardless of which pixels were deleted, and its late fusion with pose reached
0.702 (exploratory, one split). REHAB24-6 has a different confound structure: the scene
carries nothing (the person-deleted arm scores 0.5074 cross-subject and 0.5357
within-session), identity and appearance are closed, and recording position is closed
only along 16 linear directions. This plan asks two fixed questions and nothing else:
(1) does the SSv2 fine-tune change the cross-subject score, and (2) does it change how
much of the within-session ranking is linearly position-readable.

## Terminology

| term | meaning |
| --- | --- |
| arm | one feature set run through the unchanged REHAB24-6 LOSO recipe (`videomae_stage_a.run_arm`, frozen since the framing plan) |
| `full_frame_letterbox` | the whole frame padded to square with neutral grey, so the processor crops nothing; the framing experiment's best arm and the arm every later REHAB24-6 control was run on |
| `background_only` | the person's fixed per-video box (mocap union, +15%) filled from the pixels either side; scene and its drift kept, person gone |
| LOSO BA | leave-one-subject-out balanced accuracy, one fold per subject P1–P9, seeds 42 / 7 / 1234 averaged within subject before anything is compared; P10 (16 samples) excluded from the primary and reported as sensitivity |
| within-session AUC | inside one recording (same person, exercise, video), the probability that a correct repetition outranks an incorrect one; cameras averaged per repetition, seed AUCs averaged per session, sessions averaged per subject, n = 9; the identity-control statistic |
| position probe ρ | subject-macro within-class Spearman correlation between a LOSO ridge prediction and the repetition's rank among same-label repetitions of its recording; how much the features linearly encode position |
| position share | (AUC_k0 − AUC_k16) / (AUC_k0 − 0.5) after projecting out the 16 fold-fitted position directions; 4.9% for Kinetics, a floor |
| kin / ssv2 | `MCG-NJU/videomae-base-finetuned-kinetics` / `MCG-NJU/videomae-base-finetuned-ssv2` |

## Hypotheses

- **H1, checkpoint-general.** The REHAB24-6 signal is coarse person-region content that
  any video backbone reads the same way. Prediction: ΔBA and ΔAUC both inside ±0.02,
  probe ρ near the Kinetics 0.43, `background_only` at chance.
- **H2, motion-centric.** The SSv2 fine-tune reads within-repetition motion where the
  Kinetics fine-tune reads appearance. Prediction: ΔBA ≥ +0.02 with at least 6/9
  subjects positive, probe ρ_ssv2 below ρ_kin, position share at or below 4.9%,
  `background_only` at chance.
- **H3, drift-reading.** The SSv2 fine-tune's temporal sensitivity makes it a better
  reader of monotone drift inside a recording (stance, fatigue, camera settling).
  Prediction: ΔAUC > 0 with ΔBA undetermined or negative, probe ρ_ssv2 above ρ_kin,
  share above 4.9%, and `background_only` within-session AUC above the Kinetics 0.5357.

H2 and H3 both predict a within-session gain. The probe and the `background_only` arm
are what separate them, which is why both are in the plan and not optional.

## Design

Two SSv2 arms are extracted with the REHAB24-6 extractor unchanged except
`--model-name`: `full_frame_letterbox` (primary, all readouts) and `background_only`
(bound, cross-subject and within-session). Boxes, frame sampling (16 frames, stride 2,
4 clips), pooling (tokens mean-pooled through the checkpoint's own `fc_norm`, clips
averaged) and the materialize step are those of the framing plan. The Kinetics arms are
not re-extracted; their existing feature dirs are run through the same report commands in
the same invocation, so both checkpoints share folds, seeds, validation subjects and
threshold selection.

Checkpoint facts already established on Fitness-AQA and re-checked by the shared
`load_backbone` on every worker: `use_mean_pooling=True`, `fc_norm` loaded from the
checkpoint (weight mean 0.703, bias mean −0.008), image preprocessing identical to the
Kinetics checkpoint, official SSv2 sampling rate 2 = the stride already used.

`person_crop` is not extracted. On Kinetics it sat 0.0009 from `full_frame_letterbox`,
and on Fitness-AQA the crop-minus-full delta flipped sign between checkpoints inside
intervals that contained 0. A crop arm for SSv2 would be a separate plan.

## Primary comparison

**Statistic.** ΔBA = BA_ssv2 − BA_kin on `full_frame_letterbox`, paired by subject over
P1–P9, each subject's BA the mean over seeds 42 / 7 / 1234. Test: exact paired Wilcoxon,
two-sided, α = 0.05, as in the framing plan. Reported with it: mean ± SD of the nine
deltas, subjects positive, and a 9-subject bootstrap 95% interval (10,000 resamples,
seed 20260908). Practical band 0.02 balanced accuracy, unchanged from the framing plan.

**Power, stated before the run.** The framing plan's primary (letterbox minus full frame)
had nine per-subject deltas with SD 0.0567 and p = 0.570 at a mean of +0.011. A shift the
size of the Fitness-AQA column shift (+0.029) is therefore expected to come out
undetermined here. This design resolves a shift of roughly one SD, about 0.05 with 8/9
subjects concordant, and nothing smaller. An undetermined primary is the expected outcome
under H1 and is not read as a failure of the checkpoint.

**Rule, fixed now.**

| outcome | reading | consequence |
| --- | --- | --- |
| ΔBA ≥ +0.02, ≥ 6/9 subjects positive, p < 0.05 | SSv2 scores higher cross-subject on REHAB24-6 | SSv2 becomes the REHAB24-6 VideoMAE checkpoint; every later REHAB24-6 step, including any fusion pre-registration, uses it and quotes its number, carrying the secondary readouts as caveats |
| \|ΔBA\| < 0.02, or p ≥ 0.05 | undetermined at n = 9 | Kinetics remains the quoted checkpoint. The Fitness-AQA shift is neither reproduced nor refuted on REHAB24-6. No equivalence claim |
| ΔBA ≤ −0.02, p < 0.05 | SSv2 scores lower cross-subject | Reported as such; Kinetics remains |

The checkpoint choice is made by this rule and by nothing else: not by the within-session
AUC, not by a fusion score, not by any stratum.

## Secondary (reported, never decisive for adoption)

1. **S1, within-session ΔAUC** on `full_frame_letterbox`, paired by subject, exact
   Wilcoxon, bootstrap interval as above. Each arm also gets its own within-session label
   permutation null (10,000 draws, seed 20260908) and must independently clear the
   identity-control rule (mean ≥ 0.55, ≥ 6/9 subjects > 0.5, p < 0.05) to be quoted.
2. **S2, position readability of the SSv2 features.** The k = 0 probe (ρ median and mean
   with their 10,000-draw null) and the k = 16 residualization share, run with the
   position-control module unchanged, λ from the same inner sweep. Compared descriptively
   with Kinetics: ρ median 0.4324, share 4.9%. As in the position control, the probe after
   removal must reach its null on the median; if it does not, the share is a floor and is
   labelled one. Whether ρ_ssv2 is above or below ρ_kin is a direction, not a test; no
   p-value is attached to the cross-checkpoint probe difference.
3. **S3, `background_only` for SSv2.** LOSO BA against the Kinetics 0.5074 and
   within-session AUC against the Kinetics 0.5357, each with its own permutation null.
   Informative if BA ≥ 0.55 (the project's practical threshold) or within-session
   AUC − 0.5 > 0.15 (the identity note's drift-proxy margin), with p < 0.05.
4. **S4, strata.** ΔBA by camera (cam17 / cam18) and by exercise, from the framing report
   script, reported only.
5. **S5, P10-inclusive sensitivity** for the primary and S1.

Raw and Holm-corrected p-values for S1 and S3 (the family of secondary tests); S2 and S4
carry no test.

## Reading the secondaries (fixed now)

| pattern | reading |
| --- | --- |
| S1 ΔAUC > 0 with p < 0.05; S2 ρ_ssv2 < ρ_kin and share ≤ 4.9%; S3 not informative | H2 pattern. The within-session gain is reported as "SSv2 ranks repetitions inside a recording better and encodes less linear position". Still not evidence of movement reading; see Not supported |
| S1 ΔAUC > 0 with p < 0.05; S2 ρ_ssv2 > ρ_kin or share > 4.9%, or S3 informative | H3 pattern. The within-session gain is reported as drift-reading. If the primary passed, SSv2 is still adopted by the rule, but its number is never quoted without the S2/S3 figures, and the identity plan's fusion gate stays closed for SSv2 |
| S1 undetermined, primary undetermined, S3 not informative | H1 pattern. Nothing changes; both checkpoints keep their current caveats |
| S3 informative for SSv2 | A checkpoint-specific non-person path exists on REHAB24-6 that Kinetics did not have. Written into the SSv2 caveat and into the Fitness-AQA note's "nothing about REHAB24-6" line as the resolution |
| S1 ΔAUC < 0 with p < 0.05 | SSv2 ranks worse inside a recording; reported |

p ≥ 0.05 is written as undetermined throughout, never as "no difference" or "equivalent".

## Gates (all must pass before any SSv2 score is looked at)

| gate | check |
| --- | --- |
| G1 checkpoint | `load_backbone` accepts the SSv2 checkpoint through both of its gates (mean pooling, `fc_norm` not at default init), logged by every extractor worker |
| G2 completeness | 2,144 raw bundles per SSv2 arm (P1–P10, both cameras), sample-id set identical to the Kinetics `full_frame_letterbox` raw dir; provenance `model_name` = ssv2 in every bundle; 2,144 materialized `.npz` per arm |
| G3 features changed and variant applied | per sample, cosine(ssv2, kin) < 1 for all 2,144 (no bundle copied across); the framing feature audit (`audit_videomae_features.py`) passes on both SSv2 dirs as it did for Kinetics; geometry is unchanged by construction (same `videomae_boxes.json`) and is not re-run |
| G4 reproduction | the same report invocations, on the Kinetics dirs, return LOSO BA 0.6612 and within-session AUC 0.8741 to 4 decimals, and probe median 0.4324; the position control already showed the k = 0 path is byte-identical on CPU, so a miss here means the wrong feature dir or device |
| G5 frozen analysis | this plan and the checkpoint report script (below) are committed before any SSv2 classifier run on REHAB24-6 |

## Exclusions and stopping

- No sample is excluded. P10 is excluded from the primary by the standing REHAB24-6
  rule and appears only in S5.
- If G2 or G3 fails for one arm, that arm is re-extracted once. A second failure stops
  the run and is reported as a failed gate.
- No recipe hyperparameter, fold, seed, validation subject or threshold objective is
  changed for SSv2. A degenerate seed is reported, not re-tuned.
- Nothing about the primary rule, the 0.02 band, the P10 rule, camera or seed
  aggregation, or sidedness changes after any SSv2 number is seen. Any change is a plan
  deviation in the results note, with the registered analysis kept alongside.
- Fusion is out of scope. The fusion pre-registration that the identity plan allows
  (calibrated late fusion of NLF and the VideoMAE arm, validation-only calibration, one
  rule, subject-paired endpoint, stop on no improvement over NLF) names its VideoMAE
  checkpoint from this plan's primary rule, and a fusion score never feeds back into it.

## What this plan cannot show

- **Pre-training data and fine-tune labels are confounded.** The SSv2 checkpoint is also
  self-supervised pre-trained on SSv2. Any difference says "checkpoint", not "labels".
- **An undetermined primary is not equivalence.** n = 9 with SD ≈ 0.05 cannot resolve a
  0.03 shift, and the plan says so above.
- **A cross-subject gain is not movement reading on its own.** `repetition_number`
  alone reaches LOSO BA 0.7309, above both checkpoints, so position can leak
  cross-subject too. Only S2 and S3 staying flat make a gain readable as person-region
  content, and even then non-linear position paths are untested.
- **Drift that steps at the label boundary** is collinear with the label and invisible
  to every analysis here, as it was in the position control; 27 of 61 sessions are
  single-boundary.
- **Within-clip temporal order** (16-frame shuffle) remains untested for both checkpoints.
- **Boxes are mocap-derived.** Nothing here transfers to detector boxes at deployment.
- **One lab, two fixed cameras.** No cross-scene, cross-clothing or cross-day claim.

## Compute

Extraction is the cost. The Kinetics `full_frame_letterbox` arm (2,144 samples × 4 clips)
took 5 h 25 min on the GTX 1660 Ti with three workers (2026-08-15 22:41 to 08-16 04:06).
Two SSv2 arms are about 11 h of GPU, run as two overnight batches; the letterbox arm
goes first so the primary is readable before the bound is extracted. Materialize is
minutes. CPU: the framing report trains two arms × 3 seeds × 10 folds, 20–40 min per
arm; identity-control predict the same again per arm; position control k = 0 and k = 16
with both probes 40–80 min. Permutation nulls are minutes.

## Implementation

No change to the extractor, materializer, `run_arm`, the identity-control or the
position-control modules. One addition:

- `src/rehab24/videomae_checkpoint_report.py` + `scripts/rehab24/videomae_checkpoint_report.py`:
  reads the framing-report summary (primary and S4), the two within-session summaries
  (S1), the two probe and within-session summaries at k = 0 / 16 (S2), and the
  `background_only` summaries (S3); computes the paired ΔAUC with the identity module's
  `exact_wilcoxon_vs` (imported, not re-implemented) and the 9-subject bootstrap; emits
  each rule row's verdict as a boolean; unit tests on synthetic inputs under `tests/`.
  Committed under G5.

**Path discipline.** `materialize_videomae_features.py` defaults `--output-parent` to the
processed root, where it would overwrite the Kinetics Stage A dir
`videomae_mean_pool_fc_norm_mean`. Every SSv2 materialize call passes a checkpoint-scoped
parent, and every identity / position call passes its own `--output-dir`.

Planned artifacts:

```text
data/REHAB24-6/processed/videomae_raw_ssv2_full_frame_letterbox/                   # G1, G2
data/REHAB24-6/processed/videomae_raw_ssv2_background_only/
data/REHAB24-6/processed/videomae_ssv2/full_frame_letterbox/videomae_mean_pool_fc_norm_mean/
data/REHAB24-6/processed/videomae_ssv2/background_only/videomae_mean_pool_fc_norm_mean/
data/REHAB24-6/processed/videomae_ssv2_audit.json                                  # G3
data/REHAB24-6/processed/videomae_checkpoint/{seed42,seed7,seed1234,summary}.json  # primary, S4, S5
data/REHAB24-6/processed/videomae_identity_control_ssv2/                           # S1 (oof, null, summary)
data/REHAB24-6/processed/videomae_identity_control_ssv2_background_only/           # S3
data/REHAB24-6/processed/videomae_position_control_ssv2/                           # S2
data/REHAB24-6/processed/videomae_checkpoint_report.json                           # verdict rows
```

## Reproduce (planned interface, in this order)

```powershell
# 1. extraction, letterbox first; three GPU workers in parallel (.venv-cuda, torch cu126)
.venv-cuda\Scripts\python.exe scripts\rehab24\extract_videomae_features.py --model-name MCG-NJU/videomae-base-finetuned-ssv2 --variant full_frame_letterbox --output-dir data\REHAB24-6\processed\videomae_raw_ssv2_full_frame_letterbox --device cuda --num-chunks 3 --chunk-index <0|1|2>
.venv-cuda\Scripts\python.exe scripts\rehab24\extract_videomae_features.py --model-name MCG-NJU/videomae-base-finetuned-ssv2 --variant background_only --output-dir data\REHAB24-6\processed\videomae_raw_ssv2_background_only --device cuda --num-chunks 3 --chunk-index <0|1|2>

# 2. materialize into a checkpoint-scoped parent (never the default parent)
.venv\Scripts\python.exe scripts\rehab24\materialize_videomae_features.py --raw-dir data\REHAB24-6\processed\videomae_raw_ssv2_full_frame_letterbox --output-parent data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox --token-pooling mean_pool_fc_norm --aggregation mean
.venv\Scripts\python.exe scripts\rehab24\materialize_videomae_features.py --raw-dir data\REHAB24-6\processed\videomae_raw_ssv2_background_only --output-parent data\REHAB24-6\processed\videomae_ssv2\background_only --token-pooling mean_pool_fc_norm --aggregation mean

# 3. gates G2/G3
.venv\Scripts\python.exe scripts\rehab24\audit_videomae_features.py data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean data\REHAB24-6\processed\videomae_ssv2\background_only\videomae_mean_pool_fc_norm_mean --report-output data\REHAB24-6\processed\videomae_ssv2_audit.json

# 4. primary + S4 + S5 (G4 falls out of the kin arm in the same run)
.venv\Scripts\python.exe scripts\rehab24\videomae_framing_report.py --arm kin=data\REHAB24-6\processed\videomae_framing\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --arm ssv2=data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --arm ssv2_background_only=data\REHAB24-6\processed\videomae_ssv2\background_only\videomae_mean_pool_fc_norm_mean --primary ssv2:kin --secondary ssv2_background_only:kin --device cpu --output-prefix data\REHAB24-6\processed\videomae_checkpoint

# 5. S1 and S3 within-session
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py predict --output-dir data\REHAB24-6\processed\videomae_identity_control_ssv2 --arm-dir data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py analyze --output-dir data\REHAB24-6\processed\videomae_identity_control_ssv2 --permutations 10000 --permutation-seed 20260908
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py predict --output-dir data\REHAB24-6\processed\videomae_identity_control_ssv2_background_only --arm-dir data\REHAB24-6\processed\videomae_ssv2\background_only\videomae_mean_pool_fc_norm_mean --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py analyze --output-dir data\REHAB24-6\processed\videomae_identity_control_ssv2_background_only --permutations 10000 --permutation-seed 20260908

# 6. S2 position readability
# (--output-dir / --seeds are per-subcommand flags in BOTH CLIs and go AFTER the
#  subcommand. `analyze` keeps its default
#  --framing-summary: the Kinetics framing summary is the per-subject 0.6612 baseline, and
#  the helper reads only its full_frame_letterbox arm)
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 0 16 --output-dir data\REHAB24-6\processed\videomae_position_control_ssv2 --arm-dir data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --seeds 42 7 1234 --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 0 --output-dir data\REHAB24-6\processed\videomae_position_control_ssv2 --arm-dir data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 16 --output-dir data\REHAB24-6\processed\videomae_position_control_ssv2 --arm-dir data\REHAB24-6\processed\videomae_ssv2\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 0 --output-dir data\REHAB24-6\processed\videomae_position_control_ssv2 --permutations 10000 --permutation-seed 20260908
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 16 --output-dir data\REHAB24-6\processed\videomae_position_control_ssv2 --permutations 10000 --permutation-seed 20260908

# 7. verdict rows
.venv\Scripts\python.exe scripts\rehab24\videomae_checkpoint_report.py --output data\REHAB24-6\processed\videomae_checkpoint_report.json
```

The results note goes to `notes/rehab24_videomae_ssv2_checkpoint_results.md`: the five
gates, every plan deviation, the nine per-subject deltas for the primary and S1, the
permutation and Wilcoxon inference, the S2 probe before and after removal, the S3
bound, which row of each rule table was hit, and the limits above. Whichever checkpoint
the primary rule selects, the framing, identity and position notes are updated together
with one sentence each naming the outcome; not one of them alone.
