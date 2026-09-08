# Removing the linear within-class position subspace costs 0.019 of within-session AUC

*REHAB24-6 · position control · run 2026-09-06 · English companion to
[`rehab24_videomae_position_control_results.md`](rehab24_videomae_position_control_results.md)*

**Result.** After projecting out the 16 feature directions that linearly predict
within-class position, within-session ROC-AUC goes from 0.8741 to 0.8556 ± 0.0435,
9/9 subjects above chance, p = 1/10001. Share along the position subspace = 4.9%.

**Caveat on reading it.** The registered primary (k = 1) failed its positive control and
is not interpretable; k = 16 is the registered secondary promoted to primary, which is a
plan deviation. The removal is linear only.

## Background

The identity control ([note](rehab24_videomae_identity_appearance_results.md)) established
within-session AUC 0.8741 but left the recording-position path open: labels are blocked
within a recording, and `repetition_number` alone, zero fitted parameters, gives LOSO
balanced accuracy 0.7309 ± 0.0481 (9/9), above the framing headline of 0.6612.

## Method

Target is **within-class position**, not raw position: rank within (session, label class),
normalised to [0, 1], 0.5 if the class has one rep, shared by both cameras. Raw position
is near-collinear with the label inside a recording, so removing directions that predict
it would drop AUC regardless of what the model reads.

Per LOSO fold: fit ridge from features to the target on **training subjects only**;
normalise to β̂; project `x' = x − (x·β̂)β̂` on all samples; for k > 1 refit on the residual
and repeat; pass `x'` to the unchanged recipe. λ from an inner leave-one-training-subject-out
sweep over {0.1, 1, 10, 100, 1000, 10000}, fixed before results; folds selected 100–10000.

Statistics unchanged from the identity control: cameras averaged per rep, per-session AUC,
seed-averaged, subject-macro, n = 9, within-session label permutation null,
10,000 draws, seed 20260906.

## Gates

| gate | result |
| --- | --- |
| 6.1 OOF integrity | pass, all 5 arms × 3 seeds (2,128 primary rows, unique ids, finite, `person_id == test_subject`, camera pairs consistent, 61 mixed sessions) |
| 6.2 fold purity | pass; β̂ hashes pairwise distinct (10 / 40 / 160 / 10 directions for k1 / k4 / k16 / naive) |
| 6.3 probe before removal | pass; ρ median +0.4324, mean +0.4029, all 9 subjects positive, range +0.17 to +0.69, null [−0.086, +0.085], p = 1/10001 |
| 6.3 probe after removal, k = 1 | **fail**; +0.2609, p = 1/10001 → primary not interpretable |
| 6.3 probe after removal, k = 16 | pass on median (+0.0525, p = 0.243); **fail on mean** (+0.1183, p = 0.0016) |
| 6.4 reproduction | pass; k = 0 OOF byte-identical to the identity control on all 3 seeds, AUC 0.8741; `replicate-exploratory` reproduces 0.8497 / 0.8223 / 0.8487 / 0.7309 |
| 6.5 frozen analysis | pass; plan `ada39ddd`, code `917281a2`, both before any k ≥ 1 OOF |

## Results

| arm | AUC (mean ± SD) | > 0.5 | bootstrap 95% CI | p | probe median after |
| --- | ---: | ---: | --- | ---: | ---: |
| k = 0 | 0.8741 ± 0.0408 | 9/9 | [0.850, 0.899] | 1/10001 | — |
| k = 1 (registered primary, not readable) | 0.8720 ± 0.0439 | 9/9 | [0.845, 0.899] | 1/10001 | +0.2609 |
| k = 4 | 0.8681 ± 0.0436 | 9/9 | [0.843, 0.896] | 1/10001 | +0.2885 |
| **k = 16** | **0.8556 ± 0.0435** | 9/9 | [0.830, 0.883] | 1/10001 | +0.0525 |
| k = 1, naive (raw position) | 0.8696 ± 0.0407 | 9/9 | [0.845, 0.895] | 1/10001 | not run |

Per subject, k = 0 → k = 16: P1 0.837→0.840, P2 0.895→0.862, P3 0.875→0.802,
P4 0.817→0.801, P5 0.835→0.840, P6 0.945→0.944, P7 0.885→0.865, P8 0.913→0.886,
P9 0.867→0.861. Minimum 0.8009.

Share = (0.8741 − 0.8556) / (0.8741 − 0.5) = 0.049. It is a floor: the k = 16 probe mean
is still outside its null, driven by P4 (+0.32) and P8 (+0.38).

The probe requires 16 directions to reach the null (median +0.43 → +0.26 → +0.29 → +0.05
for k = 0 / 1 / 4 / 16). The plan's assumption that k = 1 would clear it was wrong.

## Secondary

Position-balanced AUC on the 34 interleaved sessions, where the position rule is 0.5000
by construction: 0.8497 (k=0), 0.8447 (k=1), 0.8464 (k=4), 0.8320 (k=16), 0.8498 (naive).
Incorrect-first pairs only, position rule 0.0000: 0.8223 / 0.8147 / 0.8186 / 0.7952 / 0.8248.
The 27 single-boundary sessions, position rule 0.9821, give 0.9267 / 0.9225 / 0.9166 /
0.9063 / 0.9193; reported only, not comparable to the 34.

Session-level check that the effect is not an artefact of blocked labels: on `Ex4_PM_027`
(P3, 27 reps, 7 label blocks) the position rule scores 0.494 while the model scores 0.854
at k = 0 and 0.803 at k = 16. On `Ex2_PM_003` (P1, single boundary) the position rule
scores 0.921 and the model 0.302, i.e. the model can be well below the trivial rule where
labels split cleanly.

LOSO balanced accuracy vs the framing baseline 0.6612: k=1 0.6643 (+0.0031, 3/9, p = 1.00),
k=4 0.6570 (−0.0042, 3/9, p = 0.652), k=16 0.6717 (+0.0105, 6/9, p = 0.359),
naive 0.6660 (+0.0048, 4/9, p = 0.570). All undetermined; n = 9 with SD ≈ 0.05 cannot
resolve deltas this size.

P10-inclusive sensitivity: k = 1 gives 0.8848 ± 0.0579 (10/10), k = 16 gives
0.8700 ± 0.0614 (10/10), both p = 1/10001.

## Drift proxies

Zero-parameter scores, same within-session statistic, two-sided p, informative if
|AUC − 0.5| > 0.15 and p < 0.05.

| proxy | AUC | \|AUC − 0.5\| | consistent | p raw / Holm | informative |
| --- | ---: | ---: | ---: | --- | --- |
| box `x0` | 0.6465 | 0.1465 | 8/9 | 1/10001 / 0.0004 | no |
| box area | 0.3743 | 0.1257 | 7/9 reversed | 1/10001 / 0.0004 | no |
| box `y0` | 0.5177 | 0.0177 | 6/9 | 0.407 / 0.814 | no, undetermined |
| luminance | 0.5069 | 0.0069 | 6/9 | 0.744 / 0.814 | no, undetermined |

Lateral position and apparent size drift with the label within a recording, both below
the pre-registered margin, so no carrier is named. Not in conflict with the framing
result that box geometry alone scores 0.5075: that is cross-subject balanced accuracy,
this is within-recording ranking.

## Deviations from the plan

| item | plan | actual | effect |
| --- | --- | --- | --- |
| which arm is read | k = 1 primary | k = 1 control failed; read k = 16 (registered secondary) | every "linear position path closed" statement inherits this label; k = 16 is the lowest-scoring of the four arms, not a favourable pick |
| probe statistic | subject-macro median | median passes, mean does not (p = 0.0016) | mean is not the registered criterion, but 4.9% is a floor |
| λ | unspecified | inner LOSO over a fixed 6-point grid, one λ per fold | fixed before results |
| null mean in 6.4 | 0.5002 ± 0.0211 | 0.4999 ± 0.0211 from the module, 0.5002 from the exploratory script | session iteration order changes RNG consumption; observed statistic and p unchanged |
| test file path | `tests/rehab24/…` | `tests/test_rehab24_videomae_position_control.py` | none |
| luminance decode | one CPU pass | every 4th frame, ffmpeg backend after OpenCV ran ~11 fps and was killed at 1h45 by the host; backends agree to 0.27 grey levels on a real video | none |

## Not supported

- Non-linear encoding of position. The projection is linear; the classifier is an MLP.
- P4 and P8 were not cleaned at k = 16, so the 4.9% share is a lower bound.
- Drift that steps at the label boundary is collinear with the label and invisible here;
  in the 27 single-boundary sessions no analysis can separate it. Data-collection limit.
- Within-clip temporal order (16-frame shuffle) is still untested.
- The five balanced-accuracy deltas are undetermined, not evidence of no effect.
- 0.7309 vs 0.6612 is not a like-for-like comparison and was never tested.
- 2026-09-08: the same procedure on the SSv2-finetuned checkpoint gives probe median
  0.3843, AUC 0.7805 → 0.7772 at k = 16, share 1.2% (a floor, mean probe p = 0.033);
  see [`rehab24_videomae_ssv2_checkpoint_results.md`](rehab24_videomae_ssv2_checkpoint_results.md).
  The 4.9% here remains the Kinetics figure.

## Reproduce

Logic `src/rehab24/videomae_position_control.py`, CLI
`scripts/rehab24/videomae_position_control.py`, 44 tests in
`tests/test_rehab24_videomae_position_control.py`. `videomae_stage_a.run_arm` gained
optional `feature_transform_factory` / `materialize_root`; default path unchanged, shown
by `predict --k 0` reproducing the earlier OOF byte for byte.

```powershell
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py replicate-exploratory --permutations 10000 --permutation-seed 20260906
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict  --k 0 --seeds 42 7 1234 --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze  --k 0 --permutations 10000 --permutation-seed 20260906
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe    --k 0 --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict  --k 1 4 16 --seeds 42 7 1234 --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe    --k 16 --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze  --k 16 --permutations 10000 --permutation-seed 20260906
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py drift-proxies --permutations 10000 --luminance-backend ffmpeg
```

CPU only, 20–40 min per arm for three seeds, ~80 MB of materialised features per arm.
Artifacts in `data/REHAB24-6/processed/videomae_position_control/`: per-arm OOF CSVs and
fold JSONs, `fold_betas.json`, `probe_summary_k*.json`, `within_session_summary_*.json`,
`permutation_null_*.npz`, `drift_proxies.json`, `luminance_per_sample.csv`.
Exploratory scripts in `notes/exploratory/rehab24_position_control_20260906/`.
