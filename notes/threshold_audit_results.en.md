# Central threshold restriction preserves the MAE–verdict reversal; full-range integration removes it

*Fit3D · retrospective threshold sensitivity audit · run 2026-09-19 · companion to [model_fusion_gate_results.md](model_fusion_gate_results.md)*

**Result.** For squat/NLF, shrinking globally mean-corrected rep readings from lambda 1 to 0.7 reduces MAE from 4.6268 to 3.7325 degrees, while central uniform-threshold disagreement rises from 9.0616% to 9.9247%. Full-range uniform disagreement instead falls from 4.5914% to 3.7039%. Thus restricting the threshold range is sufficient for this observed reversal; neither percentile weighting nor a grid of only 13 thresholds is necessary.

**Caveat on reading it.** These are descriptive results on 160 squat rep-camera readings from 40 reps and 8 subjects. The central-disagreement change is +0.8630 percentage points, subject SD 2.8762 points, conditional subject-bootstrap 95% interval [-1.0318, 2.8508]; 7/8 subjects worsen. Generalization of this difference is undetermined. All oracle calibration and lambda selection use the evaluation data. This audit tests the evaluation metric, not coaching validity.

## Background

The companion note inferred that smaller pose/cue errors do not imply better coaching decisions, and attributed the difference to ordering fidelity. Its model comparison mixed frame-level uncorrected cue error with oracle-corrected rep-level threshold disagreement. Its shrinkage experiment compared two losses under a central percentile sweep. This audit holds the observations and calibration fixed while changing only the threshold measure.

## Method

Existing NLF and MeTRAbs NPZ predictions were reduced again through `decision_eval.collect_records`, with model stride 15 and full-rate GT, using `smpl3d` predictions. Records were matched by subject, camera and rep index. Both models' full-rate GT and predicted rep minima reproduced every saved pair in `rep_extreme_decomposition.json` within numerical tolerance. No model inference was performed.

| Action | Rep-camera pairs | Unique reps | Subjects | Pair exclusions |
| --- | ---: | ---: | ---: | ---: |
| squat | 160 | 40 | 8 | 0 |
| deadlift | 164 | 41 | 8 | 0 |
| overhead_extension_thruster | 176 | 44 | 8 | 0 |

Three calibration conditions were evaluated separately: raw, global oracle mean correction, and per-camera oracle mean correction. Each metric uses the same corrected rep readings in a condition. These rep MAEs must not be confused with the earlier note's frame-level pooled MAEs or its sampled-GT extreme errors.

| Metric | Definition |
| --- | --- |
| MAE | Mean absolute error of predicted rep minimum against full-rate GT rep minimum |
| Original percentile sweep | Equal weight at 13 GT percentiles from 20 to 80 |
| Dense percentile sweep | Equal weight at 6,001 GT percentiles from 20 to 80 |
| Central uniform sweep | Exact integral with uniform angular weight from GT q20 to q80 |
| Full uniform sweep | Exact integral over one common range covering GT and every audited prediction for that action |
| Fixed 90 degrees | Disagreement under the repository's existing strict greater-than rule; a sensitivity check, not a validated cutoff for every exercise |

The full range is fixed within action across both models, all calibration conditions and all 11 shrinkage factors, with one degree padding on both sides. Normalized scores from different actions therefore have different denominators and should not be pooled. Squat's full range is [29.6296, 130.4006] degrees; its central range is [60.8207, 81.7267] degrees. The unnormalized full integral is also saved, in degrees, and equals MAE.

Subject comparisons average all cameras and reps within each subject before computing the mean difference and sample SD. Intervals use 10,000 paired subject bootstrap draws, seed 20260919. Calibration constants, threshold grids and selected lambdas are held fixed: these intervals are conditional sensitivity summaries, not held-out deployment estimates or post-selection-corrected inference. All reported model differences are NLF minus MeTRAbs; negative favors NLF. Per-subject values are in `results.json`.

This is an exploratory retrospective audit, not a preregistered study. No population-level superiority test or multiple-comparison-adjusted claim is made.

## Results

### Shrinking squat/NLF redistributes error across the threshold range

Lambda 1 is globally oracle mean-corrected NLF, not its raw deployment output. Both transformations preserve prediction ordering and Pearson correlation.

| Metric, same 160 pairs | Lambda 1 | Lambda 0.7 |
| --- | ---: | ---: |
| Rep MAE, degrees | 4.6268 | 3.7325 |
| Original 13-percentile disagreement | 10.8654% | 11.5865% |
| Dense 6,001-percentile disagreement | 10.2892% | 11.1122% |
| Exact central uniform disagreement | 9.0616% | 9.9247% |
| Exact full uniform disagreement | 4.5914% | 3.7039% |
| Fixed 90-degree disagreement | 6.8750% | 5.0000% |
| Pearson r | 0.923123 | 0.923123 |

Dispersion for the changes above, lambda 0.7 minus lambda 1:

| Metric | Subject mean change +/- SD | Conditional 95% interval | Subjects improving |
| --- | ---: | --- | ---: |
| MAE, degrees | -0.8943 +/- 2.6109 | [-2.5353, 0.7981] | 4/8 |
| Original sweep, percentage points | +0.7212 +/- 4.4591 | [-2.1154, 3.7019] | 3/8 |
| Dense sweep, percentage points | +0.8230 +/- 4.0192 | [-1.6844, 3.5613] | 2/8 |
| Central uniform, percentage points | +0.8630 +/- 2.8762 | [-1.0318, 2.8508] | 1/8 |
| Full uniform, percentage points | -0.8875 +/- 2.5909 | [-2.5159, 0.7920] | 4/8 |
| Fixed 90 degrees, percentage points | -1.8750 +/- 3.7201 | [-4.3750, 0.0000] | 2/8 |

The central integral in degrees increases from 1.8944 to 2.0749, while the integral outside the central range decreases from 2.7323 to 1.6576. Their sums give the MAEs above. This exact partition locates the descriptive reversal: improvements outside the evaluated central thresholds outweigh deterioration inside them in full MAE. Across individual rep-camera readings, 86 absolute errors improve and 74 worsen. Shrinkage is not a pointwise movement of every estimate toward its own GT.

The reversal remains under exact central integration and therefore is not solely a coarse-grid artifact. A separate independent calculation with 100,001 percentile thresholds gives 10.2892% to 11.1117%, consistent with the dense result. At the fixed 90-degree rule, both MAE and disagreement improve for this example. No universal direction across thresholds follows.

### Matching calibration and aggregation removes two of three model-ranking contradictions

The table uses per-camera oracle correction, stride 15 and the same rep minima for MAE and disagreement. Values are pooled descriptive means; subject dispersion and intervals for paired differences follow.

| Action | NLF MAE (degrees) | MeTRAbs MAE (degrees) | NLF original sweep | MeTRAbs original sweep |
| --- | ---: | ---: | ---: | ---: |
| squat | 4.7025 | 4.4435 | 11.7308% | 12.5962% |
| deadlift | 4.2548 | 5.3536 | 12.5235% | 17.3546% |
| thruster | 3.4215 | 4.6961 | 9.5717% | 13.4178% |

Only squat retains the opposing point-estimate rankings. Deadlift and thruster favor NLF under both metrics. In every full uniform comparison, the ranking agrees with the rep MAE ranking by the integral identity below.

| Action / metric | Subject mean difference +/- SD | Conditional 95% interval | Subjects favoring NLF |
| --- | ---: | --- | ---: |
| squat / MAE, degrees | +0.2590 +/- 3.0545 | [-1.7115, 2.2113] | 4/8 |
| squat / original sweep, points | -0.8654 +/- 6.7147 | [-5.1442, 3.4615] | 4/8 |
| deadlift / MAE, degrees | -1.1963 +/- 2.5028 | [-2.8703, 0.3763] | 6/8 |
| deadlift / original sweep, points | -5.0160 +/- 11.2367 | [-13.3654, 0.2885] | 5/8 |
| thruster / MAE, degrees | -1.2665 +/- 1.8330 | [-2.4316, -0.0229] | 7/8 |
| thruster / original sweep, points | -4.0385 +/- 7.0723 | [-8.9183, 0.2885] | 5/8 |

All three original-sweep intervals include zero; generalizable superiority on this metric remains undetermined. The raw and global-corrected conditions and all per-subject values are included in the JSON artifact. Under raw MAE and the original raw sweep, MeTRAbs has the better pooled point estimate for all three actions.

## Mathematical and implementation controls

For GT reading g and prediction p, the two strict greater-than decisions differ only for thresholds between g and p. The exact uniform-threshold disagreement on [a,b] is:

`mean(abs(clip(g,a,b) - clip(p,a,b))) / (b-a)`.

When [a,b] covers every value, this is `MAE / (b-a)`. The audit independently integrates actual Boolean decisions on the intervals between all GT/prediction breakpoints and checks that the result equals MAE. All 84 evaluated configurations pass this numerical identity, as well as the corresponding clipped central-range identity. Lower full-range MAE cannot produce higher full-range uniform disagreement under these matched conditions.

The original `_verdict_metrics` implementation has no model-name branch. Both models use identical GT, paired observations and threshold grids. These checks establish symmetric application of the chosen loss, not that the chosen threshold distribution represents useful coaching decisions.

## Not supported

- Pose accuracy is an incorrect optimization target, or improving pose accuracy inherently harms decisions.
- Pearson r determines verdict quality: r and all ranks remain fixed throughout each positive-lambda sweep while disagreement changes.
- An implementation rule specifically penalizes NLF. Its comparative score does depend on the chosen threshold distribution and calibration; no clinically preferred threshold distribution is established here.
- A 90-degree knee rule is validated coaching ground truth for squat, deadlift or thruster. The fixed-rule output is only a sensitivity check.
- New-subject superiority or a deployable benefit from oracle calibration or evaluation-selected shrinkage.
- That every apparent pose-to-decision mismatch is solely threshold weighting: the prior model comparison also changed temporal aggregation and calibration.
- That full-range uniform thresholds are the correct coaching objective. They are a mathematical control; an application may legitimately place more weight near particular thresholds, but must justify that choice independently.

## Reproduce

From the repository root, using the existing Windows CPU environment:

```powershell
.venv\Scripts\python.exe -m src.fit3d.threshold_audit
.venv\Scripts\python.exe -m pytest tests/test_fit3d_threshold_audit.py tests/test_fit3d_model_fusion.py tests/test_fit3d_decision_eval.py -q -p no:cacheprovider
.venv\Scripts\python.exe .claude/skills/write-experiment-note/check_note.py notes/threshold_audit_results.en.md
```

The first invocation recomputes rep summaries from existing NPZ files and saves an identifier-bearing cache; subsequent invocations reuse it. The initial collection and audit took approximately 9 seconds wall time; the cached audit approximately 1.4 seconds. No GPU inference or external API was used. Four new mathematical-contract tests plus the two existing relevant suites passed, 55 tests total.

Code: `src/fit3d/threshold_audit.py`; tests: `tests/test_fit3d_threshold_audit.py`.

Artifacts: `data/Fit3D/derived/threshold_audit_20260919/paired_records.json` and `data/Fit3D/derived/threshold_audit_20260919/results.json`. The result includes all three conditions, every lambda, conditional intervals, per-subject differences, support bounds and original threshold values. These data artifacts may be ignored by Git and must accompany the code for a cache-only reproduction.
