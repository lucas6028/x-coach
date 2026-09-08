# Plan: how much of the REHAB24-6 within-session VideoMAE signal is linearly readable recording position?

*REHAB24-6 · pre-registered position control · registered 2026-09-06 as commit `ada39ddd`,
before any residualized feature or new out-of-fold probability existed · run the same day;
results in [`rehab24_videomae_position_control_results.en.md`](rehab24_videomae_position_control_results.en.md)
(Chinese companion [`rehab24_videomae_position_control_results.md`](rehab24_videomae_position_control_results.md)) ·
siblings: [`rehab24_videomae_framing_results.md`](rehab24_videomae_framing_results.md),
[`rehab24_videomae_identity_appearance_results.md`](rehab24_videomae_identity_appearance_results.md)*

This is an English rewrite of the registered plan. The hypotheses, primary statistic,
gates, exclusion rules and interpretation table are those of commit `ada39ddd`, unchanged.

**Question.** The identity control established a within-session ROC-AUC of 0.8741 for the
`full_frame_letterbox` VideoMAE arm, and the framing experiment reads its 0.6612 as "the
signal comes from the person in the frame, not the background". Neither closes the
recording-position path: inside a recording, which repetition a clip is (its
`repetition_number`) predicts the label almost perfectly, and anything that drifts
monotonically along the recording (stance, camera settling, lighting, fatigue) is a pixel
carrier for it. The primary test removes, from the frozen VideoMAE features, the direction
that linearly predicts **within-class position**, and asks how much within-session ranking
survives. The framing and identity claims stay usable only if this passes. The plan also
registers the rule under which a fusion pre-registration may be opened.

**Status.** Pre-registration. None of the primary, mechanism or bound analyses had been run
when this was committed. Dataset REHAB24-6, subjects P1–P9 for the primary, P10 as
sensitivity only. Features: `full_frame_letterbox` with the corrected pooling
`mean_pool_fc_norm_mean`. Classifier, folds, seeds (42 / 7 / 1234), validation subjects and
threshold objective are those of `videomae_stage_a.run_arm`, not one line changed.

## Terminology

| term | meaning |
| --- | --- |
| rep / repetition | one execution of the exercise; every REHAB24-6 video holds several, each with its own correct / incorrect label |
| session | one `exercise_id + video_id`: one person, one exercise, one recording, filmed by two cameras (cam17, cam18) |
| position | `repetition_number` in `Segmentation.csv`: which rep of the recording this is |
| within-class position | rank of a rep among the same-label reps of its recording; a recording with 12 correct reps gives the correct reps within-class positions 1–12; this, not raw position, is what the plan removes (reason under Design) |
| within-session AUC | inside one recording, the probability that a randomly drawn correct rep outranks a randomly drawn incorrect one; anything constant across the recording (person, clothing, background, camera) contributes zero |
| subject-macro | one value per recording, averaged within subject, then averaged over the nine subjects; the inference unit is the subject, n = 9 |
| LOSO | leave-one-subject-out: each subject is the test set once and is unseen in training |
| drift | anything that changes monotonically along the recording's time axis and is visible in the frame: lighting, camera settling, stance, fatigue, sweat, clothing shift; these move with position and are the only pixel-level carriers by which a model could read it |
| probe | a linear model that predicts a variable from the frozen features, used to ask whether the features carry that information |
| `full_frame_letterbox` | the whole frame padded to square; person and background both present; the framing experiment's best arm |
| arm | one feature set run through the unchanged LOSO recipe |

## Motivating observations (already seen before registration)

The numbers below were produced by an exploratory analysis on 2026-09-06 that only read
the existing out-of-fold probabilities and `Segmentation.csv`; no model was retrained. They
motivate the plan and are not its results. Every confirmatory analysis registered here was
still unrun.

| observation | value | source |
| --- | ---: | --- |
| `repetition_number` alone as the score, within-session AUC | **0.1761** (0.8239 read reversed) | identity-control note, post-hoc control |
| the same rule, one threshold, LOSO subject-macro balanced accuracy | **0.7309 ± 0.0481**, 9/9 subjects > 0.5 | exploratory, `Segmentation.csv` only |
| the framing experiment's best arm, `full_frame_letterbox`, same metric and protocol | 0.6612 ± 0.0567 | framing note, results table |
| model AUC after position balancing (correct-first and correct-last pairs within a recording weighted 0.5 each) | **0.8497** on the 34 sessions where it is computable (unbalanced on the same 34: 0.8487; the position rule falls from 0.7309 to a constant 0.5000) | exploratory |
| model AUC on the 984 pairs where the incorrect rep precedes the correct one (position rule = 0.0000 there) | **0.8223** | exploratory |

Two things pull against each other in this table.

Position is a real shortcut, strong enough to beat the framing headline on the framing
experiment's own metric: one integer and one threshold give 0.7309 balanced accuracy
against 0.6612 for the whole VideoMAE pipeline. The framing note's "person, not
background" ruled out the background, not position.

On the 34 sessions where labels interleave, balancing position barely moves the model
(0.8487 to 0.8497) and the model still scores 0.8223 on pairs where position points the
wrong way. On those 34 sessions position is not the source of the signal.

The second point cannot be extended to all sessions, for a structural reason: 27 of the 61
mixed-label sessions are a single boundary, all correct reps first and then all incorrect
(identity-control note: median 3 label blocks per session, 27 sessions with exactly 2). No
correct-after-incorrect pair exists there, so the position-balanced statistic is
undefined by construction, and in exactly those 27 sessions position almost fully
determines the label.

What is missing is therefore not another paired statistic but an intervention that covers
all 61 sessions: remove position from the features directly and measure the remaining
ranking ability. That is the primary.

## Feasibility from labels and metadata alone

| scope | sessions | interleaved (balancing computable) | single boundary (balancing undefined) | (correct, incorrect) pairs | of which correct-last |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary, P1–P9, mixed-label | 61 | 34 | 27 | 4,391 | 984 |

- Each of the nine subjects has at least one interleaved session (the exploratory
  position-balanced statistic had n_subjects = 9).
- Within-class position is defined in all 61 sessions: inside a single-boundary session,
  the correct block still has an internal order.
- Drift proxies: `data/REHAB24-6/processed/box_geometry_features/` already exists (12
  dimensions, including the person box `x0, y0, x1, y1` and area); mean luminance per rep
  is new and needs one CPU decode pass.

## Design

**Why within-class position and not raw position.** Raw position and the label are nearly
the same variable inside a recording (AUC 0.1761). A feature direction that predicts raw
position is a direction that predicts the label; removing it lowers AUC even if the model
reads nothing but movement, and that drop cannot be interpreted. Within-class position
holds the label fixed: it orders the correct reps among themselves and the incorrect reps
among themselves. Directions that predict it are drift directions, orthogonal to the label
to first order. Removing them removes drift.

**Intervention, per LOSO fold.**

1. Using **training subjects only**, target = within-class position normalised to [0, 1]
   (0.5 when the class has a single rep).
2. Standardise features (the existing `FoldConfig.normalize_features`), fit ridge
   regression to the target, take the unit direction β̂.
3. Project **all** samples of the fold, training and test: x' = x − (x·β̂) β̂. Test-subject
   samples never enter the fit of β.
4. Pass x' to `run_arm` with everything else unchanged.

The primary removes **k = 1** direction. k ∈ {4, 16} (fit, remove, refit on the residual)
is secondary and measures how wide the linear drift subspace is.

## Primary comparison

Observed statistic, null and alternative are exactly those of the identity-control plan,
restated here. cam17 and cam18 probabilities are averaged into one score per rep. Each seed
gives one ROC-AUC per mixed-label session; the three seeds are averaged within session;
each subject's sessions are averaged with equal weight; the nine subjects give mean ±
sample SD. Null: permute rep labels inside each session, preserving that session's
positive and negative counts, the same permutation applied to both cameras and all three
seeds, 10,000 draws, p = (1 + #null ≥ observed) / 10,001, α = 0.05.

**Prior, fixed before the run.** On the 34 interleaved sessions position balancing moved
the model by +0.001, and the model kept 0.8223 on position-reversed pairs. Both point to a
residual of 0.02–0.07. If the 27 single-boundary sessions behave like the 34, the primary
should land in 0.80–0.86. A primary below 0.80 means the model leans on drift more heavily
in the single-boundary sessions than in the interleaved ones, which is precisely what this
plan must be able to detect.

## Mechanism probe

The primary is an intervention; the probe asks whether the thing being removed existed.

- LOSO ridge probe: fit features → within-class position on training subjects; on each
  test-subject session compute the Spearman ρ between prediction and true within-class
  position, **within class** (correct and incorrect reps separately, then averaged);
  report subject-macro median and mean.
- Null: permute positions within each session and label class, 10,000 draws, two-sided.
- The same probe runs on the **residualized** features. That run is the positive-control
  gate, not a result.

## Drift bounds

Zero-parameter scores from metadata and geometry, put through the identical
within-session statistic.

| proxy | source | what it captures |
| --- | --- | --- |
| box `x0`, `y0` (top-left corner) | existing box_geometry | stance drift |
| box area | existing box_geometry | forward / backward drift |
| mean luminance per rep | new | lighting / exposure drift |

Cameras are averaged first. p is **two-sided**: a reversed predictor is as much a shortcut
as a forward one (the lesson of the identity note's post-hoc control). "Informative"
follows that note: |AUC − 0.5| > 0.15.

## Secondary (never changes the primary reading)

- Position-balanced within-session AUC of the residualized arm (k = 1) on the 34
  interleaved sessions. This replicates the exploratory 0.8497 with a newly trained
  classifier on residualized features, so it is not a re-report of the same number.
- LOSO subject-macro balanced accuracy of the residualized arm against the
  `full_frame_letterbox` 0.6612 ± 0.0567: nine-subject paired delta and exact Wilcoxon.
  This is the direct test of whether the framing note's sentence survives.
- Within-session AUC of the residualized arm on the **27 single-boundary sessions**,
  reported only and never contrasted with the 34 (different sessions).
- A naive variant residualizing on **raw** position instead of within-class position;
  expected to over-remove for the reason given under Design, reported so the gap is
  visible.
- k ∈ {4, 16}.
- P10-inclusive sensitivity.

Multiple p-values are reported raw and Holm-corrected. No favourable k or subset may
replace the primary.

## Statistical unit

Same as the identity-control plan. sample = rep × camera; repetition = one cam17 / cam18
pair; session = one `exercise_id + video_id`; **the subject is the only inference unit,
n = 9**. Cameras, reps, sessions and seeds are not n.

## Reading the result (fixed now)

0.55 is the project's standing practical threshold; 0.80 is the prior above
(0.8741 − 0.07). Neither is an equivalence bound.

| pattern | registered reading | next step |
| --- | --- | --- |
| primary (k = 1) AUC **≥ 0.80**, ≥ 6/9 subjects > 0.5, p < 0.05, and both probe conditions hold (significant before removal, inside the null after) | the linear drift direction is not the source of the within-session signal; the linear position path is closed. The framing note's "from the person" and the identity note's 0.8741 **stand**, with the bound "linear drift excluded, residual ≤ 0.8741 − primary" | a fusion pre-registration may be opened, under the identity plan's conditions: calibrated late fusion of NLF and `full_frame_letterbox` only, validation-only calibration, one fusion rule, subject-paired endpoint, stop if no improvement over NLF |
| primary in **[0.55, 0.80)**, p < 0.05 | part of the signal shares the drift direction; report share = (0.8741 − primary) / (0.8741 − 0.5); may not be written as "mostly movement" | no fusion; run k = 4, 16 for subspace width, then decide whether non-linear removal is needed |
| primary **< 0.55**, or p ≥ 0.05, or < 6/9 subjects > 0.5 | **undetermined**: no claim that the model distinguishes movement quality inside a recording; the framing note's headline is **withdrawn** and rewritten as "the signal comes from the person and is inseparable from recording position" | stop VideoMAE claims on REHAB24-6; the next step is data (re-record with an interleaved protocol), not modelling |
| probe already inside the null **before** removal | the features do not linearly encode within-class position; the pixel-level position path is bounded by this. **The primary is not read** (the intervention is a no-op) | report the bound; the non-linear path stays open and goes under Not supported |
| any drift proxy with \|AUC − 0.5\| > 0.15 and two-sided p < 0.05 | that drift is visible and is a candidate carrier of what the intervention removed | goes into the mechanism section; the primary reading does not change |
| primary passes but the 27 single-boundary sessions score clearly below the 34 | reported, not inferred (different sessions) | goes under Not supported |

p ≥ 0.05 is written as **undetermined**, never "no difference", never "equivalent".

## Gates (all must pass before any outcome is read)

| gate | check |
| --- | --- |
| features and OOF integrity | 2,128 primary rows per seed; sample ids unique and complete; every probability finite; `person_id == test_subject` on every row (verified by the exploratory script on the existing OOF; must be re-verified on the new runs); every rep has exactly one cam17 and one cam18 row with agreeing labels; mixed-label sessions = 61 |
| fold purity | each fold's β is fit on that fold's training subjects only; the β hash is logged per fold and the ten hashes must be pairwise **distinct** (identical hashes mean the fit saw all data); any test-subject sample in the input of a β fit invalidates the whole experiment |
| positive control, removal worked | the probe on the residualized (k = 1) features must have its subject-macro within-class Spearman median inside that probe's null 95% interval; otherwise the intervention failed and the primary is not read. If the probe is already inside the null **before** removal, the intervention is a no-op, the conclusion passes to the bound (fourth row of the reading table), and a high primary may not be written as "holds after removal" |
| reproduction | the new code with k = 0 (no residualization) on `full_frame_letterbox` must return within-session AUC **0.8741** to four decimals (seed, fold and config fixed), null mean 0.5002 ± 0.0211; the formal module must also reproduce the three exploratory figures on the existing OOF to four decimals (position-balanced 0.8497, reversed pairs 0.8223, unbalanced on the same 34 sessions 0.8487), otherwise the module has a bug |
| frozen analysis | this note, the residualization code, the probe definition, the permutation seed (20260906) and the reading table are committed before any new OOF is read; after the primary is seen, k, the within-class position definition, the session definition, camera / seed aggregation and sidedness may not change; any change is a plan deviation in the results note, with the registered analysis kept alongside |

## What this plan cannot show (even if the primary passes)

- **Only linear directions are removed.** A non-linear encoding of drift is outside the
  test. k = 4, 16 widen the linear subspace and remain linear.
- **Within-class position is not position.** Drift that steps at the moment the labels
  switch (a coach pausing to adjust lighting, say) is unrelated to within-class position
  and fully collinear with the label; this design cannot see it. In the 27 single-boundary
  sessions no analysis can separate such drift from movement. That is a data limit,
  fixable only by re-recording.
- **Temporal order inside a rep is not tested.** Position is the rep's place in the
  recording, not the order of the 16 frames in a clip; the temporal-shuffle control listed
  as untested in the identity note stays untested here.
- **No cross-day, cross-clothing or cross-venue generalisation is tested.**
- **A probe inside the null does not mean the features carry no position.** That is an
  equivalence claim and needs power.
- **0.7309 vs 0.6612 is not a like-for-like comparison.** One-parameter rule against a
  full pipeline, n = 9, unpaired, untested. Its role is to show the shortcut is strong
  enough to beat the headline, not that position beats VideoMAE by 0.07.

## Implementation and compute

New, folding the exploratory scratch scripts `posctl3.py` / `posctl4.py` / `loso.py` into a
tested module:

- `src/rehab24/videomae_position_control.py`: within-class position target, in-fold
  ridge residualization, probe, drift proxies, position-balanced statistic; `midranks`,
  `build_sessions`, `observed_statistic`, `permutation_null` imported from
  `videomae_identity_control.py`, not rewritten.
- `scripts/rehab24/videomae_position_control.py`: thin CLI.
- `videomae_stage_a.run_arm` gains one **optional** parameter,
  `feature_transform_factory` (takes the fold's training sample ids, returns a callable
  applied to the (train, test) features), default None, behaviour unchanged. This is the
  only touch on the existing runner; all existing tests must stay green.
- `tests/rehab24/test_videomae_position_control.py`, on synthetic data: (a) the probe is
  zero after residualization, (b) a pure position-shortcut feature set drops to AUC 0.5
  after residualization, (c) a pure movement feature set is unchanged, (d) the
  position-balanced statistic is identically 0.5 on the position rule.

Planned artifacts:

```text
data/REHAB24-6/processed/videomae_position_control/oof_k0_seed{42,7,1234}.csv      # reproduction gate
data/REHAB24-6/processed/videomae_position_control/oof_k1_seed{42,7,1234}.csv      # primary
data/REHAB24-6/processed/videomae_position_control/oof_k{4,16}_seed*.csv           # secondary
data/REHAB24-6/processed/videomae_position_control/oof_naive_k1_seed*.csv          # raw-position variant
data/REHAB24-6/processed/videomae_position_control/fold_betas.json                 # per-fold β hash
data/REHAB24-6/processed/videomae_position_control/probe_summary.json
data/REHAB24-6/processed/videomae_position_control/drift_proxies.json
data/REHAB24-6/processed/videomae_position_control/luminance_per_sample.csv
data/REHAB24-6/processed/videomae_position_control/within_session_summary.json
data/REHAB24-6/processed/videomae_position_control/permutation_null.npz
```

Compute: CPU only, `.venv`. Ridge is 2,128 × 768; the `run_arm` MLP takes minutes per seed
for ten folds on CPU; luminance decodes 128 videos once. No GPU and no VideoMAE
re-extraction.

## Reproduce (planned interface)

Gates first, then the primary, secondaries last.

```powershell
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py replicate-exploratory   # reproduction gate, exploratory figures
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 0 --seeds 42 7 1234
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 0 --permutations 10000 --permutation-seed 20260906   # must equal 0.8741
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 0            # positive control, before removal
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 1 --seeds 42 7 1234
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 1            # positive control, after removal
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 1 --permutations 10000 --permutation-seed 20260906   # primary
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py drift-proxies --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 4 16 --naive --seeds 42 7 1234
```

The results note goes to `notes/rehab24_videomae_position_control_results.md` and reports,
item by item: the five gates, plan deviations, the primary's nine per-subject values,
the permutation inference, the probe before and after removal, the drift proxies, which
row of the reading table was hit, and the limits listed above. The framing note and the
identity-control note are updated **together** according to that row, never one alone.
