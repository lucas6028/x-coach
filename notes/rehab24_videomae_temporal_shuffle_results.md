# Reordering the 16 frames leaves REHAB24-6 within-session AUC at 0.8798 vs 0.8741 (undetermined); a single static frame drops it to 0.7002

*REHAB24-6 · pre-registered temporal-order control · run 2026-09-12 · plan
[`rehab24_videomae_temporal_shuffle_validation_plan.md`](rehab24_videomae_temporal_shuffle_validation_plan.md)
(commit `5faf4db1`) · siblings:
[`rehab24_videomae_position_control_results.en.md`](rehab24_videomae_position_control_results.en.md),
[`rehab24_videomae_identity_appearance_results.md`](rehab24_videomae_identity_appearance_results.md),
[`rehab24_videomae_framing_results.md`](rehab24_videomae_framing_results.md)*

**Result.** Showing VideoMAE the 16 frames of every clip in random order does not lower
within-session ROC-AUC: 0.8798 ± 0.0647 against the baseline 0.8741 ± 0.0408, paired
delta −0.0057 ± 0.0489, 3/9 subjects drop, two-sided exact Wilcoxon p = 0.734. That is
the plan's row 3: undetermined. Showing one frame of the clip repeated 16 times drops the
AUC to 0.7002 ± 0.0939, delta +0.1739 ± 0.0659, 9/9 subjects drop, p = 0.0039, i.e.
46.5% [35.1%, 59.9%] of the above-chance signal needs more than one frame. Reversal and
tubelet-level shuffling sit between, with drops of 0.0095 and 0.0100 (7/9 each,
undetermined).

**Caveat on reading it.** "Undetermined" is the registered reading and the only one:
this design has no equivalence bound, so the shuffle result may not be written as "the
model does not use temporal order". What the interval does say is descriptive: a drop
larger than 0.034 of AUC lies outside the bootstrap 95% interval [−0.034, +0.025]. The
cross-subject balanced accuracy of the shuffled arm is higher than the baseline's
(0.7007 vs 0.6612, 8/9 up, Holm p = 0.035), which is a secondary result and is reported,
not interpreted.

## Background

The position control ([note](rehab24_videomae_position_control_results.en.md)) removed
16 linear directions that predict within-class recording position and lost 4.9% of the
within-session signal. Every carrier it looked for is static inside a 16-frame clip:
stance, camera settling, lighting, appearance, background. Its Not supported list left
one path open: within-clip temporal order. The identity control's Not supported list
says the same thing from the other side: single-frame pose differences inside a
recording could produce the AUC without the model using time at all. This experiment
runs the shuffle and the static-frame control that both notes asked for.

## Terminology

| term | meaning |
| --- | --- |
| rep / repetition | one execution of the exercise, with its own correct / incorrect label and a first / last frame in `Segmentation.csv` |
| session | one `exercise_id + video_id`: one person, one exercise, one recording, filmed by cam17 and cam18 |
| clip | 16 frames at stride 2 from one rep, from one of 4 evenly spaced starts; VideoMAE encodes one clip per forward pass and the 4 clip embeddings are averaged into the rep's feature |
| tubelet | VideoMAE's temporal token: 2 consecutive frames; a clip is 8 tubelets |
| `full_frame_letterbox` | the whole frame padded to square; the baseline every later control is paired against; within-session AUC 0.8741, LOSO balanced accuracy 0.6612 |
| temporal arm | the same pixels and the same sampled frames as the baseline, only the order (or repetition) of the 16 frames changed before they enter the backbone |
| within-session AUC | inside one recording, the probability that a correct rep outranks an incorrect one; cameras averaged per rep, seed AUCs averaged per session, sessions averaged per subject, n = 9 |
| LOSO BA | leave-one-subject-out balanced accuracy, seeds averaged within subject, n = 9 |
| share | (baseline − arm) / (baseline − 0.5): the fraction of the above-chance signal the intervention removed |
| G1–G5 | the plan's gates: frame pairing, permutation validity, reproduction, OOF integrity, frozen analysis |

## Method

Five arms were extracted from the same videos with the extractor's new `--temporal`
flag. The reorder is applied after the letterbox transform and before the backbone, so
pixels and clip starts are the baseline's; each arm is paired frame-for-frame with it.

| arm | what enters the backbone |
| --- | --- |
| `frame_shuffle` (primary) | the 16 frames in a uniform random order, drawn per clip from SHA-256 of sample id, clip start and the tag 20260911 |
| `frame_reverse` | the 16 frames reversed |
| `tubelet_shuffle` | the 8 consecutive frame pairs in random order, each pair intact |
| `rep_static_frame` | the clip's 9th sampled frame (index 8) repeated 16 times |
| `frame_identity` | natural order through the same code path; reproduction gate only |

The one design decision a reader would otherwise get wrong: the LOSO classifier is
retrained from scratch on every arm's features, with the recipe, folds, seeds
(42 / 7 / 1234) and threshold objective frozen since the framing plan. The question is
what the frozen backbone still delivers under each input, not whether a classifier
trained on ordered features transfers to shuffled ones.

Statistics are the identity control's, imported unchanged: cameras averaged per rep,
per-session AUC, seed-averaged, subject-macro, n = 9, within-session label permutation
null with 10,000 draws and seed 20260911. The paired inference is baseline minus arm per
subject, two-sided exact Wilcoxon, bootstrap over subjects with 10,000 resamples and the
same seed. All of this was fixed in the plan before any temporal feature existed.

## Gates

| gate | result |
| --- | --- |
| G1 frame pairing | pass, all five arms: 2,144 bundles each, clip starts and frame ranges identical to the baseline bundle by bundle, provenance identical except the temporal stamp; features moved for every reordered arm (mean relative L2 vs baseline: shuffle 0.81, reverse 0.25, tubelet 0.64, static 0.84) and are byte-identical for `frame_identity` (0.0) |
| G2 permutation validity | pass, all five arms: 8,576 clips each, 0 invalid; `frame_shuffle` mean displacement 5.3057 against the uniform expectation 5.3125 (tolerance ±0.15), 0 identity draws; `tubelet_shuffle` 5.2537 with every pair adjacent and in order; reverse 8.0; static 4.0; reseeding 32 sampled clips per arm reproduced every stored permutation bit for bit |
| G3 reproduction | pass: `frame_identity` returns within-session AUC 0.8741 (delta 4.6e−5) and LOSO BA 0.6612 (delta 8.1e−6), all nine per-subject BA deltas exactly 0; the stored baseline is therefore the paired baseline and no swap was needed |
| G4 OOF integrity | pass, all arms: 2,128 primary rows per seed, unique ids, finite, `person_id == test_subject`, cam17/cam18 pairs with agreeing labels, 61 mixed-label sessions, the same 3 single-class sessions excluded (`Ex1_PM_000`, `Ex3_PM_044`, `Ex5_PM_028`) |
| G5 frozen analysis | pass: plan `5faf4db1`, code `32ad26ce`, both before any temporal arm was extracted; `frame_identity` passed G3 before any reordered arm was started (see Deviations for the two later code commits) |

## Results

Primary statistic and the registered paired inference, n = 9 subjects.

| arm | AUC (mean ± SD) | > 0.5 | bootstrap 95% CI | permutation p | paired delta (baseline − arm) | drop | delta 95% CI | Wilcoxon p | share |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `full_frame_letterbox` (baseline) | 0.8741 ± 0.0408 | 9/9 | [0.850, 0.899] | 1/10001 | — | — | — | — | — |
| **`frame_shuffle`** (primary) | **0.8798 ± 0.0647** | 9/9 | [0.838, 0.917] | 1/10001 | **−0.0057 ± 0.0489** | 3/9 | [−0.034, +0.025] | 0.734 | −1.5% [−9.2%, +6.7%] |
| `frame_reverse` | 0.8646 ± 0.0373 | 9/9 | [0.842, 0.888] | 1/10001 | +0.0095 ± 0.0164 | 7/9 | [−0.001, +0.019] | 0.129 | +2.6% [−0.3%, +4.9%] |
| `tubelet_shuffle` | 0.8641 ± 0.0500 | 9/9 | [0.833, 0.894] | 1/10001 | +0.0100 ± 0.0238 | 7/9 | [−0.005, +0.024] | 0.301 | +2.7% [−1.3%, +6.5%] |
| `rep_static_frame` | 0.7002 ± 0.0939 | 9/9 | [0.643, 0.756] | 1/10001 | +0.1739 ± 0.0659 | 9/9 | [+0.137, +0.216] | 0.0039 | +46.5% [+35.1%, +59.9%] |

Per subject, baseline → shuffle → static: P1 0.837 → 0.838 → 0.686, P2 0.895 → 0.940 → 0.623,
P3 0.875 → 0.880 → 0.697, P4 0.817 → 0.760 → 0.528, P5 0.835 → 0.909 → 0.677,
P6 0.945 → 0.937 → 0.856, P7 0.885 → 0.806 → 0.723, P8 0.913 → 0.938 → 0.796,
P9 0.867 → 0.910 → 0.716. The shuffle's per-subject deltas range from −0.074 (P5) to
+0.078 (P7); the static arm's from +0.089 (P6) to +0.289 (P4).

Reading-table row hit: **row 3, undetermined** for the primary. The shuffled arm is above
chance (0.8798, p = 1/10001), the paired drop is negative and 3/9, so neither row 1
(drop ≥ 0.10) nor row 2 (drop in (0, 0.10) with ≥ 7/9 and p < 0.05) applies. Row 5
(static ≥ shuffle in ≥ 6/9 subjects) is not triggered: 0/9. Row 6 (reverse drop ≥ 0.05
with p < 0.05) is not triggered.

The pattern to see: the ranking survives every reordering of the 16 frames within the
noise of nine subjects, and loses about half its above-chance span when only one frame
of the clip is shown. Among the three reorderings the ordering is shuffle ≥ reverse ≈
tubelet, but the three are within 0.016 of each other and none of those differences was
tested.

## Secondary

**LOSO balanced accuracy against the framing headline 0.6612 ± 0.0567**, paired,
Holm-corrected across the four arms (the registered family).

| arm | LOSO BA (mean ± SD over subjects) | per seed | delta vs 0.6612 | up | two-sided p | Holm p |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `frame_shuffle` | 0.7007 ± 0.0619 | 0.705 / 0.699 / 0.698 | +0.0395 | 8/9 | 0.0117 | 0.0352 |
| `frame_reverse` | 0.6730 ± 0.0629 | 0.692 / 0.671 / 0.656 | +0.0118 | 6/9 | 0.250 | 0.500 |
| `tubelet_shuffle` | 0.6722 ± 0.0454 | 0.669 / 0.656 / 0.691 | +0.0110 | 6/9 | 0.496 | 0.500 |
| `rep_static_frame` | 0.5685 ± 0.0505 | 0.573 / 0.560 / 0.573 | −0.0928 | 0/9 | 0.0039 | 0.0156 |

The shuffled arm's cross-subject score is above the baseline's after correction. This
was not predicted and is not explained here; it is a different statistic from the
within-session one (thresholded, cross-subject, not rank-based) and the two are not
comparable. The static arm's 0.5685 sits between the appearance-only control's 0.5241
(one mid-video frame shared by every rep) and the baseline, as expected for a
rep-specific single frame.

**Position-balanced within-session AUC on the 34 interleaved sessions**, where the
position rule scores 0.5000 by construction: baseline 0.8497, shuffle 0.8571, reverse
0.8295, tubelet 0.8514, static 0.6538. Unrestricted AUC on the same 34: 0.8487 / 0.8659 /
0.8381 / 0.8629 / 0.7001. Reordering does not change how much of the ranking survives
position balancing.

**Per camera and per exercise**, means and subject counts only, no tests.

| arm | cam17 | cam18 | Ex1 | Ex2 | Ex3 | Ex4 | Ex5 | Ex6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.8206 | 0.8670 | 0.9705 | 0.7193 | 0.9403 | 0.9088 | 0.8846 | 0.8331 |
| `frame_shuffle` | 0.8354 | 0.8567 | 0.9650 | 0.8003 | 0.8096 | 0.9367 | 0.8921 | 0.8507 |
| `frame_reverse` | 0.8129 | 0.8408 | 0.9720 | 0.6847 | 0.9111 | 0.9032 | 0.8875 | 0.8357 |
| `tubelet_shuffle` | 0.8150 | 0.8529 | 0.9485 | 0.7349 | 0.8506 | 0.9318 | 0.9070 | 0.8368 |
| `rep_static_frame` | 0.6443 | 0.6885 | 0.7734 | 0.6328 | 0.8508 | 0.5644 | 0.6815 | 0.7818 |

Ex3 (8 sessions) is the one exercise where the shuffle costs something visible
(0.940 → 0.810) and Ex2 the one where it gains (0.719 → 0.800); neither was tested.

**P10-inclusive sensitivity**, n = 10: baseline 0.8867, shuffle 0.8918, reverse 0.8781,
tubelet 0.8777, static 0.7302.

**Exploratory Holm** over the three secondary arms' within-session paired p-values
(family of 3, the primary excluded): reverse 0.258, tubelet 0.301, static 0.0117.

## Bounds

`rep_static_frame` is the registered floor for "pose configuration with no motion".
Its share, 46.5% [35.1%, 59.9%], is what a set of 16 poses adds over one pose, whatever
the order of that set. `frame_shuffle` is the registered upper bound on order use: its
drop is −0.006 with interval [−0.034, +0.025], so under the plan's own caveat (a shuffled
clip is out of distribution for the Kinetics backbone, and any OOD degradation would
only enlarge the drop) the order-dependent share is bounded above by the interval, not
estimated. The features did move under the shuffle (mean relative L2 0.81 against the
baseline, larger than the reverse's 0.25), so the null delta is not an unapplied
transform.

## Deviations from the plan

| item | plan | actual | effect |
| --- | --- | --- | --- |
| G1 content | four sampling keys equal | after an isolated code review, G1 additionally requires provenance equality with the baseline on every key except the temporal stamp, and a non-zero feature distance for reordered arms (commit `b7ac8f6c`, before any reordered arm was gated) | stricter than registered; all arms pass; `frame_identity` re-gated under the stricter rule, unchanged |
| absolute verdict | row 4 trigger is mean ≤ 0.55 or permutation p ≥ 0.05 | the first code version also required ≥ 6/9 subjects above 0.5 (inherited from the identity control); corrected to the registered two conditions in `b7ac8f6c` before any arm was read | none; every arm is 9/9 above 0.5 |
| paired inference gate | not stated | `paired` refuses an arm whose `analyze` (G4) has not run; Holm keeps the family size at the number of arms compared when a p is unavailable | none; all four p-values available |
| reproduce interface | `videomae_identity_control.py predict/analyze --arm …` | the identity CLI has no `--arm`; shipped as the temporal module's own `gates / materialize / predict / analyze / paired` subcommands | none |
| G3 on a missing folds file | "fails" | reported as unavailable, not failed, so a lost file cannot trigger the baseline-swap row | none; nothing was missing |
| extraction order | `frame_identity` gated before any other arm | as planned; the four reordered arms then ran as one chained job, and each arm's LOSO ran while the next arm extracted | none; the LOSO recipe is seeded and CPU-only |
| runtime | ~80 min per arm | 95 min for `frame_identity` on an idle machine; 4.0 h / 3.3 h / 2.5 h / 4.2 h for shuffle / reverse / tubelet / static, because extraction is CPU-bound (GPU utilisation 0%) and other applications held the CPU | none |

## Not supported

- "The model does not use temporal order." The primary is undetermined by the
  registered rule; no equivalence test was run and n = 9 cannot power one.
- "The model reads movement dynamics." Nothing in the shuffle result licenses it; the
  drop is negative. The static-frame result shows that more than one frame is needed,
  not that the frames' order is.
- Any explanation of the shuffled arm's higher LOSO balanced accuracy (+0.040, Holm
  p = 0.035). It is one secondary comparison on a different statistic, unpredicted, and
  is reported only.
- The differences among reverse, tubelet and shuffle (all within 0.016) were not tested.
- The per-exercise movements (Ex3 −0.13, Ex2 +0.08 under the shuffle) are means over
  8–12 sessions with no test.
- Which frames matter: a static frame at index 8 was the only single-frame arm; the
  clip's other frames were not tried.
- Static carriers of recording position. Whatever survives the shuffle is consistent
  with both an unordered pose set and any static carrier; the position control's 4.9%
  floor and its linear-only limit apply to it unchanged.
- Position drift that steps at the label boundary in the 27 single-boundary sessions;
  unchanged data-collection limit.
- The Kinetics checkpoint only. The SSv2 fine-tune was not run under reordering.
- Cross-day, cross-clothing, cross-venue generalisation.

## Reproduce

Logic `src/rehab24/videomae_temporal_control.py`; the extractor's `--temporal` flag and
`reorder_clip_frames` in `src/rehab24/videomae_features.py`; CLI
`scripts/rehab24/videomae_temporal_control.py`; 34 tests in
`tests/test_rehab24_videomae_temporal_control.py`. Extraction from `.venv-cuda`
(torch 2.13.0+cu126, transformers 5.5.0, OpenCV 4.13.0) on a GTX 1660 Ti with three
round-robin workers; classifier and statistics from `.venv` on CPU.

```powershell
# per arm, in this order; frame_identity first and its analyze must print "G3 reproduction: PASS"
$env:OMP_NUM_THREADS = 4
.venv-cuda\Scripts\python.exe scripts\rehab24\extract_videomae_features.py --variant full_frame_letterbox --temporal frame_identity --device cuda --num-chunks 3 --chunk-index 0 --output-dir data\REHAB24-6\processed\videomae_raw_temporal\frame_identity   # and chunk-index 1, 2
.venv\Scripts\python.exe scripts\rehab24\videomae_temporal_control.py gates       --arm frame_identity
.venv\Scripts\python.exe scripts\rehab24\videomae_temporal_control.py materialize --arm frame_identity
.venv\Scripts\python.exe scripts\rehab24\videomae_temporal_control.py predict     --arm frame_identity --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_temporal_control.py analyze     --arm frame_identity
# same five steps for frame_shuffle, frame_reverse, tubelet_shuffle, rep_static_frame, then
.venv\Scripts\python.exe scripts\rehab24\videomae_temporal_control.py paired
```

Runtime: extraction 1.5–4 h per arm depending on CPU contention (~2.2 s per bundle on an
idle machine); `predict` 20–35 min per arm for three seeds; `analyze` about 2 min;
`paired` seconds. Artifacts under `data/REHAB24-6/processed/`: raw bundles with the
stored permutations in `videomae_raw_temporal/<arm>/`, materialised features in
`videomae_temporal/<arm>/videomae_mean_pool_fc_norm_mean/`, and in
`videomae_temporal_control/`: `gates_<arm>.json`, `oof_<arm>_seed{42,7,1234}.csv`,
`folds_<arm>_seed*.json`, `within_session_summary_<arm>.json`,
`permutation_null_<arm>.npz`, `paired_deltas.json`.
