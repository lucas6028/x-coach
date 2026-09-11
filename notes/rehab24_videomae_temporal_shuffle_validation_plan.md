# Plan: how much of the REHAB24-6 within-session VideoMAE signal needs the order of the 16 frames?

*REHAB24-6 · pre-registered temporal-order control · written 2026-09-11, before any
frame-shuffled feature was extracted · siblings:
[`rehab24_videomae_position_control_results.en.md`](rehab24_videomae_position_control_results.en.md),
[`rehab24_videomae_identity_appearance_results.md`](rehab24_videomae_identity_appearance_results.md),
[`rehab24_videomae_framing_results.md`](rehab24_videomae_framing_results.md),
[`rehab24_videomae_ssv2_checkpoint_results.md`](rehab24_videomae_ssv2_checkpoint_results.md)*

**Question.** The position control left one path open in its Not supported list:
"within-clip temporal order (16-frame shuffle) is still untested". Every carrier of
recording position that the earlier controls looked for is static within a clip: stance,
camera settling, lighting, body appearance, background. None of them changes when the 16
frames of a clip are reordered. Movement dynamics do. So the part of the within-session
signal that disappears when frame order is destroyed is a part that no static carrier can
explain. It would be a floor on how much of the 0.8741 is read from how the person moves,
except that a reordered clip is also unnatural input for the backbone; the Design section
says how that is bounded. This plan asks one fixed question: how much within-session ranking survives when the
16 frames of every clip are shown to VideoMAE in random order, with every pixel
unchanged.

**Status.** Pre-registration. No temporal arm has been extracted. Dataset REHAB24-6,
subjects P1–P9 for the primary, P10 as sensitivity only. Pixels: `full_frame_letterbox`
with the corrected pooling `mean_pool_fc_norm_mean`, Kinetics-400 checkpoint. Classifier,
folds, seeds (42 / 7 / 1234), validation subjects and threshold objective are those of
`videomae_stage_a.run_arm`, not one line changed.

## Terminology

| term | meaning |
| --- | --- |
| rep / repetition | one execution of the exercise; each has its own correct / incorrect label and a first / last frame in `Segmentation.csv` |
| session | one `exercise_id + video_id`: one person, one exercise, one recording, filmed by cam17 and cam18 |
| clip | 16 frames sampled at stride 2 from one rep, starting at one of 4 evenly spaced clip starts; VideoMAE encodes one clip per forward pass, the 4 clip embeddings are mean-aggregated into the rep's feature |
| tubelet | VideoMAE's temporal token unit: 2 consecutive frames, so a 16-frame clip is 8 tubelets; motion inside a tubelet and order across tubelets are two different things to the backbone |
| `full_frame_letterbox` | the whole frame padded to square; the framing experiment's best arm and the baseline of every later control; within-session AUC 0.8741 ± 0.0408, LOSO balanced accuracy 0.6612 ± 0.0567 |
| temporal arm | the same pixels and the same sampled frames as `full_frame_letterbox`, with only the order (or repetition) of the 16 frames changed before they enter the backbone |
| within-session AUC | inside one recording, the probability that a randomly drawn correct rep outranks a randomly drawn incorrect one; cameras averaged per rep, seed AUCs averaged per session, sessions averaged per subject, n = 9; the identity-control statistic |
| LOSO BA | leave-one-subject-out balanced accuracy, one fold per subject, seeds averaged within subject; the framing-experiment statistic |
| subject-macro | one value per subject; the subject is the only inference unit, n = 9 |
| share | (baseline − arm) / (baseline − 0.5): the fraction of the above-chance signal that the intervention removed |
| position drift | anything monotone along the recording that is visible in the frame; the position control's candidate carriers, all static within a clip |

## Motivating observations (already established before registration)

| observation | value | source |
| --- | ---: | --- |
| within-session AUC, `full_frame_letterbox` | 0.8741 ± 0.0408, 9/9, p = 1/10001 | identity-control note |
| the same after projecting out 16 linear position directions | 0.8556 ± 0.0435, share 4.9% (a floor) | position-control note |
| person deleted, scene and its drift kept (`background_only`) | 0.5357 ± 0.0472 within-session; 0.5074 LOSO BA | identity-control note (person-deleted upper bound), framing note |
| one mid-video frame repeated 16× and shared by every rep of the video (`canonical_frame_repeat`) | LOSO BA 0.5241 ± 0.0185, 9/9 below baseline, p = 0.0039 | identity-control note (appearance-only negative control) |
| `repetition_number` alone, zero parameters | within-session AUC 0.1761 (0.8239 reversed); LOSO BA 0.7309 | identity-control and position-control notes |

The controls so far removed carriers one at a time and the signal stayed. All of them
were static within a clip. The identity note's Not supported list already says that
single-frame pose differences inside a recording could produce the AUC without the model
using time at all, and that a repetition-level static-frame control and a temporal-shuffle
control are needed to tell. Neither has been run. This plan runs both, with the shuffle as
primary.

## Design

**Intervention.** Frames are decoded, transformed to `full_frame_letterbox` and then
reordered, exactly at the point between `transform_frames` and `encode_clip` in
`src/rehab24/videomae_features.py`. Clip starts are unchanged, so every temporal arm
samples the identical frames as the stored baseline, and every comparison is paired
frame-for-frame. Nothing downstream changes: pooling, materialisation, `run_arm`, the
within-session statistic, the permutation null.

| arm | what enters the backbone | what it isolates |
| --- | --- | --- |
| `frame_shuffle` (**primary**) | the 16 frames in a uniform random permutation, drawn independently per clip | everything that needs order: motion inside tubelets and order across tubelets |
| `frame_reverse` | the 16 frames in reverse order | direction only; local continuity and the set of tubelet contents are intact, so it is in-distribution for the backbone |
| `tubelet_shuffle` | the 8 consecutive frame pairs kept intact, the pairs in a random order | order across tubelets only; motion inside each tubelet survives |
| `rep_static_frame` | the clip's middle sampled frame (index 8 of 16) repeated 16 times | a repetition-specific static frame: pose configuration with no motion at all, the floor the identity note asked for |
| `frame_identity` | the 16 frames in their natural order, re-extracted through the same code path | reproduction gate and, if the stored baseline cannot be matched, the paired baseline |

**Why the primary is the full shuffle and not the reverse.** Reversal keeps every tubelet
and its internal motion, so a model that reads within-tubelet velocity but ignores global
direction would be untouched by it. The full shuffle is the only arm that removes all
order information. Its cost is that a shuffled clip is out of distribution for a backbone
trained on natural video, so part of any drop can be degradation rather than "order was
used". That is why the three temporal arms are read together (below) and why the drop is
registered as an **upper bound** on the order-dependent share, not a point estimate.

**Randomness, fixed now.** Each clip's permutation comes from
`numpy.random.default_rng(seed)` with `seed = SHA-256(f"{sample_id}:{clip_start}:20260911")`
truncated to 64 bits. Both cameras and all four clips of a rep therefore get independent
permutations; re-running extraction reproduces them exactly. The permutation array
(num_clips × 16, int8) is stored in every raw bundle. A drawn permutation equal to the
identity is kept, not redrawn.

**Readout retraining.** The LOSO classifier is trained from scratch on each arm's
features, as for every previous arm. The question is what the frozen backbone still
delivers, not whether a classifier trained on ordered features transfers to shuffled ones.

## Primary comparison

Observed statistic: subject-macro within-session AUC of `frame_shuffle`, exactly as in the
identity-control plan (cameras averaged per rep, per-session AUC, seed-averaged,
subject-averaged, n = 9), and its **paired delta** against `full_frame_letterbox`,
0.8741, subject by subject.

Two inferences, both fixed:

1. Absolute: within-session label permutation null on the `frame_shuffle` arm, labels
   permuted inside each session preserving class counts, the same permutation applied to
   both cameras and all three seeds, 10,000 draws, seed 20260911,
   p = (1 + #null ≥ observed) / 10,001.
2. Paired: nine per-subject deltas (baseline − shuffle), mean ± SD, count of subjects
   with a drop, exact two-sided Wilcoxon signed-rank; bootstrap 95% interval over
   subjects, 10,000 resamples, seed 20260911.

Share = (0.8741 − shuffle AUC) / (0.8741 − 0.5), reported with the interval.

**Prior, fixed before the run.** The static controls so far removed at most 4.9% of the
signal, and `canonical_frame_repeat` (appearance, no per-rep information) is at chance.
A per-rep pose set without order is expected to keep most of the ranking, so the prior for
`frame_shuffle` is 0.76–0.86, a drop of 0.01–0.11. `frame_reverse` is expected within
0.02 of baseline. `rep_static_frame` is expected below `frame_shuffle`. A `frame_shuffle`
result below 0.70 would mean the model leans on dynamics much more than the earlier
controls implied, which is the outcome this plan must be able to detect.

**Power, stated now.** With n = 9 and a paired SD of about 0.03–0.05 (the position
control's k = 16 deltas), the exact Wilcoxon can reach p < 0.05 only if at least 8 of 9
subjects move the same way and the mean delta is about 0.03 or larger. Deltas below that
are reported as undetermined; this design cannot show that order does not matter.

## Secondary (never changes the primary reading)

- `frame_reverse` and `tubelet_shuffle`: the same two inferences as the primary. Read
  together with the primary as a triple: a drop present under `frame_shuffle` but absent
  under `tubelet_shuffle` locates the order-dependence inside tubelets (velocity); a drop
  present under both but absent under `frame_reverse` locates it in cross-tubelet
  sequence rather than direction.
- `rep_static_frame`: the same two inferences. This is the floor for "pose configuration
  alone"; the gap `frame_shuffle − rep_static_frame` is what a set of poses adds over a
  single pose.
- LOSO BA of every temporal arm against the framing headline 0.6612 ± 0.0567: paired
  nine-subject delta, exact Wilcoxon, Holm across the four arms. This is the cross-subject
  reading and is reported separately because the two statistics are not comparable
  (the identity note: 0.8741 averages cameras and seeds and ranks within a recording,
  0.6612 is a thresholded cross-subject score).
- Position-balanced within-session AUC of `frame_shuffle` on the 34 interleaved sessions
  (the position control's secondary, baseline 0.8497), and the position rule's 0.5000 on
  the same sessions as a check that the subset is unchanged.
- Per-exercise and per-camera within-session AUC of `frame_shuffle`, means and
  subject counts only, no tests.
- P10-inclusive sensitivity, n = 10.

All secondary p-values are reported raw and Holm-corrected. No favourable arm may replace
the primary.

## Statistical unit

sample = rep × camera; repetition = one cam17 / cam18 pair; session = one
`exercise_id + video_id`; **the subject is the only inference unit, n = 9**. Cameras,
reps, clips, sessions and seeds are not n.

## Reading the result (fixed now)

0.55 is the project's standing practical threshold. 0.10 of AUC is the registered
boundary between "order carries a substantial share" and "order carries little", chosen
before any number exists as roughly twice the position control's k = 16 drop (0.019) plus
its paired SD. Neither is an equivalence bound.

| pattern | registered reading | next step |
| --- | --- | --- |
| `frame_shuffle` drop **≥ 0.10**, ≥ 7/9 subjects drop, Wilcoxon p < 0.05, and the shuffled arm still > 0.55 with permutation p < 0.05 | at least that share of the within-session signal needs frame order; no static carrier (position drift, appearance, background) can produce it. The removed share is reported as an upper bound on order use, with `frame_reverse` and `tubelet_shuffle` bounding the out-of-distribution component; the surviving part stays ambiguous | write the bounded share into the framing and identity notes; the fusion pre-registration may be opened under the identity plan's conditions |
| `frame_shuffle` drop in **(0, 0.10)**, ≥ 7/9 drop, p < 0.05 | order is used, but most of the ranking is order-free: a set of poses within the rep ranks correct against incorrect. Report share; may not be written as "the model reads movement dynamics" | run nothing more on order; the pose-set reading moves the next control to per-frame pose (2D keypoints from the same frames) as an explicit comparator |
| `frame_shuffle` drop with p ≥ 0.05, or < 7/9 subjects | **undetermined**; not "order does not matter", not "the model ignores time" | report; `tubelet_shuffle` and `frame_reverse` are read for direction only |
| `frame_shuffle` **≤ 0.55** or permutation p ≥ 0.05 | the within-session signal is order-dependent almost entirely; the earlier "person, not background" readings stand but the mechanism is dynamics, and OOD degradation of the backbone cannot be separated from it | `tubelet_shuffle` and `frame_reverse` become the informative arms; if they also collapse, the backbone's OOD sensitivity is the finding, and a per-frame pose comparator is required before any mechanism claim |
| `rep_static_frame` **≥** `frame_shuffle` (paired, ≥ 6/9) | a single frame ranks as well as an unordered set; the signal is a static pose cue | reported under mechanism; the primary reading does not change |
| `frame_reverse` drop ≥ 0.05 with p < 0.05 | the backbone reads direction; unexpected under the prior | reported; no claim beyond the number |
| `frame_identity` fails to reproduce 0.8741 to four decimals | the extraction venv differs from the one that produced the stored baseline (README: transformers version dominates hardware by ~230×) | every paired delta is taken against `frame_identity`, both baselines are reported, and this is a plan deviation |

p ≥ 0.05 is written as **undetermined**, never "no difference", never "equivalent".

## Gates (all must pass before any outcome is read)

| gate | check |
| --- | --- |
| G1 frame pairing | every temporal bundle's `clip_starts`, `first_frame`, `last_frame`, `total_frames` equal the stored `full_frame_letterbox` bundle's for the same sample, all 2,144 samples; any mismatch invalidates the arm |
| G2 permutation validity | per clip, the stored permutation is a permutation of 0–15; over all clips of `frame_shuffle` the mean absolute displacement is within ±0.15 of the uniform expectation 5.3125 ((n² − 1)/(3n) for n = 16); `tubelet_shuffle` keeps every pair (2i, 2i+1) adjacent and in order; `frame_reverse` is exactly 15 − i; `rep_static_frame` stores the index 8 repeated; `frame_identity` stores 0–15; a re-run of the seeding function on 32 sampled clips reproduces the stored arrays bit for bit |
| G3 reproduction | `frame_identity` through materialise → `predict` → `analyze` returns within-session AUC 0.8741 to four decimals and LOSO BA 0.6612 to four decimals; if either fails, the baseline swap in the reading table applies and the relative L2 between stored and re-extracted features is reported |
| G4 OOF integrity | per arm and seed: 2,128 primary rows, unique sample ids, finite probabilities, `person_id == test_subject` on every row, one cam17 and one cam18 row per rep with agreeing labels, 61 mixed-label sessions |
| G5 frozen analysis | this note, the temporal transform code, the seeding rule, the permutation seed 20260911 and the reading table are committed before any temporal arm is extracted; after the primary is seen, the arm definitions, the aggregation order, the sidedness and the 0.10 boundary may not change; any change is a plan deviation in the results note, with the registered analysis kept alongside |

## What this plan cannot show (even if the primary passes)

- **A drop is an upper bound on order use.** Shuffled frames are out of distribution for
  a Kinetics-trained backbone; some of the drop is degradation. `frame_reverse` and
  `tubelet_shuffle` narrow this but do not remove it. A per-frame pose comparator on the
  same frames would; it is not in this plan.
- **Survival is not a static-shortcut verdict.** Whatever survives the shuffle is
  consistent with both pose-set reading and any static carrier. The position control's
  4.9% floor and the linear-only limit still apply to that part unchanged.
- **Position drift that steps at the label boundary** remains invisible; the 27
  single-boundary sessions cannot separate it from movement. Data-collection limit,
  carried over from the position control.
- **Nothing about which frames matter.** A drop says order matters, not whether the
  bottom of the movement, the descent or the ascent carries it.
- **No cross-day, cross-clothing or cross-venue generalisation.**
- **The n = 9 Wilcoxon cannot establish a small effect.** A delta reported as
  undetermined is not evidence that order is unused.
- **Kinetics checkpoint only.** The SSv2 fine-tune (within-session 0.7805) is not run
  here; its temporal behaviour may differ and is a separate question.

## Implementation and compute

- `src/rehab24/videomae_features.py`: one new keyword `temporal` on
  `extract_repetition_features`, default `"identity"`, applied after `transform_frames`
  and before `encode_clip`; a pure function `reorder_clip_frames(frames, temporal,
  sample_id, clip_start) -> (frames, permutation)` holding the five rules and the
  seeding; the permutation stack saved into the bundle as `frame_permutations`;
  provenance gains `temporal_transform` and `temporal_seed`. CLI flag `--temporal`,
  orthogonal to `--variant`. `assert_output_dir_matches_variant` gains the temporal name so
  a temporal arm cannot be written into the baseline's directory.
- `scripts/rehab24/videomae_temporal_control.py` and
  `src/rehab24/videomae_temporal_control.py`: gates G1–G2, the paired delta with exact
  Wilcoxon and bootstrap, the four-arm Holm table; `midranks`, `build_sessions`,
  `observed_statistic`, `permutation_null` imported from `videomae_identity_control.py`,
  the position-balanced statistic from `videomae_position_control.py`, nothing rewritten.
  `predict` and `analyze` are the identity-control CLI pointed at the temporal feature
  dirs (`--output-dir` after the subcommand).
- Tests, `tests/test_rehab24_videomae_temporal_control.py` and additions to
  `tests/test_videomae_features_extraction.py`, on synthetic frames: (a) identity is a
  no-op and stores 0–15, (b) reverse is 15 − i, (c) tubelet shuffle keeps pairs adjacent
  and ordered, (d) static repeats index 8 and yields 16 identical arrays, (e) the shuffle
  is deterministic in (sample_id, clip_start) and differs across clips and cameras,
  (f) G2's displacement statistic on 10,000 synthetic draws lands within ±0.05 of 5.3125,
  (g) the paired delta and Wilcoxon reproduce a hand-computed nine-subject case.
- Compute: extraction on this machine's GTX 1660 Ti from `.venv-cuda`, 3 workers,
  `OMP_NUM_THREADS=4`, ~2.2 s per bundle → about 80 min per arm, five arms ≈ 6.5 h GPU.
  `frame_identity` is extracted first and gated (G3) before any other arm is started.
  Classifier and statistics on CPU in `.venv`, 20–40 min per arm for three seeds.
  Both the identity arm and the temporal arms must come from the same venv (README
  rule: provenance does not record the device, a mixed arm is undetectable afterwards).

Planned artifacts:

```text
data/REHAB24-6/processed/videomae_raw_temporal/{frame_identity,frame_shuffle,frame_reverse,tubelet_shuffle,rep_static_frame}/   # raw bundles, permutations inside
data/REHAB24-6/processed/videomae_temporal/<arm>/videomae_mean_pool_fc_norm_mean/                                                 # materialised, one dir per arm
data/REHAB24-6/processed/videomae_temporal_control/oof_<arm>_seed{42,7,1234}.csv
data/REHAB24-6/processed/videomae_temporal_control/gates_{frame_pairing,permutations}.json
data/REHAB24-6/processed/videomae_temporal_control/within_session_summary_<arm>.json
data/REHAB24-6/processed/videomae_temporal_control/permutation_null_<arm>.npz
data/REHAB24-6/processed/videomae_temporal_control/paired_deltas.json                                                             # primary + Holm table
data/REHAB24-6/processed/videomae_temporal_control/loso_ba_<arm>.json
```

## Reproduce (planned interface)

Gates first, then the primary, secondaries last. All commands from the repo root.

```powershell
# G3: identity arm through the new code path, must reproduce 0.8741 / 0.6612
$env:OMP_NUM_THREADS = 4
.venv-cuda\Scripts\python.exe scripts\rehab24\extract_videomae_features.py --variant full_frame_letterbox --temporal frame_identity --device cuda --num-chunks 3 --chunk-index 0 --output-dir data\REHAB24-6\processed\videomae_raw_temporal\frame_identity   # repeat chunk-index 1, 2
.venv\Scripts\python.exe scripts\rehab24\materialize_videomae_features.py --raw-dir data\REHAB24-6\processed\videomae_raw_temporal\frame_identity --output-parent data\REHAB24-6\processed\videomae_temporal\frame_identity
.venv\Scripts\python.exe scripts\rehab24\videomae_temporal_control.py gates --arm frame_identity
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py predict --feature-dir data\REHAB24-6\processed\videomae_temporal\frame_identity\videomae_mean_pool_fc_norm_mean --seeds 42 7 1234 --device cpu --output-dir data\REHAB24-6\processed\videomae_temporal_control
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py analyze --arm frame_identity --permutations 10000 --permutation-seed 20260911 --output-dir data\REHAB24-6\processed\videomae_temporal_control   # must equal 0.8741

# primary and the three secondary arms: same three steps per arm
foreach ($arm in "frame_shuffle","frame_reverse","tubelet_shuffle","rep_static_frame") { <# extract --temporal $arm, materialize, gates, predict, analyze #> }

# paired inference, Holm table, LOSO BA deltas, position-balanced subset
.venv\Scripts\python.exe scripts\rehab24\videomae_temporal_control.py paired --baseline full_frame_letterbox --arms frame_shuffle frame_reverse tubelet_shuffle rep_static_frame --bootstrap 10000 --seed 20260911
.venv\Scripts\python.exe scripts\rehab24\videomae_framing_report.py --arm full_frame_letterbox=data\REHAB24-6\processed\videomae_framing\full_frame_letterbox\videomae_mean_pool_fc_norm_mean --arm frame_shuffle=data\REHAB24-6\processed\videomae_temporal\frame_shuffle\videomae_mean_pool_fc_norm_mean
```

The results note goes to `notes/rehab24_videomae_temporal_shuffle_results.md` and reports,
item by item: the five gates, plan deviations, the primary's nine per-subject values and
deltas, both inferences, the three secondary arms as a triple, the static-frame floor, the
LOSO BA table with Holm, which row of the reading table was hit, and the limits listed
above. The position-control note's Not supported line on the 16-frame shuffle, the
identity note's Not supported line on temporal order, and the framing note are updated
**together** according to that row, never one alone.
