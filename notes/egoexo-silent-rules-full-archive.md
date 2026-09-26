# Both silent rules' metrics rank the human-flagged actions at AUC 0.73; at the spec's own cut, Jumping Jacks still flags 51% of correct clips

*EgoExo-Fitness · pre-registered separability test of two silent pose rules · run 2026-09-26 ·
plan: [`docs/superpowers/specs/2026-09-26-egoexo-silent-rule-separability-prereg.md`](../docs/superpowers/specs/2026-09-26-egoexo-silent-rule-separability-prereg.md)*

**Result.** On the full EgoExo-Fitness archive, each rule's own metric ranks the human-flagged
actions on the fault side:

| rule | actions | flagged | participants (flagged of all) | AUC | 95% CI |
|---|---:|---:|---|---:|---|
| Jumping Jacks `jj_incomplete_leg_rom` | 121 | 12 | 8 of 30 | 0.730 | [0.591, 0.879] |
| High Knee `hk_insufficient_knee_lift` | 59 held out | 12 | 9 of 25 | 0.730 | [0.586, 0.864] |

What each result licenses under the plan:
- **Jumping Jacks: a proposal to un-silence, which goes to the user.** Every pre-registered
  robustness condition held.
  - The cut read off the labels is 1.321, essentially the spec's own 1.3, so un-silencing would
    not move the pinned constant.
  - Held out by participant, that cut scores sensitivity 0.667 and specificity 0.633.
  - At 1.3, a feedback card (at least one firing rep) would appear on 164 of 319 correct
    (action, camera) clips (51.4%) and on 28 of 35 flagged ones (80.0%).
- **High Knee: no number.** The spec's two cuts bracket the data. The cited 45° cut fires on a
  median 0% of reps in both classes; the implemented 90° cut fires on 97.6% (unflagged) and
  100% (flagged).

**Outcome.** On 2026-09-26 the user accepted the Jumping Jacks proposal:
- `jj_incomplete_leg_rom` is live at the unmoved 1.3 cut.
- The detector stays unregistered, as the plan fixed, so no production path runs it.
- The live rule, run through `run_detector` on all 356 clips, fires exactly where this analysis
  reconstructed it on 354. The other 2 are the whole-clip fallback pairs, where the rule scores
  the whole clip and the reconstruction has no reps to read (Gates).
- `hk_insufficient_knee_lift`: the user then overrode the plan's "no number from a comment label" rule and made it live at the label-derived 65.6° of hip flexion (next paragraph).

**High Knee, live at 65.6° (user decision, 2026-09-26, exploratory basis).**
- **The cut is fitted, not sourced.** It is Youden's cut on the held-out actions. Refit without each participant it lands at 64.5–65.8° (23 of 25 folds at 65.6°).
- **Held out by participant** it scores sensitivity 0.833 (10/12) and specificity 0.660 (31/47).
- **Citation.** The rule cites Alberton et al. 2015 for the target ("hip and knee flexion to 90°" in running in place; verified in the full text). `citation_support` states that the 65.6° cut was derived by this project from the EgoExo-Fitness annotations, which no paper states. Alberton's "±5° from the target angle (90°)" is a rep-selection window for the study, not a fault tolerance, and it is not used.
- **View gate.** The live rule stays silent unless the rep's median `anterior_axis_length` exceeds 0.170, the largest frontal-camera value observed.
- **Through `run_detector` on every clip:**
  - **Frontal camera:** 0 of 59 held-out clips carded.
  - **Side cameras:** 36 of 90 no-complaint clips carded (40.0%) and 19 of 22 complaint clips (86.4%).
  - **Against the ungated analysis quantity:** the live rule differs on 3 side-camera clips (2 no-complaint, 1 complaint), and the gate explains all 3. Without the gate: 38/90 and 20/22.
  - **The gate is what silences the frontal camera.** 3 of 177 analysed frontal reps pass it. With the gate removed, the rule would card 28 of 47 no-complaint and 11 of 12 complaint frontal clips.
- **These card rates are in-sample.** The 59 actions are held out from building the comment rule, not from fitting the cut. The out-of-sample figure is the leave-one-participant-out sensitivity 0.833 and specificity 0.660.

**Caveat on reading it.**
- **The labels are weak.**
  - The Jumping Jacks criterion has inter-annotator α 0.41, and 10 of its 12 flagged actions had
    a single annotator.
  - Only 2 of those 12 comments describe insufficient width. The width-only subset (5 flagged
    actions) is **undetermined**.
  - High Knee labels are free-text complaints, not checklist items.
- **High Knee's camera gate no longer separates the cameras at full n.** An exploratory check
  shows the result survives (see Secondary).
- **The plan was hashed, not committed, before decoding.**

## Background

Both rules are registered but permanently silent. Their metric was judged clean and their
number was judged wrong:

| rule | small-n finding that silenced it | note |
|---|---|---|
| Jumping Jacks | the spec's 1.3 stance cut fired on 79.1% of reps in actions humans judged correct, on 11 reachable actions | `notes/jumping-jacks-rule-validation.md` |
| High Knee | the implemented 90° cut fired on every rep of 6 reachable actions, and the cited 45° cut appeared to sort them backwards | `notes/high-knee-rule-validation.md` |

Both detectors' docstrings named the full archive as the upgrade path. Only 3 of its 21 parts
were on disk until 2026-09-26.

## Method

| term | meaning |
|---|---|
| widest stance ratio | per rep, the maximum of ankle distance over shoulder width (`stance_width_ratio`); lower means a narrower jack |
| thigh elevation | per rep, the peak of −cos(hip flexion) for the driving leg; −0.707 is 45° of flexion and 0 is the knee at hip height; lower means less lift |
| gated cameras | High Knee reads only `exo_l`/`exo_r`, the two side cameras that can see a sagittal quantity |
| majority label | Jumping Jacks: the action is flagged when a strict majority of annotators marked "Perform the jump by opening and closing your feet." FALSE |
| comment label | High Knee: a pre-registered regular-expression rule flags an action when any comment pairs a leg word with an insufficiency word; 6 actions used to build the rule are held out of the primary |
| AUC | probability that a flagged action scores on the fault side of an unflagged one, ties counted half |

**The design decision a reader would otherwise get wrong:** the unit is the **action**. EgoExo
labels whole actions, not reps. The score has two steps:
1. Per camera, take the median over the camera's scored reps.
2. Per action, take the median over usable cameras: all three exo cameras for Jumping Jacks, the
   two gated cameras for High Knee.

The CI resamples **participants**, 2,000 times, seed 20260926. A participant is grouped by the
name part of `original_actor`, because one person spans several dates. Resampling actions would
treat one person's actions as independent.

Procedure:
1. One streaming pass extracted the three exocentric cameras of every judged Jumping Jacks and
   High Knee action.
2. MediaPipe Pose ran on every frame (`model_complexity=2`, tracking mode).
3. The two existing validation harnesses replayed each clip through the production
   `run_detector` path.
4. `run_silent_rule_separability.py` computed the AUCs.

Fixed before any new frame was decoded:
- the labels;
- the metric and its aggregation;
- the statistics;
- the three verdict words: *separates* if the CI's lower bound exceeds 0.5, *sorted backwards*
  if the upper bound is below 0.5, *undetermined* otherwise;
- the conditions under which a Jumping Jacks cut may be proposed.

The decision was report-only: no rule, cut or registration changes on this run.

## Gates

| gate | result |
|---|---|
| all 21 archive parts at the Hugging Face listing sizes (20 × 3,221,225,472 + 2,895,783,140 bytes) | pass |
| all parts from one Hugging Face commit (2025-02-18), so no mixed revisions | pass |
| `gzip -t` over the concatenated stream (CRC and length) | pass, exit 0 |
| extraction complete (`--require-complete`): clean end of stream, no part after a gap, every pair within one frame of its window | pass; 554 of 554 planned (action, camera) pairs, 171,659 frames; 13 of the 567 possible pairs were never planned because those recordings lack that camera |
| pose files parse and match the extracted frame count | pass, 171,659 frames |
| plan and analysis code unchanged since the pre-decoding hash (08:23:33) | pass, 7 of 7 SHA-256 match |
| Jumping Jacks segmentation | median validity 1.000; 2 of 356 pairs on the whole-clip fallback (both side cameras of `q6ITZM_action_6`) |
| High Knee segmentation | median validity 1.000; 0 of 198 pairs on the fallback; 2,404 scored reps |
| old small-n figures reproduce on the old subsets with the new pose files | High Knee: exact (6 actions, 18 pairs, 146 scored reps, 1.31 Hz, implemented cut 100%). Jumping Jacks: close, not exact. 33 pairs vs 31, because 2 pairs were unreachable in the prefix run; 97 scored reps vs 91; per-rep fire 77.3% vs 79.1%; median widest 1.185 vs 1.163 |
| live rule reproduces the analysed quantity (after un-silencing) | pass. `fired` equals the reconstructed verdict on 354 of 356 clips; the 2 others are the whole-clip fallback pairs. Per-rep widest stance is byte-identical to the pre-change run |
| High Knee camera gate separates side from frontal cameras | **fail.** `anterior_axis_length` spans 0.066–0.441 on side cameras and 0.019–0.170 on the frontal one. 17 of 130 side pairs (14 actions) fall inside the frontal range. At n=6 the ranges did not overlap. |

## Results

| analysis | actions | flagged | flagged participants | AUC | 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| **Jumping Jacks, widest stance vs majority label (primary)** | 121 | 12 | 8 of 30 | **0.730** | [0.591, 0.879] | separates |
| **High Knee, peak thigh elevation vs held-out comment label (primary)** | 59 | 12 | 9 of 25 | **0.730** | [0.586, 0.864] | separates |

No action was lost to an empty score:
- Jumping Jacks scored all 121 actions.
- High Knee's 59 is the 62 held-out actions minus 3 whose comments were unattributable
  (pre-registered exclusion).

The two metrics separate by the same margin.
- **Jumping Jacks:** the flagged actions' median widest stance is 1.222 (IQR 1.097–1.313)
  against 1.374 (IQR 1.250–1.642) for unflagged ones. The interquartile ranges overlap.
- **High Knee:** flagged actions peak at a median 50.6° of hip flexion, unflagged ones at 77.7°.

## Secondary

**Jumping Jacks robustness.**

Two threats were found from the labels before any pixels were read (plan, "JJ label threats"):
- Most flagged comments are not about width. Three describe a front-back scissor jump, a
  different movement that a frontal width ratio also reads as narrow.
- A single-annotator action needs only one FALSE vote to be flagged, so 10 of 50
  single-annotated actions are flagged against 2 of 71 multi-annotated ones.

| analysis | actions | flagged | AUC | 95% CI | verdict |
|---|---:|---:|---:|---|---|
| any-FALSE label | 121 | 17 | 0.746 | [0.622, 0.872] | separates |
| scissor / crossed-feet flagged actions dropped | 118 | 9 | 0.723 | [0.526, 0.905] | separates |
| width and direction-free comments only | 114 | 5 | 0.719 | [0.492, 0.979] | undetermined |
| single-annotator actions only | 50 | 10 | 0.752 | [0.579, 0.917] | separates |
| the spec's 1.3 cut, per-rep fire rate as the action score | 121 | 12 | 0.744 | [0.628, 0.874] | separates |

- **The pre-registered proposal condition holds.** It required the primary to separate and
  every row above to have a point estimate above 0.5.
- **The separation is not carried by scissor jumps or by annotator count alone.** Dropping the
  scissor/crossed-feet cases, or keeping only single-annotator actions, still separates.
- **The width-only row is undetermined at 5 flagged actions from 4 participants.** Its point
  estimate is in line with the others.

**Jumping Jacks cut, read off the labels (reported, not applied).** Production merges firings by
fault, so a user sees one card if any rep of their clip fires. The unit that matters to a user is
therefore the (action, camera) clip.

| cut | correct clips carded | flagged clips carded |
|---|---:|---:|
| 1.3 (spec) | 164 / 319 (51.4%) | 28 / 35 (80.0%) |
| 1.321 (label-derived) | 173 / 319 (54.2%) | 28 / 35 (80.0%) |

- The Youden cut fitted on all data is 1.321. Youden's J is sensitivity + specificity − 1.
- Fitted without each participant and applied to that participant, pooled over 30 folds, the cut
  scores sensitivity 0.667 (8 of 12) and specificity 0.633 (69 of 109). No fold lacked a cut.
- The spec's own 1.3, applied to the action score, fires on 9 of 12 flagged actions and 36 of 109
  unflagged ones.
- Per rep, 1.3 fires on 71.3% of 101 reps in flagged actions and 37.1% of 941 reps in
  unflagged ones.

**High Knee cuts (no cut proposed).** The plan fixes that a comment label, which judges a whole
action, cannot license a threshold. Per gated camera pair:

| actions | pairs | median % of reps the cited 45° cut fires on | median % of reps the implemented 90° cut fires on |
|---|---:|---:|---:|
| unflagged | 90 | 0.0% | 97.6% |
| flagged | 22 | 0.0% | 100.0% |

- The cited cut's per-rep fire rate, used as an action score, gives AUC 0.646 [0.493, 0.798],
  which is undetermined.
- The secondary label set, all 68 actions including the 6 disclosed ones, gives AUC 0.708
  [0.568, 0.833] from 15 flagged actions: separates.

**Jumping Jacks per camera (exploratory, not pre-registered; run after the proposal).**
Production sees one camera, and the primary pooled three.

| camera | actions | flagged | AUC | 95% CI | verdict | correct clips carded at 1.3 |
|---|---:|---:|---:|---|---|---:|
| `exo_l` (side) | 120 | 12 | 0.803 | [0.678, 0.936] | separates | 59 / 108 |
| `exo_m` (frontal, the view the spec names) | 121 | 12 | 0.654 | [0.441, 0.853] | undetermined | 56 / 109 |
| `exo_r` (side) | 113 | 11 | 0.647 | [0.416, 0.836] | undetermined | 49 / 102 |

All three point estimates are above 0.5 and every interval is wide. `exo_l` and `exo_r` are
the two mirror-image side cameras, so their gap at 11–12 flagged actions is not evidence of a view
effect. No single camera, the frontal one included, is established on its own. The rule ships
with no view gate because no camera-specific effect is established.

**High Knee annotator count (exploratory, not pre-registered).** The comment label flags an
action if **any** annotator complains, so multi-annotated actions get more chances to be
flagged. That is the opposite direction to Jumping Jacks' strict majority. Flagged rates are 9 of
39 multi-annotated and 3 of 20 single-annotated held-out actions.

| stratum | actions | flagged | AUC | 95% CI | verdict |
|---|---:|---:|---:|---|---|
| multi-annotator | 39 | 9 | 0.700 | [0.471, 0.868] | undetermined |
| single-annotator | 20 | 3 | 0.824 | [0.574, 0.944] | separates (242 of 2,000 resamples had no flagged action and were skipped) |

Both point estimates are above 0.5, so neither stratum alone carries the pooled result.

**High Knee gate sensitivity (exploratory, not pre-registered).** This drops the 17 side-camera
pairs whose gate value lies inside the frontal camera's range. It leaves 57 actions with 12
flagged, AUC 0.754 [0.620, 0.869]: separates.

**A figure the Jumping Jacks harness prints on every run:** the withdrawn
`jj_knee_valgus_landing` confound, re-measured at full n.
- Of the 0.820 of open-phase frames below the 0.82 knee/ankle cut, 0.785 also fall below it with
  the knees replaced by straight-limb positions. That leaves 0.035 needing inward deviation.
- It was not analysed further. Withdrawn rules were out of scope because each still carries a
  citation failure.

## What changes from the small-n runs

| figure | small-n run | full archive |
|---|---|---|
| Jumping Jacks: spec 1.3 cut, per-rep fire rate in actions judged correct | 79.1% (11 actions) | 37.1% (109 actions, 941 reps) |
| Jumping Jacks: median widest stance per rep, correct actions | 1.163 | 1.392 |
| High Knee: cited 45° cut ranks the comment labels | backwards on 6 actions | AUC 0.646 [0.493, 0.798], undetermined, point estimate in the right direction |
| High Knee: implemented 90° cut | 100% of reps | 97.6% (unflagged) / 100% (flagged) median per pair |

Rerun on the same 11 actions with this run's pose files, the old Jumping Jacks figure comes back
at 77.3% (Gates). The drop to 37.1% is therefore which actions were reachable, not a pipeline
change. The archive prefix held a narrower-than-typical set of performances. The rule still fires
on over a third of correct reps, and on 51.4% of correct clips at the card level.

## Deviations from the plan

| item | plan | what happened | what it changes |
|---|---|---|---|
| where the plan lives | this skill asks for `notes/<stem>_validation_plan.md`, committed before results | written to `docs/superpowers/specs/`; not committed because approval to commit was pending. SHA-256 of the plan and all 7 analysis files recorded at 08:23:33, before decoding began, and verified unchanged afterwards | the pre-registration rests on a local hash file, not a commit timestamp |
| High Knee camera gate | if side and frontal ranges overlap, report it; do not re-gate | reported as a failed gate. An exploratory re-run without the 17 ambiguous pairs was added | none to the primary; the exploratory row is labelled as such |
| pose extraction workers | `run_pose_on_frame_dirs.py`, 7 shards | 4 extra workers called the same `process_directory` over Jumping Jacks shards in reverse order, skipping existing outputs | none; same function, same settings, output frame count verified |
| withdrawn-rule figures | out of scope | the Jumping Jacks harness prints the valgus confound unprompted; one line reported | none |
| small-n reproduction check | not in the plan | added after the results, to ground "What changes from the small-n runs" | a gate row; grounds a claim, changes no result |
| High Knee annotator-count check and clip-level card rates | not in the plan | added after the results, labelled exploratory | qualifies the reading; changes no pre-registered verdict |
| Jumping Jacks per-camera AUCs | not in the plan | added after the proposal was accepted, labelled exploratory | no single camera is established on its own; not known when the user decided; recorded in the rule's docstring as found afterwards |
| High Knee threshold | the plan fixed that a comment label cannot license a number | the user overrode it after seeing the results; live at the label-derived 65.6° with a 0.170 view gate | the High Knee cut rests on an exploratory, post-hoc analysis, not a pre-registered one |
| live rule's scope | the silent docstring recorded an `open`-phase scope | the live rule reads the whole rep | none to the numbers: this is the quantity the analysis measured. `open` is set by the clip's 70th percentile, so a narrow rep can hold no `open` frame |

## Not supported

- **That Jumping Jacks is ready to register.** The rule is live but the movement is not
  analyzable in the app. The measured cost is a card on 51.4% of correct clips at 1.3 and 63.3%
  held-out specificity at the action level, and no single camera's separation is established.
- **That 65.6° is a validated High Knee threshold.** It is read off 12 comment-derived positives, after the results. Its 40.0% false-card rate is in-sample, which biases it down, and "no complaint" is not a judgement that the lift was high enough, which biases it up. Neither bound holds; leave-one-participant-out specificity 0.660 is the out-of-sample figure.
- **Validation of Jumping Jacks' width fault specifically.** Only 5 flagged actions have a
  comment compatible with insufficient width, and that subset is undetermined.
- **That 1.321 or 1.3 is "the" threshold.** It is the ranking optimum on 12 positives from 8
  people, with 456×256 preprocessed frames. That resolution gives about 2.8× production's landmark
  error in normalised units.
- **That rep-level coaching would be right.** Both labels judge whole actions. An action-level
  AUC says nothing about which reps a per-rep rule should flag.
- **Registering either movement.** Each would ship one live rule at best. The decision is the
  user's.
- **Anything about the withdrawn rules.** They are blocked by citations, not data.

## Reproduce

Code:
- `src/egoexo/frame_extraction.py`
- `scripts/egoexo/extract_action_frames.py`
- `scripts/egoexo/run_pose_on_frame_dirs.py`
- `src/egoexo/{jumping_jacks,high_knee}_validation.py` and their runners
- `src/egoexo/separability.py`
- `scripts/egoexo/run_silent_rule_separability.py`

Tests: `tests/test_egoexo_separability.py`, `tests/test_frame_extraction.py`.

```
set OUT=data/EgoExo-Fitness/derived/separability_2026-09-26
.venv\Scripts\python.exe scripts/egoexo/extract_action_frames.py --movement "Jumping Jacks" ^
  --movement "High Knee" --out %OUT%/frames --report %OUT%/extraction_report.json --require-complete
.venv\Scripts\python.exe scripts/egoexo/run_pose_on_frame_dirs.py --frames-root %OUT%/frames/jumping_jacks ^
  --out %OUT%/pose/jumping_jacks --shard 0 --shards 4      (repeat for shards 1-3)
.venv\Scripts\python.exe scripts/egoexo/run_pose_on_frame_dirs.py --frames-root %OUT%/frames/high_knee ^
  --out %OUT%/pose/high_knee --shard 0 --shards 3          (repeat for shards 1-2)
.venv\Scripts\python.exe scripts/egoexo/run_jumping_jacks_validation.py --pose-dir %OUT%/pose/jumping_jacks --json %OUT%/jj_validation.json
.venv\Scripts\python.exe scripts/egoexo/run_high_knee_validation.py --pose-dir %OUT%/pose/high_knee --json %OUT%/hk_validation.json
.venv\Scripts\python.exe scripts/egoexo/run_silent_rule_separability.py --jj-json %OUT%/jj_validation.json ^
  --hk-json %OUT%/hk_validation.json --out %OUT%/separability.json
```

Runtime on this machine (CPU only):
- extraction: ~10 min for the 66 GB stream;
- pose: ~2 h 10 min for 171,659 frames, about 3.5 frames/s per worker;
- harnesses and analysis: under a minute each.

Artifacts under `data/EgoExo-Fitness/derived/separability_2026-09-26/`:
- `extraction_report.json`, `integrity.txt`, `prereg_hashes_before_decoding.txt`;
- `pose/`;
- `jj_validation.json`, `hk_validation.json`, `separability.json`.
