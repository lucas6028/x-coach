# Lunge view reconnaissance (REHAB24-6 Ex5)

Measurement task, Phase 0 of the Lunge detector plan
(`.superpowers/sdd/2026-07-30-lunge-detector/task-1-brief.md`). Answers, before writing
any detector code, whether `lunge_knee_past_toes` (hard-gated on `view_type == "side"` +
`view_confidence >= 0.20`) can ever fire on REHAB24-6's Ex5 (leg lunge) clips. This note
records a measurement, not a recommendation; no threshold or gate was changed in response
to what it found.

## Step 1: pose extraction

```
.venv\Scripts\python.exe scripts/pose/run_pose_extraction.py --dataset unlabeled --video-dir data/REHAB24-6/Ex5 --output-dir data/REHAB24-6/processed/lunge_pose_json --no-video
```

Produced 18 JSON files under `data/REHAB24-6/processed/lunge_pose_json/` (9 video ids ×
cam17/cam18), matching the 18 mp4s in `data/REHAB24-6/Ex5`. Gitignored, not committed.

## Step 2: cam18 transpose verdict

```
PM_021-Camera17-30fps:              n=2846  mean shoulder->ankle dx=-0.004 dy=+0.497
PM_021-Camera18-30fps-transposed:   n=2846  mean shoulder->ankle dx=-0.052 dy=+0.411
```

Both files: `|dy| >> |dx|`, `dy` positive (ankles below shoulders, y grows downward) — an
upright person in both. The cam18 rotation is baked into the pixels; the STOP condition
does not trigger. Sagittal results below are trustworthy on this axis.

## A finding in its own right: every Ex5 video mixes camera orientations

Before running view estimation, `Segmentation.csv`'s `cam17_orientation` column was
checked per video_id, over all 174 Ex5 reps:

```
PM_021: {'front': 10, 'half-profile': 10}
PM_028: {'front': 11, 'half-profile': 10}
PM_037: {'front': 10, 'half-profile': 10}
PM_042: {'front': 13, 'half-profile': 12}
PM_104: {'front': 10, 'half-profile': 10}
PM_112: {'front': 12, 'half-profile': 13}
PM_117a: {'front': 9}
PM_117b: {'front': 3, 'half-profile': 11}
PM_125: {'half-profile': 10, 'front': 10}
```

Every video except PM_117a mixes `front` and `half-profile` reps roughly 50/50 within the
same recording — the subject reorients partway through. This makes a **per-file**
`estimate_view_for_pose(Path)` call invalid for this dataset: it aggregates frames across
both orientations into one verdict, which is wrong for about half the reps in every mixed
video, and that verdict cannot be meaningfully compared against a per-rep label at all. A
preliminary per-file run was done before this was discovered and is **not used** below; it
is recorded only as a discarded artifact: one `side`-labeled clip read `side` at confidence
0.83, and four `half-profile`-labeled clips read `side` at 0.66–0.70. Those numbers are
superseded entirely by the per-rep-window measurement that follows.

## Step 3 (corrected): per-rep-window view estimation

Estimated view **per rep window** instead of per file, slicing each pose JSON's frames by
`first_frame:last_frame+1` (`Segmentation.csv`, Ex5 rows only) and comparing against
`src.rehab24.dataset.camera_orientation(segment, camera)` for the ground truth (the
existing cam17→cam18 mapping in `src/rehab24/dataset.py`, not restated here). The
aggregation over a rep window mirrors `estimate_view_for_pose`'s aggregation
(`src/pose/view_estimation.py:390-414`) exactly — same `frame_view_signals` /
`mean_finite` / `score_view` calls, `allow_front=False` (the production default), and the
same NaN-not-zero default for `torso_width_ratio`. Script (throwaway, not committed to
`src/` or `scripts/`; Task 8 builds the tested version):
`C:\Users\ttsh1\AppData\Local\Temp\claude\C--Users-ttsh1-code-x-coach\f2efde71-ac9d-409b-a5d3-df9c31ed5258\scratchpad\per_rep_view.py`.

Full output (174 reps × 2 cameras = 348 rows tabulated, matches expectation):

```
=== Per-rep view estimation table (label -> estimated), all 174 reps x 2 cameras ===
label=front          -> estimated=rear_oblique   n= 88 conf_range=[0.50, 0.50]
label=half-profile   -> estimated=rear_oblique   n= 88 conf_range=[0.50, 0.67]
label=half-profile   -> estimated=side           n= 84 conf_range=[0.66, 0.76]
label=side           -> estimated=side           n= 88 conf_range=[0.69, 0.99]

Total rows tabulated: 348 (expect 348 = 174 reps x 2 cameras)
```

Per-row confidence lists (as printed, `round(confidence, 2)` per rep):

- `label=front -> estimated=rear_oblique` (n=88): all exactly 0.50.
- `label=half-profile -> estimated=rear_oblique` (n=88): 0.50–0.67, e.g.
  `[0.61, 0.61, 0.62, 0.62, 0.61, 0.58, 0.59, ...]`.
- `label=half-profile -> estimated=side` (n=84): 0.66–0.76, e.g.
  `[0.69, 0.68, 0.70, 0.70, 0.70, 0.69, 0.69, ...]`.
- `label=side -> estimated=side` (n=88): 0.69–0.99, e.g.
  `[0.95, 0.97, 0.99, 0.98, 0.95, 0.79, 0.80, ...]`.

`label=front` (n=88) is the cam17 orientation column read directly; `label=side` (n=88)
is the same 88 reps read through the cam18 mapping (`front → side`, per
`CAMERA18_ORIENTATION`). `label=half-profile` appears under both cameras (half-profile
maps to itself), 86 reps × 2 cameras = 172, split 88/84 across `rear_oblique`/`side`
above.

## Step 3b: the gate answer

**Question:** on reps whose cam18 orientation the dataset calls `side`, does
`estimate_view_for_pose`'s per-rep aggregation return `side` with
`view_confidence >= 0.20`?

```
=== GATE ANSWER ===
Reps whose cam18 orientation the dataset calls 'side': 88
Of those, estimated as 'side' with view_confidence >= 0.20: 88
Proportion: 88/88 = 1.000
```

**All 88 of 88 (100%)** cam18 `side`-labeled reps are estimated `side`, with confidence
ranging 0.69–0.99 — well clear of the 0.20 floor.

## The consequence, stated explicitly

The side gate opens: on this dataset's cam18 stream, `lunge_knee_past_toes`'s
`view_type == "side"` + `view_confidence >= 0.20` gate passes on all 88 reps the dataset
labels `side` (100%), at confidences far above the floor (0.69–0.99). `lunge_knee_past_toes`
is **not** permanently silent on REHAB24-6 the way `pushup_elbow_flare` is — it is
validatable directly on production-path input (`estimate_view_for_pose` with its default
`allow_front=False`), not only in the oracle pass, for the population of reps whose
orientation the dataset calls `side`. This is a materially better outcome than the earlier,
now-superseded per-file measurement suggested, and it reverses the concern that motivated
running this task first (that the estimator might emit `side` on this dataset zero or
near-zero times, as it did across the 45 pre-existing squat pose JSONs). No threshold or
gate was changed to produce this result; the production defaults (`allow_front=False`,
`SIDE_THRESHOLD`, `view_confidence >= 0.20`) were used unmodified.

## Files changed

- Created: `notes/lunge-view-reconnaissance.md` (this file).
- Not committed (gitignored, per `data/` policy): 18 pose JSON files under
  `data/REHAB24-6/processed/lunge_pose_json/`.
- Not committed (scratchpad, throwaway): the per-rep-window analysis script under the
  session scratchpad directory (path above).
