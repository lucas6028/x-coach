# Fit3D bar geometry — can body keypoints see the barbell?

**Question.** Blind spot B of the "what do keypoints structurally miss" thread. A keypoint
skeleton encodes the *body* and nothing else, so on paper the implement is invisible to it.
Unlike blind spot A (segment axial rotation), this one is not abstract in this codebase:
`src/pose/movements/overhead_press.py:595` **withdrew** a bar-path sub-criterion, in part
because referencing the bar to the shoulders conflates it with trunk lean and in part because
no bar reference exists at all. This experiment asks whether that withdrawal was forced by a
representation limit or merely by missing effort.

## B is two claims, and only one of them is testable here

| claim | status |
|---|---|
| **Geometry** — *where* the implement is | **Measured below, for bar-in-hands lifts only.** |
| **Magnitude** — *how heavy* it is | **No ground truth exists on this machine.** |

Fit3D uses one unloaded bar for every subject and every rep; Fitness-AQA labels faults, not
loads; REHAB24-6 has no implement at all. So the Newtonian argument — identical kinematics
plus unknown external load leaves joint moments unidentifiable — remains an *argument*, not a
result. Closing it needs a force-plate-plus-video dataset, which is a data-acquisition task,
not an analysis task. **Do not report the geometry result as if it settled B.**

## Ground truth without annotation

Fit3D ships a calibrated 4-camera rig, so the bar can be *measured* rather than hand-labelled:

1. colour-threshold the (yellow) bar inside an ROI anchored on the upper body;
2. fit a 2D line per view with RANSAC — scored by `inliers × contiguous_span`;
3. back-project each 2D line to the plane through its camera centre;
4. intersect the planes across views ⇒ the 3D bar **axis**.

Planes, not points, because a barbell is a featureless rod with nothing to match between
views. Keypoints place the search ROI and nothing else; no keypoint enters the measurement.

Extraction is **bimodal**, and the honest figures are the pre-gate ones: over all 508 deadlift
frames the 4-view plane residual has median **1.98 mm** but mean **28.7 mm** and p90 **84.2 mm**.
Most frames are measured to ~2 mm; a minority fail outright. A 5 mm gate keeps 79%. The
post-gate residual (1.7 mm) is *not* an independent quality claim — the gate selected on it.

The distortion model was pinned empirically, not assumed: round-tripping GT joints through
`project_world_to_image` and `cv2.undistortPoints` reproduces the undistorted projection to
**0.068 px** with `[k1, k2, p1, p2, k3]`, versus 0.323 px with the tangential pair swapped.

## RESULT — for bar-in-hands lifts there is no blind spot to refute

The comparison that matters is not model-vs-threshold but **model vs a zero-parameter
baseline**: put the bar at the mid-hand point plus a constant offset learned on the training
folds. No regression, no features. Deadlift, 8 subjects, 401 frames, LOSO, anterior offset:

| predictor | MAE cm (anterior) | MAE cm (vertical) |
|---|---|---|
| **constant offset from mid-hand (zero parameters)** | **4.07** | **1.35** |
| constant offset from mid-wrist | 4.51 | — |
| ridge on `wrists` (6 dims) | 4.23 | 3.37 |
| ridge on `hands` (18 dims) | **3.00** | 1.94 |
| ridge on `arms` (30 dims) | 3.62 | 2.58 |
| ridge on `h36m17` (51 dims) | 3.28 | 3.45 |
| ridge on `full25` (75 dims) | 3.36 | 3.37 |

*(Models are chosen by best `R²_within`, then their MAE reported. Choosing on MAE instead
gives `hands` 2.58 cm on the anterior axis; the conclusion is unchanged either way.)*

**The bar is mechanically co-located with a tracked joint.** In a deadlift or a row the hands
*grip* the bar, so its axis passes through them by physical necessity. A constant offset
localises it to 4.07 cm anterior and 1.35 cm vertical — comfortably inside the
0.30-shoulder-width (8.4 cm) deviation the withdrawn OHP rule would have had to call a fault.
This is not "keypoints predict the implement"; it is "the implement is effectively part of the
tracked skeleton". There is no blind spot here to refute, and **nothing in this generalises to
a lift where the bar is not in the hands.**

The sharpest form of this: **on the vertical axis no learned model beats the zero-parameter
baseline at all** (1.35 cm vs 1.94 cm for the best of them). Fifty-one body keypoints and a
tuned regulariser are *worse* than "the bar is a fixed distance from the hands". On the
anterior axis learning does add something real but modest — `hands` cuts 4.07 → 3.00 cm
(−26%) — and the gain comes from the four hand-extremity points, i.e. grip width and forearm
orientation, the two things that actually modulate the hand-to-bar offset. Whole-body sets are
*worse* than `hands` on both axes, the same "extra dimensions cost more than they buy" pattern
as Exp 0f.

**Replication, and it is starker.** `barbell_row` (7 subjects, 293 frames) reproduces the
pattern, with the baseline winning even more clearly. Across both actions and both axes:

| action | axis | zero-parameter baseline | best learned model | verdict |
|---|---|---|---|---|
| deadlift | anterior | 4.07 | 3.00 (`hands`) | model −26% |
| deadlift | vertical | **1.35** | 1.94 (`hands`) | **baseline wins** |
| barbell_row | anterior | 2.80 | 2.60 (`hands`) | tie (−7%) |
| barbell_row | vertical | **1.84** | 2.37 (`hands`) | **baseline wins** |

**In three of four cells a zero-parameter constant offset ties or beats every tuned model over
every keypoint set.** Both are bar-in-hands lifts, so this strengthens the claim *within that
class* and not one step beyond it.

**Consequence for the product.** "We have no bar reference" is no longer true for bar-in-hands
movements — a constant offset from the hands gives one to ~3-4 cm with no model at all, and
fitting a model on top of it is not worth the complexity. That
does not revive the withdrawn OHP criterion as written: objection 1 in `overhead_press.py`
(referencing bar path to the *shoulder* conflates it with trunk lean) was about the choice of
reference and is untouched here.

## What this does NOT establish

- **The quality gate is strongly phase-biased.** Dropped frames cluster at lockout: on
  deadlift the mean normalised bar height is 0.73 for dropped vs 0.36 for kept, and in the top
  height quintile 61 frames are dropped against 8 kept — lockout is under-represented by
  roughly 8:1. `barbell_row` is milder but the same direction (0.79 vs 0.45). The surviving
  frames are the phase where the bar is clear of the body, plausibly also where it is easiest
  to localise. Retention ranges 57% (s10) to 87% (s03) on deadlift, 73% overall on rows.
- **λ was selected on the same LOSO folds it is scored on**, so the regression numbers are an
  *upper bound*. The zero-parameter baseline has no such problem, which is another reason to
  lead with it.
- **Bar-in-hands lifts only.** The back squat — where the bar rests on the traps, is not
  gripped for load, and where high-bar/low-bar placement could genuinely be unobservable — is
  exactly the case the extractor cannot measure (below). The interesting cell is missing.
- **Empty bar, mocap lab, n = 8 subjects.** Fit3D's bar is unloaded and identical for
  everyone: nothing here speaks to grip width under real load, plates occluding the ends, or
  gym lighting.
- **Axis only, not endpoints**, so "is the bar level / is one side high" is untested.

## The squat failure, and why the quality metric could not catch it

`squat` (a back squat) does **not** extract reliably and is excluded by default in the CLI.
The physical requirement is sharp: the bar rests on the trapezius, so the shoulder→bar-axis
distance must be ~10–15 cm and near-constant. Measured, after the improved extractor *and* the
5 mm quality gate:

| subject | kept | plane residual | shoulder→bar |
|---|---|---|---|
| s03 | 33/70 | 2.1 mm | 32.1 ± 23.1 cm |
| s04 | 33/64 | 2.3 mm | 37.0 ± 22.5 cm |
| s05 | 32/64 | 2.0 mm | 35.5 ± 23.8 cm |

With the arms up at the bar the ROI grows tall enough to admit background fixtures, and RANSAC
locks onto those. **The plane residual is small anyway** — 2.0–2.3 mm — because the wrong
structure is straight and all four cameras see the same wrong structure. On the pre-fix
extractor the residual was actively *anti*-correlated with the error (−0.55).

`clean_and_press` fails too, but **loudly**: the 5 mm gate rejects essentially every frame
(0–2 survivors of 51–91 per subject, all 8 subjects skipped). Same root cause — arms overhead
means a tall ROI full of background — but there the wrong structures differ between cameras,
so the residual blows up and the gate does its job.

That contrast is the transferable lesson: **a self-consistency metric cannot detect a
*consistent* mistake.** It caught `clean_and_press`, where the four views disagreed, and was
fooled by `squat`, where they agreed on the wrong thing. Multi-view agreement validates
precision, never correctness. The only thing that caught the squat failure was an external
physical constraint — a bar resting on the traps cannot be 35 cm from the shoulders.

## Three defects found en route

1. **A wrong headline, caught by a zero-parameter control.** The first draft of this note read
   "the geometry half of B is refuted", comparing a 3.00 cm residual against the 8.4 cm
   coaching threshold. That comparison is beatable by the identity function: the bar passes
   through the hands. Always ask what a no-feature baseline scores before claiming a
   representation predicts something it is physically attached to.
2. **`inliers × span` is gameable, and a unit test caught it before the data did.** A dense
   compact blob plus two distant stragglers spans far while covering nothing between them. The
   fix — score the longest *contiguous* run — was written for a synthetic test and turned out
   to be the same defect corrupting the real frames.
3. **The contiguity fix immediately over-corrected.** At `max_gap = 15 px` the median deadlift
   plane residual was 2.9 mm; the bar is routinely occluded mid-span by the torso, so a strict
   gap forces the fit onto one short fragment and the views stop agreeing. At 150 px it is
   1.2 mm. Contiguity must reject *disconnected* structures without demanding an unbroken one.

Endpoints were abandoned for the same occlusion reason: carving the mask along the axis gave
bar lengths of **0.79 ± 0.39 m** for a physically constant bar. Every target here is therefore
endpoint-free — a property of the infinite line, not of the segment.

## Repro

```
python scripts/fit3d/run_bar_observability.py --action deadlift \
    --json data/Fit3D/derived/bar_observability_deadlift.json
python scripts/fit3d/run_bar_observability.py --action barbell_row
```

Tracks cache under `data/Fit3D/derived/bar_tracks/`; `--refresh` recomputes. `--action squat`
is refused by default and `--force-unreliable` overrides it, but those numbers must not be
reported. Code `src/fit3d/bar_geometry.py`, tests `tests/test_fit3d_bar_geometry.py`.
OpenCV is imported lazily throughout — it is absent from `requirements-ci.txt`, and CI skips
the opencv suites, so a module-level import would break CI exactly as scipy nearly did in
Exp 0.
