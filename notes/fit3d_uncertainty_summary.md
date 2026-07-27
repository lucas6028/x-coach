# Fit3D uncertainty — does a pose estimator's confidence add anything to its keypoints?

**Question.** Blind spot C. A keypoint skeleton is a **point estimate**: it records where each
joint is and has nowhere to record how much that should be believed. Two readings with
identical coordinates but different reliability warrant different coaching decisions, so on
paper no accuracy improvement substitutes for a confidence channel — a representational gap,
like A (axial rotation) and B (the implement).

The project already has the channel and has never used it. NLF writes `unc` (F, 24), per-joint
uncertainty in **millimetres**, into every prediction npz for both Fit3D and REHAB24-6. Its
only appearance anywhere in `src/` is one docstring line in `nlf_skeleton_features.py`.

Fit3D has mocap ground truth and a **synchronised 4-camera rig**, so it can settle whether the
channel is meaningful and whether it is redundant, before any downstream work is spent on it.

## The trap that had to be cleared first

Raw per-joint MPJPE says the *worst* joints are thorax (200 mm), neck (147), head (124), hips
(111–120) — anatomically absurd, since those are the best-localised parts of the body. They are
the worst **matched**: `depth_eval.py` already flags spine/thorax as approximate SMPL→H36M
mappings, and the measured constant offsets are large.

| joint | thorax | l_hip | head | r_hip | r_ankle | l_wrist |
|---|---|---|---|---|---|---|
| convention offset (mm) | **176** | 95 | 95 | 76 | 61 | 38 |

Against that, `unc` — which correctly calls wrists and elbows hardest — looks *anti*-correlated:
Spearman **−0.15** between per-joint median error and per-joint median uncertainty. Within
joints, the same data gives **+0.44**.

So: **every comparison here is within-joint**, and error is bias-corrected by removing the
per-joint constant offset (estimated on training folds only inside LOSO). Comparing uncertainty
across joints on this data measures joint-definition mismatch, not localisation error.

A second, smaller trap: `unc` is SMPL-24 indexed and `depth_eval.resolve_lr` chooses one of two
SMPL-24→H36M-17 index lists by minimising MPJPE. Indexing `unc` with the other one pairs each
joint's uncertainty with a *different* joint's error — which produces a near-null correlation
that reads exactly like "uncertainty is uncalibrated".

## 1. The channel is real and well calibrated

Squat, 8 subjects × 4 cameras, 43 304 frame-joint pairs. Within-joint Spearman(unc, error):

| | rho | | rho |
|---|---|---|---|
| l_knee | **+0.722** | r_wrist | +0.292 |
| r_knee | +0.629 | l_wrist | +0.281 |
| spine | +0.565 | l_shoulder | +0.269 |
| l_ankle | +0.549 | l_elbow | +0.258 |
| r_ankle | +0.529 | neck | +0.249 |
| r_hip | +0.470 | head | +0.179 |

**Mean +0.403, and all 16 non-degenerate joints are positive.** (The pelvis is the root, so its
root-relative error is identically zero and it reports `nan`.) This is a genuine uncertainty
estimate, not a decoration.

## 2–3. RESULT — and it is redundant

The comparison that decides it is not "does `unc` beat nothing" but **"does `unc` beat the
pose itself"**. A self-occluded or extreme configuration is recognisable from coordinates
alone, so the keypoints may already carry their own difficulty signal.

LOSO over 8 subjects, MAE in mm on bias-corrected error, fitted per joint:

| predictor | features | squat | deadlift |
|---|---|---|---|
| per-joint constant lookup | **0** | 22.36 | 22.82 |
| `unc` alone | 1 | 19.26 (−3.10) | 18.73 (−4.09) |
| predicted pose alone | 51 | **14.16** (−8.20) | **13.69** (−9.13) |
| pose + `unc` | 52 | 14.15 (**−0.01**) | 13.52 (**−0.17**) |

`unc` genuinely carries information — it beats a zero-feature lookup by 3–4 mm. But the
**predicted pose predicts its own error far better** (14.2 vs 19.3 mm), and **adding `unc` on
top of the pose buys 0.01–0.17 mm**. Everything the confidence channel knows, the coordinates
already encode.

**Blind spot C is refuted, the same way A and B were.** The naive representational claim —
"keypoints structurally cannot express X, so an explicit X channel must add information" — has
now failed three times for three different X, and for the same reason each time: the
representation implicitly encodes X even though it cannot state it.

## 4. Even the routing use case does not survive

This was the strongest remaining cell. The project's headline finding is that reliability
depends on **fault type × viewpoint**, and that routing is currently a hand-written
`view_type` gate. Fit3D's cameras are synchronised, so for a fixed (subject, frame, joint) the
motion is *identical* across the four views and any error difference is purely view-induced —
a genuinely controlled contrast.

`unc` does know which view is unusually bad: standardised within (camera, joint), agreement
with true error is **rho +0.405** on squat — consistent across all 8 subjects (+0.31 to +0.49)
— and **+0.462** on deadlift. But against a pose-based error predictor:

| | squat | deadlift |
|---|---|---|
| rank agreement — `unc` | +0.522 | +0.581 |
| rank agreement — pose | **+0.558** | **+0.585** |
| top-1 pick best view — `unc` | **22.9%** | **23.8%** |
| top-1 pick best view — pose | 31.5% | 30.3% |
| chance | 25.0% | 25.0% |

The pose matches or beats `unc` on rank agreement, and **`unc` picks the single best view
*below chance*** (22.9% vs 25%). A positive rank correlation over four views is compatible with
an unreliable argmin — the ordering is right on average while the top of it is noise — but for
the actual product decision ("which camera should I trust right now") that distinction is the
whole point. **`unc` cannot be used to route views.**

## What this does NOT establish

- **NLF only.** `unc` is one model's uncertainty head. A different estimator, or an ensemble /
  test-time-augmentation variance, could be less redundant with its own point estimate.
- **λ was chosen on the held-out fold** for `unc`, `pose` and `pose+unc` alike, while the
  zero-feature lookup got no such help. That is generous to every learned model, so
  "lookup is close" and "unc adds nothing over pose" are both conservative readings.
- **The pose model may be capturing systematic, pose-dependent bias**, not variance. Bias
  correction removes only a per-joint *constant*. That is a fair advantage for the practical
  question ("what predicts error?") but it means the pose baseline is not purely a
  reliability estimator.
- **Downstream not tested.** `unc` also exists in the REHAB24-6 npz where fault labels live, so
  "does it change verdicts" is answerable. It is now much less worth answering: a channel that
  adds 0.01 mm over the pose at the *measurement* level is unlikely to move decisions.
- **n = 8 subjects**, two actions, mocap-lab conditions.

## Repro

```
python scripts/fit3d/run_uncertainty_eval.py --action squat \
    --json data/Fit3D/derived/uncertainty_eval_squat.json
python scripts/fit3d/run_uncertainty_eval.py --action deadlift
```

Code `src/fit3d/uncertainty_eval.py`, tests `tests/test_fit3d_uncertainty_eval.py`. The
rank-correlation helper is numpy-only with proper tie averaging and is cross-checked against
`scipy.stats.spearmanr` in a skipped-if-absent test — `argsort(argsort(x))` is not a rank
function under ties, and on the all-zero root joint it fabricated a rho of +0.229 before this
was fixed.
