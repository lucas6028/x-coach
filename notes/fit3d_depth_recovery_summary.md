# Fit3D experiment 1 — does monocular direct image->3D recover depth?

**Question.** The depth-bottleneck thread concluded that 2D->3D *lifting* cannot recover
out-of-plane depth (`lift_2d_to_3d*`), and a REHAB24-6 probe hinted NLF direct image->3D
might — but only indirectly (LOSO accuracy, n=9, p=0.20; see
`notes/rehab24_correctness_experiment_summary.md`). Fit3D's mocap ground truth lets us
measure depth recovery **directly**, per axis, on squats.

**Method.** Ran NLF (Neural Localizer Fields, `detect_smpl_batched`, half-res) on all 32
Fit3D **train squat** videos via Kaggle P100 (kernel `haoping6028/fit3d-nlf-extract`,
100% detection, ~42 min). SMPL-24 output is mapped to the Human3.6M-17 core; the L/R
convention is **resolved against the GT** (`src/fit3d/depth_eval.py::resolve_lr`), not
assumed. Errors are root-relative vs the camera-frame GT, decomposed per axis. Squat-cue
errors are placed beside the single-view **2D-projection baseline** from experiment 2
(both are "readings" of the same true 3D cue; per-frame mean-abs, matched reduction).
32 sequences, ~43k frames.

## Results

Position error vs mocap GT (mm, root-relative):

| readout | MPJPE | PA-MPJPE | err_x | err_y | **err_z (depth)** | ez/exy |
|---|---|---|---|---|---|---|
| NLF parametric (`smpl3d`) | 77.7 | 65.4 | 26.5 | 46.7 | **42.4** | **1.16** |
| NLF nonparametric (`smpl3d_np`) | 77.5 | 65.9 | 27.4 | 46.1 | **42.0** | 1.14 |

Squat-cue error vs 3D truth — NLF monocular-3D vs single-view 2D projection (per-frame):

| cue | 2D-view | NLF param | NLF nonpar | recovered? |
|---|---|---|---|---|
| knee_angle | 18.4° | **7.1°** | 6.8° | yes (~2.6x better) |
| hip_angle | 18.3° | **10.4°** | 11.8° | yes |
| torso_lean | 12.5° | **5.4°** | 4.2° | yes (~2.3x better) |
| depth_ratio | 0.06 | 0.09 | 0.09 | no (already view-stable per-frame) |

L/R resolution: all 32 sequences -> swap=True (Fit3D index-1 is anatomically **left**);
the correct orientation is **4.1x** better (77.7 vs 317.2 mm) — decisive, unanimous.

## Findings

1. **Depth is no longer the failure axis.** NLF's per-axis depth error (42 mm) is on par
   with its in-plane axes (x 27, y 47; ez/exy = 1.16) — actually *better* than the vertical
   image axis. For 2D->3D lifting the depth axis error is a multiple of in-plane; direct
   image->3D essentially removes that gap. This is the mocap-GT, well-powered (32 seqs /
   ~43k frames) confirmation of the earlier underpowered REHAB24 hint.

2. **NLF restores the cues single-view 2D destroys.** Monocular 3D roughly halves the
   per-frame error on the sagittal angle cues (knee, hip, torso lean) that experiment 2
   showed are view-corrupted under projection. So the depth bottleneck quantified in
   experiment 2 is *recoverable* — via direct image->3D, not via better 2D.

3. Parametric and nonparametric NLF readouts are equivalent here.

## Caveats

- **Absolute scale**: NLF underestimates camera distance ~18% (pelvis 3.6 m vs GT 4.4 m) —
  the monocular metric-scale ambiguity. Root-relative analysis sidesteps it; any
  absolute-distance use would need calibration.
- **Per-frame vs per-rep**: experiment 2's headline knee error (42° at the squat *bottom*)
  is the per-rep extreme; the per-frame average is 18°. The comparison here is per-frame
  matched. A per-rep-extreme NLF comparison (does NLF fix the worst-case bottom-of-squat
  reading?) is the natural follow-up. `depth_ratio` "no" is the same story — its
  view-corruption in experiment 2 was the per-rep extreme, not the per-frame mean.
- Squats only; Fit3D's 4-camera rig is all-oblique (no side view, see
  [[fit3d-dataset-facts]]).

## Implications

- The route past the depth bottleneck for x-coach is **direct image->3D perception**, not
  2D-lifting. NLF 3D can feed `pose_rule_detector` the knee/hip/torso readings that a
  single 2D view gets wrong — a concrete upgrade path for the depth-dependent squat faults.
- Next: per-rep-extreme comparison; extend beyond squats (deadlift/lunge) to test depth
  recovery on other sagittal-heavy faults; feed NLF 3D into the correctness classifier.

## Extension — deadlift + thruster (does depth recovery generalise?)

Ran NLF on deadlift and overhead_extension_thruster (64 videos, 100% detection) to test
whether direct image->3D recovers the *different* defining cue of each movement that
experiment 2 flagged as view-corrupted (deadlift = hip/torso, thruster = knee/depth).

| exercise | per-axis depth ez/exy | corrupted cue (exp 2) | 2D-view err | **NLF err** |
|---|---|---|---|---|
| squat | 1.16 | knee_angle | 18.4° | **7.1°** |
| deadlift | 1.17 | hip_angle | 17.8° | **7.9°** |
| deadlift | — | torso_lean | 13.9° | **5.5°** |
| thruster | 0.95 | knee_angle | 17.4° | **5.9°** |
| thruster | — | torso_lean | 6.9° | **4.0°** |

**NLF recovers each movement's defining cue, knee-dominant or hip-dominant.** Per-axis
depth error stays on par with in-plane across all three exercises (ez/exy 0.95–1.17), and
NLF roughly halves the single-view-2D error on the cue that matters most for each — the
deadlift's **hip hinge + torso lean** as cleanly as the squat's knee flexion. So the
"direct image->3D beats 2D" conclusion is not a squat artefact; it holds across lower-body
movement patterns. (depth_ratio is again unchanged — its corruption is the per-rep
extreme, not the per-frame mean.) Per-action JSON: `data/Fit3D/derived/depth_eval_<action>_nlf.json`.

Reproduce: `python scripts/fit3d/run_depth_eval.py --pred-root data/Fit3D/derived/preds/nlf`
(result JSON: `data/Fit3D/derived/depth_eval_squat_nlf.json`; raw NLF npz under
`data/Fit3D/derived/preds/nlf/`).
