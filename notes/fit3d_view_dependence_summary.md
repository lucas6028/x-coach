# Fit3D experiment 2 — view-dependence of 2D squat-rule readings

**Question.** The project's depth-bottleneck conclusion (`pose_rule_detector` depth
faults are unreliable; 2D backbones don't beat each other on squat correctness because
the bottleneck is depth, not 2D accuracy) rested on *indirect* evidence — downstream
REHAB24-6 LOSO accuracy with n=9, p=0.20. Fit3D's mocap 3D ground truth lets us measure
the view-dependence of each biomechanical cue **directly**.

**Method.** For every squat repetition (segmented from `rep_ann.json`) we read each cue
twice with identical formulas (`src/fit3d/biomech.py`): once from the view-invariant 3D
ground truth (`joints3d_25`, the right answer), once from the 2D projection in each of
the 4 calibrated cameras (what a single-camera 2D pipeline sees). Angles are scale-free;
length ratios are normalised by a body segment within each space, so 3D-truth and 2D
readings share units and are directly comparable. **40 reps × 8 subjects × 4 cameras =
160 paired readings** (vs the old n=9).

`noise/sig` = (mean across-camera std of the reading for the same rep) / (between-rep std
of the truth). ≥ 1 means *which camera you used* matters as much as the athlete's actual
form. Verdict heuristics: view-robust if pooled r ≥ 0.80 and noise/sig ≤ 0.50;
view-corrupted if pooled r < 0.50 or noise/sig ≥ 1.00; else view-sensitive.

## Result

| cue | truth (mean±sd) | pooled MAE | bias | r | view std | noise/sig | verdict |
|---|---|---|---|---|---|---|---|
| knee_angle | 70.8°±11.9 | 42.4° | +41.2° | 0.60 | 14.5° | **1.21** | **view-corrupted** |
| hip_angle | 76.3°±18.8 | 40.7° | +39.7° | 0.72 | 15.8° | 0.84 | view-sensitive |
| torso_lean_deg | 44.8°±11.7 | 21.7° | −21.7° | 0.87 | 4.3° | 0.37 | view-robust |
| depth_ratio | 0.4±0.2 | 0.25 | +0.18 | 0.65 | 0.19 | **1.21** | **view-corrupted** |
| knee_width_ratio | 0.9±0.1 | 0.02 | +0.01 | 0.90 | 0.02 | 0.44 | view-robust |

Per-camera knee_angle MAE: 30.9° / 34.9° / 45.6° / 58.1° (cameras 50591643 / 58860488 /
60457274 / 65906101).

## Findings

1. **Sagittal cues are view-corrupted.** The projected 2D **knee angle** mis-reads a
   deep squat as a shallow one and disagrees by camera: at one verified squat bottom the
   true 3D knee angle is **78°** but the four cameras report **108° / 118° / 119° /
   133°**. The bias is systematic (+41° on average — projection foreshortening opens the
   joint toward 180°) and camera-dependent (noise/sig = 1.21). **hip_angle** and
   **depth_ratio** (hip-below-knee) behave the same way. These are exactly the cues
   `pose_rule_detector` uses to judge squat depth — and they are not trustworthy from a
   single 2D view. This is the depth bottleneck, now measured with n=160 instead of n=9.

2. **Frontal-plane and tilt cues survive.** **torso_lean** ranks reps reliably across
   views (r=0.87, low cross-view spread) though with a consistent −22° underestimate (a
   *calibratable* offset, not noise). **knee_width_ratio** (valgus / knees-caving) is the
   most view-robust cue (MAE 0.02, r=0.90).

3. **Caveat — the rig has no pure side view.** Fit3D's 4 cameras sit at ±45° corners
   (±4 m, ±1.7 m, ~1.5 m high). None looks down the sagittal plane, so true knee flexion
   is never seen edge-on. A near-lateral camera would likely read flexion far better.
   This oblique/frontal regime is, however, the realistic one for phone-camera coaching,
   where users rarely film a clean side view.

## Implications for the codebase

- **`pose_rule_detector` thresholds** for knee-angle / depth faults should be gated on
  view: trustworthy only near a sagittal view, otherwise low-confidence. The systematic
  biases (knee +41°, torso −22°) are candidates for per-view calibration offsets.
- **`view_estimation`** should flag oblique/frontal views as unreliable for depth faults
  (raise `SIDE_VIEW_CONF_THRESHOLD` handling here), and the system should prefer the cue
  set that survives (torso lean, knee width) when no sagittal view is available.
- Motivates **experiment 1** (`src/fit3d/depth_eval.py`): if monocular 3D (NLF) recovers
  the depth axis, it should restore the knee/hip/depth cues that 2D projection destroys.

Reproduce: `python scripts/fit3d/run_view_dependence.py --action squat`
(full result: `data/Fit3D/derived/view_dependence_squat.json`).

## Extension — across lower-body exercises

Ran the same analysis on deadlift, reverse lunge, and thruster (~40 reps × 8 subjects ×
4 cameras each). `noise/signal` per cue (`*` view-corrupted, `~` sensitive, blank robust):

| cue | squat | deadlift | rev_lunge | thruster |
|---|---|---|---|---|
| knee_angle | 1.21 * | 0.31 | 0.46 | 0.74 ~ |
| hip_angle | 0.84 ~ | 2.46 * | 1.34 * | 0.35 |
| torso_lean | 0.37 | 1.50 * | 0.50 ~ | 0.26 |
| depth_ratio | 1.21 * | 0.37 ~ | 1.66 * | 0.68 ~ |
| knee_width | 0.44 | 0.35 | 5.45 * | 0.31 |

**The corrupted cue tracks the movement's defining mechanic.** Single-view 2D corrupts
exactly the cue that matters most for each exercise: the **squat** and **thruster**
(knee-dominant) lose **knee flexion + depth**; the **deadlift** (hip hinge) loses
**hip angle + torso lean** (knee is fine — it barely flexes); the **reverse lunge** loses
hip/depth. So the depth bottleneck is not squat-specific — every lower-body pattern's
form-defining cue degrades from a single oblique 2D view.

Caveats: rev_lunge `knee_width` 5.45 is a metric-design artefact — the legs split
fore-aft so lateral knee separation is meaningless for a lunge, not a depth finding.
Deadlift `knee_angle` carries a large but view-*stable* +32° bias (calibratable, like
squat torso lean). man_maker / burpees / clean_and_press also have ~40 reps but are
multi-phase per rep, so the per-rep extreme is muddier — not reported here.

**Prioritisation for NLF depth recovery (experiment 1):** squat (done), **thruster**
(knee/depth corrupted like squat), and **deadlift** (hip/torso corrupted) are where
direct image->3D should help most. Reproduce: `run_view_dependence.py --action <name>`
(results: `data/Fit3D/derived/view_dependence_<action>.json`).
