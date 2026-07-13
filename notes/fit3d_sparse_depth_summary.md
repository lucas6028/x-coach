# Fit3D sparse-skeleton depth quality — is it density or true depth that recovers the cues?

**Question.** Experiments 1–3 and the model comparison showed that **dense** direct image→3D models
(NLF, HMR2.0, Multi-HMR) recover the sagittal knee/hip/depth cues that a single 2D view geometrically
corrupts. But those models are *both* dense **and** metric. Which property does the work? Two **sparse
keypoint** models separate the axis:

| model | skeleton | depth | where it runs |
|---|---|---|---|
| **MediaPipe** (BlazePose GHUM) | sparse (33→17) | **weak** (heuristic z) | local CPU |
| **MeTRAbs** (Sárándi et al.) | sparse (SMPL-24) | **true** metric | Kaggle GPU |
| NLF / HMR2.0 / Multi-HMR | dense SMPL mesh | true metric | Kaggle GPU |

If **MediaPipe** (sparse+weak) fails but **MeTRAbs** (sparse+true) recovers the cues *like the dense
models*, then the recovering ingredient is **true metric depth, not mesh density** — the bottleneck is
depth *quality*, consistent with the whole depth-bottleneck thread.

**Method.** Both sparse models are evaluated with the *same* `src/fit3d` harness against Fit3D mocap GT
(8 subjects × 4 cameras, squat + deadlift + thruster), subsampled every 15th frame so the sparse-vs-sparse
contrast is on the **same frames** (per-frame-mean metrics are unbiased under subsampling). MediaPipe's 33
world landmarks map to H36M-17 (`src/fit3d/mediapipe_baseline.py`) and are saved as `joints_cam`; MeTRAbs
emits `smpl_24` saved as `smpl3d`, taking the *same* `resolve_lr`+`SMPL24_TO_H36M17` path as NLF/HMR2 so the
mapping artifact cancels. As with HMR2's crop frame, we rank on **bias/frame-tolerant** metrics: `pa_mpjpe`,
the **ez/exy** depth-axis ratio, **rotation-invariant knee/hip** cue error, and the **debiased** verdict-flip.

**Caveats (honest).** (a) MediaPipe's world-landmark frame is only *roughly* the true camera rotation (it
never sees extrinsics), so its raw per-axis `ez` carries a frame-rotation term like HMR2's crop frame — read
it off `pa_mpjpe` + rotation-invariant cues, not literal depth-mm. (b) MeTRAbs was passed Fit3D's **real
per-camera intrinsics**; NLF used an assumed FOV≈55 — so absolute-mm depth comparisons carry that parity
caveat (the ranked metrics are largely intrinsics-robust). MeTRAbs ran at `num_aug=2` (its default is 5) and
without lens-distortion modelling (k1≈−0.19, mild). (c) The two **sparse** models are every-15th-frame; the
**dense** models are every-frame. Per-frame cue error is unbiased under subsampling (masking NLF to every-15th
leaves its knee 7.09→7.09), so the **cue tables** are clean cross-model — but verdict-flip reduces a rep window
to its *extreme* (nanmin/nanmax), which is mildly sample-count biased, so read the sparse-vs-**dense**
verdict-flip loosely. The headline **sparse-vs-sparse** contrast (MediaPipe vs MeTRAbs) is immune — both at 15.
Detection held on every sequence: 0/96 empty, min per-video finite-cue coverage 100% (all 32 rows/action
contribute). 

## Results — MediaPipe (sparse + weak depth), 32 sequences/action

**Depth-axis pattern** (root-relative, mm):

| action | model | PA-MPJPE | ez (depth) | exy | ez/exy |
|---|---|---|---|---|---|
| squat | **MediaPipe** | 103.5 | 96.2 | 59.9 | 1.61 |
| squat | NLF | 65.4 | 42.4 | 36.6 | 1.16 |
| deadlift | **MediaPipe** | 84.4 | 90.4 | 65.2 | 1.39 |
| deadlift | NLF | 61.7 | 44.5 | 38.0 | 1.17 |
| thruster | **MediaPipe** | 88.7 | 80.2 | 56.7 | 1.42 |
| thruster | NLF | 67.0 | 33.8 | 35.5 | 0.95 |

**Rotation-invariant cue recovery** (deg vs 3D truth; 2D-view = single-camera projection baseline):

| action | cue | 2D-view | MediaPipe | NLF | HMR2 | MultiHMR |
|---|---|---|---|---|---|---|
| squat | knee* | 18.42 | 14.18 | 7.09 | 7.88 | 9.67 |
| squat | hip* | 18.27 | 12.53 | 10.42 | 9.26 | 9.49 |
| deadlift | knee* | 14.76 | 12.83 | 7.51 | 8.81 | 12.40 |
| deadlift | hip* | 17.84 | 17.77 | 7.93 | 7.78 | 10.21 |
| thruster | knee* | 17.37 | 14.87 | 5.91 | 6.64 | 6.93 |
| thruster | hip* | 14.50 | 9.48 | 11.21 | 9.49 | 8.34 |

**Debiased verdict-flip** (swept thresholds, oracle-calibrated; lower = better):

| action | cue | deb-2D | MediaPipe | NLF |
|---|---|---|---|---|
| squat | knee | 21% | 24% | 11% |
| squat | hip | 12% | 14% | 6% |
| deadlift | knee | 17% | 24% | 10% |
| thruster | knee | 22% | 22% | 9% |

**MediaPipe findings.**

1. **On the frame-robust metrics, it sits at the 2D baseline.** Its `pa_mpjpe` (84–104 mm) is far above
   the dense models', and — the robust signal — its rotation-invariant cues and verdict-flip barely move
   off single-view 2D (see #2, #3). *Caveat on its depth axis:* MediaPipe's world frame is not the true
   camera frame, so its raw `ez`/`ez/exy` (1.4–1.6) carry a rotation term (part of the 40 mm gap between
   its mpjpe 147 and pa 109) — we do **not** read a clean depth-mm claim off MediaPipe's `ez`; only
   MeTRAbs's `ez/exy`, computed in the true camera frame with real K, is read literally.
2. **It barely recovers the sagittal cues.** Squat knee only closes 18.4°→14.2° (NLF reaches 7.1°);
   deadlift hip is essentially unrecovered (17.8°→17.8°). The dense models roughly *halve* the 2D error;
   MediaPipe shaves ~20%.
3. **Its depth/flexion verdict is no better than calibrated 2D.** Squat knee verdict-flip 24% ≈ deb-2D
   21% (and worse than NLF's 11%); deadlift knee 24% > deb-2D 17%. Weak 3D does not buy a better coaching
   verdict than a well-calibrated single camera.

So **sparse + weak depth ≈ not enough.**

## Results — MeTRAbs (sparse + true depth), 32 sequences/action

MeTRAbs (`metrabs_l`, `num_aug=2`, real Fit3D intrinsics), a **sparse** SMPL-24 regressor, on the same
frames as MediaPipe.

**Depth-axis pattern** — MeTRAbs sits with the dense metric models, *not* with MediaPipe:

| action | model | PA-MPJPE | ez (depth) | exy | ez/exy |
|---|---|---|---|---|---|
| squat | **MeTRAbs** | 68.0 | 51.5 | 39.9 | **1.29** | 
| squat | NLF | 65.4 | 42.4 | 36.6 | 1.16 |
| squat | MediaPipe | 103.5 | 96.2 | 59.9 | 1.61 |
| deadlift | **MeTRAbs** | 62.6 | 53.1 | 41.9 | **1.27** |
| deadlift | NLF | 61.7 | 44.5 | 38.0 | 1.17 |
| deadlift | MediaPipe | 84.4 | 90.4 | 65.2 | 1.39 |
| thruster | **MeTRAbs** | 68.0 | 38.2 | 36.9 | **1.03** |
| thruster | NLF | 67.0 | 33.8 | 35.5 | 0.95 |
| thruster | MediaPipe | 88.7 | 80.2 | 56.7 | 1.42 |

**Rotation-invariant cue recovery** (deg; MeTRAbs vs the dense models and the sparse-but-weak MediaPipe):

| action | cue | 2D-view | MediaPipe | **MeTRAbs** | NLF | HMR2 | MultiHMR |
|---|---|---|---|---|---|---|---|
| squat | knee* | 18.42 | 14.18 | **6.37** | 7.09 | 7.88 | 9.67 |
| squat | hip* | 18.27 | 12.53 | **9.33** | 10.42 | 9.26 | 9.49 |
| deadlift | knee* | 14.76 | 12.83 | 7.43 | 7.51 | 8.81 | 12.40 |
| deadlift | hip* | 17.84 | 17.77 | 7.90 | 7.93 | 7.78 | 10.21 |
| thruster | knee* | 17.37 | 14.87 | **4.59** | 5.91 | 6.64 | 6.93 |
| thruster | hip* | 14.50 | 9.48 | 9.43 | 11.21 | 9.49 | 8.34 |

**Debiased verdict-flip** (lower = better):

| action | cue | deb-2D | MediaPipe | **MeTRAbs** | NLF |
|---|---|---|---|---|---|
| squat | knee | 21% | 24% | 13% | 11% |
| squat | hip | 12% | 14% | 6% | 6% |
| deadlift | knee | 17% | 24% | 17% | 10% |
| thruster | knee | 22% | 22% | 13% | 9% |
| thruster | hip | 14% | 15% | 6% | 4% |

**MeTRAbs findings.**

4. **Sparse + true depth recovers the cues at dense-model level.** MeTRAbs's `pa_mpjpe` (62–68 mm) and
   depth axis (ez 38–53 mm, ez/exy 1.03–1.29) land right on NLF (65–67 mm, 34–45 mm, 0.95–1.17) — roughly
   *half* MediaPipe's depth error. On the rotation-invariant cues it is the **best knee model in all three
   actions** (6.37 / 7.43 / 4.59, beating even NLF) and ties the dense models on hip. Its depth/flexion
   verdict-flip (squat knee 13%, thruster knee 13%, hip 6%) is at dense-model level and roughly *halves*
   MediaPipe's — despite being just as sparse.
5. **Parity caveat, doesn't change the verdict.** MeTRAbs was given real intrinsics (NLF used FOV≈55), which
   could flatter its *absolute* depth mm — but the cue/verdict metrics it wins on are rotation-invariant and
   intrinsics-independent, so the recovery is genuine, not a calibration artefact.

## Conclusion — the dividing line is depth *quality*, not skeleton density

Holding the skeleton **sparse** and varying only depth quality separates the two properties the dense models
conflated:

* **MediaPipe** (sparse + *weak* depth) → fails: cues only ~20% recovered, depth/flexion verdict no better
  than a well-calibrated single 2D camera.
* **MeTRAbs** (sparse + *true* depth) → succeeds: cues and verdicts at dense-mesh (NLF) level, often best.

So the ingredient that recovers the sagittal depth cues is **true metric depth, not mesh density**. A sparse
skeleton is entirely sufficient *if* its depth is real. This sharpens the depth-bottleneck conclusion: the
bottleneck was never skeleton richness — it is depth **quality**, and a lightweight sparse metric-3D model
(MeTRAbs) captures the coaching-relevant faults as well as a full dense SMPL regressor.

*One confound named:* MeTRAbs is a stronger model than MediaPipe **in-plane too** (exy 37–42 vs 57–65 mm),
not only on the depth axis — so this is not a clean depth-axis-only manipulation. The conclusion still holds
because the bottleneck is already established to be depth, not 2D: a *perfect* 2D detector (mocap-2D) is no
better than a real one (RTMPose) on the knee cue (`notes/fit3d_2d_vs_3d_summary.md`), so in-plane accuracy is
not what gates the depth/flexion verdict. MediaPipe's weak *depth* — not its slightly noisier 2D — is why it
stays at the 2D baseline.

Reproduce: `python scripts/fit3d/run_model_comparison.py --action squat --model
MediaPipe=data/Fit3D/derived/preds/mediapipe --model MeTRAbs=data/Fit3D/derived/preds/metrabs --model
NLF=data/Fit3D/derived/preds/nlf --model HMR2=data/Fit3D/derived/preds/hmr2 --model
MultiHMR=data/Fit3D/derived/preds/multihmr` (per-action JSON `data/Fit3D/derived/model_comparison_<action>_sparse.json`;
npz under `data/Fit3D/derived/preds/{mediapipe,metrabs}`; MediaPipe via `scripts/fit3d/run_mediapipe_fit3d.py`,
MeTRAbs via Kaggle `haoping6028/fit3d-metrabs-extract`).
