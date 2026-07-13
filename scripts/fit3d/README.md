# Fit3D pipeline (`scripts/fit3d`)

Fit3D (AIFit, CVPR'21) ships **mocap-grade 3D ground truth** (`joints3d_25`, world
metres, Z up), 4 calibrated camera views, SMPLX meshes, and per-action repetition
annotations for 8 train subjects × 47 actions. We use the 3D ground truth to attack
the project's depth-bottleneck question *directly* (with mocap truth) instead of
indirectly (via downstream LOSO accuracy, n=9).

Data is already extracted under `data/Fit3D/{train,test}/<subj>/`. Logic lives in
`src/fit3d/`; run everything from the repo root.

## Joint layout

`joints3d_25` is the **Human3.6M-17 convention** for indices 0..16 plus 8 extremity
points (feet/hands), verified against the official limb connectivity in
`sminchisescu-research/imar_vision_datasets_tools`. Constants and the world→camera→image
projection (ported from their `util/dataset_util.py`) live in `src/fit3d/dataset.py`.

## Experiment 2 — view-dependence of 2D squat-rule readings (runs locally, no GPU)

For each squat rep we read every biomechanical cue from the view-invariant 3D truth
and from each camera's 2D projection (identical formulas, `src/fit3d/biomech.py`), then
report which cues survive single-view 2D and which need 3D.

```bash
python scripts/fit3d/run_view_dependence.py --action squat \
    --json data/Fit3D/derived/view_dependence_squat.json \
    --csv  data/Fit3D/derived/view_dependence_squat.csv
```

Headline result (40 reps, 8 subjects, 4 cameras) is in
`notes/fit3d_view_dependence_summary.md`: projected 2D **knee/hip angle** and
**hip-below-knee depth** are *view-corrupted* (a deep 78° squat reads as 108–133°
depending on camera); **torso lean** and **knee-width/valgus** rank reliably across
views. Fit3D's 4 cameras are all ~45° obliques — there is no pure side view, which is
the realistic phone-camera coaching regime.

## Experiment 1 — monocular-3D depth recovery vs mocap truth

`src/fit3d/depth_eval.py` decomposes a monocular 3D prediction's error against GT into
per-axis components (in-plane x/y vs **depth** z) — the split that separates "lifting
already solves this" from the actual bottleneck — plus Procrustes-aligned MPJPE and the
squat-cue errors set beside experiment 2's single-view 2D baseline. NLF SMPL-24 output is
mapped to Human3.6M-17 with the **L/R convention resolved against the GT** (not assumed).

```bash
python scripts/fit3d/run_depth_eval.py --pred-root data/Fit3D/derived/preds/nlf \
    --json data/Fit3D/derived/depth_eval_squat_nlf.json
```

Result (32 train-squat sequences, NLF on Kaggle P100, full writeup in
`notes/fit3d_depth_recovery_summary.md`): NLF per-axis **depth error 42 mm is on par with
in-plane** (ez/exy = 1.16) — depth is no longer the failure axis — and NLF roughly
**halves** the per-frame knee/hip/torso-lean error that single-view 2D projection
introduces. The route past the depth bottleneck is direct image->3D, not 2D-lifting.

## Experiment 3 — coaching-verdict fidelity (2D vs NLF 3D, runs locally)

`src/fit3d/decision_eval.py` translates the cue findings into the pass/fault **verdict** a
coaching app emits, and asks how often a single 2D camera flips that verdict vs the mocap
truth — and whether NLF fixes it. The 2D arm is reported raw **and** after oracle per-view
debiasing (the cap on what calibration can buy), with the same debiasing applied to NLF, so
the deciding comparison is fair.

```bash
python scripts/fit3d/run_decision_eval.py --pred-root data/Fit3D/derived/preds/nlf \
    --action squat --json data/Fit3D/derived/decision_eval_squat_nlf.json
```

Result (full writeup `notes/fit3d_decision_fidelity_summary.md`): as deployed, single-view
2D **false-fails 82% of good squats** on the depth verdict; even oracle per-view calibration
leaves it flipping the verdict 16% (14% false-alarm + 42% missed shallow reps — one offset
can't fix both), vs **7% for NLF**. The needs-3D map (squat/deadlift/thruster) is consistent
with experiment 2: **direct 3D rescues the sagittal depth/flexion verdicts** (knee, hip,
hips-below-knee) it geometrically corrupts, while **calibrated 2D is better for valgus** and
ties on torso-lean — so route by fault type, don't replace 2D wholesale.

## Model comparison — is depth recovery an NLF quirk or general? (runs locally)

`src/fit3d/model_comparison.py` compares direct image->3D models (NLF vs HMR2.0 vs ...) on the
same Fit3D videos. Because body conventions differ per model, it ranks on **bias-tolerant**
metrics — ez/exy depth-axis pattern, rotation-invariant knee/hip cues, pa_mpjpe, and the
**debiased** verdict-flip — not raw MPJPE.

```bash
python scripts/fit3d/run_model_comparison.py --action squat \
    --model NLF=data/Fit3D/derived/preds/nlf \
    --model HMR2=data/Fit3D/derived/preds/hmr2
```

Result (`notes/fit3d_model_comparison_summary.md`): HMR2.0 (a parametric-SMPL transformer,
architecturally distinct from NLF's localizer field) **independently recovers the sagittal
knee/hip cues** single-view 2D corrupts — knee within ~1deg of NLF, hip often better — across
squat/deadlift/thruster. So direct image->3D depth recovery is a **general mechanism, not an NLF
quirk** (closing the REHAB24 thread that was confounded there by 75% detection). Confirmed with a
**third** architecture, Multi-HMR (full-frame SMPL-X, ECCV'24) — all three recover the knee/hip
cues. NLF remains the best depth model (ez/exy ~1.0 vs HMR2.0/Multi-HMR's ~1.2–1.6); full-frame
Multi-HMR does not close the gap, so NLF's advantage is genuine, not just HMR2.0's crop handicap.

## 2D-vs-3D-vs-mocap decomposition — detector error vs projection error (runs locally, CPU)

`src/fit3d/twod_vs_threed.py` decomposes a single-view 2D pipeline's cue error into
**detector error (real-2D − mocap-2D)** + **projection error (mocap-2D − GT-3D)**, where real-2D is
RTMPose (`rtmlib`, extracted by `run_rtmpose_fit3d.py`), mocap-2D is the GT projection (a *perfect*
detector), and 3D is NLF/HMR2.0/Multi-HMR.

```bash
pip install rtmlib onnxruntime
python scripts/fit3d/run_rtmpose_fit3d.py --mode balanced --subsample 15   # ~one-time CPU extraction
python scripts/fit3d/run_twod_vs_threed.py --action squat \
    --model NLF=data/Fit3D/derived/preds/nlf --model HMR2=data/Fit3D/derived/preds/hmr2 \
    --model MultiHMR=data/Fit3D/derived/preds/multihmr
```

Result (`notes/fit3d_2d_vs_3d_summary.md`): on the **knee** cue the **detector error is ~0** across
all three actions (a strong real detector, RTMPose, is already as good as a *perfect* one) — the
whole ~14–18° error is projection geometry, and only 3D (~6–8°) fixes it. This is the direct mocap-GT
proof of "depth is the bottleneck, not 2D accuracy". **Valgus** is the mirror image (detector-
dominated → a better 2D detector helps, 3D not needed).

## Sparse-skeleton depth quality — MediaPipe (weak depth) vs MeTRAbs (true depth)

The dense SMPL regressors (NLF/HMR2.0/Multi-HMR) recover the sagittal depth cues. Is that because
they are **dense**, or just because they emit **true metric depth**? Two *sparse* keypoint models
isolate the axis:

* **MediaPipe** (BlazePose GHUM) — sparse + **weak** depth (a heuristic `z`). Runs locally on CPU;
  33 world landmarks mapped to H36M-17 (`src/fit3d/mediapipe_baseline.py`), saved as `joints_cam`
  (F,17,3) mm so the eval resolves L/R against the GT (`depth_eval.resolve_lr_h36m17`).
* **MeTRAbs** (Sárándi et al.) — sparse + **true** metric depth. Kaggle GPU kernel
  (`.kaggle_tmp/metrabs_extract`, `haoping6028/fit3d-metrabs-extract`); `smpl_24` skeleton saved as
  `smpl3d` (mm, camera frame) — the *same* code path as NLF/HMR2, so the mapping artifact cancels.
  Passed Fit3D's **real per-camera intrinsics** (baked into the kernel), unlike NLF's assumed FOV≈55.

Both are subsampled every 15th frame (per-frame-mean metrics are unbiased) so the sparse-vs-sparse
contrast is on the *same* frames.

```bash
.venv\Scripts\python.exe -m pip install mediapipe
.venv\Scripts\python.exe scripts/fit3d/run_mediapipe_fit3d.py \
    --actions squat deadlift overhead_extension_thruster --subsample 15
# MeTRAbs: push .kaggle_tmp/metrabs_extract (SMOKE=True first), then pull to preds/metrabs
python scripts/fit3d/run_model_comparison.py --action squat \
    --model MediaPipe=data/Fit3D/derived/preds/mediapipe \
    --model MeTRAbs=data/Fit3D/derived/preds/metrabs \
    --model NLF=data/Fit3D/derived/preds/nlf
```

Result (`notes/fit3d_sparse_depth_summary.md`): MediaPipe's 3D is weak everywhere (depth error ~2× NLF)
and barely recovers the sagittal cues — its depth/flexion verdict-flip is no better than calibrated 2D.
MeTRAbs, sparse but metric, recovers the cues **like the dense models** (pa_mpjpe and knee/hip cue
NLF-level). So **true depth — not dense mesh — is what buys the recovery**; the bottleneck is depth
quality, not skeleton density.

### Kaggle GPU extraction

The NLF kernel (`scratchpad`/`haoping6028/fit3d-nlf-extract`) mirrors the proven
`rehab24-nlf-extract-c1` recipe (torch 2.5.1+cu121, `detect_smpl_batched`, half-res,
largest-area box), reads the `fit3d-squat-nlf-input` dataset, processes all GT frames
(~42 min, 100% detection), and saves per-video SMPL-24 npz named `<subj>__squat__<cam>.npz`.
To extend beyond squats, add the action's videos to the input dataset + manifest and
re-run. (Kaggle MCP note: `dataset_create_new` / `kernel_push` work; `kernel_pull` /
`kernel_status` / `kernel_output` / `datasets_list` are broken — use `uv run --with kaggle
-- kaggle ...` for those.)
