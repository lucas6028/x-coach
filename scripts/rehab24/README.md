# REHAB24-6 Scripts

Build the repetition-level manifest and subject-wise splits:

```bash
python scripts/rehab24/build_manifest.py
```

Extract lightweight skeleton features locally (from the dataset's Vicon mocap skeletons):

```bash
python scripts/rehab24/extract_skeleton_features.py
```

Extract the same geometric skeleton features from the RGB videos via MediaPipe Pose
(monocular estimated joints instead of mocap; output is drop-in compatible with the
correctness classifier via `--feature-dir`):

```bash
python scripts/rehab24/extract_mediapipe_skeleton_features.py
# smoke test on one video: --video-limit 1 --output-dir /tmp/mp_smoke
```

Extract the same geometric features from the RGB videos via RTMPose / RTMW
(monocular **2D** keypoints — more accurate joints than MediaPipe but no learned
depth, so it runs the 2D image branch only; feature dim 1188 vs MediaPipe's 2970).
Unlike MediaPipe, RTMPose uses the GPU, so run it on Colab — see
`notebooks/rehab24_rtmpose_colab.ipynb`:

```bash
# locally (needs rtmlib + onnxruntime); on Colab the notebook drives this for you
python scripts/rehab24/extract_rtmpose_skeleton_features.py \
  --runtime rtmlib --model balanced --device cuda:0
# smoke test on one video: --video-limit 1
```

Extract the same 2D features with a **stronger backbone (HRNet whole-body)** instead of
RTMPose. HRNet isn't in `rtmlib`, so it runs through the `mmpose` runtime (full OpenMMLab
stack). It runs locally on a modest GPU — measured on a GTX 1660 Ti: ~6.4 fps, **0.42 GB**
peak VRAM, **~16 h** for the full 130-video set (extraction is resumable). It stays
2D-only, so compare it against RTMPose-2D, not MediaPipe — see
`notebooks/rehab24_hrnet_colab.ipynb` for the paired-LOSO rationale.

One-time local GPU env (Windows / PowerShell). The versions are a **locked set** — bumping
any one forces mmcv to compile from source. The cu118 wheels run on older drivers
(≥ 452.39) via CUDA minor-version compatibility, so **no NVIDIA driver update is needed**:

```powershell
py -3.11 -m venv .venv-mmpose          # mmcv/torch wheels need Python 3.11, not 3.12+
.\.venv-mmpose\Scripts\Activate.ps1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim wheel "setuptools<81" "numpy<2" opencv-python pycocotools
#   setuptools<81 : >=81 removes pkg_resources, which mmengine reload()s at runtime
#   numpy<2       : torch 2.1 is built against numpy 1.x
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
pip install --no-build-isolation "chumpy==0.70"   # mmpose dep; setup.py needs wheel + no isolation
pip install --no-deps "mmpose==1.3.2"             # --no-deps skips xtcocotools (no Windows wheel);
                                                  #   src/pose aliases pycocotools for it at runtime
# verify (run from repo root): prints "GPU True"
python -c "import torch; from src.pose.rtmpose_pose_extraction import import_mmpose_inferencer; import_mmpose_inferencer(); print('GPU', torch.cuda.is_available())"
```

Then extract (resumable — re-run the same line to continue after a stop). 6 GB VRAM is
plenty; use `w32` here, or `td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288` for the
more accurate (slower) backbone:

```powershell
$env:PYTHONPATH = (Get-Location)
python scripts/rehab24/extract_rtmpose_skeleton_features.py `
  --runtime mmpose --model td-hm_hrnet-w32_dark-8xb64-210e_coco-wholebody-256x192 `
  --device cuda:0 --output-dir data/REHAB24-6/processed/hrnet_skeleton_features
# smoke test on one video: --video-limit 1
```

Extract repetition-level VideoMAE features on Colab/GPU:

```bash
python scripts/rehab24/extract_videomae_features.py --device cuda
```

This writes *raw* bundles to `data/REHAB24-6/processed/videomae_raw/`, holding the
per-clip stacks for **both** token-pooling modes from a single forward pass:

- `mean_pool_fc_norm` — mean over patch tokens + the pretrained `fc_norm`, i.e. what
  `VideoMAEForVideoClassification` actually does.
- `legacy_first_token` — `last_hidden_state[:, 0, :]`, the historical extraction.
  VideoMAE has **no CLS token**, so this is the first patch of the first tubelet, not a
  clip representation. Kept only so a paired comparison can isolate the pooling fix.

Aggregation over clips stays an offline choice (see `src/video/videomae_pooling.py` for
why `max` is wrong after `fc_norm`). Materialize the LOSO-ready dirs, then audit them:

```bash
python scripts/rehab24/materialize_videomae_features.py
python scripts/rehab24/audit_videomae_features.py \
  data/REHAB24-6/processed/videomae_mean_pool_fc_norm_mean
```

`materialize` writes one dir per (token pooling x clip aggregation) combination, each
storing `video_feature` — the key the LOSO drivers already read, so they consume these
via `--feature-dir` unmodified. `audit` exits non-zero on incomplete coverage, duplicate
stems, mixed dims/dtypes, non-finite values, split mismatches or mixed provenance.

Run the Stage A evidence matrix (paired deltas, within-subject permuted-label null
control, and camera/exercise stratification) in one pass:

```bash
python scripts/rehab24/videomae_stage_a.py --device cpu
```

### Framing arms (letterbox / person crop / background only)

Pre-registration: `notes/rehab24_videomae_framing_validation_plan.md`. Run the steps in
this order — the gates exist so nobody can look at an accuracy first and then decide
which arm was "really" showing the whole athlete.

```bash
# 1. One fixed box per SOURCE VIDEO, from the dataset's own mocap 2D skeletons.
#    Never one box per repetition: that would encode how far the rep travelled, which
#    is a function of its correctness.
python scripts/rehab24/build_videomae_boxes.py

# 2. Geometry gate. Exact arithmetic on frame sizes and boxes; no decoding, no model.
#    Must pass, and be READ, before any features exist.
python scripts/rehab24/videomae_framing_geometry.py

# 3. Extract each arm into its OWN raw dir (the extractor refuses to mix them).
#    On this machine use .venv-cuda (see below); --num-chunks splits the manifest
#    round-robin into disjoint sets, so workers never write the same file.
OMP_NUM_THREADS=4 .venv-cuda/Scripts/python.exe scripts/rehab24/extract_videomae_features.py \
  --variant full_frame_letterbox \
  --output-dir data/REHAB24-6/processed/videomae_raw_full_frame_letterbox \
  --device cuda --num-chunks 3 --chunk-index 0   # repeat for chunk-index 1 and 2

# 4. Pairing gate: same ids, splits, clip starts and metadata; different pixels.
python scripts/rehab24/videomae_framing_pairing.py \
  --baseline-dir data/REHAB24-6/processed/videomae_raw_full_frame_local \
  --candidate-dir data/REHAB24-6/processed/videomae_raw_full_frame_letterbox

# 5. Materialize the primary representation only, once per variant. BOTH arms:
#    the local full_frame baseline is an arm like any other.
for variant in full_frame full_frame_letterbox; do
  python scripts/rehab24/materialize_videomae_features.py \
    --raw-dir data/REHAB24-6/processed/videomae_raw_${variant/full_frame/full_frame_local} \
    --output-parent data/REHAB24-6/processed/videomae_framing/$variant \
    --token-pooling mean_pool_fc_norm --aggregation mean
done

# 6. Audit the MATERIALIZED dirs. Not a duplicate of step 4: the audit needs the
#    `video_feature` key raw bundles do not have, and it is what independently
#    checks split placement against the manifest and catches constant features.
python scripts/rehab24/audit_videomae_features.py \
  data/REHAB24-6/processed/videomae_framing/full_frame/videomae_mean_pool_fc_norm_mean \
  data/REHAB24-6/processed/videomae_framing/full_frame_letterbox/videomae_mean_pool_fc_norm_mean

# 7. Paired LOSO across arms, three seeds, one pre-registered primary test.
python scripts/rehab24/videomae_framing_report.py \
  --arm full_frame=data/REHAB24-6/processed/videomae_framing/full_frame/videomae_mean_pool_fc_norm_mean \
  --arm full_frame_letterbox=data/REHAB24-6/processed/videomae_framing/full_frame_letterbox/videomae_mean_pool_fc_norm_mean \
  --arm kaggle_full_frame=data/REHAB24-6/processed/videomae_mean_pool_fc_norm_mean \
  --primary full_frame_letterbox:full_frame --device cpu
```

`kaggle_full_frame` is declared as an arm but is **not** a `--secondary` comparison. It
is a quality check, not a framing hypothesis: the runner scores every declared arm, so
its seed-averaged balanced accuracy lands beside the local `full_frame` in the summary
and answers whether the local re-extraction reproduces the published 0.657 — the way
Stage A checked its legacy reproduction against the historical 0.536. Putting it in
`--secondary` would drag a QC check into the Holm family.

The baseline arm is re-extracted **locally** rather than reused from the archived
`videomae_raw/`: those bundles came from a Kaggle kernel on transformers 5.0.0, and a
local re-run of the same code differs by ~1e-3 relative L2 (cosine 0.9999996). Small,
but it would sit inside the measured delta on one arm only. Re-extracting removes the
environment from the comparison and doubles as a reproduction check against 0.657.

Extraction is resumable: it skips any bundle already on disk, so re-invoking the same
command continues rather than restarting.

#### Extracting on the GPU (`.venv-cuda`)

`.venv` holds a CPU-only torch and must stay that way — it serves the backend and the
whole test suite. Extraction gets its own venv, the same way `--runtime mmpose` does:

```bash
py -3.12 -m venv .venv-cuda
.venv-cuda/Scripts/python.exe -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
.venv-cuda/Scripts/python.exe -m pip install numpy==2.4.3 transformers==5.5.0 \
  opencv-python==4.13.0.92 pillow==12.2.0
```

`cu126`, not `cu13x`: this machine's card is a GTX 1660 Ti (Turing, **sm_75**) and CUDA
13.x has been dropping older architectures. Check `torch.cuda.get_arch_list()` contains
`sm_75` before trusting a build — and note `resolve_device` runs a real strided `conv3d`
probe, so a wheel without the right kernels falls back to CPU in two seconds instead of
dying six hours in (the Kaggle P100/sm_60 failure).

Measured here, per repetition bundle: CPU 1 worker **9.30 s**, GPU 3 workers **2.19 s**
(4.25x). Three workers because the profile is CPU-bound, not GPU-bound — decode is 42%
of a clip, the processor's resize most of the rest, and the forward leaves a 1660 Ti at
32% with 1.8 of 6 GB used. Cap `OMP_NUM_THREADS` per worker so three processes do not
oversubscribe 12 logical cores; that is numerically safe *because* the forward is on the
GPU, so CPU threads touch only decode and the processor, neither of which reduces across
threads.

**Both arms must come from the same venv.** Provenance records the transformers version
but not the device, so a mixed arm is undetectable afterwards. Measured on 8 identical
samples: CPU vs GPU differ by 4.1e-06 relative L2 (cosine 1.00000000), while Kaggle 5.0.0
vs local 5.5.0 differ by 9.4e-04 — the library version dominates the hardware by ~230x.

Fuse skeleton and VideoMAE features:

```bash
python scripts/rehab24/fuse_features.py
```

Train the correctness classifier (point `--feature-dir` at any feature set):

```bash
# mocap skeleton features (default)
python scripts/rehab24/train_correctness_classifier.py

# MediaPipe (RGB-estimated) skeleton features
python scripts/rehab24/train_correctness_classifier.py \
  --feature-dir data/REHAB24-6/processed/mediapipe_skeleton_features

# RTMPose / RTMW (RGB-estimated 2D) skeleton features
python scripts/rehab24/train_correctness_classifier.py \
  --feature-dir data/REHAB24-6/processed/rtmpose_skeleton_features
```

Cross-validate any feature set with Leave-One-Subject-Out (10 subjects rotate
through the test position; reports mean±std — the trustworthy yardstick given the
fixed split only tests 2 subjects):

```bash
python scripts/rehab24/loso_cross_validation.py \
  --feature-dir data/REHAB24-6/processed/rtmpose_skeleton_features \
  --summary-output data/REHAB24-6/processed/correctness_loso_rtmpose.json

# HRNet 2D features (same command, different feature dir). For the HRNet-vs-RTMPose
# paired comparison, the RTMPose features above must exist too — see the HRNet notebook.
python scripts/rehab24/loso_cross_validation.py \
  --feature-dir data/REHAB24-6/processed/hrnet_skeleton_features \
  --summary-output data/REHAB24-6/processed/correctness_loso_hrnet.json
```

Export metadata for Colab:

```bash
python scripts/rehab24/export_colab_package.py --include-skeleton-features
```

Validate the **Lunge** rule detector against Ex5's 174 labeled repetitions (the only movement
in this repo checked against human-labeled ground truth). Needs the Ex5 pose corpus under
`data/REHAB24-6/processed/lunge_pose_json/` first — see
`notes/lunge-view-reconnaissance.md` for the extraction command:

```bash
python scripts/rehab24/validate_lunge_rules.py \
  --pose-dir data/REHAB24-6/processed/lunge_pose_json \
  --segmentation data/REHAB24-6/Segmentation.csv \
  --out data/REHAB24-6/processed/lunge_rule_validation.json
# ~15 min; add --report-only to re-print the report from the saved JSON in ~1 s
```

Results, method and caveats: `notes/lunge-rule-validation.md`. The rules are replayed one
labeled repetition at a time, twice — once with the view label the production estimator really
produces, once with the dataset's ground-truth orientation — so a gate failure can be told
apart from a rule failure. **The labels are binary (correct/incorrect) and never name the
fault**, so this measures whether a rule's signal carries information about rep correctness,
not per-fault precision.
