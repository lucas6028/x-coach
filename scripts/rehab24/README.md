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
