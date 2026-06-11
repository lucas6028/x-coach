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

Extract the same geometric features from the RGB videos via MMPose / RTMPose
(monocular **2D** keypoints — more accurate joints than MediaPipe but no learned
depth, so it runs the 2D image branch only; feature dim 1188 vs MediaPipe's 2970).
Unlike MediaPipe, MMPose uses the GPU, so run it on Colab — see
`notebooks/rehab24_mmpose_colab.ipynb`:

```bash
# locally (needs rtmlib + onnxruntime); on Colab the notebook drives this for you
python scripts/rehab24/extract_mmpose_skeleton_features.py \
  --runtime rtmlib --model balanced --device cuda:0
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

# MMPose / RTMPose (RGB-estimated 2D) skeleton features
python scripts/rehab24/train_correctness_classifier.py \
  --feature-dir data/REHAB24-6/processed/mmpose_skeleton_features
```

Cross-validate any feature set with Leave-One-Subject-Out (10 subjects rotate
through the test position; reports mean±std — the trustworthy yardstick given the
fixed split only tests 2 subjects):

```bash
python scripts/rehab24/loso_cross_validation.py \
  --feature-dir data/REHAB24-6/processed/mmpose_skeleton_features \
  --summary-output data/REHAB24-6/processed/correctness_loso_mmpose.json
```

Export metadata for Colab:

```bash
python scripts/rehab24/export_colab_package.py --include-skeleton-features
```
