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
```

Export metadata for Colab:

```bash
python scripts/rehab24/export_colab_package.py --include-skeleton-features
```
