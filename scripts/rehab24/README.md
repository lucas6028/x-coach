# REHAB24-6 Scripts

Build the repetition-level manifest and subject-wise splits:

```bash
python scripts/rehab24/build_manifest.py
```

Extract lightweight skeleton features locally:

```bash
python scripts/rehab24/extract_skeleton_features.py
```

Extract repetition-level VideoMAE features on Colab/GPU:

```bash
python scripts/rehab24/extract_videomae_features.py --device cuda
```

Fuse skeleton and VideoMAE features:

```bash
python scripts/rehab24/fuse_features.py
```

Train the correctness classifier:

```bash
python scripts/rehab24/train_correctness_classifier.py
```

Export metadata for Colab:

```bash
python scripts/rehab24/export_colab_package.py --include-skeleton-features
```
