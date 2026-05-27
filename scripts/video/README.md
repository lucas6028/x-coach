# Video Scripts

Video workflow entry points for VideoMAE features, lightweight classifier training, experiment grids, and prediction error analysis.

## Feature Extraction

```bash
python scripts/video/run_videomae_feature_extraction.py
```

## Classifier Training

```bash
python scripts/video/train_videomae_classifier.py
```

## Experiment Grid

```bash
python scripts/video/run_videomae_experiment_grid.py \
  --feature-dir data/Squat/Labeled_Dataset/pose_features \
  --train-keys data/Squat/Labeled_Dataset/Splits/train_keys.json \
  --val-keys data/Squat/Labeled_Dataset/Splits/val_keys.json \
  --test-keys data/Squat/Labeled_Dataset/Splits/test_keys.json \
  --forward-labels data/Squat/Labeled_Dataset/Labels/error_knees_forward.json \
  --inward-labels data/Squat/Labeled_Dataset/Labels/error_knees_inward.json \
  --output-root data/Squat/pose_classifier_experiments
```

## Error Analysis

```bash
python scripts/video/analyze_classifier_errors.py
```
