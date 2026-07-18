# Pose Scripts

Pose workflow entry points for MediaPipe/RTMPose extraction, pose features, view metadata, rule detection, evaluation, and backend comparison.

## Pose Extraction

```bash
# Extract MediaPipe pose landmarks from unlabeled videos.
python scripts/pose/run_pose_extraction.py --dataset unlabeled --limit 5

# Extract pose JSON for labeled clips without writing annotated videos.
python scripts/pose/run_pose_extraction.py --dataset labeled --no-video

# Extract RTMPose-compatible pose JSON (rtmlib runtime by default; --runtime mmpose runs HRNet).
python scripts/pose/run_rtmpose_pose_extraction.py
```

## Pose Features And View Metadata

```bash
python scripts/pose/run_pose_feature_extraction.py
python scripts/pose/run_view_estimation.py
```

## Pose Rule Detection

```bash
# Run detector over the default labeled pose dataset.
python scripts/pose/run_pose_rule_detection.py \
  --no-retrieval \
  --summary-output data/Squat/Labeled_Dataset/pose_rule_detections_summary.csv

# Run detector on one pose JSON file.
python scripts/pose/run_pose_rule_detection.py \
  --pose-json data/Squat/Unlabeled_Dataset/processed_poses/25195_3.json \
  --output-json results/single_detection.json

# Select the movement detector explicitly (default: Squat; also supports "Overhead Press").
# (illustrative path — no Overhead Press dataset is checked in yet)
python scripts/pose/run_pose_rule_detection.py \
  --pose-json path/to/overhead_press_pose.json \
  --output-json results/single_detection.json \
  --movement "Overhead Press"
```

`--movement "<Name>"` picks which registered detector (`src/pose/movements/registry.py`) processes
the pose JSON; it defaults to `Squat`. Currently `Squat` and `Overhead Press` are registered — OHP
thresholds are spec-derived and not yet validated against labeled data (see
`docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` §8).

## Evaluation And Analysis

```bash
python scripts/pose/evaluate_pose_rule_detection.py

python scripts/pose/evaluate_pose_rule_detection.py \
  --detections-dir data/Squat/Labeled_Dataset/pose_rule_detections \
  --view-metadata data/Squat/Labeled_Dataset/view_metadata.csv \
  --output data/Squat/Labeled_Dataset/pose_rule_validation_metrics.csv

python scripts/pose/analyze_predictions_by_view.py \
  --predictions-dir data/Squat/pose_classifier_experiments_normalize/predictions \
  --output data/Squat/pose_classifier_experiments_normalize/view_analysis.csv \
  --summary-output data/Squat/pose_classifier_experiments_normalize/view_analysis_summary.csv

python scripts/pose/compare_pose_backends.py
```
