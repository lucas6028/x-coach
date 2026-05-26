# MMPose Result Analysis and Backend Comparison

Date: 2026-05-26

## Source Artifacts

This note summarizes the executed results in:

- `notebooks/run_mmpose_pose_comparison.ipynb`
- `data/Squat/Labeled_Dataset/pose_rule_validation_metrics.csv`
- `data/Squat/pose_classifier_experiments_normalize/metrics/experiment_summary.csv`
- `notes/pose_only_classifier_experiment_summary.md`
- `notes/videomae_classifier_experiment_summary.md`

The MMPose classifier metrics were parsed from the notebook output because the generated Colab MMPose artifact directory is not present in the local workspace.

## MMPose Setup

The MMPose run uses an RTMPose/RTMW-family whole-body model through `rtmlib` and ONNX Runtime GPU. The model outputs COCO-WholeBody keypoints, which are adapted into the existing MediaPipe-compatible 33-landmark JSON format. This keeps the downstream feature extraction, rule detection, view estimation, and classifier training code unchanged.

Important implementation detail: MMPose provides 2D normalized image landmarks in this pipeline, but not MediaPipe-style `world_landmarks`. Therefore, the same 654-dimensional pose feature extractor runs, but MMPose features depend on 2D image geometry rather than MediaPipe's 3D/world landmark fallback.

## Rule-Based Detection

All rows below use the full labeled set (`n=1623`) and compare the `ALL` view rows.

| Backend | Class | TP | FP | TN | FN | Precision | Recall | Specificity | Balanced accuracy | F1 | Mean segment IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MediaPipe | knees_forward | 1 | 0 | 514 | 1108 | 1.000 | 0.001 | 1.000 | 0.500 | 0.002 | 0.000 |
| MMPose | knees_forward | 5 | 0 | 514 | 1104 | 1.000 | 0.005 | 1.000 | 0.502 | 0.009 | 0.001 |
| MediaPipe | knees_inward | 79 | 185 | 1206 | 153 | 0.299 | 0.341 | 0.867 | 0.604 | 0.319 | 0.045 |
| MMPose | knees_inward | 116 | 397 | 994 | 116 | 0.226 | 0.500 | 0.715 | 0.607 | 0.311 | 0.055 |

Rule interpretation:

- `knees_forward` rule detection is essentially non-functional for both backends. MMPose finds 5 positives instead of MediaPipe's 1, but recall is still only `0.005`.
- `knees_inward` is the only useful rule comparison. MMPose improves recall from `0.341` to `0.500`, but false positives more than double from `185` to `397`.
- MMPose and MediaPipe have nearly identical inward balanced accuracy (`0.607` vs `0.604`), but they fail differently: MediaPipe is conservative, while MMPose is more sensitive and noisier.
- For rule-based coaching feedback, MMPose should not be treated as a drop-in improvement. It needs backend-specific thresholds, especially for inward knee detection.

## Pose-Only Classifier Results

The table below uses test split, selected threshold, 5-seed mean. MediaPipe numbers are from the normalized pose-only experiment. MMPose numbers are from the MMPose notebook run, which also used `--normalize-features`.

| Label mode | Backend | Balanced accuracy | Macro F1 | Recall | Specificity | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| combined | MediaPipe | 0.635 | 0.622 | 0.717 | 0.553 | 0.750 |
| combined | MMPose | 0.628 | 0.623 | 0.779 | 0.478 | 0.778 |
| knees_forward | MediaPipe | 0.615 | 0.599 | 0.714 | 0.517 | 0.735 |
| knees_forward | MMPose | 0.627 | 0.627 | 0.795 | 0.459 | 0.780 |
| knees_inward | MediaPipe | 0.608 | 0.526 | 0.578 | 0.638 | 0.315 |
| knees_inward | MMPose | 0.702 | 0.568 | 0.789 | 0.614 | 0.394 |

Delta, MMPose minus MediaPipe:

| Label mode | Balanced accuracy | Macro F1 | Recall | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| combined | -0.007 | +0.001 | +0.062 | -0.075 | +0.028 |
| knees_forward | +0.012 | +0.028 | +0.082 | -0.059 | +0.045 |
| knees_inward | +0.094 | +0.042 | +0.211 | -0.023 | +0.079 |

Classifier interpretation:

- MMPose is clearly better for `knees_inward` classification. The main gain is recall: `0.789` vs MediaPipe `0.578`.
- For `knees_forward`, MMPose is slightly better on balanced accuracy and macro F1, but the improvement comes with lower specificity.
- For `combined`, MediaPipe is still slightly better on balanced accuracy because it preserves more normal-class specificity.
- MMPose generally pushes the classifier toward positive predictions. This helps recall and positive-class F1, but hurts normal squat recognition.

## Comparison With Previous Baselines

| Model / backend | combined bal. acc. | knees_forward bal. acc. | knees_inward bal. acc. | Main read |
| --- | ---: | ---: | ---: | --- |
| VideoMAE-only | ~0.53-0.59 | ~0.51-0.56 | ~0.48-0.61 | Weak research baseline; generic RGB embedding is not enough. |
| MediaPipe pose-only, no normalization | 0.581 | 0.573 | 0.570 | Biomechanically meaningful, but over-predicts error. |
| MediaPipe pose-only, normalized | 0.635 | 0.615 | 0.608 | Best current MediaPipe baseline; better specificity. |
| MMPose pose-only, normalized | 0.628 | 0.627 | 0.702 | Best inward-knee detector; stronger recall but lower specificity. |

Overall ranking depends on the target:

- Best general balanced classifier: normalized MediaPipe pose-only for `combined`.
- Best `knees_forward` classifier: MMPose pose-only by a small margin.
- Best `knees_inward` classifier: MMPose pose-only by a large margin.
- Best conservative rule detector: MediaPipe for inward, because it has fewer false positives.
- Best sensitive rule detector: MMPose for inward, because it catches more positives.

## Practical Conclusion

MMPose is useful, but not as a universal replacement for MediaPipe.

For classifier-based squat error detection, MMPose is worth keeping because it substantially improves `knees_inward` and slightly improves `knees_forward`. The tradeoff is lower specificity, so it may produce more false alarms in normal squat videos.

For rule-based detection, MMPose does not solve the rule problem. `knees_forward` remains almost undetected, and `knees_inward` improves recall at the cost of many additional false positives. The existing rule thresholds were tuned around MediaPipe geometry and should be recalibrated separately for MMPose.

The strongest next experimental direction is not choosing only one backend. A better path is:

1. Keep normalized MediaPipe pose-only as the main balanced baseline.
2. Keep normalized MMPose pose-only as the stronger inward-knee baseline.
3. Add backend-specific rule thresholds before making rule-based claims.
4. Compare late fusion or ensemble logic: MediaPipe for specificity, MMPose for inward recall, and VideoMAE only as supporting RGB context.

