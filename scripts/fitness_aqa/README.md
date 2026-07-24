# Fitness-AQA: does depth matter in the wild?

Fit3D and REHAB24-6 have mocap ground truth, so the depth bottleneck can be *measured*
there (`scripts/fit3d/`, `scripts/rehab24/`). Fitness-AQA has none — it is real gym
footage from a single phone-grade camera. What it does have is
`Shallow_Squat_Error_Dataset`: 3,611 bottom-of-squat frames labelled `0` = no error /
`1` = shallow, with **video-disjoint** train/val/test splits. That is the in-the-wild
version of the squat-depth verdict, so the question becomes a downstream one: with the
same detector and the same cue features, does adding the depth channel change the
classification?

Everything runs on the release's own `crops_unaligned` person crops. Matching a crop
back to its source video frame is ambiguous to ±1 frame at the bottom of a squat
(adjacent frames are near-identical), and every arm must see byte-identical input.

## Arms

| npz | source | space | what it is |
|---|---|---|---|
| `nlf_3d` | NLF `joints3d` | `cam3d` (mm) | regressed depth |
| `nlf_2d` | NLF `joints2d` | `image2d` (px) | **same forward pass**, depth removed |
| `mediapipe_3d` | BlazePose world landmarks | `cam3d` (mm) | heuristic depth |
| `mediapipe_2d` | BlazePose image landmarks | `image2d` (px) | same pass, depth removed |
| `rtmpose_2d` | RTMPose COCO-17 | `image2d` (px) | strong 2D reference |

`nlf_3d` vs `nlf_2d` is the measurement. Everything else is context: a cross-model
comparison confounds depth with detector quality, which is exactly the confound the
Fit3D error decomposition was built to avoid.

## Run

```powershell
# 2D + heuristic-3D arms (CPU, ~20 min for all 3,611 crops)
.venv\Scripts\python.exe scripts/fitness_aqa/run_shallow_pose_extraction.py --backend mediapipe
.venv\Scripts\python.exe scripts/fitness_aqa/run_shallow_pose_extraction.py --backend rtmpose

# strong-depth arm (GPU strongly preferred; --resume picks up a partial run)
.venv\Scripts\python.exe scripts/fitness_aqa/run_shallow_nlf_extraction.py --device cuda --batch 8

# compare
.venv\Scripts\python.exe scripts/fitness_aqa/run_shallow_depth_classification.py --mlp-seeds 5
```

Arms land in `data/Fitness-AQA/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/derived/pose/`;
the report is written next to them as `shallow_depth_classification.json`.

## How the comparison is kept fair

- **Same rows.** A crop any arm failed on is dropped from *every* arm, so no arm gets an
  easier subset.
- **Same features.** `src/fitness_aqa/cue_features.py` computes 14 scale-free angles and
  ratios with the formulas in `src/fit3d/biomech.py`. Identical dimension for 2D and 3D:
  a 3D win cannot come from having more columns.
- **Same "up".** `cam3d` is rotated so vertical is image `−y` in both spaces. Neither
  space assumes a level camera.
- **Same classifier.** L2 logistic regression, threshold picked on val for balanced
  accuracy. A weak model is deliberate — the question is about the features.
- **Video-level bootstrap.** Frames from one clip are correlated; resampling frames
  would fake precision. Confidence intervals and arm-vs-arm deltas resample whole videos,
  paired on the same replicates.

## Reading the result

This is downstream evidence, not a depth-error measurement. Fitness-AQA cannot say how
many millimetres or degrees wrong a 2D reading is — only whether the verdict changes.
Sanity check before trusting any 3D lift: the 2D arms should cluster together. If
`nlf_2d` beats `mediapipe_2d` by a lot, something in the pipeline differs beyond depth
and the 3D gap is not interpretable.
