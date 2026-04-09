# MediaPipe Video Processing Pipeline Walkthrough

## Summary of Changes
- Created [src/process_videos.py](file:///c:/Users/Hao/Code/my-project/x-coach/src/process_videos.py): The core script that uses `cv2` and `mediapipe` to process videos frame-by-frame and extract the 33 2D and 3D pose landmarks into a structured JSON file.
- Created [scripts/run_pose_extraction.py](file:///c:/Users/Hao/Code/my-project/x-coach/scripts/run_pose_extraction.py): A batch runner script that extracts the dataset from [videos.zip](file:///c:/Users/Hao/Code/my-project/x-coach/data/Squat/Unlabeled_Dataset/videos.zip) and iterates through all extracted [.mp4](file:///c:/Users/Hao/Code/my-project/x-coach/data/Squat/Unlabeled_Dataset/processed_poses/25195_3_annotated.mp4) files.
- Added support for `--limit` and `--no-video` flags in the batch script to allow for quick testing and space-saving when running on the entire 4,970 video dataset.
- Added dependency resolution to the `.venv` by installing `mediapipe==0.10.14` and `opencv-python`. An older version of MediaPipe was used to avoid the loss of the `solutions` API in newer versions on Windows.

## Verification
- Extracted the Squat unlabeled dataset.
- Ran the pipeline for a single sample video (`25195_3.mp4`).
- **Results**:
  - Successfully outputted [data/Squat/Unlabeled_Dataset/processed_poses/25195_3.json](file:///c:/Users/Hao/Code/my-project/x-coach/data/Squat/Unlabeled_Dataset/processed_poses/25195_3.json) containing framewise metadata and localized MediaPipe `landmarks` and `world_landmarks`. File size: ~620 KB.
  - Successfully outputted an annotated test video [25195_3_annotated.mp4](file:///c:/Users/Hao/Code/my-project/x-coach/data/Squat/Unlabeled_Dataset/processed_poses/25195_3_annotated.mp4) with a drawn skeleton overlay showing the pose detection functionality. File size: ~2.1 MB.

## Next Steps
You can process all the videos in your entire dataset by running:
> [!TIP]
> From the project root, run:
> `.venv\Scripts\python.exe scripts\run_pose_extraction.py --no-video`

The `--no-video` flag is recommended when running all 4,970 videos to save disk space and processing time. If you only want to process a small number (e.g., 100), you can use the `--limit 100` flag.
