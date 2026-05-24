# UI-PRMD vs. MediaPipe 
• Not exactly. UI-PRMD skeleton data and MediaPipe pose output are both human pose time series, but they are not the same representation.

  UI-PRMD provides 3D skeleton data from Kinect and Vicon:

  - positions: 3D joint coordinates over time.
  - angles: joint-angle features derived from the skeleton.
  - Kinect data is depth-camera based.
  - Vicon data is marker-based motion capture and usually cleaner/more accurate.

  MediaPipe Pose usually gives:

  - 33 body landmarks.
  - 2D image landmarks: x, y, plus visibility.
  - Optional normalized/world landmarks: x, y, z, visibility.
  - A different joint set from Kinect/Vicon.
  - Coordinates in a different reference system and scale.

  So the raw data is not directly interchangeable. The joint names, joint count, coordinate frame, scaling, and noise characteristics differ.

  You can use UI-PRMD for this project if your model is designed around skeleton time series, but you should treat it as a pretraining /
  prototyping / benchmarking dataset, not as a perfect drop-in replacement for MediaPipe data.

  A practical way to use it:

  1. Pick a common subset of joints shared by UI-PRMD and MediaPipe, such as shoulders, elbows, wrists, hips, knees, ankles, neck/head
     approximation.
  2. Convert both datasets into the same feature format.
  3. Normalize coordinates, for example center on pelvis/hip midpoint and scale by torso length or shoulder-hip distance.
  4. Prefer relative features over raw coordinates: joint angles, limb vectors, pairwise distances, temporal velocity.
  5. Train/test carefully because UI-PRMD comes from rehab exercises, while your project may involve a different movement domain.

  For this project, the safest usage is:

  - Use UI-PRMD to train movement-quality ideas: correct vs incorrect, repetition-level classification, skeleton sequence modeling.
  - Use MediaPipe data from your actual videos for final calibration or fine-tuning.
  - Do not assume a model trained directly on UI-PRMD Kinect/Vicon coordinates will work well on MediaPipe landmarks without adaptation.

  In short: compatible conceptually, incompatible directly. Use it after converting both sources into a shared pose-feature representation.

#
# Implementation Plan
 
 Yes. You can collect similar skeleton time-series data without Kinect or Vicon. The best practical path is to use markerless pose estimation
  from normal RGB video, then convert the output into a UI-PRMD-like format.

  For this project, I would use MediaPipe Pose Landmarker first.

  It gives you 33 body landmarks per frame, including normalized image coordinates and optional 3D world landmarks in meters with the hip midpoint
  as origin. That is close enough for an AI-coach prototype if you normalize and convert it consistently. Source: Google MediaPipe Pose Landmarker
  docs.

  Recommended Setup
  Use:

  - Phone or webcam video, 30 FPS if possible.
  - One camera for easiest collection.
  - MediaPipe Pose Landmarker for skeleton extraction.
  - Store one CSV/NPY/JSON file per repetition or exercise trial.
  - Normalize all poses around the hip midpoint.
  - Convert landmarks into joint angles, limb vectors, and relative distances.

  A UI-PRMD-like row/frame could look like:

  frame,timestamp,
  left_shoulder_x,left_shoulder_y,left_shoulder_z,
  left_elbow_x,left_elbow_y,left_elbow_z,
  left_wrist_x,left_wrist_y,left_wrist_z,
  ...
  label_exercise,label_correctness,subject_id,rep_id

  For your project, I would not try to exactly reproduce Vicon. Instead, define your own common format:

```json
  sample = {
    "fps": 30,
    "source": "mediapipe",
    "exercise": "squat",
    "correctness": "correct",
    "landmarks": [num_frames, num_joints, 4],
    "features": {
      "angles": [num_frames, num_angles],
      "velocities": [num_frames, num_joints, 3]
    }
  }
```


  The 4 channels are usually:

  x, y, z, visibility

  Tool Options

| Tool | Hardware | Output | Best For |
| :--- | :--- | :--- | :--- |
| MediaPipe Pose | phone/webcam | 33 landmarks, 2D + pseudo/world 3D | easiest project data collection |
| OpenPose | webcam/video | 2D body keypoints, optional body/hand/face | research-style 2D pose data |
| MMPose | webcam/video + GPU preferred | 2D/3D pose models | more flexible, more setup |
| Pose2Sim | 2+ cameras preferred | triangulated 3D + OpenSim-style kinematics | closest to biomechanics/Vicon-like workflow |


If you want data closer to Vicon/Kinect 3D, use multi-camera recording + Pose2Sim. Pose2Sim is designed for markerless 3D kinematics from calibrated cameras and connects pose estimation with OpenSim-style biomechanical modeling. It is more work, but it gives more physically meaningful 3D motion than a single webcam.

  Best Recommendation
  For this project:
  
  1.  Start with MediaPipe Pose using one phone camera.
  2. Collect correct/incorrect examples for each exercise.
  3. Convert landmarks to normalized joint-angle features.
  4. Train on those features instead of raw x,y,z.
  5. Later, if accuracy is not enough, upgrade to two-camera Pose2Sim.

  This gives you the same kind of usable information as UI-PRMD: skeleton movement over time plus correctness labels, without needing `Kinect` or `Vicon`.
