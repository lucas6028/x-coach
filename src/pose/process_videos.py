import cv2
import mediapipe as mp
import os
import json
import numpy as np
import argparse
from tqdm import tqdm

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video(input_path, output_json_path, output_video_path=None, model_complexity: int = 2):
    """
    Processes a video using MediaPipe Pose, saving landmarks to JSON and optionally an annotated video.
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return False

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    pose_data = {
        "metadata": {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames
        },
        "frames": []
    }

    landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=4)
    connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Processing video frames", leave=False)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Draw the pose annotation on the image.
            image.flags.writeable = True

            frame_data = {
                "frame_index": frame_idx,
                "landmarks": None,
                "world_landmarks": None
            }

            if results.pose_landmarks:
                # 2D normalized landmarks [0.0, 1.0]
                landmarks_list = []
                for lm in results.pose_landmarks.landmark:
                    landmarks_list.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    })
                frame_data["landmarks"] = landmarks_list

                # 3D world landmarks
                if results.pose_world_landmarks:
                    world_landmarks_list = []
                    for lm in results.pose_world_landmarks.landmark:
                        world_landmarks_list.append({
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility
                        })
                    frame_data["world_landmarks"] = world_landmarks_list

                if out_video:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=landmark_spec,
                        connection_drawing_spec=connection_spec)

            pose_data["frames"].append(frame_data)
            
            if out_video:
                out_video.write(image)

            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    if out_video:
        out_video.release()

    # Save JSON data
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(pose_data, f, separators=(',', ':'))

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to extract MediaPipe Pose landmarks")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video file path")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file path for landmarks")
    parser.add_argument("--output_video", type=str, default=None, help="Output annotated video path (optional)")
    
    args = parser.parse_args()
    
    print(f"Processing {args.input}...")
    success = process_video(args.input, args.output_json, args.output_video)
    if success:
        print(f"Successfully processed video. JSON saved to {args.output_json}")
        if args.output_video:
            print(f"Annotated video saved to {args.output_video}")
    else:
        print("Failed to process video.")
