import os
import subprocess
import glob
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch process videos for MediaPipe poses")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    parser.add_argument("--no-video", action="store_true", help="Do not generate annotated videos")
    args = parser.parse_args()

    base_dir = Path(r"c:\Users\Hao\Code\my-project\x-coach")
    video_dir = base_dir / "data" / "Squat" / "Unlabeled_Dataset" / "videos"
    output_dir = base_dir / "data" / "Squat" / "Unlabeled_Dataset" / "processed_poses"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process videos
    video_files = glob.glob(str(video_dir / "**" / "*.mp4"), recursive=True)
    print(f"Found {len(video_files)} videos to process.")
    
    script_path = base_dir / "src" / "process_videos.py"
    
    for i, video_file in enumerate(video_files):
        if args.limit is not None and i >= args.limit:
            print(f"Reached limit of {args.limit} videos. Stopping.")
            break
            
        video_path = Path(video_file)
        video_name = video_path.stem
        
        json_path = output_dir / f"{video_name}.json"
        
        if json_path.exists():
            print(f"Skipping {video_name}, already processed.")
            continue
            
        print(f"\nProcessing {video_name}...")
        
        cmd = [
            sys.executable, str(script_path),
            "--input", str(video_path),
            "--output_json", str(json_path)
        ]
        
        if not args.no_video:
            annotated_video_path = output_dir / f"{video_name}_annotated.mp4"
            cmd.extend(["--output_video", str(annotated_video_path)])
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_name}: {e}")

if __name__ == "__main__":
    main()
