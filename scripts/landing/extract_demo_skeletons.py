"""Extract MediaPipe Pose landmarks for the landing-page demo clips.

Runs over the web-optimized clips in ``frontend/public/demo/*.mp4`` and writes a
compact per-clip JSON (``<name>.pose.json``) next to them, consumed by the
landing ``SkeletonStage`` overlay. Output is intentionally small: normalized
[x, y, visibility] per landmark, rounded, with ``null`` for undetected frames.

This is a one-off asset generator, not part of the core pipeline. The repo's main
``.venv`` could not run mediapipe (corrupted packages), so use a clean throwaway env:

    python -m venv .venv-demo
    .venv-demo/Scripts/python -m pip install --only-binary=:all: \
        numpy "protobuf==4.25.9" opencv-contrib-python matplotlib absl-py \
        attrs flatbuffers sounddevice
    .venv-demo/Scripts/python -m pip install --no-deps "mediapipe==0.10.14"
    .venv-demo/Scripts/python scripts/landing/extract_demo_skeletons.py
"""

import json
from pathlib import Path

import cv2
import mediapipe as mp

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "frontend" / "public" / "demo"
CLIPS = ["squat", "pushups", "highknee", "situps"]

mp_pose = mp.solutions.pose


def extract(clip: str) -> None:
    src = DEMO_DIR / f"{clip}.mp4"
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list = []
    detected = 0
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            image.flags.writeable = False
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                detected += 1
                frames.append(
                    [
                        [round(lm.x, 4), round(lm.y, 4), round(lm.visibility, 2)]
                        for lm in results.pose_landmarks.landmark
                    ]
                )
            else:
                frames.append(None)
    cap.release()

    out = DEMO_DIR / f"{clip}.pose.json"
    payload = {"fps": round(fps, 3), "width": width, "height": height, "frames": frames}
    out.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = out.stat().st_size / 1024
    print(f"{clip:10s} {len(frames):4d} frames, {detected:4d} detected -> {out.name} ({size_kb:.0f} KB)")


def main() -> None:
    for clip in CLIPS:
        extract(clip)


if __name__ == "__main__":
    main()
