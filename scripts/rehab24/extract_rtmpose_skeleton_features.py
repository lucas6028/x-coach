"""Thin CLI entry point for REHAB24-6 RTMPose/RTMW 2D skeleton feature extraction."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rehab24.rtmpose_skeleton_features import main

if __name__ == "__main__":
    main()
