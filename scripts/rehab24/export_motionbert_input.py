"""Thin CLI: export H36M-17 2D input for the MotionBERT Kaggle lifting kernel.

Writes a compact npz of screen-normalized H36M-17 2D (from the cached MediaPipe
landmarks) to upload to Kaggle. See ``src/rehab24/motionbert_lift.py``.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rehab24.motionbert_lift import main

if __name__ == "__main__":
    main()
