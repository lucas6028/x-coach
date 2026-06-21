"""Thin CLI entry point for REHAB24-6 pretrained (VideoPose3D) 2D->3D lifting.

Lifts cached MediaPipe 2D to H36M-17 3D with the pretrained VideoPose3D model and
writes the vp3d_lifted + vp3d_2d skeleton features. See
``src/rehab24/lift_2d_to_3d_pretrained.py`` for the design and required weights.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rehab24.lift_2d_to_3d_pretrained import main

if __name__ == "__main__":
    main()
