"""Thin CLI entry point for the REHAB24-6 Vicon 2D->3D lifting experiment.

Trains a temporal lifter on the native 26-joint Vicon 2D/3D arrays and writes the
lifted3d-vicon + vicon2d skeleton features. See ``src/rehab24/lift_2d_to_3d_vicon.py``.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rehab24.lift_2d_to_3d_vicon import main

if __name__ == "__main__":
    main()
