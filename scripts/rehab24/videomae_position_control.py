"""REHAB24-6 VideoMAE within-class position control (pre-registered).

Plan: ``notes/rehab24_videomae_position_control_validation_plan.md``.
Run the gate first; do not read any new arm's outcome before it passes:

    .venv\\Scripts\\python.exe scripts/rehab24/videomae_position_control.py replicate-exploratory
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_position_control.py predict --k 0 --seeds 42 7 1234
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_position_control.py analyze --k 0 \\
        --permutations 10000 --permutation-seed 20260906
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_position_control.py probe --k 0

Then the primary arm, then the bound:

    .venv\\Scripts\\python.exe scripts/rehab24/videomae_position_control.py predict --k 1 --seeds 42 7 1234
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_position_control.py probe --k 1
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_position_control.py analyze --k 1 \\
        --permutations 10000 --permutation-seed 20260906
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_position_control.py drift-proxies --permutations 10000

Logic lives in ``src/rehab24/videomae_position_control.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rehab24.videomae_position_control import main


if __name__ == "__main__":
    main()
