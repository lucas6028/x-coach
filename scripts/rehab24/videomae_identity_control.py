"""REHAB24-6 VideoMAE identity/appearance control (pre-registered).

Plan: ``notes/rehab24_videomae_identity_appearance_validation_plan.md``.
Run the primary chain first, and do not look at subgroup outcomes before it:

    .venv\\Scripts\\python.exe scripts/rehab24/videomae_identity_control.py audit --labels-only
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_identity_control.py predict --seeds 42 7 1234
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_identity_control.py audit
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_identity_control.py analyze \\
        --permutations 10000 --permutation-seed 20260823

Then the supporting appearance-only control:

    .venv\\Scripts\\python.exe scripts/rehab24/videomae_identity_control.py extract-appearance \\
        --variant canonical_frame_repeat
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_identity_control.py evaluate-appearance --seeds 42 7 1234

Logic lives in ``src/rehab24/videomae_identity_control.py`` and
``src/rehab24/videomae_appearance_only.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rehab24.videomae_identity_control import main


if __name__ == "__main__":
    main()
