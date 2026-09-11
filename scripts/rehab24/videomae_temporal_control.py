"""REHAB24-6 VideoMAE temporal-order (16-frame shuffle) control, pre-registered.

Plan: ``notes/rehab24_videomae_temporal_shuffle_validation_plan.md``. Extract each arm
with ``extract_videomae_features.py --variant full_frame_letterbox --temporal <arm>``
into ``data/REHAB24-6/processed/videomae_raw_temporal/<arm>``, then per arm, gates
first:

    .venv\\Scripts\\python.exe scripts/rehab24/videomae_temporal_control.py gates --arm frame_identity
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_temporal_control.py materialize --arm frame_identity
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_temporal_control.py predict --arm frame_identity --device cpu
    .venv\\Scripts\\python.exe scripts/rehab24/videomae_temporal_control.py analyze --arm frame_identity   # G3: must be 0.8741

``frame_identity`` must pass G3 before any other arm is extracted. Then the same four
steps for frame_shuffle (primary), frame_reverse, tubelet_shuffle, rep_static_frame, and:

    .venv\\Scripts\\python.exe scripts/rehab24/videomae_temporal_control.py paired

Logic lives in ``src/rehab24/videomae_temporal_control.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rehab24.videomae_temporal_control import main


if __name__ == "__main__":
    main()
