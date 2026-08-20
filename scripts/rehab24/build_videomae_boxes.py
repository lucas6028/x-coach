"""Build REHAB24-6's fixed per-video person boxes for the VideoMAE framing arms.

    .venv\\Scripts\\python.exe scripts/rehab24/build_videomae_boxes.py

One box per source video, from the dataset's own mocap 2D skeletons -- never one box
per repetition. See ``src/rehab24/videomae_boxes.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rehab24.videomae_boxes import main


if __name__ == "__main__":
    main()
