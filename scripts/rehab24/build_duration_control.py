"""Derive REHAB24-6's duration-only (n_frames) control from the box-geometry control.

    .venv\\Scripts\\python.exe scripts/rehab24/build_duration_control.py

One number per repetition, no pixels. It is the floor `background_only` has to beat
before its above-chance score can be read as scene information. See
``src/rehab24/duration_control.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rehab24.duration_control import main


if __name__ == "__main__":
    main()
