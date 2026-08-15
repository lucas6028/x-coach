"""Check that two REHAB24-6 VideoMAE framing arms are paired sample-for-sample.

    .venv\\Scripts\\python.exe scripts/rehab24/videomae_framing_pairing.py \\
        --baseline-dir data/REHAB24-6/processed/videomae_raw_full_frame_local \\
        --candidate-dir data/REHAB24-6/processed/videomae_raw_full_frame_letterbox

Must pass before any classifier is trained on the pair. See
``src/rehab24/videomae_framing_pairing.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rehab24.videomae_framing_pairing import main


if __name__ == "__main__":
    main()
