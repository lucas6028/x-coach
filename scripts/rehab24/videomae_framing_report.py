"""Paired LOSO across REHAB24-6 VideoMAE framing arms.

    .venv\\Scripts\\python.exe scripts/rehab24/videomae_framing_report.py \\
        --arm full_frame=data/REHAB24-6/processed/videomae_framing/full_frame/videomae_mean_pool_fc_norm_mean \\
        --arm full_frame_letterbox=data/REHAB24-6/processed/videomae_framing/full_frame_letterbox/videomae_mean_pool_fc_norm_mean \\
        --primary full_frame_letterbox:full_frame --device cpu

Arms are explicit name=path pairs because every variant's features share one leaf
directory name. See ``src/rehab24/videomae_framing_report.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rehab24.videomae_framing_report import main


if __name__ == "__main__":
    main()
