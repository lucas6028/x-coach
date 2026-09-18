"""Thin CLI entry point for the frozen VideoMAE-vs-V-JEPA-2 comparison on REHAB24-6."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rehab24.video_backbone_comparison import main

if __name__ == "__main__":
    main()
