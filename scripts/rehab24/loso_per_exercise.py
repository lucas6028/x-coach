"""Thin CLI entry point for REHAB24-6 per-exercise LOSO correctness breakdown."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rehab24.loso_per_exercise import main

if __name__ == "__main__":
    main()
