"""Write frontend/src/lib/faultCitations.json from the movement detectors' citations."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose.fault_citations import main


if __name__ == "__main__":
    main()
