from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.egoexo.agreement import compute_agreement_main


if __name__ == "__main__":
    compute_agreement_main()
