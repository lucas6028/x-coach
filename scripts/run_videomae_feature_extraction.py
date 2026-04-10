from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.videomae_feature_extraction import main


if __name__ == "__main__":
    # Default entrypoint for the squat dataset used in this repo.
    # Override the CLI arguments if you want to point at another split or video root.
    main()
