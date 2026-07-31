"""Replay REHAB24-6 Ex5's 174 labeled repetitions through the production lunge rules.

Thin entry point. Every decision -- loading pose JSON, slicing rep windows, estimating the
view per window, running the detector in both the production and oracle passes, and assembling
the report -- lives in `src/rehab24/lunge_rule_validation.py`.

    .venv\\Scripts\\python.exe scripts/rehab24/validate_lunge_rules.py \\
        --pose-dir data/REHAB24-6/processed/lunge_pose_json \\
        --segmentation data/REHAB24-6/Segmentation.csv \\
        --out data/REHAB24-6/processed/lunge_rule_validation.json

Add `--report-only` to re-print the report from an existing `--out` file without re-running.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rehab24.lunge_rule_validation import main


if __name__ == "__main__":
    raise SystemExit(main())
