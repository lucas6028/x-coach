r"""REHAB24-6 checkpoint comparison: SSv2-finetuned vs Kinetics-finetuned VideoMAE.

Plan: ``notes/rehab24_videomae_ssv2_checkpoint_validation_plan.md``. Reads the summaries
written by the framing report, the identity control and the position control for each
checkpoint and evaluates the plan's rule tables. Sections whose inputs are missing are
reported as unavailable, so this can run after the letterbox arm alone:

    .venv\Scripts\python.exe scripts/rehab24/videomae_checkpoint_report.py \
        --output data/REHAB24-6/processed/videomae_checkpoint_report.json

Logic lives in ``src/rehab24/videomae_checkpoint_report.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rehab24.videomae_checkpoint_report import main


if __name__ == "__main__":
    main()
