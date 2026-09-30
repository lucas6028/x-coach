"""Thin CLI for the NLF 3D-view worker (home GPU PC). Logic: ``src/nlf_worker/runtime/app.py``.

Run from the repo root under the GPU venv:

    .venv-nlf\\Scripts\\python.exe scripts/nlf_worker/run_worker.py            # poll until Ctrl+C
    .venv-nlf\\Scripts\\python.exe scripts/nlf_worker/run_worker.py --once     # one poll
    .venv-nlf\\Scripts\\python.exe scripts/nlf_worker/run_worker.py --dry-run CLIP --out DIR \\
        [--movement squat] [--segment START_S END_S ...]                        # offline, no network

The polling modes read credentials from data/models/nlf/worker.env (or --env); --dry-run needs none.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlf_worker.runtime.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
