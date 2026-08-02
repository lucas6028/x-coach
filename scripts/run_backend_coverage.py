"""Measure and print line+branch coverage for the web backend (``backend/app``).

Thin CLI wrapper around coverage.py that runs the backend test suite under measurement
and reports the percentage. Run it from the repository root:

    python scripts/run_backend_coverage.py              # term report + total %
    python scripts/run_backend_coverage.py --html       # also write htmlcov/index.html
    python scripts/run_backend_coverage.py tests/test_backend.py tests/test_backend_analysis.py

By default it measures ``tests/test_backend.py`` (the self-contained suite that mocks the
heavy ML / retrieval stack). Pass test paths as positional args to widen the run.

Requires coverage.py (``pip install coverage`` or ``pip install -r requirements.txt``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default target: the complete, dependency-light backend suite (network/ML/Supabase all mocked).
_DEFAULT_TESTS = [
    "tests/test_backend.py",
    "tests/test_analyze_pose_endpoint.py",
    "tests/test_chat_endpoint.py",
    "tests/test_backend_line_auth.py",
    "tests/test_backend_line_webhook.py",
    "tests/test_backend_admin_line.py",
    "tests/test_storage.py",
    "tests/test_upload_staging.py",
    "tests/test_upload_urls.py",
    "tests/test_delete_reaping.py",
]
# Package(s) to measure coverage for.
_SOURCE = ["backend.app"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tests", nargs="*", default=None, help="Test files/dirs to run (default: tests/test_backend.py).")
    parser.add_argument("--html", action="store_true", help="Also write an HTML report under htmlcov/.")
    parser.add_argument("--fail-under", type=float, default=0.0, help="Exit non-zero if total coverage is below this percent.")
    args = parser.parse_args(argv)

    try:
        import coverage
    except ImportError:
        print(
            "coverage.py is not installed. Install it with:\n"
            "    pip install coverage\n"
            "(it is also listed in requirements.txt).",
            file=sys.stderr,
        )
        return 2

    try:
        import pytest
    except ImportError:
        print("pytest is not installed. Install it with: pip install pytest", file=sys.stderr)
        return 2

    test_targets = args.tests or _DEFAULT_TESTS

    # Run from the repo root so ``backend.app`` / ``src`` imports resolve, mirroring how the
    # backend itself is launched.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    cov = coverage.Coverage(source=_SOURCE, branch=True)
    cov.start()
    # Import-time module code in backend.app.* must execute *after* cov.start() to be counted, so
    # let pytest do the importing (this script imports no backend module at top level).
    exit_code = pytest.main(["-q", *test_targets])
    cov.stop()
    cov.save()

    print("\n=== backend coverage (backend/app) ===")
    total = cov.report(show_missing=True)
    print(f"\nTOTAL backend coverage: {total:.1f}%")

    if args.html:
        out_dir = PROJECT_ROOT / "htmlcov"
        cov.html_report(directory=str(out_dir))
        print(f"HTML report: {out_dir / 'index.html'}")

    if exit_code != 0:
        print(f"\nWARNING: test run returned exit code {exit_code} (some tests failed/errored).", file=sys.stderr)
    if args.fail_under and total < args.fail_under:
        print(f"FAIL: coverage {total:.1f}% < required {args.fail_under:.1f}%", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
