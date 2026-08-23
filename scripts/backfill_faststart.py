"""Make already-stored clips streamable (thin CLI over ``backend.app.services.faststart``).

New uploads are fixed on the way in by ``analysis.stage_upload``. Clips uploaded BEFORE that
existed are not: their ``moov`` index still sits after the media, so a browser downloads the whole
file before it can show one frame and cannot seek at all until it has. That is every video in
every user's history. This script rewrites them in place.

Run it from the repository root, against whichever store the environment is configured for — R2
when the ``R2_*`` settings are present, the local filesystem store otherwise::

    .venv\\Scripts\\python.exe scripts/backfill_faststart.py --dry-run
    .venv\\Scripts\\python.exe scripts/backfill_faststart.py
    .venv\\Scripts\\python.exe scripts/backfill_faststart.py --prefix uploads/<user-id>

Start with ``--dry-run``. It downloads and remuxes exactly as a real run does and simply skips the
upload, so its counts are the counts you will get — including how many clips ffmpeg cannot handle.

Needs ffmpeg on PATH. Without it every clip is reported as failed and nothing is written, which is
the correct outcome but a slow way to discover a missing dependency.

SAFE TO RERUN. A rewritten clip is already streamable, so a second pass rewrites nothing. A clip
that cannot be remuxed is left byte-for-byte as it was — this script never deletes and never
replaces an object with a worse one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--prefix",
        default="uploads",
        help="Key prefix to walk (default: uploads). Narrow it to one user to try a small batch first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except write the result back.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Imported here, not at module scope, so `--help` works without settings or credentials.
    from backend.app.services import faststart, storage

    store = storage.get_object_store()
    print(f"Store: {type(store).__name__}   prefix: {args.prefix}   dry-run: {args.dry_run}")
    report = faststart.backfill(store, prefix=args.prefix, dry_run=args.dry_run)

    verb = "would rewrite" if args.dry_run else "rewrote"
    print(f"\nscanned            {report.scanned}")
    print(f"{verb:<18} {report.rewritten}")
    print(f"already streamable {report.already_streamable}")
    print(f"skipped            {report.skipped}")
    print(f"failed             {report.failed}")
    if report.failed_keys:
        print("\nLeft unchanged (ffmpeg could not remux, or the write failed):")
        for key in report.failed_keys:
            print(f"  {key}")
    # A non-zero exit on failures, so this is usable from a deploy step without anyone having to
    # read the output to notice that half the bucket was left alone.
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
