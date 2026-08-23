"""Make a stored clip streamable — the one-time remux that turns a download into playback.

The problem this exists for is not bandwidth, it is WHERE THE INDEX LIVES.

An MP4 keeps its seek table in the ``moov`` atom. Muxers that do not know the final duration up
front (every phone camera, every ``ffmpeg`` writing to a pipe) can only write ``moov`` once the
media is finished, so it lands AFTER the multi-megabyte ``mdat``. A browser cannot decode a single
frame until it has that atom, so ``<video src=...>`` against such a file downloads the whole clip
before the first frame appears and cannot seek at all before then — the exact "it buffers the
entire video" symptom that separates this from how YouTube feels. ``-movflags +faststart`` moves
``moov`` to the front; the browser then reads a few KB of header and range-requests only the bytes
it is about to play. Every demo clip in ``data/demo/`` is moov-last, so this is measured, not
theoretical.

WebM has the same shape of problem for a different reason: ``MediaRecorder`` live-muxes with an
UNKNOWN segment size and no ``Cues`` index (this is the same defect behind ``duration ===
Infinity``, see ``frontend/src/lib/useVideoPlayback.ts``). Such a file plays but cannot be seeked.
Remuxing to a file — where the size IS known — writes both.

Two properties are load-bearing:

* **Stream copy only, never a re-encode.** ``SkeletonOverlay`` maps landmarks onto the video's
  ``currentTime``, and the detections carry frame indices from the ORIGINAL file. A re-encode may
  resample timestamps or drop frames, which would silently drift the skeleton away from the body.
  ``-c copy`` moves bytes and rewrites the index; the timebase is preserved.
* **Every failure degrades to the original bytes.** ffmpeg is in ``backend/Dockerfile`` but is not
  guaranteed on a development machine, and an exotic container may simply refuse to remux. A clip
  that streams poorly is worth having; an upload that 500s is not.
"""

from __future__ import annotations

import logging
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# How long a stream copy may run before it is abandoned and the original bytes are kept. A copy is
# I/O bound — a phone clip finishes in well under a second — so anything near this is a hang, not
# slow work, and waiting longer only delays an analysis that will succeed anyway.
REMUX_TIMEOUT_SECONDS = 60

# Suffixes worth inspecting at all. Anything else is passed through untouched rather than handed
# to ffmpeg on the chance it helps.
_MP4_SUFFIXES = frozenset({".mp4", ".m4v", ".mov"})
_WEBM_SUFFIXES = frozenset({".webm"})

# Reading the top-level atom chain only needs the first few headers, and a `mdat` is skipped by
# its size rather than read. This bounds how much of an upload the scan touches when a file is
# malformed enough that the chain does not terminate.
_MAX_ATOMS_SCANNED = 64

# The EBML "unknown size" vint that a live mux writes for its Segment element: one length byte
# followed by seven 0xFF. A file muxed to disk carries a real size here.
_EBML_UNKNOWN_SIZE = b"\x01\xff\xff\xff\xff\xff\xff\xff"


def _iter_mp4_atoms(data: bytes):
    """Yield ``(type, offset)`` for each TOP-LEVEL atom, stopping at the first malformed header.

    Deliberately does not recurse: the only question here is whether ``moov`` precedes ``mdat``,
    which is decided entirely at the top level. Not recursing also means a corrupt inner box
    cannot make this raise.
    """
    offset = 0
    total = len(data)
    for _ in range(_MAX_ATOMS_SCANNED):
        if offset + 8 > total:
            return
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        atom_type = data[offset + 4 : offset + 8]
        header = 8
        if size == 1:
            # 64-bit size: the real length follows the type as an 8-byte big-endian integer.
            if offset + 16 > total:
                return
            size = struct.unpack(">Q", data[offset + 8 : offset + 16])[0]
            header = 16
        elif size == 0:
            # "Extends to end of file" — always the last atom, so its type is the final answer.
            yield atom_type, offset
            return
        if size < header:
            return
        yield atom_type, offset
        offset += size


def needs_faststart(data: bytes) -> bool:
    """True when this MP4's ``moov`` index sits after its media, i.e. playback must wait.

    A file with no ``moov`` at all in the scanned prefix is reported as needing the move: either
    the index is far past the atoms we walked (which is the very condition being detected) or the
    file is not the MP4 its suffix claims, and ffmpeg failing on it is already a no-op.
    """
    seen_mdat = False
    for atom_type, _ in _iter_mp4_atoms(data):
        if atom_type == b"moov":
            return seen_mdat
        if atom_type == b"mdat":
            seen_mdat = True
    return seen_mdat


def needs_webm_index(data: bytes) -> bool:
    """True when this WebM was live-muxed, so it carries no ``Cues`` and cannot be seeked.

    Detected from the Segment element's unknown-size marker rather than by hunting for ``Cues``:
    the marker sits in the first few dozen bytes, while ``Cues`` may legitimately live at the end
    of a large file that we would then have to read in full to rule out.
    """
    return _EBML_UNKNOWN_SIZE in data[:1024]


def _remux(data: bytes, suffix: str, extra_args: list[str]) -> bytes:
    """Run one stream copy through ffmpeg, returning the original bytes on any failure."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.info("ffmpeg is not available; storing %s upload as received", suffix)
        return data
    with tempfile.TemporaryDirectory(prefix="faststart_") as tmp:
        src = Path(tmp) / f"in{suffix}"
        dst = Path(tmp) / f"out{suffix}"
        src.write_bytes(data)
        cmd = [
            ffmpeg,
            "-nostdin",
            "-loglevel", "error",
            "-y",
            "-i", str(src),
            "-c", "copy",
            "-map", "0",
            *extra_args,
            str(dst),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=REMUX_TIMEOUT_SECONDS, check=False
            )
        except (subprocess.TimeoutExpired, OSError):
            logger.exception("Remux of a %s upload failed to run; keeping the original", suffix)
            return data
        if proc.returncode != 0 or not dst.is_file():
            logger.warning(
                "Remux of a %s upload exited %s; keeping the original. stderr: %s",
                suffix, proc.returncode, proc.stderr.decode("utf-8", "replace")[-500:],
            )
            return data
        out = dst.read_bytes()
    # A remux that produced nothing (or an implausibly tiny file) means ffmpeg wrote a header and
    # gave up on the media. Trusting the byte count here is what keeps a silent failure from
    # replacing a playable clip with an unplayable stub.
    if len(out) < 1024:
        logger.warning("Remux of a %s upload produced %d bytes; keeping the original", suffix, len(out))
        return data
    return out


def is_streamable(data: bytes, suffix: str) -> bool:
    """Whether a browser can already start and seek this clip without fetching all of it.

    Trivially true for any container this module does not act on: "we have nothing to fix here"
    and "this is fine" are the same answer to every caller.

    Split out from ``optimize_for_streaming`` because that function returns the ORIGINAL bytes for
    two different reasons — nothing to do, and the remux failed — and the backfill has to tell
    those apart to report honestly.
    """
    suffix = suffix.lower()
    if suffix in _MP4_SUFFIXES:
        return not needs_faststart(data)
    if suffix in _WEBM_SUFFIXES:
        return not needs_webm_index(data)
    return True


def optimize_for_streaming(data: bytes, suffix: str) -> bytes:
    """Return ``data`` rewritten so a browser can start playing and seek it without the whole file.

    A no-op — same object back, no subprocess — for a clip that is already streamable, which is
    the common case for anything already processed by a desktop encoder. Only the two containers
    with a known index-placement failure are inspected at all.
    """
    suffix = suffix.lower()
    if is_streamable(data, suffix):
        return data
    if suffix in _WEBM_SUFFIXES:
        # No flag needed: writing to a seekable file is itself what makes the muxer emit a real
        # Segment size and a Cues index.
        return _remux(data, suffix, [])
    return _remux(data, suffix, ["-movflags", "+faststart"])


# ---------------------------------------------------------------------------------------------
# Backfill: the same treatment for clips that were already in the bucket before this existed.
# ---------------------------------------------------------------------------------------------

# The stored source object carries no file extension (see this module's key-layout note), so the
# suffix the remux needs has to come from the ContentType R2 replays. The reverse of
# ``storage.video_content_type``, restricted to what this module can actually act on.
_SUFFIX_BY_CONTENT_TYPE = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}


@dataclass
class BackfillReport:
    """What one backfill pass did. Every counter is reported, including the ones nobody wants.

    ``failed`` is the honest half: a clip ffmpeg could not remux is left exactly as it was and
    counted here, so a run that could not fix everything cannot be mistaken for one that did.
    """

    scanned: int = 0
    rewritten: int = 0
    already_streamable: int = 0
    skipped: int = 0
    failed: int = 0
    rewritten_keys: list[str] = field(default_factory=list)
    failed_keys: list[str] = field(default_factory=list)


def backfill(store, *, prefix: str = "uploads", dry_run: bool = False) -> BackfillReport:
    """Rewrite every already-stored source clip that a browser cannot stream.

    ``stage_upload`` only fixes clips on the way IN, which leaves every video uploaded before this
    existed exactly as unplayable-until-fully-downloaded as it was. That is not a small residue:
    it is the entire existing history of every user, and it is what someone testing "is playback
    faster now?" on an old analysis would actually hit.

    Only ``.../source`` objects are considered — pose JSON and thumbnails share the prefix and are
    not video. A clip whose ContentType is not one this module handles is SKIPPED, not guessed at.

    ``dry_run`` skips only the PUT — it still downloads and still remuxes, so the report
    distinguishes clips that would be fixed from clips ffmpeg cannot handle. A cheaper dry run
    that only counted moov-last clips would answer a question nobody has; the one worth asking
    before rewriting a whole bucket is "would this actually work", and only doing the work
    answers it.

    Idempotent: a second run finds every clip already streamable and rewrites nothing.
    """
    report = BackfillReport()
    for key in store.iter_keys(prefix):
        if not key.endswith("/source"):
            continue
        report.scanned += 1
        found = store.get(key)
        if found is None:
            # Listed but gone: a user deleted the analysis between the listing and here.
            report.skipped += 1
            continue
        data, content_type = found
        suffix = _SUFFIX_BY_CONTENT_TYPE.get(content_type.split(";")[0].strip().lower())
        if suffix is None:
            logger.info("Skipping %s: content type %r is not remuxable here", key, content_type)
            report.skipped += 1
            continue
        if is_streamable(data, suffix):
            report.already_streamable += 1
            continue
        optimized = optimize_for_streaming(data, suffix)
        # ``optimize_for_streaming`` hands back the original on every failure, which is the right
        # behaviour on the upload path and the wrong thing to COUNT as success here: the clip was
        # known to need work a line ago, so unchanged bytes mean ffmpeg could not do it.
        if optimized is data:
            report.failed += 1
            report.failed_keys.append(key)
            continue
        if dry_run:
            report.rewritten += 1
            report.rewritten_keys.append(key)
            continue
        try:
            store.put(key, optimized, content_type=content_type)
        except Exception:  # noqa: BLE001 — one unwritable object must not end the pass
            logger.exception("Failed to rewrite %s; leaving the original in place", key)
            report.failed += 1
            report.failed_keys.append(key)
            continue
        report.rewritten += 1
        report.rewritten_keys.append(key)
    return report
