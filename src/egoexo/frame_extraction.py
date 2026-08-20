"""Pure helpers for pulling one movement's action frames out of ``frames_open``.

The archive is a single gzip stream split into 3 GiB parts named ``frames_open.tar.gz.aa``,
``.ab``, ``.ac``, ... On this machine ``.ac`` has never been downloaded while ``.ad`` has, so the
parts on disk are ``{aa, ab, ad}``. A gzip stream cannot be resumed across a hole: only the
CONTIGUOUS PREFIX from ``.aa`` decodes, and appending ``.ad`` after the hole would feed the
decompressor bytes from the wrong offset. :func:`contiguous_prefix` therefore stops at the first
gap instead of taking everything the glob returns.

Everything here is I/O-free except :func:`concatenated_parts`, which only opens the files it is
handed, so the interesting logic is unit-testable without the 6.4 GiB archive.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# ``./frames_open/<record>/<view>/frame_0000000001.jpg`` -- the frame number is 1-based.
_MEMBER = re.compile(r"(?:^|/)frames_open/([^/]+)/([^/]+)/frame_(\d+)\.jpg$")


def parse_member_path(name: str) -> tuple[str, str, int] | None:
    """``(record_id, view, frame_index)`` for a frame member, else None.

    Returns None for directory entries, ``.DS_Store``, AppleDouble ``._`` siblings and anything
    else that is not a numbered frame -- the archive carries all of those.
    """
    match = _MEMBER.search(name)
    if match is None:
        return None
    record, view, digits = match.groups()
    if view.startswith(".") or record.startswith("."):
        return None
    return record, view, int(digits)


def part_suffix_order(path: Path) -> str:
    """The two-letter split suffix (``aa``, ``ab``, ...) of a ``frames_open`` part."""
    return path.name.rsplit(".", 1)[-1]


def contiguous_prefix(parts: Iterable[Path]) -> list[Path]:
    """The parts forming an unbroken run from ``aa``, in order; stops at the first gap.

    ``split`` names parts in strict lexical succession, so the run is defined by string
    successor, not by "everything that matched the glob". A part after a hole is unusable and
    including it would corrupt the decompressor's input rather than extend it.
    """
    by_suffix = {part_suffix_order(p): p for p in parts}
    ordered: list[Path] = []
    suffix = "aa"
    while suffix in by_suffix:
        ordered.append(by_suffix[suffix])
        suffix = _next_suffix(suffix)
    return ordered


def _next_suffix(suffix: str) -> str:
    letters = list(suffix)
    index = len(letters) - 1
    while index >= 0:
        if letters[index] != "z":
            letters[index] = chr(ord(letters[index]) + 1)
            return "".join(letters)
        letters[index] = "a"
        index -= 1
    return "a" + "".join(letters)


class _ConcatReader(io.RawIOBase):
    """Read several files end to end as one stream. Read-only, forward-only."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = list(paths)
        self._index = 0
        self._handle = open(self._paths[0], "rb") if self._paths else None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # type: ignore[override]
        while self._handle is not None:
            count = self._handle.readinto(buffer)
            if count:
                return count
            self._handle.close()
            self._index += 1
            self._handle = (
                open(self._paths[self._index], "rb")
                if self._index < len(self._paths)
                else None
            )
        return 0

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def concatenated_parts(parts: Iterable[Path]) -> io.BufferedReader:
    """A buffered stream over the contiguous prefix of ``parts``."""
    ordered = contiguous_prefix(parts)
    if not ordered:
        raise FileNotFoundError("no frames_open.tar.gz.aa part found")
    return io.BufferedReader(_ConcatReader(ordered), buffer_size=1 << 20)


@dataclass(frozen=True)
class ExtractionPlan:
    """Which (record, view, frame) triples belong to which judged action.

    ``by_record[(record, view)]`` is a list of ``(first, last, sample_id)`` INCLUSIVE frame
    windows. Inclusive on both ends deliberately: the manifest's ``st_frame``/``ed_frame`` differ
    by exactly ``num_frames_segment`` and the archive numbers frames from 1, so the two possible
    conventions (0-based half-open, 1-based half-open) are one frame apart. Taking both endpoints
    costs one frame and cannot drop a real one.
    """

    by_record: dict[tuple[str, str], list[tuple[int, int, str]]]
    expected: dict[str, int]

    def lookup(self, record: str, view: str, frame_index: int) -> list[str]:
        """Sample ids whose window contains this frame (usually zero or one; actions of the
        same record never overlap, but the caller must not assume that)."""
        windows = self.by_record.get((record, view))
        if not windows:
            return []
        return [sid for first, last, sid in windows if first <= frame_index <= last]


def build_plan(rows: Sequence[dict], views: Sequence[str]) -> ExtractionPlan:
    """Build an :class:`ExtractionPlan` from manifest rows, restricted to ``views``.

    A row's ``views`` column lists the cameras that record actually has; a requested view the
    record does not carry is skipped rather than planned and silently never filled.
    """
    by_record: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    expected: dict[str, int] = {}
    for row in rows:
        record = row["record_id"]
        available = {v.strip() for v in row["views"].split(";") if v.strip()}
        first = int(row["st_frame"])
        last = int(row["ed_frame"])
        for view in views:
            if view not in available:
                continue
            by_record.setdefault((record, view), []).append((first, last, row["sample_id"]))
            expected[f"{row['sample_id']}__{view}"] = last - first + 1
    return ExtractionPlan(by_record=by_record, expected=expected)
