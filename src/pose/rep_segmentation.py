"""Split a movement clip into repetitions from a single 1-D metric series.

WHY THIS EXISTS. Every `assign_phases` in `src/pose/movements/` was written as if a clip
contained exactly one repetition: a global argmin for the bottom frame, global percentile
thresholds, and hard 15%/85% slices for setup/lockout. On a multi-rep clip that mislabels
every rep after the first. Segmenting first makes those same functions correct, unchanged,
by applying them to one rep at a time.

PORTABILITY CONTRACT (see the RS-SP1 spec §7). RS-SP2 reimplements this in TypeScript so the
browser can decide which spans of a recording to extract densely. Therefore: pure functions,
no I/O, no clock, no global state, and every threshold is a named constant below — never an
inline literal. `tests/fixtures/rep_segmentation_cases.json` pins both implementations to the
same outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Robust bounds for the signal's dynamic range. Percentiles rather than min/max so one
# mis-detected frame cannot define the range the thresholds are derived from.
PERCENTILE_LOW = 5.0
PERCENTILE_HIGH = 95.0

# Hysteresis band, as fractions of the dynamic range measured from the effort-peak end.
# Two thresholds rather than one: a single threshold would split one rep into several
# whenever the signal wobbles across it near the bottom.
ENTER_FRACTION = 0.35
EXIT_FRACTION = 0.65

# A static-but-jittery signal has a frame-to-frame step of roughly uniform size everywhere,
# so its typical step is a fair estimate of pure noise. A genuine excursion is not uniform: it
# decelerates through its turnarounds (top and bottom), so its CALMEST frame-to-frame steps
# approach the same small size jitter has, even while its fast-moving frames are large. Using
# the median step would conflate that fast-moving portion with noise and, on a fast-cadence
# movement sampled at only a few frames per rep, wrongly call the whole excursion noise. So
# noise is read from the low percentile of steps (the calm ones) via `NOISE_PERCENTILE`, and
# compared against the signal's dynamic range: below this ratio the range is noise, and
# segmenting it would invent repetitions.
NOISE_PERCENTILE = 5.0
MIN_RANGE_TO_NOISE = 6.0

# Movement-agnostic floor on repetition duration. Fast cyclic movements (jumping jacks, high
# knees) legitimately run below this and must lower it — see the spec's §3.4 audit.
DEFAULT_MIN_REP_SECONDS = 0.4

_POLARITIES = ("min", "max")
_REP_STARTS = ("extended", "flexed")


@dataclass(frozen=True)
class RepWindow:
    """One repetition.

    `start`/`end` are inclusive POSITIONS IN THE PASSED SEQUENCE, not `frame_index` values —
    the caller holds the mapping back to frames. `index` is 1-based because it is what a user
    is told ("your 3rd rep").
    """

    index: int
    start: int
    end: int
    partial: bool


def _oriented(signal: Sequence[float], polarity: str, rectify: bool) -> np.ndarray:
    """Normalise any movement's signal to the convention "the effort peak is a LOW value"."""
    values = np.asarray(signal, dtype=np.float64).copy()
    if rectify:
        # A bipolar signal (torso twist: centre -> A -> centre -> B) has two excursions in
        # opposite directions. Rectifying makes each swing its own excursion from zero.
        values = np.abs(values)
    if polarity == "max":
        values = -values
    return values


def _runs_at_or_below(values: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Maximal inclusive runs where the signal is at/below `threshold`, skipping NaN."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue  # an unmeasurable frame neither opens nor closes a run
        if value <= threshold and start is None:
            start = index
        elif value > threshold and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _last_at_or_above(values: np.ndarray, threshold: float, before: int) -> int | None:
    for index in range(before - 1, -1, -1):
        if np.isfinite(values[index]) and values[index] >= threshold:
            return index
    return None


def _first_at_or_above(values: np.ndarray, threshold: float, after: int) -> int | None:
    for index in range(after + 1, len(values)):
        if np.isfinite(values[index]) and values[index] >= threshold:
            return index
    return None


def _climb_backward(values: np.ndarray, index: int) -> int:
    """Walk back from an `exit_` crossing to the top of the excursion.

    The crossing is only where the hysteresis band was pierced; the rep actually begins at the
    standing/extended peak above it. Using the crossing as the boundary would drop the opening
    third of every rep outside the window.
    """
    while index > 0 and np.isfinite(values[index - 1]) and values[index - 1] >= values[index]:
        index -= 1
    return index


def _climb_forward(values: np.ndarray, index: int) -> int:
    """The mirror of `_climb_backward` for the end of a rep."""
    last = len(values) - 1
    while index < last and np.isfinite(values[index + 1]) and values[index + 1] >= values[index]:
        index += 1
    return index


def segment_reps(
    signal: Sequence[float],
    *,
    fps: float,
    polarity: str = "min",
    rectify: bool = False,
    rep_start: str = "extended",
    min_rep_seconds: float = DEFAULT_MIN_REP_SECONDS,
) -> list[RepWindow]:
    """Segment `signal` into repetitions.

    Returns `[]` — never a guess — when the signal carries no repetition structure. The caller
    is required to fall back to whole-clip analysis in that case, NOT to report no faults.
    """
    if polarity not in _POLARITIES:
        raise ValueError(f"polarity must be one of {_POLARITIES}, got {polarity!r}")
    if rep_start not in _REP_STARTS:
        raise ValueError(f"rep_start must be one of {_REP_STARTS}, got {rep_start!r}")

    values = _oriented(signal, polarity, rectify)
    finite = values[np.isfinite(values)]
    min_frames = max(3, int(round(min_rep_seconds * max(fps, 1.0))))
    if finite.size < 2 * min_frames:
        return []

    low = float(np.percentile(finite, PERCENTILE_LOW))
    high = float(np.percentile(finite, PERCENTILE_HIGH))
    span = high - low
    if span <= 0.0:
        return []
    noise = float(np.percentile(np.abs(np.diff(finite)), NOISE_PERCENTILE))
    if noise > 0.0 and span < MIN_RANGE_TO_NOISE * noise:
        return []

    enter = low + ENTER_FRACTION * span
    exit_ = low + EXIT_FRACTION * span
    deep_runs = _runs_at_or_below(values, enter)
    if not deep_runs:
        return []

    if rep_start == "flexed":
        return _windows_from_valleys(values, deep_runs, min_frames)
    return _windows_from_plateaus(values, deep_runs, exit_, min_frames)


def _windows_from_plateaus(
    values: np.ndarray, deep_runs: list[tuple[int, int]], exit_: float, min_frames: int
) -> list[RepWindow]:
    """Boundaries at the EXTENDED end: a rep runs standing -> bottom -> standing.

    Two deep runs with no return above `exit_` between them are one rep, not two — they
    produce the same (start, end) pair here and collapse in the de-duplication below. That
    collapse is the whole point of the hysteresis band.
    """
    spans: list[tuple[int, int, bool]] = []
    for deep_start, deep_end in deep_runs:
        before = _last_at_or_above(values, exit_, deep_start)
        after = _first_at_or_above(values, exit_, deep_end)
        # Cross the band to identify the rep, then climb to the peak to bound it. Two jobs,
        # two mechanisms -- see `_climb_backward`.
        start = 0 if before is None else _climb_backward(values, before)
        end = len(values) - 1 if after is None else _climb_forward(values, after)
        spans.append((start, end, before is None or after is None))
    return _finalize(spans, min_frames)


def _windows_from_valleys(
    values: np.ndarray, deep_runs: list[tuple[int, int]], min_frames: int
) -> list[RepWindow]:
    """Boundaries at the FLEXED end: a rep runs floor -> lockout -> floor (deadlift).

    The span before the first valley and the span after the last are incomplete reps by
    construction, so they are emitted as partial rather than silently dropped.
    """
    valleys: list[int] = []
    for deep_start, deep_end in deep_runs:
        window = values[deep_start : deep_end + 1]
        offset = int(np.nanargmin(np.where(np.isfinite(window), window, np.inf)))
        valleys.append(deep_start + offset)

    spans: list[tuple[int, int, bool]] = []
    if valleys[0] > 0:
        spans.append((0, valleys[0] - 1, True))
    for earlier, later in zip(valleys, valleys[1:]):
        spans.append((earlier, later - 1, False))
    if valleys[-1] < len(values) - 1:
        spans.append((valleys[-1], len(values) - 1, True))
    return _finalize(spans, min_frames)


def _finalize(spans: list[tuple[int, int, bool]], min_frames: int) -> list[RepWindow]:
    """De-duplicate, resolve shared boundaries, drop noise-length spans, and number the rest.

    Adjacent reps meet at a single frame -- the peak between them belongs to the rep that
    STARTS there -- so the earlier window gives it up. Without this, one frame would be
    phase-assigned twice and scored twice.
    """
    unique: list[tuple[int, int, bool]] = []
    seen: set[tuple[int, int]] = set()
    for span in sorted(spans):
        if span[:2] in seen:
            continue
        seen.add(span[:2])
        unique.append(span)

    windows: list[RepWindow] = []
    for position, (start, end, partial) in enumerate(unique):
        if position + 1 < len(unique):
            end = min(end, unique[position + 1][0] - 1)
        if end - start + 1 < min_frames:
            continue
        windows.append(RepWindow(index=len(windows) + 1, start=start, end=end, partial=partial))
    return windows
