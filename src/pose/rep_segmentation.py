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

# Movement-agnostic floor on repetition duration, and the ONLY thing that separates a real
# excursion from an anomalous one. It works because a window measures the excursion and nothing
# else -- see `_climb_backward` for the plateau rule that makes that true. Fast cyclic movements
# (jumping jacks, high knees) legitimately run below this and must lower it — see the spec's
# §3.4 audit.
#
# There is deliberately NO separate noise-vs-range gate. Four attempts at one (median step, low
# percentile of steps, then a moving-step fraction over the clip and over the active span) each
# false-rejected some ordinary training signal -- a paused rep, an inter-rep rest, an idle
# preamble, a fast cadence -- because all four measured the DISTRIBUTION OF STEPS over a region
# that legitimately contains still frames. Duration does not have that failure mode: a pause
# makes a rep longer, never shorter, and idle time sits outside the window entirely.
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
    """Walk back from an `exit_` crossing to the top of the excursion, STOPPING AT A PLATEAU.

    The crossing is only where the hysteresis band was pierced; the rep actually begins at the
    standing/extended peak above it. Using the crossing as the boundary would drop the opening
    third of every rep outside the window.

    The climb requires each earlier frame to be STRICTLY higher (`>`, not `>=`), and that
    strictness is load-bearing rather than a detail. A run of equal values is not part of the
    excursion -- it is the athlete standing there, before the set or between reps -- and its
    LAST frame, the one the descent leaves from, is the top just as much as its first is. A
    non-strict climb cannot tell the two apart, so it walks the whole flat run and the window
    inherits every idle frame in front of the rep. That is how a one-frame detection glitch in
    an otherwise static clip used to become a "rep" spanning the entire clip: the glitch's own
    deep run is a single frame, but the window climbed outward across all 90 flat frames before
    its length was ever measured, so the `min_frames` filter in `_finalize` -- which is a
    duration gate on the repetition -- was handed the clip's duration instead of the
    excursion's. Stopping at the plateau makes a window's length mean what `min_frames` assumes
    it means, and that single change is what lets an anomalous blip be rejected on duration
    without any noise estimate, and therefore without the false rejections every noise estimate
    tried here brought with it.

    Two consequences, both intended: an idle preamble and an inter-rep rest now fall OUTSIDE
    every window instead of being absorbed into the neighbouring rep (they are rest, and phase
    assignment should not see them as setup), and a repeated frame mid-descent -- a dropped
    capture -- stops the climb early, trimming a few frames off the top of that one window.
    """
    while index > 0 and np.isfinite(values[index - 1]) and values[index - 1] > values[index]:
        index -= 1
    return index


def _climb_forward(values: np.ndarray, index: int) -> int:
    """The mirror of `_climb_backward` for the end of a rep, plateau rule included."""
    last = len(values) - 1
    while index < last and np.isfinite(values[index + 1]) and values[index + 1] > values[index]:
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
    """De-duplicate, resolve shared boundaries, drop too-short spans, and number the rest.

    Adjacent reps meet at a single frame -- the peak between them belongs to the rep that
    STARTS there -- so the earlier window gives it up. Without this, one frame would be
    phase-assigned twice and scored twice.

    The `min_frames` filter here is the module's ONLY rejection of an anomalous excursion, and
    it is a duration gate: a span is dropped when the movement it describes, measured top to
    top, is shorter than the caller's `min_rep_seconds`. It can only mean that because
    `_climb_backward` refuses to extend a window across a plateau -- see there. Note the
    boundary this implies: a dip is kept once its top-to-top extent reaches `min_frames`, which
    is roughly `min_frames - 2` frames of dip plus the frames it descends and ascends through.
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
