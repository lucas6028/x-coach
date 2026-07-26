# RS-SP1 逐 rep 規則偵測 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Segment a clip into repetitions, assign phases and run fault rules per repetition (default: the first, middle and last of at most 3), and merge repeated faults — instead of treating the whole clip as one repetition.

**Architecture:** A new pure-function segmenter (`src/pose/rep_segmentation.py`) turns a 1-D metric series into rep windows using dynamic-range-relative hysteresis thresholds. `MovementDetector` gains five data fields naming which metric to segment on and how; `run_detector` computes raw metrics and smoothing globally (unchanged), then assigns phases per rep window and runs each rule on each *selected* window's slice. Frames outside every rep get phase `"rest"`, which no rule's active-phase gate matches, so walk-in/rack/rest frames stop being scored. Detections carry the rep they came from and are merged per `fault_id`. Zero detected reps falls back to today's whole-clip behavior.

**Tech Stack:** Python 3.11/3.12, numpy, `unittest.TestCase` under `tests/`, FastAPI (backend), pytest.

## Global Constraints

- **Python interpreter is always `.venv\Scripts\python.exe` from the repo root.** Never bare `python`/`pip`, never `source .venv/bin/activate` (POSIX-only, fails on this Windows machine).
- **Tests always scoped to `tests/`:** `.venv\Scripts\python.exe -m pytest tests/`
- **Coverage gate (CI enforces 95%):** `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
- **Run everything from the repository root.** Modules import by absolute package path (`from src.pose... import ...`).
- **Dependency-light style:** stdlib + numpy only in `src/pose/`. No scipy.
- **Tests are `unittest.TestCase` classes under `tests/`.**
- **`src/pose/rep_segmentation.py` must stay a pure function module** — no file I/O, no clock, no global state, nothing beyond numpy. RS-SP2 ports it to TypeScript; every threshold must be a named module constant, never an inline literal. (Spec §7)
- **Payload changes are additive only.** Do not change `quality`'s existing denominators and do not reduce the number of rows in `frame_metrics`. (Spec §5)
- **Never return empty detections on a segmentation failure** — always fall back to whole-clip analysis. Reporting a segmentation failure as "no faults found" is the failure mode this codebase explicitly rejects. (Spec §4.2)
- Commit messages end with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

## Spec

`docs/superpowers/specs/2026-07-26-rep-segmentation-sp1-design.md`

## File Structure

| File | Responsibility |
|---|---|
| `src/pose/rep_segmentation.py` (**create**) | `RepWindow`, `segment_reps`, `select_reps`, and every threshold constant. Pure, numpy-only, TS-portable. |
| `tests/test_rep_segmentation.py` (**create**) | Unit tests for the segmenter, driven partly by the shared fixture. |
| `tests/fixtures/rep_segmentation_cases.json` (**create**) | Shared synthetic signal → expected windows. Consumed by Python now, by RS-SP2's vitest later. |
| `src/pose/movements/base.py` (**modify**) | `MovementDetector` gains 5 rep fields; `run_detector` becomes per-rep and returns a `RunResult`; `merge_by_fault` lives here. |
| `src/pose/pose_rule_detector.py` (**modify**) | `PoseRuleDetection` gains 3 rep fields; `detect_pose_rules_from_*` gain `max_reps` and emit the `reps` payload block; CLI gains `--max-reps`. |
| `src/pose/movements/{squat,pushup,overhead_press}.py` (**modify**) | One-line each: pass the rep signal/polarity to their `MovementDetector`. |
| `backend/app/config.py` (**modify**) | `DEFAULT_MAX_REPS`. |
| `backend/app/services/analysis.py` (**modify**) | Thread `max_reps` to the detector. |
| `backend/app/routers/analyze.py` (**modify**) | Optional `max_reps` form field + validation on both endpoints. |
| `tests/test_movement_registry.py` (**modify**) | Update `run_detector` unpacking; add the multi-rep regression test. |

---

### Task 1: The segmenter core — `segment_reps`

**Files:**
- Create: `src/pose/rep_segmentation.py`
- Create: `tests/test_rep_segmentation.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `RepWindow(index: int, start: int, end: int, partial: bool)` — frozen dataclass. `start`/`end` are **positions in the passed sequence** (not `frame_index` values), both inclusive. `index` is 1-based.
  - `segment_reps(signal: Sequence[float], *, fps: float, polarity: str = "min", rectify: bool = False, rep_start: str = "extended", min_rep_seconds: float = DEFAULT_MIN_REP_SECONDS) -> list[RepWindow]`
  - Constants: `PERCENTILE_LOW=5.0`, `PERCENTILE_HIGH=95.0`, `ENTER_FRACTION=0.35`, `EXIT_FRACTION=0.65`, `MIN_RANGE_TO_NOISE=6.0`, `DEFAULT_MIN_REP_SECONDS=0.4`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rep_segmentation.py`:

```python
from __future__ import annotations

import math
import unittest

from src.pose.rep_segmentation import RepWindow, segment_reps


def sine_reps(n_reps: int, frames_per_rep: int = 30, low: float = 60.0, high: float = 170.0) -> list[float]:
    """A clean multi-rep signal shaped like a knee angle: starts extended (high), dips to
    `low` at the bottom of each rep, returns to `high`. Exactly `n_reps` full cycles."""
    values: list[float] = []
    mid = (high + low) / 2.0
    amp = (high - low) / 2.0
    for i in range(n_reps * frames_per_rep):
        theta = 2.0 * math.pi * (i % frames_per_rep) / frames_per_rep
        values.append(mid + amp * math.cos(theta))
    return values


class SegmentRepsTests(unittest.TestCase):
    def test_three_clean_reps_are_segmented(self) -> None:
        reps = segment_reps(sine_reps(3), fps=30.0)
        self.assertEqual(len(reps), 3)
        self.assertEqual([r.index for r in reps], [1, 2, 3])
        self.assertTrue(all(not r.partial for r in reps))
        # Ordered and strictly non-overlapping — a shared boundary frame would be phased and
        # scored twice.
        for earlier, later in zip(reps, reps[1:]):
            self.assertLess(earlier.end, later.start)

    def test_windows_reach_the_top_of_each_excursion(self) -> None:
        """The boundary must sit at the actual top of the rep, not at the `exit` crossing.

        `EXIT_FRACTION` is a hysteresis crossing detector — its job is to stop a wobble near
        the bottom from splitting one rep in two. If it also defined the boundary, the window
        would open 35% of the dynamic range BELOW the top, so the whole opening third of every
        rep would fall outside every window, be labelled `rest`, and never be scored — taking
        the standing frames `rule_heel_rise` reads for its setup baseline with it.
        """
        signal = sine_reps(3)
        reps = segment_reps(signal, fps=30.0)
        covered = sum(rep.end - rep.start + 1 for rep in reps)
        self.assertGreaterEqual(covered, int(0.9 * len(signal)))
        # Each window opens near that rep's own maximum.
        for rep in reps:
            window = signal[rep.start : rep.end + 1]
            self.assertAlmostEqual(window[0], max(window), delta=0.1 * (max(window) - min(window)))

    def test_single_rep_is_one_window(self) -> None:
        reps = segment_reps(sine_reps(1), fps=30.0)
        self.assertEqual(len(reps), 1)

    def test_static_signal_yields_no_reps(self) -> None:
        self.assertEqual(segment_reps([120.0] * 60, fps=30.0), [])

    def test_jittery_static_signal_yields_no_reps(self) -> None:
        # Range comes only from noise, not from an excursion.
        noisy = [120.0 + (1.0 if i % 2 else -1.0) for i in range(60)]
        self.assertEqual(segment_reps(noisy, fps=30.0), [])

    def test_empty_and_tiny_inputs_yield_no_reps(self) -> None:
        self.assertEqual(segment_reps([], fps=30.0), [])
        self.assertEqual(segment_reps([170.0, 60.0], fps=30.0), [])

    def test_trailing_truncated_rep_is_partial(self) -> None:
        # Two full reps, then a descent the clip cuts off before it comes back up.
        signal = sine_reps(2) + sine_reps(1)[:15]
        reps = segment_reps(signal, fps=30.0)
        self.assertEqual(len(reps), 3)
        self.assertFalse(reps[0].partial)
        self.assertFalse(reps[1].partial)
        self.assertTrue(reps[2].partial)

    def test_leading_truncated_rep_is_partial(self) -> None:
        # Clip starts already at the bottom of a rep.
        signal = sine_reps(2)[15:]
        reps = segment_reps(signal, fps=30.0)
        self.assertTrue(reps[0].partial)

    def test_polarity_max_mirrors_polarity_min(self) -> None:
        base = sine_reps(3)
        flipped = [-v for v in base]
        self.assertEqual(
            segment_reps(base, fps=30.0, polarity="min"),
            segment_reps(flipped, fps=30.0, polarity="max"),
        )

    def test_rectify_splits_a_bipolar_signal_into_two_reps(self) -> None:
        # Torso twist shape: centre -> side A -> centre -> side B -> centre. Each swing is a rep.
        n = 30
        signal = [math.sin(2.0 * math.pi * i / (2 * n)) * 40.0 for i in range(2 * n)]
        self.assertEqual(len(segment_reps(signal, fps=30.0, rectify=True, polarity="max")), 2)

    def test_rep_start_flexed_places_boundaries_at_valleys(self) -> None:
        # Deadlift shape: a rep runs floor -> lockout -> floor, so boundaries sit at the minima.
        signal = sine_reps(3)
        extended = segment_reps(signal, fps=30.0)
        flexed = segment_reps(signal, fps=30.0, rep_start="flexed")
        self.assertTrue(flexed)
        # Every non-partial flexed window starts at a local minimum of the signal.
        for rep in [r for r in flexed if not r.partial]:
            window = signal[rep.start : rep.end + 1]
            self.assertEqual(min(window), window[0])
        # The phase differs from the extended convention.
        self.assertNotEqual([r.start for r in extended], [r.start for r in flexed])

    def test_short_blips_are_not_reps(self) -> None:
        signal = [170.0] * 60
        signal[30] = 60.0  # one-frame spike
        self.assertEqual(segment_reps(signal, fps=30.0), [])

    def test_faster_cadence_needs_a_smaller_min_rep_seconds(self) -> None:
        # Jumping-jack / high-knee shape: 10 frames per rep at 30fps.
        fast = sine_reps(4, frames_per_rep=10)
        self.assertEqual(segment_reps(fast, fps=30.0), [])
        self.assertEqual(len(segment_reps(fast, fps=30.0, min_rep_seconds=0.2)), 4)

    def test_nan_frames_do_not_break_segmentation(self) -> None:
        signal = sine_reps(2)
        signal[5] = float("nan")
        signal[40] = float("nan")
        self.assertEqual(len(segment_reps(signal, fps=30.0)), 2)

    def test_rejects_unknown_polarity_and_rep_start(self) -> None:
        with self.assertRaises(ValueError):
            segment_reps(sine_reps(1), fps=30.0, polarity="sideways")
        with self.assertRaises(ValueError):
            segment_reps(sine_reps(1), fps=30.0, rep_start="middle")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rep_segmentation.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.pose.rep_segmentation'`

- [ ] **Step 3: Implement the segmenter**

Create `src/pose/rep_segmentation.py`:

```python
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

# A real repetition's excursion spans many frames, so its full range is far larger than the
# typical frame-to-frame step. A static-but-jittery signal has a range of roughly a few steps.
# Below this ratio the "range" is noise, and segmenting it would invent repetitions.
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
    noise = float(np.median(np.abs(np.diff(finite))))
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rep_segmentation.py -v`
Expected: all PASS.

If `test_rectify_splits_a_bipolar_signal_into_two_reps` fails with 1 rep instead of 2, the two swings are being merged because the signal never returns above `exit_` at the centre — check that `sin` actually crosses zero at the midpoint of the fixture; do NOT loosen `EXIT_FRACTION` to make it pass.

- [ ] **Step 5: Commit**

```bash
git add src/pose/rep_segmentation.py tests/test_rep_segmentation.py
git commit -m "feat(pose): add a pure rep segmenter over a 1-D metric series

Hysteresis over the signal's own dynamic range, so the thresholds are
movement-, body- and camera-distance-agnostic. Returns [] rather than a guess
when there is no repetition structure; the caller must fall back to whole-clip
analysis on that, never to reporting no faults.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Rep selection + the shared fixture

**Files:**
- Modify: `src/pose/rep_segmentation.py`
- Modify: `tests/test_rep_segmentation.py`
- Create: `tests/fixtures/rep_segmentation_cases.json`

**Interfaces:**
- Consumes: `RepWindow`, `segment_reps` from Task 1.
- Produces:
  - `select_reps(reps: Sequence[RepWindow], max_reps: int | None) -> list[RepWindow]`
  - `tests/fixtures/rep_segmentation_cases.json` with schema:
    `{"cases": [{"name": str, "signal": [float], "fps": float, "polarity": str, "rectify": bool, "rep_start": str, "min_rep_seconds": float, "expected": [{"index": int, "start": int, "end": int, "partial": bool}]}]}`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rep_segmentation.py` (and add `import json` / `from pathlib import Path` to the imports at the top, plus `select_reps` to the `src.pose.rep_segmentation` import):

```python
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "rep_segmentation_cases.json"


def window(index: int, start: int, end: int, partial: bool = False) -> RepWindow:
    return RepWindow(index=index, start=start, end=end, partial=partial)


class SelectRepsTests(unittest.TestCase):
    def test_five_reps_capped_at_three_takes_first_middle_last(self) -> None:
        reps = [window(i + 1, i * 10, i * 10 + 9) for i in range(5)]
        self.assertEqual([r.index for r in select_reps(reps, 3)], [1, 3, 5])

    def test_fewer_reps_than_the_cap_are_all_kept(self) -> None:
        reps = [window(1, 0, 9), window(2, 10, 19)]
        self.assertEqual([r.index for r in select_reps(reps, 3)], [1, 2])

    def test_zero_and_none_mean_every_rep(self) -> None:
        reps = [window(i + 1, i * 10, i * 10 + 9) for i in range(7)]
        self.assertEqual(len(select_reps(reps, 0)), 7)
        self.assertEqual(len(select_reps(reps, None)), 7)

    def test_partial_reps_are_excluded_when_complete_ones_exist(self) -> None:
        reps = [window(1, 0, 9, partial=True), window(2, 10, 19), window(3, 20, 29)]
        self.assertEqual([r.index for r in select_reps(reps, 3)], [2, 3])

    def test_partial_reps_are_kept_when_they_are_all_there_is(self) -> None:
        reps = [window(1, 0, 9, partial=True), window(2, 10, 19, partial=True)]
        self.assertEqual([r.index for r in select_reps(reps, 3)], [1, 2])

    def test_empty_input_selects_nothing(self) -> None:
        self.assertEqual(select_reps([], 3), [])

    def test_seven_reps_capped_at_three_spans_the_whole_set(self) -> None:
        reps = [window(i + 1, i * 10, i * 10 + 9) for i in range(7)]
        self.assertEqual([r.index for r in select_reps(reps, 3)], [1, 4, 7])


class SharedFixtureTests(unittest.TestCase):
    """The SAME file RS-SP2's vitest will read. Any threshold change reddens both suites."""

    def test_every_fixture_case_matches(self) -> None:
        cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]
        self.assertTrue(cases, "fixture file must not be empty")
        for case in cases:
            with self.subTest(case=case["name"]):
                actual = segment_reps(
                    case["signal"],
                    fps=case["fps"],
                    polarity=case["polarity"],
                    rectify=case["rectify"],
                    rep_start=case["rep_start"],
                    min_rep_seconds=case["min_rep_seconds"],
                )
                self.assertEqual(
                    [
                        {"index": r.index, "start": r.start, "end": r.end, "partial": r.partial}
                        for r in actual
                    ],
                    case["expected"],
                )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rep_segmentation.py -v`
Expected: `ImportError: cannot import name 'select_reps'`

- [ ] **Step 3: Implement `select_reps`**

Append to `src/pose/rep_segmentation.py`:

```python
def select_reps(reps: Sequence[RepWindow], max_reps: int | None) -> list[RepWindow]:
    """Choose which repetitions to actually analyze.

    First / middle / last rather than "the first N" or "the middle N": the first rep carries
    warm-up errors, the middle one represents steady state, and the last one carries fatigue
    breakdown. Sampling only the middle systematically hides the fault a lifter most needs
    told. Partial reps (a clip that starts or ends mid-repetition) are skipped when complete
    ones exist, but kept when they are all there is — analyzing a truncated rep beats
    analyzing nothing.
    """
    candidates = [rep for rep in reps if not rep.partial] or list(reps)
    if not candidates:
        return []
    if not max_reps or max_reps <= 0 or len(candidates) <= max_reps:
        return candidates
    positions = sorted({int(round(value)) for value in np.linspace(0, len(candidates) - 1, max_reps)})
    return [candidates[position] for position in positions]
```

- [ ] **Step 4: Generate the shared fixture**

Run this from the repo root to write the fixture from the implementation's current behavior, then **read the output and confirm each `expected` block is what the case name claims** before committing:

```bash
.venv\Scripts\python.exe -c "
import json, math, pathlib
from src.pose.rep_segmentation import segment_reps

def sine(n, fpr=30, low=60.0, high=170.0):
    mid, amp = (high+low)/2.0, (high-low)/2.0
    return [mid + amp*math.cos(2*math.pi*(i%fpr)/fpr) for i in range(n*fpr)]

raw = [
    ('three_clean_reps', sine(3), 30.0, 'min', False, 'extended', 0.4),
    ('single_rep', sine(1), 30.0, 'min', False, 'extended', 0.4),
    ('static_no_reps', [120.0]*60, 30.0, 'min', False, 'extended', 0.4),
    ('trailing_partial', sine(2)+sine(1)[:15], 30.0, 'min', False, 'extended', 0.4),
    ('leading_partial', sine(2)[15:], 30.0, 'min', False, 'extended', 0.4),
    ('polarity_max', [-v for v in sine(3)], 30.0, 'max', False, 'extended', 0.4),
    ('bipolar_rectified', [math.sin(2*math.pi*i/60)*40.0 for i in range(60)], 30.0, 'max', True, 'extended', 0.4),
    ('deadlift_flexed_start', sine(3), 30.0, 'min', False, 'flexed', 0.4),
    ('fast_cadence', sine(4, fpr=10), 30.0, 'min', False, 'extended', 0.2),
]
cases = []
for name, signal, fps, polarity, rectify, rep_start, mrs in raw:
    reps = segment_reps(signal, fps=fps, polarity=polarity, rectify=rectify, rep_start=rep_start, min_rep_seconds=mrs)
    cases.append({
        'name': name, 'signal': [round(float(v), 6) for v in signal], 'fps': fps,
        'polarity': polarity, 'rectify': rectify, 'rep_start': rep_start, 'min_rep_seconds': mrs,
        'expected': [{'index': r.index, 'start': r.start, 'end': r.end, 'partial': r.partial} for r in reps],
    })
out = pathlib.Path('tests/fixtures/rep_segmentation_cases.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'cases': cases}, indent=2), encoding='utf-8')
for c in cases:
    print(c['name'], '->', c['expected'])
"
```

Confirm from the printed summary: `three_clean_reps` → 3 non-partial; `single_rep` → 1; `static_no_reps` → `[]`; `trailing_partial` → 3 with the last partial; `leading_partial` → first partial; `polarity_max` → 3; `bipolar_rectified` → 2; `deadlift_flexed_start` → non-empty with different starts than `three_clean_reps`; `fast_cadence` → 4. **If any disagrees, the implementation is wrong — fix Task 1, do not edit the fixture by hand.**

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rep_segmentation.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pose/rep_segmentation.py tests/test_rep_segmentation.py tests/fixtures/rep_segmentation_cases.json
git commit -m "feat(pose): select first/middle/last reps and pin the segmenter with a shared fixture

Sampling across the set rather than the first N: fatigue breakdown shows up in
the last rep, so analyzing only the opening reps hides the fault that most needs
telling.

The fixture is the artifact RS-SP2's TypeScript port will read, so a threshold
change reddens both suites instead of letting the two implementations drift.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `MovementDetector` rep fields

**Files:**
- Modify: `src/pose/movements/base.py:37-47` (the `MovementDetector` dataclass)
- Modify: `src/pose/movements/squat.py:320-327`
- Modify: `src/pose/movements/overhead_press.py` (the `OHP_DETECTOR = MovementDetector(...)` call near line 705)
- Modify: `src/pose/movements/pushup.py:1592-1600`
- Modify: `tests/test_movement_registry.py`

**Interfaces:**
- Consumes: `DEFAULT_MIN_REP_SECONDS` from `src.pose.rep_segmentation`.
- Produces: `MovementDetector` with five new keyword fields, all defaulted so no positional call site changes:
  `rep_signal: str | None = None`, `rep_polarity: str = "min"`, `rep_rectify: bool = False`, `rep_start: str = "extended"`, `min_rep_seconds: float = DEFAULT_MIN_REP_SECONDS`.
  `rep_signal=None` means **segmentation disabled for this movement** → whole-clip fallback.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_movement_registry.py` inside `MovementRegistryTests`:

```python
    def test_registered_detectors_declare_their_rep_signal(self) -> None:
        """A detector's rep signal must be one of the metrics it actually emits, or the
        segmenter would read NaN for every frame and silently find zero reps."""
        expected = {
            "Squat": ("avg_knee_angle", "min"),
            "Push-up": ("min_elbow_angle", "min"),
            "Overhead Press": ("avg_elbow_angle", "max"),
        }
        for name, (signal, polarity) in expected.items():
            with self.subTest(movement=name):
                detector = registry.get_detector(name)
                self.assertEqual(detector.rep_signal, signal)
                self.assertEqual(detector.rep_polarity, polarity)
                self.assertIn(detector.rep_signal, detector.metric_keys)
                # The other knobs exist for movements RS-SP1 does not implement (spec §3.4);
                # these three use the defaults.
                self.assertFalse(detector.rep_rectify)
                self.assertEqual(detector.rep_start, "extended")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py::MovementRegistryTests::test_registered_detectors_declare_their_rep_signal -v`
Expected: FAIL — `AttributeError: 'MovementDetector' object has no attribute 'rep_signal'`

- [ ] **Step 3: Add the fields**

In `src/pose/movements/base.py`, add the import and extend the dataclass:

```python
from src.pose.rep_segmentation import DEFAULT_MIN_REP_SECONDS
```

```python
@dataclass(frozen=True)
class MovementDetector:
    name: str
    metric_keys: tuple[str, ...]
    compute_raw: Callable[[Sequence[object], float], list[dict]]
    assign_phases: Callable[[list[dict]], list[str]]
    rules: tuple[RuleFn, ...]
    # Whether this detector's rules have been checked against labeled ground truth. Defaults to
    # False so a newly registered detector surfaces as Beta in the UI rather than silently
    # presenting as validated; Squat opts in explicitly.
    validated: bool = False
    # How this movement's repetitions are found. `rep_signal` names the metric (it MUST be one
    # of `metric_keys`) whose excursion defines a rep; None disables segmentation for this
    # movement and takes the whole-clip fallback. The remaining knobs exist because the 16
    # movements in the rule spec do not all share one shape -- see the spec's §3.4 audit:
    # `rep_rectify` for bipolar signals (torso twist swings to both sides), `rep_start="flexed"`
    # for movements whose rep starts at the bottom (deadlift, from the floor), and
    # `min_rep_seconds` for fast cyclic movements (high knees run ~3Hz, about 10 frames per rep
    # at 30fps, which the default would discard as noise).
    rep_signal: str | None = None
    rep_polarity: str = "min"
    rep_rectify: bool = False
    rep_start: str = "extended"
    min_rep_seconds: float = DEFAULT_MIN_REP_SECONDS
```

- [ ] **Step 4: Declare the signal on each of the three detectors**

`src/pose/movements/squat.py` — change the `SQUAT_DETECTOR` construction to add the keyword after `validated=True`:

```python
SQUAT_DETECTOR = MovementDetector(
    "Squat",
    METRIC_KEYS,
    compute_raw,
    assign_phases,
    (rule_knees_inward, rule_knees_forward, rule_shallow_depth, rule_forward_lean, rule_heel_rise),
    validated=True,
    rep_signal="avg_knee_angle",
    rep_polarity="min",
)
```

`src/pose/movements/pushup.py` — add to the `PUSHUP_DETECTOR` construction:

```python
    rep_signal="min_elbow_angle",
    rep_polarity="min",
```

`src/pose/movements/overhead_press.py` — add to the `OHP_DETECTOR` construction. Polarity is `"max"` because the effort peak is the overhead lockout, where the elbows are most EXTENDED (the largest angle), the mirror image of a squat's bottom:

```python
    rep_signal="avg_elbow_angle",
    rep_polarity="max",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py -v`
Expected: all PASS (the new test plus the existing registry tests, which are unaffected because every new field is defaulted).

- [ ] **Step 6: Commit**

```bash
git add src/pose/movements/base.py src/pose/movements/squat.py src/pose/movements/pushup.py src/pose/movements/overhead_press.py tests/test_movement_registry.py
git commit -m "feat(pose): let each movement declare how its reps are found

Five defaulted fields on MovementDetector rather than branches inside the
segmenter: the 16 movements in the rule spec do not share one shape, and a
segmenter that grows per-movement conditionals is one RS-SP2 has to port to
TypeScript conditional-for-conditional.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Per-rep phases and per-rep rule execution

**Files:**
- Modify: `src/pose/movements/base.py:50-81` (`run_detector`)
- Modify: `src/pose/pose_rule_detector.py:616` (the one production caller)
- Modify: `tests/test_movement_registry.py:52` (the `run_detector` unpacking in `test_squat_via_registry_matches_legacy`)
- Modify: `tests/test_pushup.py` (any `run_detector` unpacking — locate with grep in Step 4)

**Interfaces:**
- Consumes: `segment_reps`, `select_reps`, `RepWindow` from Tasks 1–2; the `MovementDetector` rep fields from Task 3.
- Produces:
  - `RunResult(core: list[CoreFrame], detections: list[PoseRuleDetection], reps: list[RepWindow], analyzed: list[RepWindow], fallback: str | None)` — frozen dataclass in `base.py`.
  - `run_detector(detector, frames, fps, view_type, view_confidence, *, max_reps: int | None = 3) -> RunResult` — **return type changes from a 2-tuple to `RunResult`**; every call site must be updated.
  - `REST_PHASE = "rest"` in `base.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_detector_per_rep.py`:

```python
from __future__ import annotations

import math
import unittest

from src.pose.movements import registry
from src.pose.movements.base import REST_PHASE, run_detector
from tests.test_pose_rule_detector import frame


def squat_reps(n_reps: int, frames_per_rep: int = 30) -> list[dict]:
    """`n_reps` squats built from the shared frame fixture: the hip rides from standing
    (hip_y 0.45, above the knee) down to a deep bottom (hip_y 0.92) and back, so
    avg_knee_angle traces one excursion per rep."""
    frames: list[dict] = []
    for index in range(n_reps * frames_per_rep):
        theta = 2.0 * math.pi * (index % frames_per_rep) / frames_per_rep
        hip_y = 0.685 - 0.235 * math.cos(theta)  # 0.45 at the top, 0.92 at the bottom
        frames.append(frame(hip_y=hip_y, knee_y=0.70, frame_index=index))
    return frames


class RunDetectorPerRepTests(unittest.TestCase):
    def test_three_reps_are_detected_and_all_analyzed_by_default(self) -> None:
        result = run_detector(registry.get_detector("Squat"), squat_reps(3), 30.0, "rear", 0.8)
        self.assertEqual(len(result.reps), 3)
        self.assertEqual(len(result.analyzed), 3)
        self.assertIsNone(result.fallback)

    def test_every_rep_gets_its_own_descent_bottom_and_ascent(self) -> None:
        """The bug this change fixes: with one global argmin, reps 2 and 3 had no descent."""
        result = run_detector(registry.get_detector("Squat"), squat_reps(3), 30.0, "rear", 0.8)
        for rep in result.reps:
            phases = {c.phase for c in result.core[rep.start : rep.end + 1]}
            with self.subTest(rep=rep.index):
                self.assertIn("descent", phases)
                self.assertIn("bottom", phases)
                self.assertIn("ascent", phases)

    def test_five_reps_are_capped_to_first_middle_last(self) -> None:
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(5), 30.0, "rear", 0.8, max_reps=3
        )
        self.assertEqual(len(result.reps), 5)
        self.assertEqual([r.index for r in result.analyzed], [1, 3, 5])

    def test_max_reps_zero_analyzes_every_rep(self) -> None:
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(5), 30.0, "rear", 0.8, max_reps=0
        )
        self.assertEqual(len(result.analyzed), 5)

    def test_frames_outside_every_rep_are_rest_and_are_never_scored(self) -> None:
        """Walk-in / rack / rest frames must not be scored -- that is the noise this fixes.

        The idle frames are built with a KNEES-FORWARD posture so they WOULD fire a rule if
        they were scored; a neutral idle stretch would make this test pass for the wrong
        reason.
        """
        idle = [frame(left_knee_x=0.48, right_knee_x=0.88, frame_index=i) for i in range(20)]
        working = squat_reps(2)
        for offset, item in enumerate(working):
            item["frame_index"] = 20 + offset
        result = run_detector(
            registry.get_detector("Squat"), idle + working, 30.0, "side", 0.8, max_reps=0
        )

        self.assertTrue(result.reps, "the working stretch must still segment")
        self.assertTrue(all(c.phase == REST_PHASE for c in result.core[:20]))

        # Detections are reported in frame_index units; rep windows are sequence positions.
        # Convert once, then require every detection to lie inside some ANALYZED rep.
        analyzed_ranges = [
            (result.core[rep.start].frame_index, result.core[rep.end].frame_index)
            for rep in result.analyzed
        ]
        self.assertTrue(analyzed_ranges)
        for detection in result.detections:
            with self.subTest(fault=detection.fault_id):
                self.assertTrue(
                    any(
                        start <= detection.start_frame and detection.end_frame <= end
                        for start, end in analyzed_ranges
                    ),
                    f"{detection.fault_id} spans {detection.start_frame}-{detection.end_frame}, "
                    f"outside every analyzed rep {analyzed_ranges}",
                )
                # Nothing may reach back into the idle stretch (frame_index 0-19).
                self.assertGreaterEqual(detection.start_frame, 20)

    def test_only_partial_reps_are_still_reported_even_though_the_clip_is_analyzed_whole(self) -> None:
        """A clip trimmed to one mid-rep excerpt -- the labeled research dataset's shape."""
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(1)[8:22], 30.0, "rear", 0.8
        )
        if result.fallback == "only_partial_reps":
            self.assertTrue(result.reps, "partial reps must still be reported")
            self.assertEqual(result.analyzed, [])
            self.assertTrue(result.detections, "the clip must still be analyzed whole")

    def test_static_clip_falls_back_to_whole_clip_analysis(self) -> None:
        """A segmentation failure must never present as 'no faults found'."""
        frames = [frame(left_knee_x=0.48, right_knee_x=0.88, frame_index=i) for i in range(14)]
        result = run_detector(registry.get_detector("Squat"), frames, 30.0, "side", 0.8)
        self.assertEqual(result.reps, [])
        self.assertEqual(result.fallback, "no_reps_detected")
        self.assertTrue(result.detections, "fallback must still produce detections")
        self.assertFalse(any(c.phase == REST_PHASE for c in result.core))

    def test_segmentation_disabled_detector_falls_back(self) -> None:
        from dataclasses import replace

        detector = replace(registry.get_detector("Squat"), rep_signal=None)
        result = run_detector(detector, squat_reps(3), 30.0, "rear", 0.8)
        self.assertEqual(result.reps, [])
        self.assertEqual(result.fallback, "segmentation_disabled")

    def test_detections_carry_absolute_frame_indices(self) -> None:
        """Rules run on a SLICE, so a bug here would report REP-RELATIVE indices.

        Bounds-checking alone would not catch that -- rep-relative indices are also in range.
        The discriminating assertion is that each detection's frames fall inside the window of
        the rep it says it came from: a rep-relative index would land in rep 1's range while
        `rep_index` claimed rep 3.
        """
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(3), 30.0, "rear", 0.8, max_reps=0
        )
        self.assertEqual(len(result.reps), 3)
        ranges = {
            rep.index: (result.core[rep.start].frame_index, result.core[rep.end].frame_index)
            for rep in result.reps
        }
        self.assertTrue(any(index > 1 for index in ranges), "need more than one rep to discriminate")

        for detection in result.detections:
            with self.subTest(fault=detection.fault_id):
                self.assertLessEqual(detection.start_frame, detection.peak_frame)
                self.assertLessEqual(detection.peak_frame, detection.end_frame)
                start, end = ranges[detection.rep_index]
                self.assertGreaterEqual(detection.start_frame, start)
                self.assertLessEqual(detection.end_frame, end)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_run_detector_per_rep.py -v`
Expected: FAIL — `ImportError: cannot import name 'REST_PHASE'`

- [ ] **Step 3: Rewrite `run_detector`**

Replace `run_detector` in `src/pose/movements/base.py` (keep everything above it) with:

```python
# Phase for frames belonging to no repetition: walking in, racking, resting between reps.
# Every rule gates on its movement's active phases, so "rest" frames are never scored -- that
# suppression is the point, not a side effect.
REST_PHASE = "rest"

DEFAULT_MAX_REPS = 3


@dataclass(frozen=True)
class RunResult:
    core: list[CoreFrame]
    detections: list[PoseRuleDetection]
    reps: list[RepWindow]
    analyzed: list[RepWindow]
    # None when reps were segmented normally; otherwise why the whole clip was analyzed
    # instead: "no_reps_detected", "only_partial_reps", or "segmentation_disabled".
    fallback: str | None


def run_detector(
    detector: MovementDetector,
    frames: Sequence[object],
    fps: float,
    view_type: str,
    view_confidence: float,
    *,
    max_reps: int | None = DEFAULT_MAX_REPS,
) -> RunResult:
    """Compute metrics over the whole clip, then phase and score one repetition at a time.

    Smoothing stays GLOBAL (all frames exist here), and only phase assignment and rule
    execution are per-rep. RS-SP2 extracts only the selected windows, at which point smoothing
    necessarily becomes per-padded-window -- that is an SP2 change, not a constraint on this
    one.
    """
    raw = detector.compute_raw(frames, fps)
    smoothed = {
        key: centered_median([float(item.get(key, np.nan)) for item in raw], window=5)
        for key in detector.metric_keys
    }

    reps: list[RepWindow] = []
    fallback: str | None = None
    if detector.rep_signal is None:
        fallback = "segmentation_disabled"
    else:
        reps = segment_reps(
            smoothed[detector.rep_signal],
            fps=fps,
            polarity=detector.rep_polarity,
            rectify=detector.rep_rectify,
            rep_start=detector.rep_start,
            min_rep_seconds=detector.min_rep_seconds,
        )
        if not reps:
            fallback = "no_reps_detected"
        elif all(rep.partial for rep in reps):
            # A tightly-trimmed single-rep clip (the labeled research dataset) looks like this.
            # Analyzing it whole is exactly the pre-existing behavior, which is correct for it.
            fallback = "only_partial_reps"

    # `reps` is what was FOUND and gets reported; `segmented` is what is actually used to phase
    # and score. They differ on the only_partial_reps path, where the payload should still say
    # what was there rather than claiming the clip held nothing.
    segmented = reps if fallback is None else []

    # Phases: per-rep when segmented, whole-clip on any fallback (today's behavior).
    if segmented:
        phases = [REST_PHASE] * len(raw)
        for rep in segmented:
            phases[rep.start : rep.end + 1] = detector.assign_phases(raw[rep.start : rep.end + 1])
    else:
        phases = detector.assign_phases(raw)

    core: list[CoreFrame] = []
    for i, item in enumerate(raw):
        core.append(
            CoreFrame(
                frame_index=int(item.get("frame_index", i) or i),
                time=float(item.get("time", 0.0) or 0.0),
                phase=phases[i],
                valid=bool(item.get("valid", False)),
                lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
                metrics={key: float(smoothed[key][i]) for key in detector.metric_keys},
            )
        )

    min_frames = max(3, int(math.ceil(max(fps, 1.0) * 0.20)))
    ctx = RuleContext(fps=fps, view_type=view_type, view_confidence=view_confidence, min_frames=min_frames)
    analyzed = select_reps(segmented, max_reps)

    detections: list[PoseRuleDetection] = []
    if analyzed:
        for rep in analyzed:
            # A SLICE, not a mask: `contiguous_true_segments` over a rep-gated global mask would
            # weld a fault at the end of rep 2 to one at the start of rep 3 into a single
            # detection spanning the gap between them. CoreFrame carries absolute frame_index
            # and time, so slicing does not disturb the reported timestamps.
            window = core[rep.start : rep.end + 1]
            for rule in detector.rules:
                for detection in rule(window, ctx):
                    detections.append(
                        replace(detection, rep_index=rep.index, occurred_reps=(rep.index,), rep_count=1)
                    )
    else:
        for rule in detector.rules:
            detections.extend(rule(core, ctx))

    detections = merge_by_fault(detections)
    detections.sort(key=lambda d: (d.observability == "low", -d.severity, d.start_frame))
    return RunResult(core=core, detections=detections, reps=reps, analyzed=analyzed, fallback=fallback)
```

Add to the imports at the top of `base.py`:

```python
from dataclasses import dataclass, field, replace

from src.pose.rep_segmentation import DEFAULT_MIN_REP_SECONDS, RepWindow, segment_reps, select_reps
```

`merge_by_fault` does not exist yet — Task 5 writes it. For this task only, add a temporary
pass-through directly above `run_detector` so the module imports:

```python
def merge_by_fault(detections: list[PoseRuleDetection]) -> list[PoseRuleDetection]:
    return detections
```

- [ ] **Step 4: Update every `run_detector` call site**

Find them:

```bash
git grep -n "run_detector(" -- src tests
```

`src/pose/pose_rule_detector.py:616` — replace:

```python
    core, detections = run_detector(detector, frames, fps if fps > 0 else 30.0, view_type, view_confidence)
```

with:

```python
    run = run_detector(detector, frames, fps if fps > 0 else 30.0, view_type, view_confidence)
    core, detections = run.core, run.detections
```

`tests/test_movement_registry.py:52` — replace:

```python
            _, new = run_detector(registry.get_detector("Squat"), frames, 30.0, view_type, 0.8)
```

with:

```python
            new = run_detector(registry.get_detector("Squat"), frames, 30.0, view_type, 0.8).detections
```

Apply the same `.core` / `.detections` change to any hit in `tests/test_pushup.py` or elsewhere.

- [ ] **Step 5: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`

Expected: all PASS, including `test_movement_registry.py::test_squat_via_registry_matches_legacy`. That test's fixture is 14 identical frames — a static signal — so the new path takes the `no_reps_detected` fallback and reproduces the legacy result exactly. **If it fails, the fallback is not reproducing whole-clip behavior; fix that rather than relaxing the test.**

- [ ] **Step 6: Commit**

```bash
git add src/pose/movements/base.py src/pose/pose_rule_detector.py tests/test_run_detector_per_rep.py tests/test_movement_registry.py tests/test_pushup.py
git commit -m "fix(pose): phase and score one repetition at a time

assign_phases was written as if a clip held exactly one rep -- a global argmin
for the bottom frame, global percentile thresholds, hard 15%/85% setup/lockout
slices -- so on a multi-rep clip every descent after the first was labelled
ascent and shallow reps never registered a bottom. Applying the SAME function to
one rep at a time makes it correct with no change to any movement module.

Frames in no rep get phase 'rest', which no rule's active-phase gate matches, so
walk-in and rack frames stop being scored. Rules run on a slice rather than a
gated mask so a fault at the end of one rep cannot weld to one at the start of
the next.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Merge repeated faults across reps

**Files:**
- Modify: `src/pose/pose_rule_detector.py:83-100` (`PoseRuleDetection`)
- Modify: `src/pose/movements/base.py` (replace the pass-through `merge_by_fault`)
- Create: `tests/test_detection_merge.py`

**Interfaces:**
- Consumes: `PoseRuleDetection`, `run_detector` from Task 4.
- Produces:
  - `PoseRuleDetection` with `rep_index: int = 0`, `occurred_reps: tuple[int, ...] = ()`, `rep_count: int = 0`.
  - `merge_by_fault(detections: list[PoseRuleDetection]) -> list[PoseRuleDetection]` — one entry per `fault_id`, the representative being the highest-severity occurrence, ties broken by the earliest `start_frame` for determinism.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detection_merge.py`:

```python
from __future__ import annotations

import unittest

from src.pose.movements.base import merge_by_fault
from src.pose.pose_rule_detector import PoseRuleDetection


def detection(fault_id: str, severity: float, rep: int, start_frame: int) -> PoseRuleDetection:
    return PoseRuleDetection(
        fault_id=fault_id,
        fault_name=fault_id.replace("_", " ").title(),
        kg_query=fault_id,
        retrieval_mode="kg",
        severity=severity,
        confidence=severity,
        observability="high",
        start_time=start_frame / 30.0,
        end_time=(start_frame + 10) / 30.0,
        start_frame=start_frame,
        end_frame=start_frame + 10,
        peak_frame=start_frame + 5,
        phase="bottom",
        evidence={"primary_value": severity},
        rep_index=rep,
        occurred_reps=(rep,),
        rep_count=1,
    )


class MergeByFaultTests(unittest.TestCase):
    def test_same_fault_in_two_reps_becomes_one_entry(self) -> None:
        merged = merge_by_fault([detection("knees_inward", 0.4, 1, 0), detection("knees_inward", 0.7, 3, 60)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].rep_count, 2)
        self.assertEqual(merged[0].occurred_reps, (1, 3))

    def test_representative_is_the_worst_occurrence(self) -> None:
        merged = merge_by_fault([detection("knees_inward", 0.4, 1, 0), detection("knees_inward", 0.7, 3, 60)])
        self.assertEqual(merged[0].severity, 0.7)
        self.assertEqual(merged[0].rep_index, 3)
        self.assertEqual(merged[0].start_frame, 60)
        self.assertEqual(merged[0].evidence["primary_value"], 0.7)

    def test_distinct_faults_are_not_merged(self) -> None:
        merged = merge_by_fault([detection("knees_inward", 0.4, 1, 0), detection("shallow_depth", 0.5, 1, 0)])
        self.assertEqual({d.fault_id for d in merged}, {"knees_inward", "shallow_depth"})

    def test_occurred_reps_are_sorted_and_deduplicated(self) -> None:
        merged = merge_by_fault(
            [detection("knees_inward", 0.4, 3, 60), detection("knees_inward", 0.5, 1, 0), detection("knees_inward", 0.3, 3, 70)]
        )
        self.assertEqual(merged[0].occurred_reps, (1, 3))
        self.assertEqual(merged[0].rep_count, 2)

    def test_equal_severity_ties_break_on_the_earlier_frame(self) -> None:
        merged = merge_by_fault([detection("knees_inward", 0.5, 3, 60), detection("knees_inward", 0.5, 1, 0)])
        self.assertEqual(merged[0].start_frame, 0)

    def test_empty_input(self) -> None:
        self.assertEqual(merge_by_fault([]), [])

    def test_whole_clip_fallback_detections_pass_through(self) -> None:
        """Fallback detections carry no rep, and must survive with rep_count 0."""
        item = detection("knees_inward", 0.5, 0, 0)
        item = item.__class__(**{**item.__dict__, "occurred_reps": (), "rep_count": 0})
        merged = merge_by_fault([item])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].rep_count, 0)
        self.assertEqual(merged[0].occurred_reps, ())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_detection_merge.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'rep_index'`

- [ ] **Step 3: Add the fields to `PoseRuleDetection`**

In `src/pose/pose_rule_detector.py`, append to the dataclass (after `citation_support`, so every field stays defaulted):

```python
    citation: str = ""
    citation_support: str = ""
    # Which repetition this fault came from. `rep_index` is the representative (worst)
    # occurrence; `occurred_reps` lists every analyzed rep it fired in, so the UI can say
    # "2 of the 3 reps we looked at" instead of showing three near-identical cards. All zero /
    # empty on the whole-clip fallback path, where there are no reps to attribute to.
    rep_index: int = 0
    occurred_reps: tuple[int, ...] = ()
    rep_count: int = 0
```

- [ ] **Step 4: Implement `merge_by_fault`**

Replace the pass-through in `src/pose/movements/base.py`:

```python
def merge_by_fault(detections: list[PoseRuleDetection]) -> list[PoseRuleDetection]:
    """Collapse each fault to its worst occurrence, recording which reps it fired in.

    One card per fault rather than one per rep: three near-identical "knees inward" entries
    read as three problems. Severity, timing and evidence all come from the SAME (worst)
    occurrence so the surfaced numbers stay internally consistent -- a merged entry must never
    pair one rep's severity with another rep's evidence.
    """
    worst: dict[str, PoseRuleDetection] = {}
    reps: dict[str, set[int]] = {}
    order: list[str] = []
    for detection in detections:
        fault_id = detection.fault_id
        if fault_id not in worst:
            order.append(fault_id)
            reps[fault_id] = set()
        incumbent = worst.get(fault_id)
        if incumbent is None or (detection.severity, -detection.start_frame) > (
            incumbent.severity,
            -incumbent.start_frame,
        ):
            worst[fault_id] = detection
        reps[fault_id].update(detection.occurred_reps)

    merged: list[PoseRuleDetection] = []
    for fault_id in order:
        occurred = tuple(sorted(reps[fault_id]))
        merged.append(replace(worst[fault_id], occurred_reps=occurred, rep_count=len(occurred)))
    return merged
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_detection_merge.py tests/test_run_detector_per_rep.py -v`
Expected: all PASS.

- [ ] **Step 6: Add and run the end-to-end merge assertion**

Append to `tests/test_run_detector_per_rep.py` inside `RunDetectorPerRepTests`:

```python
    def test_a_fault_firing_in_several_reps_is_reported_once(self) -> None:
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(3), 30.0, "rear", 0.8, max_reps=0
        )
        fault_ids = [d.fault_id for d in result.detections]
        self.assertEqual(len(fault_ids), len(set(fault_ids)), "each fault must appear once")
        repeated = [d for d in result.detections if d.rep_count > 1]
        self.assertTrue(repeated, "a fault present in every rep must record rep_count > 1")
        for detection in repeated:
            with self.subTest(fault=detection.fault_id):
                self.assertEqual(detection.rep_count, len(detection.occurred_reps))
                self.assertIn(detection.rep_index, detection.occurred_reps)
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_run_detector_per_rep.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pose/pose_rule_detector.py src/pose/movements/base.py tests/test_detection_merge.py tests/test_run_detector_per_rep.py
git commit -m "feat(pose): merge a fault across reps into one entry with its rep count

Three near-identical 'knees inward' cards read as three problems. Severity,
timing and evidence all come from the same worst occurrence, so a merged entry
never pairs one rep's number with another rep's evidence.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The `reps` payload block

**Files:**
- Modify: `src/pose/pose_rule_detector.py:584-659` (`detect_pose_rules_from_payload`)
- Create: `tests/test_reps_payload.py`

**Interfaces:**
- Consumes: `RunResult` from Task 4.
- Produces: `detect_pose_rules_from_payload` result gains a top-level `reps` key:
  `{"detected": int, "analyzed": [int], "max_reps": int | None, "fallback": str | None, "segments": [{"index", "start_frame", "end_frame", "start_time", "end_time", "analyzed", "partial"}]}`
  and `quality` gains `analyzed_frames: int` and `analyzed_frame_ratio: float`. **No existing key changes.**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reps_payload.py`:

```python
from __future__ import annotations

import json
import unittest

from src.pose.pose_rule_detector import detect_pose_rules_from_payload
from tests.test_pose_rule_detector import frame
from tests.test_run_detector_per_rep import squat_reps


def payload(frames: list[dict]) -> dict:
    return {"metadata": {"fps": 30.0, "width": 640, "height": 480}, "frames": frames}


class RepsPayloadTests(unittest.TestCase):
    def test_reps_block_describes_every_segment(self) -> None:
        result = detect_pose_rules_from_payload(payload(squat_reps(5)), movement="Squat", max_reps=3)
        reps = result["reps"]
        self.assertEqual(reps["detected"], 5)
        self.assertEqual(reps["analyzed"], [1, 3, 5])
        self.assertEqual(reps["max_reps"], 3)
        self.assertIsNone(reps["fallback"])
        self.assertEqual(len(reps["segments"]), 5)
        for segment in reps["segments"]:
            self.assertEqual(
                set(segment),
                {"index", "start_frame", "end_frame", "start_time", "end_time", "analyzed", "partial"},
            )
        self.assertEqual([s["index"] for s in reps["segments"] if s["analyzed"]], [1, 3, 5])

    def test_quality_gains_analyzed_counters_without_changing_the_old_ones(self) -> None:
        frames = squat_reps(5)
        result = detect_pose_rules_from_payload(payload(frames), movement="Squat", max_reps=3)
        quality = result["quality"]
        self.assertEqual(quality["total_frames"], len(frames))
        self.assertEqual(quality["valid_frames"], len(frames))
        self.assertEqual(quality["valid_frame_ratio"], 1.0)
        self.assertLess(quality["analyzed_frames"], quality["total_frames"])
        self.assertGreater(quality["analyzed_frames"], 0)
        self.assertAlmostEqual(
            quality["analyzed_frame_ratio"], quality["analyzed_frames"] / quality["total_frames"], places=4
        )

    def test_frame_metrics_still_has_one_row_per_frame(self) -> None:
        frames = squat_reps(3)
        result = detect_pose_rules_from_payload(payload(frames), movement="Squat")
        self.assertEqual(len(result["frame_metrics"]), len(frames))

    def test_static_clip_reports_the_fallback_and_still_analyzes(self) -> None:
        frames = [frame(left_knee_x=0.48, right_knee_x=0.88, frame_index=i) for i in range(14)]
        result = detect_pose_rules_from_payload(payload(frames), movement="Squat")
        self.assertEqual(result["reps"]["detected"], 0)
        self.assertEqual(result["reps"]["fallback"], "no_reps_detected")
        self.assertEqual(result["reps"]["segments"], [])
        self.assertEqual(result["quality"]["analyzed_frames"], result["quality"]["total_frames"])

    def test_payload_survives_strict_json_dumps(self) -> None:
        """postgrest serialises with allow_nan=False; a NaN here would drop the analysis."""
        result = detect_pose_rules_from_payload(payload(squat_reps(3)), movement="Squat")
        json.dumps(result, allow_nan=False)

    def test_partial_only_clip_still_lists_what_was_found(self) -> None:
        """`fallback` explains why the clip was analyzed whole; it must not also erase the
        evidence that repetitions were there."""
        result = detect_pose_rules_from_payload(payload(squat_reps(1)[8:22]), movement="Squat")
        if result["reps"]["fallback"] == "only_partial_reps":
            self.assertGreater(result["reps"]["detected"], 0)
            self.assertTrue(result["reps"]["segments"])
            self.assertTrue(all(s["partial"] for s in result["reps"]["segments"]))
            self.assertFalse(any(s["analyzed"] for s in result["reps"]["segments"]))
            self.assertEqual(result["reps"]["analyzed"], [])
            self.assertEqual(
                result["quality"]["analyzed_frames"], result["quality"]["total_frames"]
            )

    def test_empty_frame_list_does_not_raise(self) -> None:
        result = detect_pose_rules_from_payload(payload([]), movement="Squat")
        self.assertEqual(result["reps"]["detected"], 0)
        self.assertEqual(result["quality"]["analyzed_frames"], 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reps_payload.py -v`
Expected: FAIL — `TypeError: detect_pose_rules_from_payload() got an unexpected keyword argument 'max_reps'`

- [ ] **Step 3: Emit the block**

In `src/pose/pose_rule_detector.py`, add `max_reps: int | None = DEFAULT_MAX_REPS` to `detect_pose_rules_from_payload`'s keyword-only parameters, importing the default alongside `run_detector`:

```python
    from src.pose.movements import registry
    from src.pose.movements.base import DEFAULT_MAX_REPS, run_detector
```

(Import `DEFAULT_MAX_REPS` at module scope is not possible without a circular import — `base.py` imports `PoseRuleDetection` from this module. Keep the import inside the function, where `run_detector` is already imported, and give the parameter a sentinel default of `-1` meaning "use the module default":)

```python
def detect_pose_rules_from_payload(
    payload: dict[str, Any],
    *,
    pose_json_path: Path | None = None,
    video_id: str | None = None,
    include_retrieval: bool = False,
    graph_file: Path = DEFAULT_GRAPH_FILE,
    rag_db_dir: Path = DEFAULT_RAG_DB_DIR,
    movement: str | None = None,
    # -1 (not None) is the "caller said nothing" sentinel: None is a meaningful value here,
    # meaning "analyze every rep".
    max_reps: int | None = -1,
) -> dict[str, Any]:
```

Replace the `run_detector` call and the `result` construction:

```python
    from src.pose.movements import registry
    from src.pose.movements.base import DEFAULT_MAX_REPS, run_detector

    detector = registry.get_detector(movement)
    effective_max_reps = DEFAULT_MAX_REPS if max_reps == -1 else max_reps
    run = run_detector(
        detector,
        frames,
        fps if fps > 0 else 30.0,
        view_type,
        view_confidence,
        max_reps=effective_max_reps,
    )
    core, detections = run.core, run.detections

    analyzed_indices = [rep.index for rep in run.analyzed]
    analyzed_frames = sum(rep.end - rep.start + 1 for rep in run.analyzed) or len(core)
    valid_frames = [c for c in core if c.valid]
```

Add the `reps` block to the returned dict (after `"quality"`), and the two `quality` counters:

```python
        "quality": {
            "total_frames": len(frames),
            "valid_frames": len(valid_frames),
            "valid_frame_ratio": round(len(valid_frames) / len(frames), 4) if frames else 0.0,
            "lower_body_visibility_mean": round(float(np.mean([c.lower_body_visibility for c in core])), 4)
            if core
            else 0.0,
            # ADDITIVE. The existing denominators above stay whole-clip on purpose -- they are a
            # compatibility surface for backend/app/services/analysis.py, the frontend, and
            # src/knowledge/perception_to_graph.py.
            "analyzed_frames": analyzed_frames if core else 0,
            "analyzed_frame_ratio": round(analyzed_frames / len(frames), 4) if frames else 0.0,
        },
        # Which repetitions were found and which were actually scored. `segments` exists so a UI
        # can show which spans were examined: when whole stretches of a clip are never looked
        # at, the interface must not imply they were clean.
        "reps": {
            "detected": len(run.reps),
            "analyzed": analyzed_indices,
            "max_reps": effective_max_reps,
            "fallback": run.fallback,
            "segments": [
                {
                    "index": rep.index,
                    "start_frame": core[rep.start].frame_index,
                    "end_frame": core[rep.end].frame_index,
                    "start_time": round(core[rep.start].time, 3),
                    "end_time": round(core[rep.end].time, 3),
                    "analyzed": rep.index in set(analyzed_indices),
                    "partial": rep.partial,
                }
                for rep in run.reps
            ],
        },
```

Thread the same parameter through `detect_pose_rules_from_json`: add `max_reps: int | None = -1` to its signature and pass `max_reps=max_reps` to `detect_pose_rules_from_payload`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reps_payload.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pose/pose_rule_detector.py tests/test_reps_payload.py
git commit -m "feat(pose): report which reps were found and which were scored

Additive only: quality's existing denominators stay whole-clip and frame_metrics
keeps one row per frame, because both are a compatibility surface for the
backend service, the frontend and perception_to_graph.

reps.segments exists so a UI can mark which spans were examined -- when whole
stretches of a clip are never looked at, the interface must not imply they were
clean.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `max_reps` on the CLI and the API

**Files:**
- Modify: `src/pose/pose_rule_detector.py:793-855` (`main`)
- Modify: `backend/app/config.py` (near `DEFAULT_ANALYSIS_MOVEMENT`, line 30)
- Modify: `backend/app/services/analysis.py:66-98` (`analyze_video_file`), `:101-116` (`_run_detector`), `:141-171` (`analyze_pose_payload`)
- Modify: `backend/app/routers/analyze.py:83-88` (`analyze`), `:148-154` (`analyze_pose`)
- Modify: `tests/test_backend.py`

**Interfaces:**
- Consumes: `max_reps` on `detect_pose_rules_from_json` / `detect_pose_rules_from_payload` from Task 6.
- Produces:
  - `parse_max_reps(value: str) -> int | None` in `pose_rule_detector.py` — accepts `"all"` (case-insensitive) or `"0"` → `None`; any other non-negative integer → itself; anything else → `argparse.ArgumentTypeError`.
  - `config.DEFAULT_MAX_REPS: int = 3`
  - `analysis.analyze_video_file(..., max_reps: int | None = -1)` and `analysis.analyze_pose_payload(..., max_reps: int | None = -1)`
  - Both endpoints accept an optional `max_reps: int | None = Form(None)`; values outside `0..20` → HTTP 400.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backend.py` (match the file's existing client/fixture conventions — locate an existing `/api/analyze` test and mirror its setup):

```python
class MaxRepsPlumbingTests(unittest.TestCase):
    def test_parse_max_reps_accepts_all_and_zero_as_unlimited(self) -> None:
        from src.pose.pose_rule_detector import parse_max_reps

        self.assertIsNone(parse_max_reps("all"))
        self.assertIsNone(parse_max_reps("ALL"))
        self.assertIsNone(parse_max_reps("0"))
        self.assertEqual(parse_max_reps("3"), 3)

    def test_parse_max_reps_rejects_junk(self) -> None:
        import argparse

        from src.pose.pose_rule_detector import parse_max_reps

        for bad in ("-1", "three", "", "2.5"):
            with self.subTest(value=bad):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_max_reps(bad)

    def test_backend_default_max_reps_is_three(self) -> None:
        from backend.app import config

        self.assertEqual(config.DEFAULT_MAX_REPS, 3)
```

And a router validation test, mirroring the existing upload-endpoint tests in the file:

```python
    def test_analyze_rejects_out_of_range_max_reps(self) -> None:
        response = self.client.post(
            "/api/analyze",
            files={"file": ("clip.mp4", b"not-a-real-video", "video/mp4")},
            data={"movement": "Squat", "max_reps": "99"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("max_reps", response.json()["detail"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend.py -k MaxReps -v`
Expected: FAIL — `ImportError: cannot import name 'parse_max_reps'`

- [ ] **Step 3: Add the CLI flag**

In `src/pose/pose_rule_detector.py`, next to `parse_split_names`:

```python
def parse_max_reps(value: str) -> int | None:
    """Parse ``--max-reps``. ``all`` and ``0`` both mean every repetition."""
    text = (value or "").strip().lower()
    if text == "all":
        return None
    if not text.isdigit():
        raise argparse.ArgumentTypeError(
            f"--max-reps must be a non-negative integer or 'all', got {value!r}"
        )
    count = int(text)
    return None if count == 0 else count
```

In `main()`, after the `--movement` argument:

```python
    parser.add_argument(
        "--max-reps",
        type=parse_max_reps,
        default=3,
        help="How many repetitions to analyze (first/middle/last are sampled). "
             "Use 0 or 'all' to analyze every repetition.",
    )
```

Pass `max_reps=args.max_reps` to **both** `detect_pose_rules_from_json` calls in `main()`.

- [ ] **Step 4: Thread it through the backend**

`backend/app/config.py`, next to `DEFAULT_ANALYSIS_MOVEMENT`:

```python
# How many repetitions the web path analyzes. Sampled first/middle/last, not the first N.
DEFAULT_MAX_REPS = 3
```

`backend/app/services/analysis.py` — add `max_reps: int | None = -1` to `analyze_video_file` and `analyze_pose_payload`, resolve the sentinel, and pass it down:

```python
def analyze_video_file(
    source_path: Path,
    *,
    video_id: str | None = None,
    movement: str | None = None,
    max_reps: int | None = -1,
) -> dict[str, Any]:
```

and in its `detect_pose_rules_from_json(...)` call add:

```python
        max_reps=config.DEFAULT_MAX_REPS if max_reps == -1 else max_reps,
```

Give `_run_detector` a `max_reps` parameter and forward it the same way; `analyze_pose_payload` passes its own through to `_run_detector`.

`backend/app/routers/analyze.py` — add a shared validator above `analyze`:

```python
def _validated_max_reps(max_reps: int | None) -> int | None:
    """Resolve an optional client-supplied rep cap. 0 means 'every rep'.

    Bounded so a client cannot ask for an unbounded amount of per-rep work on the shared
    analysis semaphore.
    """
    if max_reps is None:
        return config.DEFAULT_MAX_REPS
    if max_reps < 0 or max_reps > 20:
        raise HTTPException(status_code=400, detail="max_reps must be between 0 and 20.")
    return None if max_reps == 0 else max_reps
```

Add `max_reps: int | None = Form(None)` to both endpoint signatures, call `_validated_max_reps(max_reps)` **before** `save_upload` (a bad request must cost no compute — same reason `_validated_movement` is called there), and pass the result into the `analysis.*` calls.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend.py -v`
Expected: all PASS.

- [ ] **Step 6: Verify the CLI end to end**

Run: `.venv\Scripts\python.exe scripts/pose/run_pose_rule_detection.py --help`
Expected: `--max-reps` appears with its help text.

- [ ] **Step 7: Commit**

```bash
git add src/pose/pose_rule_detector.py backend/app/config.py backend/app/services/analysis.py backend/app/routers/analyze.py tests/test_backend.py
git commit -m "feat: expose max_reps on the CLI and both analyze endpoints

Default 3 lives in exactly one place per surface (argparse default for the CLI,
config.DEFAULT_MAX_REPS for the web path). The -1 sentinel distinguishes 'caller
said nothing' from the meaningful None, which asks for every rep.

Validated before save_upload so a bad request costs no compute, and bounded at
20 so a client cannot buy unbounded per-rep work on the shared semaphore.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Regression proof + coverage gate

**Files:**
- Modify: `tests/test_movement_registry.py`
- Modify: `scripts/pose/README.md`
- Modify: `docs/superpowers/specs/2026-07-26-rep-segmentation-sp1-design.md` (status line only)

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: no new production interface. A test that fails if the multi-rep bug ever returns.

- [ ] **Step 1: Write the regression test**

Add to `tests/test_movement_registry.py`:

```python
    def test_multi_rep_clip_is_mis_phased_by_the_legacy_path_and_fixed_by_the_new_one(self) -> None:
        """Pins BOTH sides of the fix.

        The legacy whole-clip path takes one global argmin for the bottom frame, so on a
        three-rep clip everything after the first bottom is labelled `ascent` -- reps 2 and 3
        get no descent at all. The per-rep path must give every rep its own descent.
        """
        from src.pose.pose_rule_detector import compute_frame_metrics
        from tests.test_run_detector_per_rep import squat_reps

        frames = squat_reps(3)

        legacy_phases = [m.phase for m in compute_frame_metrics(frames, fps=30.0)]
        # Rep 3 lives in the final third; under one global argmin it never descends.
        self.assertNotIn("descent", legacy_phases[60:], "fixture is not multi-rep enough")

        result = run_detector(registry.get_detector("Squat"), frames, 30.0, "rear", 0.8)
        self.assertEqual(len(result.reps), 3)
        for rep in result.reps:
            phases = {c.phase for c in result.core[rep.start : rep.end + 1]}
            with self.subTest(rep=rep.index):
                self.assertIn("descent", phases)
```

- [ ] **Step 2: Run it to confirm it captures the fix**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py -k multi_rep -v`
Expected: PASS.

**If the first assertion fails** (`fixture is not multi-rep enough`), the fixture's later reps are not producing a real excursion — fix `squat_reps` so the hip actually travels, do not delete the assertion. It is what proves the test would have caught the original bug.

- [ ] **Step 3: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: all PASS. Report the exact count.

- [ ] **Step 4: Run the coverage gate**

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS at ≥95%.

If `src/pose/rep_segmentation.py` drags coverage down, add the missing cases to `tests/test_rep_segmentation.py` — every branch in that module is reachable from a synthetic 1-D signal, so there is no excuse for an uncovered line there.

- [ ] **Step 5: Document the flag**

Add to the rule-detection section of `scripts/pose/README.md`:

```markdown
`--max-reps N` limits how many repetitions are analyzed (default 3, sampled
first/middle/last rather than the first N — fatigue breakdown shows up in the last
rep). `--max-reps all` (or `0`) analyzes every repetition. Clips with no detectable
repetition structure fall back to whole-clip analysis and report why in
`reps.fallback`.
```

- [ ] **Step 6: Mark the spec implemented**

In `docs/superpowers/specs/2026-07-26-rep-segmentation-sp1-design.md`, change the status line to:

```markdown
Status: **已實作** · Created 2026-07-26
```

- [ ] **Step 7: Commit**

```bash
git add tests/test_movement_registry.py scripts/pose/README.md docs/superpowers/specs/2026-07-26-rep-segmentation-sp1-design.md
git commit -m "test(pose): pin both sides of the multi-rep phase fix

Asserts the legacy whole-clip path still mis-phases a three-rep clip and the
per-rep path does not. The legacy assertion is the part that matters: without it
the test would pass even if segmentation silently stopped running.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §1 the bug | 4 (fix), 8 (regression proof) |
| §3.1 interface + knobs | 1 (`segment_reps`, constants), 2 (`select_reps`) |
| §3.2 hysteresis algorithm | 1 |
| §3.3 the three movements' signals | 3 |
| §3.4 16-movement audit | 3 (fields exist), 1 (`rectify`/`rep_start`/`min_rep_seconds` tests) |
| §4 per-rep run_detector, 4 decisions | 4 |
| §4.1 first/middle/last selection | 2 |
| §4.2 fallback table (3 values) | 4 (`no_reps_detected`, `only_partial_reps`, `segmentation_disabled`) |
| §4.3 merge + 3 new detection fields | 5 |
| §5 payload `reps` + quality counters + `"rest"` safety | 6 |
| §6 CLI/config/API plumbing | 7 |
| §7 pure function, named constants, shared fixture | 1 (purity + constants), 2 (fixture) |
| §7.1 open question for SP2 | No task — deliberately out of scope for RS-SP1. |
| §8 tests incl. multi-rep regression | 1, 2, 4, 5, 6, 8 |
| §9 risks | 4 (fallbacks), 3 (knobs as data) |

**Gap accepted:** §7.1 (who computes the rep signal on the TS side) is explicitly deferred to the RS-SP2 spec and correctly has no task here.

**2. Placeholder scan:** No TBD/TODO. Every code step carries the actual code. Task 7's `tests/test_backend.py` additions say "mirror the file's existing client conventions" rather than inventing a fixture — that is a pointer to a concrete existing pattern in the file being edited, not a placeholder for logic.

**3. Type consistency (checked across tasks):**
- `RepWindow(index, start, end, partial)` — identical in Tasks 1, 2, 4, 6.
- `segment_reps(signal, *, fps, polarity, rectify, rep_start, min_rep_seconds)` — Task 1 defines; Task 2's fixture generator and Task 4's `run_detector` call it with exactly these keywords.
- `MovementDetector` field names `rep_signal` / `rep_polarity` / `rep_rectify` / `rep_start` / `min_rep_seconds` — Task 3 defines; Task 4 reads all five. Note the detector field is `rep_rectify` while `segment_reps`' parameter is `rectify`; Task 4's call maps `rectify=detector.rep_rectify`.
- `RunResult(core, detections, reps, analyzed, fallback)` — Task 4 defines; Tasks 4 and 6 read `.core`, `.detections`, `.reps`, `.analyzed`, `.fallback`.
- `merge_by_fault(detections) -> list` — Task 4 stubs, Task 5 implements; same name and signature.
- `PoseRuleDetection` extra fields `rep_index` / `occurred_reps` / `rep_count` — Task 5 defines; Task 4's `replace(...)` sets all three, Task 6 does not touch them, Task 5's tests read all three.
- `max_reps` sentinel `-1` — used identically in Task 6 (`detect_pose_rules_from_*`) and Task 7 (`analysis.*`). The routers resolve to a real value before calling, so the sentinel never crosses the HTTP boundary.

**Ordering note:** Task 4 introduces a temporary pass-through `merge_by_fault` so `base.py` imports before Task 5 replaces it. Task 4's assertions do not depend on merging; Task 5 adds the end-to-end merge assertion once the real implementation lands.

**4. Review pass (defects found and fixed in this document, recorded so execution does not reintroduce them):**

- **The hysteresis threshold was doing two jobs.** `EXIT_FRACTION = 0.65` is a *crossing detector* — it exists so a wobble near the bottom does not split one rep in two. Using the crossing as the *window boundary* would have opened each window 35% of the dynamic range below the top of the excursion, putting the opening third of every rep outside every window, labelling it `rest`, and never scoring it — including the standing frames `rule_heel_rise` reads for its setup baseline. Split into two mechanisms: cross the band to identify the rep (`_last_at_or_above`), then climb to the peak to bound it (`_climb_backward` / `_climb_forward`). `test_windows_reach_the_top_of_each_excursion` pins it; none of the phase-based tests would have caught it.
- **Shared boundary frames.** Climbing to peaks (and the valley convention) make adjacent windows meet at one frame, which would be phase-assigned and scored twice. `_finalize` now trims each window to end one frame before the next begins, and the non-overlap assertion is `assertLess(earlier.end, later.start)` rather than the tautological `< later.start + 1`.
- **A vacuous noise-suppression assertion.** The original `assertIn(detection.start_frame - 0, rep_positions | {detection.start_frame})` cannot fail — the right-hand set contains the left-hand value by construction — and it compared sequence positions against `frame_index` values. It was also the *only* check on this change's headline claim. Rewritten to convert units once and require every detection to lie inside an analyzed rep, with idle frames built in a fault-triggering posture so the test cannot pass for the wrong reason.
- **A vacuous absolute-index assertion.** `end_frame < 90` is satisfied by rep-relative indices too. Now each detection's frames must lie inside the window of the rep its `rep_index` names.
- **`only_partial_reps` erased its own evidence.** Setting `reps = []` made the payload report `detected: 0` with empty `segments` even though partial reps were found. `reps` (reported) and `segmented` (used for phasing/scoring) are now separate.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-26-rep-segmentation-sp1.md`. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.
