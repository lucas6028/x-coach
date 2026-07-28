"""Pin what the coarse pass costs, measured on real clips rather than synthetic ones.

RS-SP2 samples every third frame to find repetitions, then extracts only the selected ones
densely. This file pins the two quantities that design rests on -- and exists because the
SYNTHETIC fixture badly underestimated both: decimating tests/fixtures/rep_segmentation_cases.json
suggested boundary error stayed within 3 frames, while real footage puts p95 at 15 and max at 45.

The measurements below (46 clips, 70 reps) are what REP_PADDING_FRAMES = 24 and the valley anchor
in frontend/src/lib/repSpans.ts are derived from. If this test goes red, that constant is wrong.

Data lives under data/ and is gitignored, so this skips in CI and bites locally -- the same
arrangement as tests/test_view_regression_corpus.py.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.pose.geometry import centered_median
from src.pose.movements import registry
from src.pose.rep_segmentation import RepWindow, segment_reps

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIRS = (
    REPO_ROOT / "data" / "runtime" / "pose_json",
    REPO_ROOT / "data" / "Fitness-AQA" / "Squat" / "Labeled_Dataset" / "pose_json",
)

# Kept in sync with frontend/src/lib/repSpans.ts -- these ARE those constants.
COARSE_STRIDE = 3
COARSE_SMOOTH_WINDOW = 3
DENSE_SMOOTH_WINDOW = 5
REP_PADDING_FRAMES = 24

MIN_CLIPS = 20            # below this the percentiles below mean nothing
MAX_VALLEY_ERROR = 5      # measured max across 70 reps
MIN_SPAN_COVERAGE = 0.98  # measured 98.6%
MAX_COUNT_MISMATCHES = 2
MIN_REFINED_EXACT = 0.95  # measured 95.7% of reps refine to the whole-clip boundary exactly
MAX_REFINED_P95 = 0.0     # measured: p95 of the refined error is zero frames
MAX_CLIPPED_SHARE = 0.03  # measured 1.4%


def _clips() -> list[tuple[str, float, list[float]]]:
    detector = registry.get_detector("Squat")
    out: list[tuple[str, float, list[float]]] = []
    for directory in CORPUS_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            frames = payload.get("frames") or []
            if len(frames) < 40:
                continue
            fps = float((payload.get("metadata") or {}).get("fps", 30.0) or 30.0)
            raw = detector.compute_raw(frames, fps)
            out.append((path.name, fps, [float(r.get("avg_knee_angle", np.nan)) for r in raw]))
    return out


def _valley(signal, window: RepWindow) -> int:
    chunk = np.asarray(signal[window.start : window.end + 1], dtype=float)
    return window.start + int(np.nanargmin(np.where(np.isfinite(chunk), chunk, np.inf)))


CLIPS = _clips()


@unittest.skipUnless(len(CLIPS) >= MIN_CLIPS, f"needs >= {MIN_CLIPS} local squat pose JSONs")
class CoarseSegmentationCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valley_errors: list[int] = []
        cls.refined_errors: list[int] = []
        cls.covered = 0
        cls.clipped = 0
        cls.total_reps = 0
        cls.mismatches = 0
        for _name, fps, values in CLIPS:
            dense_signal = centered_median(values, window=DENSE_SMOOTH_WINDOW)
            coarse_signal = centered_median(values[::COARSE_STRIDE], window=COARSE_SMOOTH_WINDOW)
            dense = segment_reps(dense_signal, fps=fps)
            coarse = segment_reps(coarse_signal, fps=fps / COARSE_STRIDE)
            if len(dense) != len(coarse):
                cls.mismatches += 1
                continue
            last = len(values) - 1
            for d, c in zip(dense, coarse):
                cls.total_reps += 1
                valley = _valley(coarse_signal, c) * COARSE_STRIDE
                cls.valley_errors.append(abs(_valley(dense_signal, d) - valley))
                half = (c.end - c.start + 1) * COARSE_STRIDE // 2 + REP_PADDING_FRAMES
                if valley - half <= d.start and d.end <= valley + half:
                    cls.covered += 1

                # Refinement: re-segment the padded span, then take the window overlapping the
                # coarse one most (padding can catch a neighbour). The slice is smoothed from RAW
                # values WITHIN the span -- not sliced out of the whole-clip-smoothed dense_signal
                # -- because that is what the browser's refineWindow does: it only ever sees the
                # extracted span, so the ~2 frames at each edge (DENSE_SMOOTH_WINDOW radius) get a
                # shrunken window there, and this must reproduce that rather than borrow context
                # from frames the client never extracted.
                span_start, span_end = max(0, valley - half), min(last, valley + half)
                coarse_start, coarse_end = c.start * COARSE_STRIDE, c.end * COARSE_STRIDE
                span_signal = centered_median(values[span_start : span_end + 1], window=DENSE_SMOOTH_WINDOW)
                windows = segment_reps(span_signal, fps=fps)
                if not windows:
                    # Mirrors refineWindow's fallback: when re-segmentation finds nothing, production
                    # returns the coarse boundary with refined=False rather than dropping the rep --
                    # exactly the poor result this file exists to measure, so it must count as an
                    # error here too rather than shrink the denominator.
                    cls.refined_errors.append(max(abs(coarse_start - d.start), abs(coarse_end - d.end)))
                    continue
                best = max(windows, key=lambda w: max(
                    0, min(span_start + w.end, coarse_end) - max(span_start + w.start, coarse_start) + 1))
                start, end = span_start + best.start, span_start + best.end
                cls.refined_errors.append(max(abs(start - d.start), abs(end - d.end)))
                if (start == span_start and span_start > 0) or (end == span_end and span_end < last):
                    cls.clipped += 1

    def test_valley_location_survives_decimation(self) -> None:
        """The span's anchor. If this drifts, anchoring on the valley stops being the cheap option."""
        self.assertLessEqual(max(self.valley_errors), MAX_VALLEY_ERROR)

    def test_padding_contains_the_dense_window(self) -> None:
        """REP_PADDING_FRAMES = 24 is justified by THIS number and nothing else."""
        coverage = self.covered / self.total_reps
        self.assertGreaterEqual(coverage, MIN_SPAN_COVERAGE, f"coverage {coverage:.3f}")

    def test_coarse_rep_count_rarely_disagrees(self) -> None:
        """A miscount is the one error neither padding nor refinement can repair: a missed rep is
        never extracted, and the user is told a rep count that is simply wrong (spec §8)."""
        self.assertLessEqual(self.mismatches, MAX_COUNT_MISMATCHES)

    def test_refinement_recovers_the_whole_clip_boundary(self) -> None:
        """The claim refinement rests on, and the obvious way it could fail.

        Refining re-derives the hysteresis band from ONE span's percentiles rather than the whole
        clip's, which is narrower and could plausibly shift the boundary -- the very thing
        refinement exists to get right, since assign_phases takes a window's first 15% as setup.
        Measured on this corpus it does not: the refined boundary is EXACTLY the whole-clip one for
        95.7% of reps, against the coarse boundary's p50 2 / p95 21 / max 45 frames. If this goes
        red, refinement needs the whole-clip band passed in rather than re-derived per span.
        """
        exact = sum(1 for error in self.refined_errors if error == 0)
        self.assertGreaterEqual(exact / len(self.refined_errors), MIN_REFINED_EXACT)
        self.assertLessEqual(float(np.percentile(self.refined_errors, 95)), MAX_REFINED_P95)

    def test_spans_rarely_cut_a_rep_short(self) -> None:
        """`clipped` counts ONLY a span edge that is not also the clip's edge -- nothing exists
        beyond the clip to extract, so touching that is not a padding failure. Conflating the two
        reported 43% of these reps as clipped instead of the true 1.4%."""
        self.assertLessEqual(self.clipped / self.total_reps, MAX_CLIPPED_SHARE)


if __name__ == "__main__":
    unittest.main()
