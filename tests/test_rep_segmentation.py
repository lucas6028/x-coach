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
