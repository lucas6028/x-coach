from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from src.pose.rep_segmentation import RepWindow, segment_reps, select_reps


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


def paused_rep(
    hold_frames: int, frames_per_rep: int = 30, jitter: float = 0.0
) -> list[float]:
    """One rep of `sine_reps` with the bottom held for `hold_frames` frames.

    A paused squat or a paused deadlift: the athlete stops at the bottom under load. With
    `jitter=0` the hold is an exact plateau (a frozen reading); with `jitter>0` the held frames
    wobble by that much, which is what a real pose estimator produces while a human holds a
    position. Both shapes must segment identically.
    """
    base = sine_reps(1, frames_per_rep)
    bottom_at = frames_per_rep // 2
    bottom = base[bottom_at]
    hold = [bottom + (jitter if i % 2 else -jitter) for i in range(hold_frames)]
    return base[:bottom_at] + hold + base[bottom_at:]


def _static_with_glitch() -> list[float]:
    """Ninety frames of standing still, carrying one 6-frame detection glitch.

    Not a repeated movement -- a single bad detection in an otherwise static clip. Shared by
    the extended and flexed tests so both conventions are judged on the identical signal.
    """
    signal = [170.0] * 90
    for index in range(40, 46):
        signal[index] = 60.0
    return signal


def _static_with_glitch_and_drift() -> list[float]:
    """`_static_with_glitch` plus one near-duplicate pair with a sub-threshold, non-zero drift."""
    signal = _static_with_glitch()
    signal[70] = 170.0 + 1e-6
    return signal


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

        Note the scope of the coverage assertion below: this fixture is back-to-back reps with
        no idle, so the windows tile it almost completely. That is NOT a general property. The
        climb stops at a plateau, so on a clip with an idle preamble or a rest between reps
        those idle frames are deliberately in no window at all -- they are rest, and the rep a
        window describes is only the excursion. What generalises is the second assertion: a
        window opens AT the top of its own excursion, whether that top is a lone peak or the
        last frame of a plateau the athlete stood through.
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

    def test_flexed_leading_and_trailing_windows_are_partial(self) -> None:
        """`_windows_from_valleys` has a leading- and a trailing-partial branch (the clip can
        start or end away from a valley); nothing previously asserted on `.partial` or on rep
        count for `rep_start="flexed"`, so a flipped or off-by-one flag there would pass
        silently. This mirrors `test_leading_truncated_rep_is_partial` /
        `test_trailing_truncated_rep_is_partial`, which make the equivalent check for the
        "extended" convention.
        """
        # Starts and ends at the extended peak (not a valley), so both the leading and the
        # trailing window are incomplete reps.
        signal = sine_reps(3)
        reps = segment_reps(signal, fps=30.0, rep_start="flexed")
        self.assertEqual(len(reps), 4)
        self.assertTrue(reps[0].partial)
        self.assertFalse(reps[1].partial)
        self.assertFalse(reps[2].partial)
        self.assertTrue(reps[3].partial)
        # The leading window runs from the very start of the clip...
        self.assertEqual(reps[0].start, 0)
        # ...and the trailing window runs to the very end of it.
        self.assertEqual(reps[-1].end, len(signal) - 1)
        # The two full (non-partial) reps between them still start at a local minimum.
        for rep in reps[1:-1]:
            window = signal[rep.start : rep.end + 1]
            self.assertEqual(min(window), window[0])

    def test_wobble_that_stays_inside_the_hysteresis_band_does_not_split_a_rep(self) -> None:
        """The module's own leading comment gives the reason two thresholds (`enter`/`exit_`)
        exist: "a single threshold would split one rep into several whenever the signal
        wobbles across it near the bottom." No prior test constructed a signal that actually
        wobbles across `enter` without recovering past `exit_` -- the two-deep-runs-collapse
        path in `_finalize`'s `seen`-based de-dup was only exercised indirectly.
        """
        signal = sine_reps(2)
        # A noisy frame at the very bottom of the first rep: it rises back above `enter`
        # (splitting what would otherwise be one continuous deep run into two) but stays well
        # below `exit_`, so it must never read as a full recovery to standing.
        signal[15] = 110.0
        reps = segment_reps(signal, fps=30.0)
        # Still 2 reps, not 3 -- the wobble must not manufacture an extra one.
        self.assertEqual(len(reps), 2)
        self.assertFalse(reps[0].partial)
        self.assertFalse(reps[1].partial)
        self.assertEqual(reps[0].start, 0)
        self.assertEqual(reps[0].end, 29)
        self.assertEqual(reps[1].start, 30)
        self.assertEqual(reps[1].end, 59)

    def test_a_static_clip_with_one_glitch_yields_no_reps(self) -> None:
        """A clip of someone standing still, carrying a single bad detection, must yield no
        reps rather than invent one. (Renamed from
        `test_duplicate_frames_do_not_zero_out_the_noise_floor`: it was written against a
        noise-floor estimate that no longer exists, and a stale name is what shows up in CI
        output and greps. The fixture and the assertion are unchanged.)

        What rejects it is duration. The 6-frame dip climbs out to an 8-frame window rather
        than one spanning the whole 90-frame clip, because `_climb_backward` will not cross the
        flat idle around it, and 8 frames is short of the 12 that `min_rep_seconds=0.4`
        demands. Load-bearing: with the plateau rule reverted this reports
        `RepWindow(index=1, start=0, end=89)`.

        SCOPE OF THIS GUARANTEE, precisely. What is pinned is the EXACT-PLATEAU idle case. The
        protection degrades on jittery idle, because jitter makes short stretches locally
        monotonic and the climb walks further: measured at sigma=0.2 over 40 seeds, this same
        6-frame glitch produces a rep in 5/40 raw and 20/40 after the caller median-5 smooths
        it (smoothing lengthens the monotonic stretches, so it makes this worse, not better).
        That is a real limit and it is not claimed away here. Two things bound it: a purely
        jittery static clip with no glitch at all already false-positives at a similar rate
        (13/30) and did so before any of this work, so jittery signals were never covered; and
        closing it would need an estimate of how noisy the signal is -- exactly the family of
        gate that was tried four times and false-rejected paused reps, inter-rep rests, idle
        preambles and fast cadences in turn. Narrowing the claim is the honest option, not
        adding a fifth mechanism.
        """
        # Ninety static frames (many exact repeats, i.e. zero-diff steps) plus one isolated
        # 6-frame glitch -- not a repeated movement, just a single bad detection.
        self.assertEqual(segment_reps(_static_with_glitch(), fps=30.0), [])

    def test_a_stray_sub_threshold_drift_does_not_make_a_glitch_a_rep(self) -> None:
        """The same static-clip-plus-glitch shape, plus one near-duplicate pair whose step is
        tiny but not exactly zero -- float rounding between two otherwise-frozen frames, or a
        quantised reading landing one ULP off its neighbour. (Renamed from
        `test_a_single_tiny_step_does_not_set_the_noise_floor` for the same reason as the test
        above; fixture and assertion unchanged.)

        Every noise-floor estimate this module tried was vulnerable to exactly this value:
        whichever one is smallest sets the floor, so a single 1e-6 step collapses it and any
        span then reads as "not noise". The duration rule is not vulnerable to it at all, and
        structurally rather than by threshold: the drift at frame 70 never crosses `enter`, so
        it is not part of any excursion and cannot influence any window's length. Nothing has
        to be tuned to ignore it. Load-bearing: with the plateau rule reverted this reports
        `RepWindow(index=1, start=0, end=70)`. The exact-plateau caveat on the test above
        applies here too.
        """
        # Same mostly-frozen clip and glitch as the test above, plus ONE additional near-duplicate
        # pair with a sub-threshold (not exactly zero) drift -- the variant a bare `min()` of the
        # nonzero steps would treat as "the noise floor", but which is not itself repetition
        # structure.
        self.assertEqual(segment_reps(_static_with_glitch_and_drift(), fps=30.0), [])

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

    def test_idle_preamble_does_not_dilute_real_reps_away(self) -> None:
        """Every real clip has idle frames before the first rep and after the last -- a live
        camera is recording before the set starts. If the noise/movement estimate is measured
        over the WHOLE clip, a long enough idle preamble outnumbers two perfectly genuine reps
        and gets them rejected outright -- a worse failure than any of the noise-floor bugs this
        estimate exists to catch, since it discards real data instead of accepting bad data.
        """
        reps_signal = sine_reps(2)
        for preamble_frames in (0, 30, 60, 90, 120):
            with self.subTest(preamble_frames=preamble_frames):
                signal = [170.0] * preamble_frames + reps_signal
                reps = segment_reps(signal, fps=30.0)
                self.assertEqual(len(reps), 2)

    def test_idle_at_both_ends_does_not_dilute_real_reps_away(self) -> None:
        # Recording starts before the set and keeps rolling after it ends -- idle on both sides
        # of the two real reps, not just a leading preamble.
        signal = [170.0] * 60 + sine_reps(2) + [170.0] * 60
        reps = segment_reps(signal, fps=30.0)
        self.assertEqual(len(reps), 2)

    def test_a_paused_rep_is_still_one_rep(self) -> None:
        """A paused squat or paused deadlift -- held at the bottom under load -- is ordinary
        training, not an artefact. Every gate this module has tried that measured the
        DISTRIBUTION OF FRAME-TO-FRAME STEPS rejected it, because a hold is precisely a stretch
        of near-zero steps: the longer the athlete pauses, the more the clip looks "frozen" to
        such a gate, and the harder it argues the rep is not real. Duration is the opposite kind
        of measure -- a pause makes the rep LONGER, so it can only push a real rep further past
        `min_rep_seconds`, never below it.
        """
        for hold_frames in (12, 14, 20, 28, 30, 60):
            for jitter in (0.0, 0.3):
                with self.subTest(hold_frames=hold_frames, jitter=jitter):
                    signal = paused_rep(hold_frames, jitter=jitter)
                    reps = segment_reps(signal, fps=30.0)
                    self.assertEqual(len(reps), 1)
                    # The whole rep, hold included, is inside the one window.
                    self.assertEqual(reps[0].start, 0)
                    self.assertEqual(reps[0].end, len(signal) - 1)
                    self.assertFalse(reps[0].partial)

    def test_a_rest_between_reps_does_not_dilute_them_away(self) -> None:
        """Standing idle BETWEEN two reps -- catching a breath, resetting the bar -- is the same
        shape as an idle preamble, only it falls between the first and last excursion instead of
        before them. A gate scoped to "the active span" therefore still swallows it, and long
        enough rests still rejected both perfectly real reps.

        Also pins where the rest frames end up: outside BOTH windows. They are not part of
        either repetition, and phase assignment must not read a 2-second stand as the second
        rep's setup.
        """
        for rest_frames in (30, 60, 90, 150, 300):
            with self.subTest(rest_frames=rest_frames):
                signal = sine_reps(1) + [170.0] * rest_frames + sine_reps(1)
                reps = segment_reps(signal, fps=30.0)
                self.assertEqual(len(reps), 2)
                self.assertFalse(reps[0].partial)
                self.assertFalse(reps[1].partial)
                # The first rep ends at the top it climbs back to (the first frame of the
                # rest), and the second starts at the top it descends from (the last frame of
                # it) -- so the rest itself belongs to neither.
                self.assertEqual(reps[0].end, 30)
                self.assertEqual(reps[1].start, 30 + rest_frames)

    def test_flexed_rejects_a_static_clip_with_one_glitch(self) -> None:
        """The flexed convention needs its OWN anomaly rejection, and for a while it had none.

        `min_rep_seconds` only rejects an anomalous excursion where a window IS an excursion,
        which is true of the extended path and false here: `_windows_from_valleys` builds spans
        valley to valley, so a window's length is the rep-to-rep PERIOD. One stray dip in a
        still clip therefore yields two long windows -- both "partial", both spanning half the
        clip -- and they sail past a `min_frames` filter that is measuring the wrong thing.
        That breaks the module's headline promise of returning [] rather than a guess, on the
        path the brief names for deadlifts.

        Same two fixtures as the extended tests above, so the two conventions are held to the
        same standard on the same signals.
        """
        for label, signal in (("glitch", _static_with_glitch()), ("glitch+drift", _static_with_glitch_and_drift())):
            with self.subTest(fixture=label):
                self.assertEqual(segment_reps(signal, fps=30.0, rep_start="flexed"), [])
                # Held to the same standard as the extended convention on the same signal.
                self.assertEqual(segment_reps(signal, fps=30.0), [])

    def test_flexed_still_segments_real_reps_and_truncated_partials(self) -> None:
        """The other half of the anomaly rejection above: it must discard a lone glitch WITHOUT
        discarding a legitimately truncated leading or trailing rep, which is the case it is
        easiest to break. Both look "incomplete"; what separates them is that a truncated rep
        still contains a genuine descent or ascent, so the excursion it belongs to is long,
        while a glitch's is a handful of frames however the clip is cut.
        """
        # Clean reps are untouched: 3 valleys -> 2 full reps between them, plus a leading and a
        # trailing partial.
        reps = segment_reps(sine_reps(3), fps=30.0, rep_start="flexed")
        self.assertEqual(len(reps), 4)
        self.assertTrue(reps[0].partial)
        self.assertTrue(reps[-1].partial)
        # A clip cut off mid-descent: its final, truncated valley must still count as a rep
        # boundary. Boundaries are pinned rather than only the count, because the count alone
        # does NOT discriminate -- filtering on deep-run length instead of excursion length
        # (round 5's falsified variant C, which drops that valley) yields three windows too,
        # just the wrong three. What changes is the last window: kept here as a complete rep
        # ending at the truncated valley, versus swallowed into a trailing partial.
        trailing = segment_reps(sine_reps(2) + sine_reps(1)[:15], fps=30.0, rep_start="flexed")
        self.assertEqual(
            trailing,
            [
                RepWindow(index=1, start=0, end=14, partial=True),
                RepWindow(index=2, start=15, end=44, partial=False),
                RepWindow(index=3, start=45, end=73, partial=False),
            ],
        )
        # A clip that starts already at the bottom: the leading valley is at frame 0, so there
        # is no leading partial, and the truncated tail is the partial one.
        leading = segment_reps(sine_reps(2)[15:], fps=30.0, rep_start="flexed")
        self.assertEqual(
            leading,
            [
                RepWindow(index=1, start=0, end=29, partial=False),
                RepWindow(index=2, start=30, end=44, partial=True),
            ],
        )
        # And a paused rep is a rep here too, for the same reason as on the extended path.
        self.assertEqual(len(segment_reps(paused_rep(20) + paused_rep(20), fps=30.0, rep_start="flexed")), 3)
        # THE BOUNDARY BAND (see `_windows_from_valleys`'s docstring). `frames_per_rep=30` above
        # gives a 15-frame half-period against `min_frames=12` -- a 3-frame margin that never
        # exercises the boundary discard. Here `frames_per_rep=16` puts the half-period at 8
        # frames, so `rep_period (16) < 2 * min_frames (24)` and the accepted trade fires: the
        # clip ends exactly on a true bottom (a genuine, fully-captured final descent -- not a
        # glitch), but the trailing deep run is a boundary run, `_excursion_bounds` can only
        # measure its unreturned half (well under `min_frames`), so it is discarded from
        # `real_runs` and that valley never becomes a boundary. The would-be third complete rep
        # (40-55) therefore never forms; instead the second valley (40) opens a trailing partial
        # that runs to the clip's actual end (56), one frame past where the discarded valley sat.
        # Pinned as-is per the accepted trade, not as a target to fix.
        boundary = segment_reps(sine_reps(3, 16) + sine_reps(1, 16)[:9], fps=30.0, rep_start="flexed")
        self.assertEqual(
            boundary,
            [
                RepWindow(index=1, start=8, end=23, partial=False),
                RepWindow(index=2, start=24, end=39, partial=False),
                RepWindow(index=3, start=40, end=56, partial=True),
            ],
        )

    def test_reps_separated_by_rests_are_all_found(self) -> None:
        # The full shape of a real recording: idle, then reps with a 2-second stand between
        # each, then idle. Every earlier gate returned [] for this.
        signal = (
            [170.0] * 45
            + sine_reps(1)
            + [170.0] * 60
            + sine_reps(1)
            + [170.0] * 60
            + sine_reps(1)
            + [170.0] * 45
        )
        self.assertEqual(len(segment_reps(signal, fps=30.0)), 3)


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


if __name__ == "__main__":
    unittest.main()
