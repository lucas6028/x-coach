"""Pure helpers of the EgoExo frame extractor.

The 6.4 GiB archive is not in the repository, which is exactly why this logic was written to be
testable without it: the part-ordering walk, the member-path filter and the window plan are all
pure functions.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from src.egoexo.frame_extraction import (
    build_plan,
    contiguous_prefix,
    parse_member_path,
    part_suffix_order,
)


class MemberPathTest(unittest.TestCase):
    def test_it_parses_a_frame_member(self) -> None:
        self.assertEqual(
            parse_member_path("./frames_open/zT0YQO/exo_r/frame_0000000001.jpg"),
            ("zT0YQO", "exo_r", 1),
        )
        self.assertEqual(
            parse_member_path("frames_open/wNsRwL/exo_l/frame_0000012345.jpg"),
            ("wNsRwL", "exo_l", 12345),
        )

    def test_it_rejects_everything_the_archive_carries_that_is_not_a_frame(self) -> None:
        for name in (
            "./frames_open",
            "./frames_open/08ALrC",
            "./frames_open/.DS_Store",
            "./frames_open/._.DS_Store",
            "./frames_open/.hidden/exo_r/frame_0000000001.jpg",
            "./frames_open/zT0YQO/.DS_Store/frame_0000000001.jpg",
            "./frames_open/zT0YQO/exo_r/thumb.png",
            "./frames_open/zT0YQO/exo_r/frame_.jpg",
        ):
            with self.subTest(name=name):
                self.assertIsNone(parse_member_path(name))

    def test_the_frame_number_is_one_based_as_the_archive_writes_it(self) -> None:
        self.assertEqual(parse_member_path("frames_open/a/exo_m/frame_0000000001.jpg")[2], 1)


class ContiguousPrefixTest(unittest.TestCase):
    """THE `.ac` HOLE IS THE WHOLE POINT. `.ad` is on disk and sits AFTER the gap; feeding it to
    the decompressor would supply bytes from the wrong offset rather than extend the stream."""

    def test_it_stops_at_the_gap_and_never_includes_a_part_after_it(self) -> None:
        parts = [Path(f"frames_open.tar.gz.{s}") for s in ("aa", "ab", "ad")]
        self.assertEqual(
            [p.name for p in contiguous_prefix(parts)],
            ["frames_open.tar.gz.aa", "frames_open.tar.gz.ab"],
        )

    def test_an_unbroken_run_is_kept_whole_and_in_order(self) -> None:
        parts = [Path(f"frames_open.tar.gz.{s}") for s in ("ac", "aa", "ad", "ab")]
        self.assertEqual(
            [part_suffix_order(p) for p in contiguous_prefix(parts)], ["aa", "ab", "ac", "ad"]
        )

    def test_a_missing_first_part_yields_nothing_rather_than_starting_midstream(self) -> None:
        self.assertEqual(contiguous_prefix([Path("frames_open.tar.gz.ab")]), [])
        self.assertEqual(contiguous_prefix([]), [])

    def test_the_suffix_successor_carries_like_split_does(self) -> None:
        parts = [Path(f"frames_open.tar.gz.{a}{b}")
                 for a in "ab" for b in "abcdefghijklmnopqrstuvwxyz"]
        # az -> ba must be a successor, or a >26-part archive would truncate at 'az'.
        self.assertEqual(len(contiguous_prefix(parts)), 52)


class BuildPlanTest(unittest.TestCase):
    def _rows(self):
        return [
            {"record_id": "recA", "sample_id": "recA_action_1", "st_frame": "100",
             "ed_frame": "110", "views": "ego_l;exo_l;exo_m;exo_r"},
            {"record_id": "recA", "sample_id": "recA_action_2", "st_frame": "200",
             "ed_frame": "205", "views": "ego_l;exo_l;exo_m;exo_r"},
            {"record_id": "recB", "sample_id": "recB_action_0", "st_frame": "5",
             "ed_frame": "7", "views": "exo_r"},
        ]

    def test_a_frame_inside_a_window_resolves_to_its_action(self) -> None:
        plan = build_plan(self._rows(), ("exo_l", "exo_r"))
        self.assertEqual(plan.lookup("recA", "exo_l", 105), ["recA_action_1"])
        self.assertEqual(plan.lookup("recA", "exo_l", 200), ["recA_action_2"])

    def test_the_window_is_inclusive_at_both_ends(self) -> None:
        """Inclusive on purpose: the manifest's st/ed differ by exactly num_frames_segment while
        the archive numbers frames from 1, so the two plausible conventions are one frame apart.
        Taking both endpoints costs one frame and cannot DROP a real one."""
        plan = build_plan(self._rows(), ("exo_l",))
        self.assertEqual(plan.lookup("recA", "exo_l", 100), ["recA_action_1"])
        self.assertEqual(plan.lookup("recA", "exo_l", 110), ["recA_action_1"])
        self.assertEqual(plan.lookup("recA", "exo_l", 99), [])
        self.assertEqual(plan.lookup("recA", "exo_l", 111), [])

    def test_a_view_the_record_does_not_carry_is_not_planned(self) -> None:
        """Otherwise the report would show a pair that was never going to be filled, and
        `complete_pairs` would understate what the stream actually reached."""
        plan = build_plan(self._rows(), ("exo_l", "exo_r"))
        self.assertEqual(plan.lookup("recB", "exo_l", 6), [])
        self.assertEqual(plan.lookup("recB", "exo_r", 6), ["recB_action_0"])
        self.assertNotIn("recB_action_0__exo_l", plan.expected)

    def test_expected_counts_match_the_inclusive_window(self) -> None:
        plan = build_plan(self._rows(), ("exo_r",))
        self.assertEqual(plan.expected["recA_action_1__exo_r"], 11)
        self.assertEqual(plan.expected["recB_action_0__exo_r"], 3)

    def test_an_unknown_record_or_view_resolves_to_nothing(self) -> None:
        plan = build_plan(self._rows(), ("exo_r",))
        self.assertEqual(plan.lookup("nope", "exo_r", 6), [])
        self.assertEqual(plan.lookup("recA", "ego_l", 105), [])


if __name__ == "__main__":
    unittest.main()
