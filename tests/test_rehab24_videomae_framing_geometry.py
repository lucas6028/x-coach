from __future__ import annotations

import unittest

from src.rehab24.videomae_framing_geometry import (
    REHAB24_FRAMING_VARIANTS,
    box_inside_frame,
    build_report,
    camera_frame_sizes,
    frame_count_findings,
    framing_rows,
    gate_checks,
)

#: The real REHAB24-6 frame sizes. Both are non-square, which is why the letterbox arm
#: has no legitimate no-op here (Fitness-AQA had 768 already-square videos).
CAM17 = [1920, 1080]
CAM18 = [1080, 1920]


def entry(camera: str, frame_size: list[int], box: list[int], video_frames: int = 1000, skeleton_frames: int = 1000) -> dict:
    return {
        "camera": camera,
        "frame_size": frame_size,
        "box": box,
        "video_frames": video_frames,
        "skeleton_frames": skeleton_frames,
        "landmark_box": box,
    }


def index(videos: dict[str, dict]) -> dict:
    return {"box_source": "test", "margin": 0.15, "n_videos": len(videos), "videos": videos}


def manifest_row(sample_id: str, video_path: str, camera: str, exercise_id: str = "6") -> dict[str, str]:
    return {"sample_id": sample_id, "video_path": video_path, "camera": camera, "exercise_id": exercise_id}


class FramingRowsTests(unittest.TestCase):
    def test_is_sample_weighted_not_video_weighted(self) -> None:
        """The LOSO unit is the repetition: a video with 30 reps must count 30 times,
        or the percentiles describe videos rather than what the classifier sees."""
        rows = framing_rows(
            [manifest_row(f"s{i}", "a.mp4", "cam17") for i in range(30)] + [manifest_row("s30", "b.mp4", "cam18")],
            index({"a.mp4": entry("cam17", CAM17, [100, 100, 900, 1000]), "b.mp4": entry("cam18", CAM18, [100, 100, 900, 1800])}),
        )
        self.assertEqual(len(rows), 31)
        self.assertEqual(sum(row["camera"] == "cam17" for row in rows), 30)

    def test_every_repetition_of_a_video_gets_the_identical_box(self) -> None:
        rows = framing_rows(
            [manifest_row("s1", "a.mp4", "cam17"), manifest_row("s2", "a.mp4", "cam17")],
            index({"a.mp4": entry("cam17", CAM17, [100, 100, 900, 1000])}),
        )
        self.assertEqual(rows[0]["box"], rows[1]["box"])

    def test_fails_closed_when_a_sample_has_no_box_entry(self) -> None:
        with self.assertRaises(KeyError):
            framing_rows([manifest_row("s1", "missing.mp4", "cam17")], index({}))


class FrameCountFindingsTests(unittest.TestCase):
    def test_tolerates_a_one_frame_container_header_gap(self) -> None:
        """86 of 130 real videos have a skeleton one frame longer than the container.
        That is a decoder/header off-by-one, not two files of different footage."""
        findings = frame_count_findings(index({"a.mp4": entry("cam17", CAM17, [0, 0, 10, 10], 1000, 1001)}))
        self.assertEqual(findings["n_within_tolerance_but_unequal"], 1)
        self.assertEqual(findings["over_tolerance"], [])

    def test_reports_a_real_disagreement(self) -> None:
        findings = frame_count_findings(index({"a.mp4": entry("cam17", CAM17, [0, 0, 10, 10], 1000, 1200)}))
        self.assertEqual(len(findings["over_tolerance"]), 1)


class BoxSanityTests(unittest.TestCase):
    def test_accepts_a_box_inside_its_frame(self) -> None:
        rows = [{"video_id": "s1", "frame_size": CAM17, "box": [10, 20, 1900, 1000]}]
        self.assertEqual(box_inside_frame(rows), [])

    def test_reports_a_box_that_leaves_the_frame(self) -> None:
        """background_only's mask must cover its box; a box past the edge means part of
        the athlete is never painted over."""
        rows = [{"video_id": "s1", "frame_size": CAM17, "box": [10, 20, 5000, 1000]}]
        self.assertEqual(len(box_inside_frame(rows)), 1)

    def test_reports_an_inverted_box(self) -> None:
        rows = [{"video_id": "s1", "frame_size": CAM17, "box": [500, 20, 100, 1000]}]
        self.assertEqual(len(box_inside_frame(rows)), 1)

    def test_camera_frame_sizes_are_grouped_per_camera(self) -> None:
        sizes = camera_frame_sizes(
            index({"a.mp4": entry("cam17", CAM17, [0, 0, 10, 10]), "b.mp4": entry("cam18", CAM18, [0, 0, 10, 10])})
        )
        self.assertEqual(sizes, {"cam17": [CAM17], "cam18": [CAM18]})


class GateTests(unittest.TestCase):
    def report(self, videos: dict[str, dict], rows: list[dict[str, str]]) -> dict:
        return build_report(rows, index(videos), REHAB24_FRAMING_VARIANTS)

    def test_a_realistic_pair_of_cameras_passes(self) -> None:
        report = self.report(
            {
                "a17.mp4": entry("cam17", CAM17, [400, 40, 1500, 1040]),
                "a18.mp4": entry("cam18", CAM18, [140, 200, 940, 1750]),
            },
            [manifest_row("s1", "a17.mp4", "cam17"), manifest_row("s2", "a18.mp4", "cam18")],
        )
        self.assertTrue(report["passed"], report["checks"])

    def test_the_portrait_camera_is_the_one_full_frame_truncates(self) -> None:
        """The premise of the whole plan: cam18's centre crop takes the athlete's feet,
        cam17's keeps their full height. Measured, not assumed."""
        report = self.report(
            {
                "a17.mp4": entry("cam17", CAM17, [400, 40, 1500, 1040]),
                "a18.mp4": entry("cam18", CAM18, [140, 200, 940, 1750]),
            },
            [manifest_row("s1", "a17.mp4", "cam17"), manifest_row("s2", "a18.mp4", "cam18")],
        )
        cam17 = report["by_camera"]["cam17"]["arms"]["full_frame"]
        cam18 = report["by_camera"]["cam18"]["arms"]["full_frame"]
        self.assertEqual(cam17["bottom_loss"]["median"], 0.0)
        self.assertGreater(cam18["bottom_loss"]["median"], 0.0)

    def test_letterbox_restores_the_whole_athlete_on_both_cameras(self) -> None:
        report = self.report(
            {
                "a17.mp4": entry("cam17", CAM17, [400, 40, 1500, 1040]),
                "a18.mp4": entry("cam18", CAM18, [140, 200, 940, 1750]),
            },
            [manifest_row("s1", "a17.mp4", "cam17"), manifest_row("s2", "a18.mp4", "cam18")],
        )
        self.assertTrue(report["checks"]["letterbox_truncates_nothing"])
        for camera in ("cam17", "cam18"):
            self.assertEqual(report["by_camera"][camera]["arms"]["full_frame_letterbox"]["n_truncated"], 0)

    def test_letterbox_costs_body_area_and_the_report_says_so(self) -> None:
        """Two arms scoring the same is consistent with a completeness gain cancelling
        a resolution loss. That reading must be available before the accuracy is."""
        report = self.report(
            {"a18.mp4": entry("cam18", CAM18, [140, 200, 940, 1750])},
            [manifest_row("s1", "a18.mp4", "cam18")],
        )
        arms = report["overall"]["arms"]
        self.assertLess(
            arms["full_frame_letterbox"]["body_area_fraction"]["median"],
            arms["full_frame"]["body_area_fraction"]["median"],
        )

    def test_the_gate_fails_when_a_camera_has_two_frame_sizes(self) -> None:
        report = self.report(
            {
                "a.mp4": entry("cam17", CAM17, [400, 40, 1500, 1040]),
                "b.mp4": entry("cam17", [1280, 720], [200, 20, 1000, 700]),
            },
            [manifest_row("s1", "a.mp4", "cam17"), manifest_row("s2", "b.mp4", "cam17")],
        )
        self.assertFalse(report["checks"]["each_camera_has_one_frame_size"])
        self.assertFalse(report["passed"])

    def test_the_gate_fails_when_a_letterboxed_sample_is_untransformed(self) -> None:
        """A square frame would make the arm a no-op. REHAB24-6 has none, so a sample
        reporting one means the pipeline was fed the wrong frame size."""
        checks = gate_checks(
            {
                "camera_frame_sizes": {"cam17": [[1920, 1920]]},
                "frame_counts": {"over_tolerance": []},
                "boxes_outside_frame": [],
                "overall": {
                    "n_boxless": 0,
                    "arms": {"full_frame_letterbox": {"n_truncated": 0, "n_identical_to_source": 3}},
                },
                "by_camera": {},
            },
            ("full_frame_letterbox",),
        )
        self.assertFalse(checks["letterbox_transforms_every_sample"])


if __name__ == "__main__":
    unittest.main()
