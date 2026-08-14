from __future__ import annotations

import subprocess
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts.pose.run_pose_extraction import PoseRequest, process_requests


def make_request(root: Path, video_id: str) -> PoseRequest:
    return PoseRequest(
        video_id=video_id,
        video_path=root / f"{video_id}.mp4",
        json_path=root / "pose_json" / f"{video_id}.json",
        annotated_video_path=None,
    )


class ProcessRequestsTests(unittest.TestCase):
    def test_existing_outputs_are_skipped_unless_overwrite(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = [make_request(root, "a"), make_request(root, "b")]
            requests[0].json_path.parent.mkdir(parents=True, exist_ok=True)
            requests[0].json_path.write_text("{}", encoding="utf-8")

            with mock.patch("scripts.pose.run_pose_extraction.run_request") as run_request:
                processed, skipped, failed = process_requests(
                    script_path=root / "process_videos.py",
                    requests=requests,
                    overwrite=False,
                    jobs=1,
                )

            self.assertEqual((processed, skipped, failed), (1, 1, 0))
            self.assertEqual(
                [call.kwargs["request"].video_id for call in run_request.call_args_list],
                ["b"],
            )

            with mock.patch("scripts.pose.run_pose_extraction.run_request") as run_request:
                processed, skipped, failed = process_requests(
                    script_path=root / "process_videos.py",
                    requests=requests,
                    overwrite=True,
                    jobs=1,
                )

            self.assertEqual((processed, skipped, failed), (2, 0, 0))

    def test_one_failing_video_is_counted_without_aborting_the_rest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = [make_request(root, name) for name in ("a", "b", "c")]

            def fake_run(script_path: Path, request: PoseRequest, capture_output: bool = False) -> None:
                if request.video_id == "b":
                    raise subprocess.CalledProcessError(returncode=1, cmd=["mediapipe"])

            with mock.patch("scripts.pose.run_pose_extraction.run_request", side_effect=fake_run):
                processed, skipped, failed = process_requests(
                    script_path=root / "process_videos.py",
                    requests=requests,
                    overwrite=False,
                    jobs=2,
                )

            self.assertEqual((processed, skipped, failed), (2, 0, 1))

    def test_jobs_above_one_keeps_several_videos_in_flight_and_captures_child_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = [make_request(root, f"v{index}") for index in range(3)]

            barrier = threading.Barrier(3, timeout=10)
            capture_flags: list[bool] = []

            def fake_run(script_path: Path, request: PoseRequest, capture_output: bool = False) -> None:
                capture_flags.append(capture_output)
                # Blocks until three workers arrive; times out if the pool is serial.
                barrier.wait()

            with mock.patch("scripts.pose.run_pose_extraction.run_request", side_effect=fake_run):
                processed, skipped, failed = process_requests(
                    script_path=root / "process_videos.py",
                    requests=requests,
                    overwrite=False,
                    jobs=3,
                )

            self.assertEqual((processed, skipped, failed), (3, 0, 0))
            self.assertTrue(all(capture_flags))

    def test_serial_mode_leaves_child_output_on_the_terminal(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("scripts.pose.run_pose_extraction.run_request") as run_request:
                process_requests(
                    script_path=root / "process_videos.py",
                    requests=[make_request(root, "a")],
                    overwrite=False,
                    jobs=1,
                )

            self.assertFalse(run_request.call_args_list[0].kwargs["capture_output"])

    def test_zero_jobs_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            process_requests(script_path=Path("x"), requests=[], overwrite=False, jobs=0)


if __name__ == "__main__":
    unittest.main()
