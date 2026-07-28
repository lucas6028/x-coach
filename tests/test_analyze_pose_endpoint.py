"""/api/analyze/pose: accept client pose JSON + video, run the detector off the event loop."""
from __future__ import annotations

import asyncio
import io
import json
import threading
import unittest
from pathlib import Path

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service

_GOOD_POSE = json.dumps({"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": 0}, "frames": []})


def _upload(filename: str = "clip.webm", data: bytes = b"fake") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


class AnalyzePoseEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_save = analysis_service.save_upload
        self._orig_analyze = analysis_service.analyze_pose_payload
        analysis_service.save_upload = lambda data, suffix=".mp4": ("upload_test", Path(f"upload_test{suffix}"))
        analysis_service.analyze_pose_payload = (
            lambda payload, *, movement, video_id=None, max_reps=-1, rep_plan=None: {
                "video_id": video_id, "source": "upload", "movement": movement, "detections": [],
            }
        )

    def tearDown(self) -> None:
        analysis_service.save_upload = self._orig_save
        analysis_service.analyze_pose_payload = self._orig_analyze

    # These tests invoke ``analyze_pose`` directly (not via FastAPI), so the ``max_reps``/``reps``
    # Form defaults are not resolved by FastAPI's DI -- pass both explicitly, since an unresolved
    # ``Form(...)`` sentinel would otherwise reach the validators verbatim.

    def test_happy_path_returns_analysis(self) -> None:
        result = asyncio.run(
            analyze_router.analyze_pose(
                "Squat", _GOOD_POSE, _upload(), max_reps=None, reps=None, user=None
            )
        )
        self.assertEqual(result["video_id"], "upload_test")
        self.assertEqual(result["movement"], "Squat")

    def test_rejects_bad_json(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", "{not json", _upload(), max_reps=None, reps=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_pose_without_frames_list(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", json.dumps({"metadata": {}}), _upload(),
                    max_reps=None, reps=None, user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_malformed_landmarks(self) -> None:
        bad = json.dumps({"metadata": {}, "frames": [{"landmarks": [{"x": 1}]}]})
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", bad, _upload(), max_reps=None, reps=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_unsupported_suffix(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", _GOOD_POSE, _upload("x.txt"), max_reps=None, reps=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_empty_file(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", _GOOD_POSE, _upload(data=b""), max_reps=None, reps=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_runtime_error_maps_to_422(self) -> None:
        def boom(payload, *, movement, video_id=None, max_reps=-1, rep_plan=None):
            raise RuntimeError("boom")

        analysis_service.analyze_pose_payload = boom
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", _GOOD_POSE, _upload(), max_reps=None, reps=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_runs_off_the_event_loop(self) -> None:
        seen: dict[str, threading.Thread] = {}

        def record(payload, *, movement, video_id=None, max_reps=-1, rep_plan=None):
            seen["t"] = threading.current_thread()
            return {"video_id": video_id, "source": "upload", "detections": []}

        analysis_service.analyze_pose_payload = record
        asyncio.run(
            analyze_router.analyze_pose(
                "Squat", _GOOD_POSE, _upload(), max_reps=None, reps=None, user=None
            )
        )
        self.assertIsNot(seen["t"], threading.main_thread())


_LANDMARKS = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]


def _pose(extracted: range, total: int = 60) -> str:
    """A full-length frame list with landmarks only inside `extracted` — the RS-SP2 shape."""
    frames = [
        {"frame_index": i,
         "landmarks": _LANDMARKS if i in extracted else None,
         "world_landmarks": _LANDMARKS if i in extracted else None}
        for i in range(total)
    ]
    return json.dumps({"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": total},
                       "frames": frames})


def _segment(index: int, start: int, end: int, *, analyzed: bool = True) -> dict:
    return {"index": index, "start_frame": start, "end_frame": end,
            "partial": False, "analyzed": analyzed, "refined": True}


def _reps(segments: list[dict], fallback: str | None = None) -> str:
    return json.dumps({"max_reps": 3, "fallback": fallback, "segments": segments})


class AnalyzePoseRepsValidationTests(unittest.TestCase):
    """`reps` is client-supplied and the backend now trusts it for rep boundaries, so every
    violation must be a 400 -- silently ignoring one would leave the backend re-segmenting a
    signal full of holes and emitting plausible-looking, wrong windows (spec §4.3)."""

    def setUp(self) -> None:
        self._orig_save = analysis_service.save_upload
        self._orig_analyze = analysis_service.analyze_pose_payload
        analysis_service.save_upload = lambda data, suffix=".mp4": ("upload_test", Path(f"upload_test{suffix}"))
        analysis_service.analyze_pose_payload = (
            lambda payload, *, movement, video_id=None, max_reps=-1, rep_plan=None: {
                "video_id": video_id, "source": "upload", "movement": movement,
                "detections": [], "rep_plan": rep_plan,
            }
        )

    def tearDown(self) -> None:
        analysis_service.save_upload = self._orig_save
        analysis_service.analyze_pose_payload = self._orig_analyze

    def _run(self, pose: str, reps: str | None):
        return asyncio.run(
            analyze_router.analyze_pose("Squat", pose, _upload(), max_reps=None, reps=reps, user=None)
        )

    def _assert_400(self, pose: str, reps: str | None) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self._run(pose, reps)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_accepts_a_well_formed_plan_and_forwards_it(self) -> None:
        result = self._run(_pose(range(0, 30)), _reps([_segment(1, 0, 29)]))
        plan = result["rep_plan"]
        self.assertEqual([w.index for w in plan.reps], [1])
        self.assertEqual([w.index for w in plan.analyzed], [1])
        self.assertIsNone(plan.fallback)

    def test_rejects_a_window_past_the_end_of_the_clip(self) -> None:
        self._assert_400(_pose(range(0, 30)), _reps([_segment(1, 0, 9999)]))

    def test_rejects_start_after_end(self) -> None:
        self._assert_400(_pose(range(0, 30)), _reps([_segment(1, 20, 5)]))

    def test_rejects_overlapping_windows(self) -> None:
        self._assert_400(
            _pose(range(0, 60)), _reps([_segment(1, 0, 29), _segment(2, 20, 49)])
        )

    def test_rejects_indices_that_do_not_run_1_to_n(self) -> None:
        self._assert_400(_pose(range(0, 30)), _reps([_segment(2, 0, 29)]))

    def test_rejects_an_unknown_fallback_value(self) -> None:
        self._assert_400(_pose(range(0, 30)), _reps([], fallback="because_i_said_so"))

    def test_rejects_more_segments_than_the_cap(self) -> None:
        segments = [_segment(i + 1, i, i) for i in range(analyze_router.MAX_REP_SEGMENTS + 1)]
        self._assert_400(_pose(range(0, 300), total=300), _reps(segments))

    def test_rejects_an_analyzed_window_over_unextracted_frames(self) -> None:
        """The violation every ordering/range/overlap check passes. Scoring all-invalid frames
        produces an EMPTY detection list -- a clean verdict from data nothing measured, which is
        the exact failure frontend/src/lib/quality.ts exists to prevent."""
        self._assert_400(_pose(range(40, 60)), _reps([_segment(1, 0, 29)]))

    def test_allows_an_UNanalyzed_window_over_unextracted_frames(self) -> None:
        """Reps that were found but not scored legitimately have no landmarks — that is the whole
        point of SP2, and `segments[].analyzed=False` is how the payload says so."""
        result = self._run(
            _pose(range(0, 30), total=60),
            _reps([_segment(1, 0, 29), _segment(2, 30, 59, analyzed=False)]),
        )
        self.assertEqual([w.index for w in result["rep_plan"].analyzed], [1])

    def test_rejects_an_analyzed_segment_alongside_any_fallback(self) -> None:
        """A fallback means the whole clip was analysed as one unit (run_detector forces
        whole-clip phase assignment whenever `fallback is not None`). A segment marked
        `analyzed=True` on top of that would be scored per-rep against phases that were never
        assigned per-rep -- the mis-phasing this whole line of work exists to eliminate, reached
        through `reps` instead of through re-segmentation. All three fallback strings must be
        guarded, not just one -- looping so a partial guard still fails this test."""
        for fallback in ("no_reps_detected", "only_partial_reps", "segmentation_disabled"):
            with self.subTest(fallback=fallback):
                self._assert_400(
                    _pose(range(0, 30)), _reps([_segment(1, 0, 29)], fallback=fallback)
                )

    def test_rejects_malformed_reps_json(self) -> None:
        self._assert_400(_pose(range(0, 30)), "{not json")

    def test_omitting_reps_keeps_todays_behaviour(self) -> None:
        """The CLI, the research datasets and old clients send no `reps` and must be unaffected."""
        result = self._run(_pose(range(0, 30)), None)
        self.assertIsNone(result["rep_plan"])


if __name__ == "__main__":
    unittest.main()
