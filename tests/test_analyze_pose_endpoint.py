"""/api/analyze/pose: accept client pose JSON + video, run the detector off the event loop."""
from __future__ import annotations

import asyncio
import io
import json
import threading
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service
from backend.app.services import runtime_config
from backend.app.services import storage
from backend.app.services import store

_GOOD_POSE = json.dumps({"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": 0}, "frames": []})


def _upload(filename: str = "clip.webm", data: bytes = b"fake") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


class AnalyzePoseEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_stage = analysis_service.stage_upload
        self._orig_artifacts = analysis_service.store_artifacts
        self._orig_discard = analysis_service.discard_stage
        self._orig_analyze = analysis_service.analyze_pose_payload

        # No real disk or object-store I/O: hand back a deterministic StagedUpload and record
        # what the router does with it.
        self.staged = analysis_service.StagedUpload(
            video_id="upload_test",
            prefix="uploads/anon/upload_test",
            video_path=Path("upload_test.mp4"),
            pose_path=Path("pose.json"),
        )
        self.artifacts: list[dict] = []
        self.discarded: list[object] = []
        analysis_service.stage_upload = lambda data, *, suffix=".mp4", owner="anon": self.staged
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: (
            self.artifacts.append({"staged": staged, "thumbnail": thumbnail}) or 0
        )
        analysis_service.discard_stage = lambda staged: self.discarded.append(staged)
        analysis_service.analyze_pose_payload = (
            lambda payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1, rep_plan=None: {
                "video_id": video_id, "source": "upload", "movement": movement, "detections": [],
            }
        )

        # Presigning is a storage concern; stub it so these tests stay offline.
        presign = mock.patch.object(
            analyze_router, "_source_url", side_effect=lambda prefix: f"https://signed/{prefix}"
        )
        presign.start()
        self.addCleanup(presign.stop)

        # Reaping a failed upload's objects is a storage concern too: unpatched it resolves the
        # LIVE object store, which is an R2 client (and a real network delete) on any machine
        # whose env carries R2 credentials. See AnalyzePoseStorageTests for what it should do.
        reap = mock.patch.object(store, "_reap_objects")
        reap.start()
        self.addCleanup(reap.stop)

        # KEEP THESE TESTS OFFLINE, as the module docstring promises. ``analyze_pose`` calls
        # ``settings.allowed_upload_suffixes()``, which reads the admin overrides via
        # ``runtime_config.get_overrides()`` -- and that does a REAL Supabase round-trip whenever
        # auth is configured. ``{}`` is exactly what ``get_overrides`` returns when auth is
        # unconfigured, so this runs the same code path CI does rather than a bespoke stub.
        overrides = mock.patch.object(runtime_config, "get_overrides", return_value={})
        overrides.start()
        self.addCleanup(overrides.stop)

    def tearDown(self) -> None:
        analysis_service.stage_upload = self._orig_stage
        analysis_service.store_artifacts = self._orig_artifacts
        analysis_service.discard_stage = self._orig_discard
        analysis_service.analyze_pose_payload = self._orig_analyze

    # These tests invoke ``analyze_pose`` directly (not via FastAPI), so the ``max_reps``/``reps``/
    # ``thumbnail`` Form/File defaults are not resolved by FastAPI's DI -- pass all of them
    # explicitly, since an unresolved ``Form(...)``/``File(...)`` sentinel would otherwise reach
    # the validators (``_validated_max_reps`` / ``_validate_reps`` / ``_read_thumbnail``) verbatim.

    def test_happy_path_returns_analysis(self) -> None:
        result = asyncio.run(
            analyze_router.analyze_pose(
                "Squat", _GOOD_POSE, _upload(),
                max_reps=None, reps=None, thumbnail=None, user=None,
            )
        )
        self.assertEqual(result["video_id"], "upload_test")
        self.assertEqual(result["movement"], "Squat")

    def test_rejects_bad_json(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", "{not json", _upload(),
                    max_reps=None, reps=None, thumbnail=None, user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_oversized_pose_upload_before_staging(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat",
                    _upload("pose.json", b"x" * (analyze_router.MAX_POSE_JSON_BYTES + 1)),
                    _upload(),
                    max_reps=None,
                    reps=None,
                    thumbnail=None,
                    user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 413)

    def test_rejects_pose_without_frames_list(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", json.dumps({"metadata": {}}), _upload(),
                    max_reps=None, reps=None, thumbnail=None, user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_malformed_landmarks(self) -> None:
        bad = json.dumps({"metadata": {}, "frames": [{"landmarks": [{"x": 1}]}]})
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", bad, _upload(),
                    max_reps=None, reps=None, thumbnail=None, user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_unsupported_suffix(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", _GOOD_POSE, _upload("x.txt"),
                    max_reps=None, reps=None, thumbnail=None, user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_empty_file(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", _GOOD_POSE, _upload(data=b""),
                    max_reps=None, reps=None, thumbnail=None, user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_runtime_error_maps_to_422(self) -> None:
        def boom(payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1, rep_plan=None):
            raise RuntimeError("boom")

        analysis_service.analyze_pose_payload = boom
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", _GOOD_POSE, _upload(),
                    max_reps=None, reps=None, thumbnail=None, user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_runs_off_the_event_loop(self) -> None:
        seen: dict[str, threading.Thread] = {}

        def record(payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1, rep_plan=None):
            seen["t"] = threading.current_thread()
            return {"video_id": video_id, "source": "upload", "detections": []}

        analysis_service.analyze_pose_payload = record
        asyncio.run(
            analyze_router.analyze_pose(
                "Squat", _GOOD_POSE, _upload(),
                max_reps=None, reps=None, thumbnail=None, user=None,
            )
        )
        self.assertIsNot(seen["t"], threading.main_thread())

    def test_stores_artifacts_even_for_the_analysis_pending_skeleton(self) -> None:
        """A movement with no detector still has a source video and a thumbnail worth keeping."""
        analysis_service.analyze_pose_payload = (
            lambda payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1, rep_plan=None: {
                "video_id": video_id,
                "source": "upload",
                "analysis_pending": True,
                "detections": [],
            }
        )
        asyncio.run(
            analyze_router.analyze_pose(
                movement="High Knee",
                pose=json.dumps({"frames": []}),
                file=_upload(),
                max_reps=None,
                reps=None,
                thumbnail=None,
                user=None,
            )
        )
        self.assertEqual(len(self.artifacts), 1)


class AnalyzePoseStorageTests(unittest.TestCase):
    """The object-storage contract of /api/analyze/pose: mirrors ``AnalyzeStorageTests`` in
    ``tests/test_analyze_endpoint.py`` -- both endpoints now share ``_stage_analyze_persist``,
    but each endpoint's own coverage still has to exist independently: nothing stops a future
    change to ``analyze_pose``'s call site (e.g. dropping the ``user``/``thumb`` kwargs it passes
    into the shared helper) from breaking this endpoint alone while `/api/analyze`'s tests stay
    green.
    """

    def setUp(self) -> None:
        self._orig = {
            name: getattr(analysis_service, name)
            for name in ("stage_upload", "store_artifacts", "discard_stage", "analyze_pose_payload")
        }
        overrides = mock.patch.object(runtime_config, "get_overrides", return_value={})
        overrides.start()
        self.addCleanup(overrides.stop)

        self.staged = analysis_service.StagedUpload(
            video_id="upload_test",
            prefix="uploads/anon/upload_test",
            video_path=Path("upload_test.mp4"),
            pose_path=Path("pose.json"),
        )
        self.artifacts: list[dict] = []
        self.discarded: list[object] = []
        self.stage_calls: list[dict] = []

        def _stage_upload(data, *, suffix=".mp4", owner="anon"):
            self.stage_calls.append({"data": data, "suffix": suffix, "owner": owner})
            return self.staged

        analysis_service.stage_upload = _stage_upload
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: (
            self.artifacts.append({"staged": staged, "thumbnail": thumbnail}) or 0
        )
        analysis_service.discard_stage = lambda staged: self.discarded.append(staged)
        analysis_service.analyze_pose_payload = (
            lambda payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1, rep_plan=None: {
                "video_id": video_id, "source": "upload", "movement": movement, "detections": [],
            }
        )
        presign = mock.patch.object(
            analyze_router, "_source_url", side_effect=lambda prefix: f"https://signed/{prefix}"
        )
        presign.start()
        self.addCleanup(presign.stop)

        # Stubbed for the whole class, both to keep the reap offline (unpatched it resolves the
        # LIVE object store) and so every test can assert on WHETHER it fired, not only the ones
        # that expect it to.
        reap = mock.patch.object(store, "_reap_objects")
        self.reap = reap.start()
        self.addCleanup(reap.stop)

        # KEEP OFFLINE: the quota gate reads the caller's usage for any signed-in user, and
        # unpatched ``get_storage_used`` builds a LIVE Supabase client. These tests assert on
        # staging/reaping/persistence, not on the quota — 0 used means every upload fits.
        used = mock.patch.object(store, "get_storage_used", return_value=0)
        used.start()
        self.addCleanup(used.stop)

    def tearDown(self) -> None:
        for name, value in self._orig.items():
            setattr(analysis_service, name, value)

    def _run(self, **kwargs):
        params = {
            "movement": "Squat",
            "pose": _GOOD_POSE,
            "file": _upload(),
            "max_reps": None,
            "reps": None,
            "thumbnail": None,
            "user": None,
        }
        params.update(kwargs)
        return asyncio.run(analyze_router.analyze_pose(**params))

    def test_returns_a_presigned_video_url(self) -> None:
        result = self._run()
        self.assertEqual(result["video_url"], "https://signed/uploads/anon/upload_test")

    def test_a_storage_failure_before_analysis_is_a_503(self) -> None:
        def boom(data, *, suffix=".mp4", owner="anon"):
            raise storage.StorageError("R2 down")

        analysis_service.stage_upload = boom
        ran = []
        analysis_service.analyze_pose_payload = lambda *a, **k: ran.append(1) or {}
        with self.assertRaises(HTTPException) as ctx:
            self._run()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ran, [], "no CPU may be spent on a clip whose video could not be stored")

    def test_stores_the_derived_artifacts_after_a_successful_analysis(self) -> None:
        self._run()
        self.assertEqual(len(self.artifacts), 1)
        self.assertIs(self.artifacts[0]["staged"], self.staged)

    def test_forwards_the_thumbnail_bytes(self) -> None:
        """The mirror of ``/api/analyze``'s own test, for the reason this class exists: nothing
        stops a change to ``analyze_pose``'s call site from dropping the ``thumb`` kwarg it
        passes into the shared helper while the other endpoint's tests stay green."""
        thumb = UploadFile(file=io.BytesIO(b"jpeg-bytes"), filename="t.jpg",
                           headers=Headers({"content-type": "image/jpeg"}))
        self._run(thumbnail=thumb)
        self.assertEqual(self.artifacts[0]["thumbnail"], b"jpeg-bytes")

    def test_accepts_the_image_jpg_content_type_alias(self) -> None:
        thumb = UploadFile(file=io.BytesIO(b"jpeg-bytes"), filename="t.jpg",
                           headers=Headers({"content-type": "image/jpg"}))
        self._run(thumbnail=thumb)
        self.assertEqual(self.artifacts[0]["thumbnail"], b"jpeg-bytes")

    def test_drops_a_thumbnail_with_an_unusable_type_instead_of_failing(self) -> None:
        thumb = UploadFile(file=io.BytesIO(b"png"), filename="t.png",
                           headers=Headers({"content-type": "image/png"}))
        result = self._run(thumbnail=thumb)
        self.assertEqual(result["video_id"], "upload_test")
        self.assertIsNone(self.artifacts[0]["thumbnail"], "the bad thumbnail must be dropped")

    def test_drops_an_oversized_thumbnail_instead_of_failing(self) -> None:
        big = b"x" * (analyze_router.MAX_THUMBNAIL_BYTES + 1)
        thumb = UploadFile(file=io.BytesIO(big), filename="t.jpg",
                           headers=Headers({"content-type": "image/jpeg"}))
        result = self._run(thumbnail=thumb)
        self.assertEqual(result["video_id"], "upload_test")
        self.assertIsNone(self.artifacts[0]["thumbnail"], "the oversized thumbnail is dropped")

    def test_reaps_the_stored_objects_when_the_analysis_fails(self) -> None:
        """The source is stored BEFORE the analysis runs and a failed analysis writes no
        ``videos`` row, so nothing else would ever delete it."""
        def boom(*args, **kwargs):
            raise RuntimeError("detector failed")

        analysis_service.analyze_pose_payload = boom
        with self.assertRaises(HTTPException):
            self._run()
        self.reap.assert_called_once_with([self.staged.prefix])

    def test_reaps_the_stored_objects_when_the_analysis_fails_unexpectedly(self) -> None:
        def boom(*args, **kwargs):
            raise ValueError("something nobody predicted")

        analysis_service.analyze_pose_payload = boom
        with self.assertRaises(ValueError):
            self._run()
        self.reap.assert_called_once_with([self.staged.prefix])

    def test_never_reaps_after_a_successful_analysis(self) -> None:
        """Pins WHERE the reap sits: one that also fired on success would delete the clip the
        caller is being handed a live ``video_url`` for."""
        self._run()
        self.reap.assert_not_called()

    def test_never_reaps_when_only_the_history_write_failed(self) -> None:
        """A documented, accepted orphan: the analysis succeeded and the client holds a live
        playback URL, so deleting the source inline would break a working session."""
        user = mock.Mock(id="u1", token="tok")
        with mock.patch.object(store, "persist_analysis", side_effect=RuntimeError("db down")):
            result = self._run(user=user)
        self.assertIsNone(result["analysis_id"])
        self.reap.assert_not_called()

    def test_discards_the_stage_even_when_the_analysis_fails(self) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("detector failed")

        analysis_service.analyze_pose_payload = boom
        with self.assertRaises(HTTPException):
            self._run()
        self.assertEqual(self.discarded, [self.staged])
        self.assertEqual(self.artifacts, [], "a failed analysis stores no derived artifacts")

    def test_discards_the_stage_after_a_successful_analysis(self) -> None:
        self._run()
        self.assertEqual(self.discarded, [self.staged])

    def test_video_url_is_not_written_into_the_persisted_result(self) -> None:
        """A presigned URL in the history row would be expired the moment it is replayed."""
        persisted: list[dict] = []

        def fake_persist(**kwargs):
            persisted.append(dict(kwargs["result"]))
            return "analysis-1"

        user = mock.Mock(id="u1", token="tok")
        with mock.patch.object(store, "persist_analysis", side_effect=fake_persist):
            result = self._run(user=user)
        self.assertNotIn("video_url", persisted[0])
        self.assertIn("video_url", result)

    def test_persists_the_storage_prefix_as_the_storage_key(self) -> None:
        seen: list[dict] = []
        user = mock.Mock(id="u1", token="tok")
        with mock.patch.object(store, "persist_analysis", side_effect=lambda **kw: seen.append(kw) or "id"):
            self._run(user=user)
        self.assertEqual(seen[0]["storage_key"], "uploads/anon/upload_test")

    def test_stages_under_the_anon_owner_for_an_anonymous_caller(self) -> None:
        self._run()
        self.assertEqual(self.stage_calls[0]["owner"], "anon")

    def test_stages_under_the_authenticated_users_id(self) -> None:
        user = mock.Mock(id="u1", token="tok")
        with mock.patch.object(store, "persist_analysis", return_value="id"):
            self._run(user=user)
        self.assertEqual(self.stage_calls[0]["owner"], "u1")


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
        # Mirrors AnalyzePoseStorageTests' setUp: analyze_pose now stages/persists via
        # stage_upload/store_artifacts/discard_stage (R2), not the old save_upload() pair, so
        # this class's rep-plan-focused tests need the same offline stand-ins to reach
        # analyze_pose_payload at all.
        self._orig = {
            name: getattr(analysis_service, name)
            for name in ("stage_upload", "store_artifacts", "discard_stage", "analyze_pose_payload")
        }
        overrides = mock.patch.object(runtime_config, "get_overrides", return_value={})
        overrides.start()
        self.addCleanup(overrides.stop)

        self.staged = analysis_service.StagedUpload(
            video_id="upload_test",
            prefix="uploads/anon/upload_test",
            video_path=Path("upload_test.mp4"),
            pose_path=Path("pose.json"),
        )
        analysis_service.stage_upload = lambda data, *, suffix=".mp4", owner="anon": self.staged
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: 0
        analysis_service.discard_stage = lambda staged: None
        analysis_service.analyze_pose_payload = (
            lambda payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1, rep_plan=None: {
                "video_id": video_id, "source": "upload", "movement": movement,
                "detections": [], "rep_plan": rep_plan,
            }
        )

        presign = mock.patch.object(
            analyze_router, "_source_url", side_effect=lambda prefix: f"https://signed/{prefix}"
        )
        presign.start()
        self.addCleanup(presign.stop)

        reap = mock.patch.object(store, "_reap_objects")
        reap.start()
        self.addCleanup(reap.stop)

    def tearDown(self) -> None:
        for name, value in self._orig.items():
            setattr(analysis_service, name, value)

    def _run(self, pose: str, reps: str | None):
        return asyncio.run(
            analyze_router.analyze_pose(
                "Squat", pose, _upload(), max_reps=None, reps=reps, thumbnail=None, user=None
            )
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

    def test_accepts_an_analyzed_window_with_partial_landmark_coverage(self) -> None:
        """The inverse of the above: a window with landmarks on SOME but not all of its frames is
        accepted, not rejected. This represents a real MediaPipe dropout mid-rep -- motion blur at
        the bottom of a squat, brief self-occlusion, a barbell crossing the hips -- where the frame
        was sampled but no pose was detected for it. The browser client (`poseExtract.ts`'s
        `sampleFrames`) calls `landmarksToFrame` for every index in the dense span unconditionally,
        so a sampled span can legitimately carry a `landmarks: null` frame or two without that
        meaning the span itself was never extracted. Requiring every frame to carry landmarks
        (an earlier `all(...)` form of this guard) would 400 this kind of ordinary clip. Fails
        against that `all(...)` form."""
        result = self._run(
            _pose(range(500, 501), total=600), _reps([_segment(1, 0, 500)])
        )
        self.assertEqual([w.index for w in result["rep_plan"].analyzed], [1])

    def test_rejects_a_non_fallback_plan_with_nothing_analyzed(self) -> None:
        """`fallback: null` asserts "I segmented this clip myself" -- if it then analyzes NOTHING,
        `run_detector` still takes the per-rep phasing path (`segmented = reps`) because `reps` is
        non-empty, but scores an empty `analyzed` list, i.e. rules run over the WHOLE sparse clip.
        The extracted frames are valid, so `valid_frames > 0`, `wasMeasured` reads true, and an
        empty detection list renders as a clean rep on a clip that was mostly never measured --
        the exact failure frontend/src/lib/quality.ts exists to prevent."""
        self._assert_400(
            _pose(range(0, 30), total=60),
            _reps([_segment(1, 0, 29, analyzed=False), _segment(2, 30, 59, analyzed=False)]),
        )

    def test_accepts_a_fallback_plan_with_nothing_analyzed(self) -> None:
        """The legitimate counterpart to the two tests above: `fallback: "only_partial_reps"` (or
        either other fallback string) with a non-empty `reps` and an empty `analyzed` is Task 9's
        accepted shape -- the span WAS scored, as part of the whole-clip fallback. The new
        'must analyze at least one segment' guard is scoped to `fallback is None` and must not
        reject this, or a later tightening to an unconditional `if not analyzed: raise` would break
        a real client path while this suite stayed green."""
        result = self._run(
            _pose(range(0, 30)),
            _reps([_segment(1, 0, 29, analyzed=False)], fallback="only_partial_reps"),
        )
        plan = result["rep_plan"]
        self.assertEqual(plan.fallback, "only_partial_reps")
        self.assertEqual(plan.analyzed, ())

    def test_rejects_a_non_fallback_plan_with_no_segments_at_all(self) -> None:
        """The degenerate case of the same guard: an empty `segments` list with `fallback: null`
        analyzes nothing just as surely as an all-unanalyzed list does."""
        self._assert_400(_pose(range(0, 30)), _reps([]))

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

    def test_clipped_segments_are_logged(self) -> None:
        """`refined: "clipped"` never reaches `RepWindow` (no such field on the dataclass), so
        without a log line it is write-only -- the spec's claim that it "makes an insufficient
        REP_PADDING_FRAMES visible" would otherwise be false in production. This is the smallest
        fix that makes it true: log, don't change behaviour."""
        clipped = {"index": 1, "start_frame": 0, "end_frame": 29,
                   "partial": False, "analyzed": True, "refined": "clipped"}
        with self.assertLogs(analyze_router.logger, level="INFO") as ctx:
            self._run(_pose(range(0, 30)), _reps([clipped]))
        self.assertTrue(any("clipped" in message for message in ctx.output))

    def test_non_clipped_segments_do_not_log(self) -> None:
        with self.assertRaises(AssertionError):
            # assertLogs itself raises AssertionError when nothing was logged at INFO+ -- the
            # happy-path plan (refined: true, via `_segment`) must not emit this line.
            with self.assertLogs(analyze_router.logger, level="INFO"):
                self._run(_pose(range(0, 30)), _reps([_segment(1, 0, 29)]))

    def test_omitting_reps_keeps_todays_behaviour(self) -> None:
        """The CLI, the research datasets and old clients send no `reps` and must be unaffected."""
        result = self._run(_pose(range(0, 30)), None)
        self.assertIsNone(result["rep_plan"])


if __name__ == "__main__":
    unittest.main()
