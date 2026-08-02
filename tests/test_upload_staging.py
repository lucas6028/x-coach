"""Tests for upload staging: object-store put first, temp files for the pipeline, then cleanup.

Offline throughout — the object store is a ``LocalObjectStore`` rooted in a tmp dir, or a stub
that raises, so no R2 credentials or network are involved.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.app.services import analysis, storage


class _FailingStore:
    """An object store whose every put fails, for the fail-fast path."""

    def put(self, key, data, *, content_type):
        raise storage.StorageError("nope")

    def presigned_url(self, key, *, expires_in=storage.DEFAULT_URL_TTL):
        raise storage.StorageError("nope")

    def delete_prefix(self, prefix):
        raise storage.StorageError("nope")


class StageUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.store = storage.LocalObjectStore(self.root)
        patcher = mock.patch.object(storage, "get_object_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_puts_the_source_video_under_an_owner_scoped_prefix(self) -> None:
        staged = analysis.stage_upload(b"video-bytes", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, staged)
        self.assertEqual(staged.prefix, f"uploads/u1/{staged.video_id}")
        found = self.store.open_object(f"{staged.prefix}/source")
        self.assertIsNotNone(found)
        self.assertEqual(found[0].read_bytes(), b"video-bytes")

    def test_source_content_type_follows_the_suffix(self) -> None:
        staged = analysis.stage_upload(b"v", suffix=".webm", owner="anon")
        self.addCleanup(analysis.discard_stage, staged)
        self.assertEqual(self.store.open_object(f"{staged.prefix}/source")[1], "video/webm")

    def test_video_id_is_an_upload_slug(self) -> None:
        staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, staged)
        self.assertTrue(staged.video_id.startswith("upload_"))

    def test_stages_a_temp_copy_the_pipeline_can_open(self) -> None:
        staged = analysis.stage_upload(b"video-bytes", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, staged)
        self.assertTrue(staged.video_path.is_file())
        self.assertEqual(staged.video_path.read_bytes(), b"video-bytes")
        self.assertEqual(staged.video_path.suffix, ".mp4")

    def test_pose_path_is_in_the_same_temp_dir_and_not_yet_written(self) -> None:
        staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, staged)
        self.assertEqual(staged.pose_path.parent, staged.video_path.parent)
        self.assertFalse(staged.pose_path.exists())

    def test_a_failed_source_put_raises_before_any_temp_file_is_written(self) -> None:
        """Fail fast: the video is the one artifact that cannot be recomputed."""
        with mock.patch.object(storage, "get_object_store", return_value=_FailingStore()):
            with self.assertRaises(storage.StorageError):
                analysis.stage_upload(b"v", suffix=".mp4", owner="u1")


class StoreArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.store = storage.LocalObjectStore(self.root)
        patcher = mock.patch.object(storage, "get_object_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, self.staged)

    def test_uploads_pose_json_when_the_analysis_produced_one(self) -> None:
        self.staged.pose_path.write_text(json.dumps({"frames": []}), encoding="utf-8")
        analysis.store_artifacts(self.staged)
        found = self.store.open_object(f"{self.staged.prefix}/pose.json")
        self.assertIsNotNone(found)
        self.assertEqual(found[1], "application/json")

    def test_skips_pose_json_when_the_analysis_never_wrote_one(self) -> None:
        """`analyze_pose_payload` returns the analysis_pending skeleton without writing pose
        JSON for any movement with no registered detector — which is most of the registry."""
        analysis.store_artifacts(self.staged)
        self.assertIsNone(self.store.open_object(f"{self.staged.prefix}/pose.json"))

    def test_uploads_the_thumbnail_as_jpeg(self) -> None:
        analysis.store_artifacts(self.staged, thumbnail=b"jpeg-bytes")
        found = self.store.open_object(f"{self.staged.prefix}/thumb.jpg")
        self.assertIsNotNone(found)
        self.assertEqual(found[0].read_bytes(), b"jpeg-bytes")
        self.assertEqual(found[1], "image/jpeg")

    def test_skips_the_thumbnail_when_absent(self) -> None:
        analysis.store_artifacts(self.staged, thumbnail=None)
        self.assertIsNone(self.store.open_object(f"{self.staged.prefix}/thumb.jpg"))

    def test_never_raises_when_a_derived_put_fails(self) -> None:
        """A storage hiccup must not discard an already-completed (expensive) analysis."""
        self.staged.pose_path.write_text("{}", encoding="utf-8")
        with mock.patch.object(storage, "get_object_store", return_value=_FailingStore()):
            analysis.store_artifacts(self.staged, thumbnail=b"jpeg")  # must not raise


class DiscardStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        patcher = mock.patch.object(storage, "get_object_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_removes_the_temp_directory(self) -> None:
        staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        tmp_dir = staged.video_path.parent
        analysis.discard_stage(staged)
        self.assertFalse(tmp_dir.exists())

    def test_is_idempotent(self) -> None:
        staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        analysis.discard_stage(staged)
        analysis.discard_stage(staged)  # must not raise

    def test_leaves_the_stored_objects_alone(self) -> None:
        """Cleanup is about temp files; the object store is the system of record."""
        staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        analysis.discard_stage(staged)
        self.assertIsNotNone(self.store.open_object(f"{staged.prefix}/source"))


if __name__ == "__main__":
    unittest.main()
