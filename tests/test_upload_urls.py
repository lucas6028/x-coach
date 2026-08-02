"""Tests for the ownership-checked upload URL endpoints, and for the IDOR they close.

`GET /api/video-file/{video_id}` used to fall back to any user's upload with no auth at all.
It now serves library demo clips only; uploads go through the endpoints below, which resolve
`videos.storage_key` with the CALLER'S OWN JWT so Postgres RLS performs the ownership check.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routers import videos as videos_router
from backend.app.services import storage, store


class _User:
    id = "u1"
    token = "tok"


class UploadUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.local = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        patcher = mock.patch.object(storage, "get_object_store", return_value=self.local)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_owner_gets_signed_source_and_thumbnail_urls(self) -> None:
        with mock.patch.object(store, "get_storage_key", return_value="uploads/u1/upload_a"):
            body = videos_router.get_upload_urls("upload_a", user=_User())
        self.assertEqual(body["video_url"], "/api/local-object/uploads/u1/upload_a/source")
        self.assertEqual(body["thumbnail_url"], "/api/local-object/uploads/u1/upload_a/thumb.jpg")
        self.assertEqual(body["expires_in"], storage.DEFAULT_URL_TTL)

    def test_a_row_the_caller_does_not_own_is_a_404(self) -> None:
        """RLS hides someone else's row, so 'not yours' and 'does not exist' are one answer."""
        with mock.patch.object(store, "get_storage_key", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                videos_router.get_upload_urls("upload_a", user=_User())
        self.assertEqual(ctx.exception.status_code, 404)

    def test_a_signing_failure_is_a_503(self) -> None:
        class _Failing:
            def presigned_url(self, key, *, expires_in=storage.DEFAULT_URL_TTL):
                raise storage.StorageError("down")

        with mock.patch.object(storage, "get_object_store", return_value=_Failing()):
            with mock.patch.object(store, "get_storage_key", return_value="uploads/u1/upload_a"):
                with self.assertRaises(HTTPException) as ctx:
                    videos_router.get_upload_urls("upload_a", user=_User())
        self.assertEqual(ctx.exception.status_code, 503)


class UploadUrlBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.local = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        patcher = mock.patch.object(storage, "get_object_store", return_value=self.local)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_one_entry_per_owned_id(self) -> None:
        keys = {"upload_a": "uploads/u1/upload_a", "upload_b": "uploads/u1/upload_b"}
        with mock.patch.object(store, "get_storage_keys", return_value=keys):
            body = videos_router.get_upload_urls_batch(
                videos_router.UploadUrlBatch(video_ids=["upload_a", "upload_b"]), user=_User()
            )
        self.assertEqual(set(body["items"]), {"upload_a", "upload_b"})
        self.assertEqual(
            body["items"]["upload_a"]["video_url"], "/api/local-object/uploads/u1/upload_a/source"
        )

    def test_ids_the_caller_does_not_own_are_simply_absent(self) -> None:
        with mock.patch.object(store, "get_storage_keys", return_value={}):
            body = videos_router.get_upload_urls_batch(
                videos_router.UploadUrlBatch(video_ids=["someone_elses"]), user=_User()
            )
        self.assertEqual(body["items"], {})

    def test_rejects_an_oversized_batch(self) -> None:
        ids = [f"upload_{i}" for i in range(videos_router.MAX_URL_BATCH + 1)]
        with self.assertRaises(HTTPException) as ctx:
            videos_router.get_upload_urls_batch(
                videos_router.UploadUrlBatch(video_ids=ids), user=_User()
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_an_empty_batch_short_circuits_without_a_db_call(self) -> None:
        with mock.patch.object(store, "get_storage_keys") as lookup:
            body = videos_router.get_upload_urls_batch(
                videos_router.UploadUrlBatch(video_ids=[]), user=_User()
            )
        lookup.assert_not_called()
        self.assertEqual(body["items"], {})


class UploadUrlAuthTests(unittest.TestCase):
    """These two endpoints REPLACE the IDOR this branch closes, so their auth is the whole
    security thesis -- and every other test in this file calls the handlers as plain Python
    functions with a ``_User()`` stand-in, which cannot notice ``Depends(get_current_user)``
    being deleted. These go through the real app over HTTP, where the dependency itself is
    what is under test.

    Verified by mutation, not by assumption: with ``= Depends(get_current_user)`` removed from
    ``videos.get_upload_urls``, ``test_get_upload_urls_requires_auth`` fails.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)
        # A sibling suite that forgot to clear its override would otherwise silently authenticate
        # these requests and turn both assertions green for the wrong reason.
        self.assertEqual(app.dependency_overrides, {}, "a leaked dependency override would hide a 401 regression")

    def test_get_upload_urls_requires_auth(self) -> None:
        resp = self.client.get("/api/uploads/upload_a/url")
        self.assertEqual(resp.status_code, 401, resp.text)

    def test_batch_upload_urls_requires_auth(self) -> None:
        # A VALID body on purpose: with a malformed one, a 422 from request validation could
        # stand in for the 401 this test exists to pin, depending on validation order.
        resp = self.client.post("/api/uploads/urls", json={"video_ids": ["upload_a"]})
        self.assertEqual(resp.status_code, 401, resp.text)

    def test_neither_endpoint_reaches_the_database_without_auth(self) -> None:
        """401 must come from the dependency, before any storage-key lookup runs -- otherwise the
        endpoint would be doing owner-scoped work for an unauthenticated caller."""
        with mock.patch.object(store, "get_storage_key") as one, mock.patch.object(
            store, "get_storage_keys"
        ) as many:
            self.client.get("/api/uploads/upload_a/url")
            self.client.post("/api/uploads/urls", json={"video_ids": ["upload_a"]})
        one.assert_not_called()
        many.assert_not_called()


class VideoFileIsLibraryOnlyTests(unittest.TestCase):
    """The regression test for the closed IDOR."""

    def test_an_upload_id_is_a_404(self) -> None:
        from backend.app.services import library

        with mock.patch.object(library, "video_path", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                videos_router.get_video_file("upload_someoneelse")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_uploaded_video_path_no_longer_exists(self) -> None:
        """The lookup itself is gone, so no future caller can re-open the hole."""
        from backend.app.services import library

        self.assertFalse(hasattr(library, "uploaded_video_path"))


if __name__ == "__main__":
    unittest.main()
