"""Tests for the object-storage service (local + R2 implementations).

Fully offline: the local store writes into a tmp dir, and the R2 store is exercised against a
hand-rolled fake client so no credentials, network, or boto3 behaviour is required.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.app.services import storage


class KeyValidationTests(unittest.TestCase):
    def test_rejects_traversal_and_absolute_keys(self) -> None:
        store = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        for bad in ("", "/etc/passwd", "uploads/../../etc/passwd", "uploads/\x00/x"):
            with self.subTest(key=bad), self.assertRaises(storage.StorageError):
                store.put(bad, b"x", content_type="text/plain")

    def test_accepts_a_normal_upload_key(self) -> None:
        root = Path(tempfile.mkdtemp())
        store = storage.LocalObjectStore(root)
        store.put("uploads/u1/upload_abc/source", b"x", content_type="video/mp4")
        self.assertTrue((root / "uploads/u1/upload_abc/source").is_file())


class UploadPrefixTests(unittest.TestCase):
    def test_builds_owner_scoped_prefix(self) -> None:
        self.assertEqual(storage.upload_prefix("u1", "upload_abc"), "uploads/u1/upload_abc")

    def test_anonymous_owner(self) -> None:
        self.assertEqual(storage.upload_prefix("anon", "upload_abc"), "uploads/anon/upload_abc")


class VideoContentTypeTests(unittest.TestCase):
    def test_known_suffixes(self) -> None:
        self.assertEqual(storage.video_content_type(".mp4"), "video/mp4")
        self.assertEqual(storage.video_content_type(".webm"), "video/webm")
        self.assertEqual(storage.video_content_type(".mov"), "video/quicktime")

    def test_unknown_suffix_falls_back_to_mp4(self) -> None:
        self.assertEqual(storage.video_content_type(".xyz"), "video/mp4")


class LocalObjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.store = storage.LocalObjectStore(self.root)

    def test_round_trip_preserves_bytes_and_content_type(self) -> None:
        self.store.put("uploads/u1/v1/source", b"video-bytes", content_type="video/webm")
        found = self.store.open_object("uploads/u1/v1/source")
        self.assertIsNotNone(found)
        path, ctype = found
        self.assertEqual(path.read_bytes(), b"video-bytes")
        self.assertEqual(ctype, "video/webm")

    def test_open_object_returns_none_for_a_missing_key(self) -> None:
        self.assertIsNone(self.store.open_object("uploads/u1/v1/nope"))

    def test_presigned_url_points_at_the_dev_endpoint(self) -> None:
        url = self.store.presigned_url("uploads/u1/v1/source")
        self.assertEqual(url, "/api/local-object/uploads/u1/v1/source")

    def test_delete_prefix_removes_every_object_under_it(self) -> None:
        self.store.put("uploads/u1/v1/source", b"a", content_type="video/mp4")
        self.store.put("uploads/u1/v1/pose.json", b"{}", content_type="application/json")
        self.store.delete_prefix("uploads/u1/v1")
        self.assertFalse((self.root / "uploads/u1/v1").exists())

    def test_delete_prefix_is_a_noop_when_absent(self) -> None:
        self.store.delete_prefix("uploads/u1/gone")  # must not raise

    def test_delete_prefix_does_not_touch_a_sibling_with_a_shared_stem(self) -> None:
        """`uploads/u1/v1` must not reap `uploads/u1/v10`."""
        self.store.put("uploads/u1/v1/source", b"a", content_type="video/mp4")
        self.store.put("uploads/u1/v10/source", b"b", content_type="video/mp4")
        self.store.delete_prefix("uploads/u1/v1")
        self.assertTrue((self.root / "uploads/u1/v10/source").is_file())


class FakeS3Client:
    """Minimal stand-in for the boto3 S3 client surface ``R2ObjectStore`` touches."""

    def __init__(self) -> None:
        self.puts: list[dict] = []
        self.presigns: list[dict] = []
        self.deleted: list[list[dict]] = []
        self.pages: list[dict] = [{"Contents": []}]
        self.raise_on_put = False

    def put_object(self, **kwargs):
        if self.raise_on_put:
            raise ValueError("boom")
        self.puts.append(kwargs)

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        self.presigns.append({"operation": operation, "Params": Params, "ExpiresIn": ExpiresIn})
        return f"https://signed/{Params['Key']}?exp={ExpiresIn}"

    def get_paginator(self, name):
        pages = self.pages

        class _Paginator:
            def paginate(self, **kwargs):
                self.kwargs = kwargs
                return pages

        self._paginator = _Paginator()
        return self._paginator

    def delete_objects(self, *, Bucket, Delete):
        self.deleted.append(Delete["Objects"])


class R2ObjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeS3Client()
        self.store = storage.R2ObjectStore(
            account_id="acc", access_key_id="k", secret_access_key="s", bucket="b"
        )
        self.store._client = self.fake  # bypass the deferred boto3 build

    def test_put_sends_bucket_key_body_and_content_type(self) -> None:
        self.store.put("uploads/u1/v1/source", b"bytes", content_type="video/mp4")
        self.assertEqual(
            self.fake.puts[0],
            {
                "Bucket": "b",
                "Key": "uploads/u1/v1/source",
                "Body": b"bytes",
                "ContentType": "video/mp4",
            },
        )

    def test_put_wraps_client_failure_in_storage_error(self) -> None:
        self.fake.raise_on_put = True
        with self.assertRaises(storage.StorageError):
            self.store.put("uploads/u1/v1/source", b"bytes", content_type="video/mp4")

    def test_presigned_url_requests_get_object_with_the_ttl(self) -> None:
        url = self.store.presigned_url("uploads/u1/v1/source", expires_in=120)
        self.assertEqual(self.fake.presigns[0]["operation"], "get_object")
        self.assertEqual(self.fake.presigns[0]["ExpiresIn"], 120)
        self.assertIn("uploads/u1/v1/source", url)

    def test_presigned_url_default_ttl(self) -> None:
        self.store.presigned_url("uploads/u1/v1/source")
        self.assertEqual(self.fake.presigns[0]["ExpiresIn"], storage.DEFAULT_URL_TTL)

    def test_delete_prefix_lists_with_a_trailing_slash_and_deletes_every_key(self) -> None:
        """The trailing slash is what stops `.../v1` from also reaping `.../v10`."""
        self.fake.pages = [{"Contents": [{"Key": "uploads/u1/v1/source"}, {"Key": "uploads/u1/v1/pose.json"}]}]
        self.store.delete_prefix("uploads/u1/v1")
        self.assertEqual(self.fake._paginator.kwargs["Prefix"], "uploads/u1/v1/")
        self.assertEqual(
            self.fake.deleted[0],
            [{"Key": "uploads/u1/v1/source"}, {"Key": "uploads/u1/v1/pose.json"}],
        )

    def test_delete_prefix_skips_the_delete_call_when_nothing_matches(self) -> None:
        self.fake.pages = [{"Contents": []}]
        self.store.delete_prefix("uploads/u1/v1")
        self.assertEqual(self.fake.deleted, [])


class GetObjectStoreTests(unittest.TestCase):
    def tearDown(self) -> None:
        storage.get_object_store.cache_clear()

    def _settings(self, **kwargs):
        base = {
            "r2_account_id": "",
            "r2_access_key_id": "",
            "r2_secret_access_key": "",
            "r2_bucket": "",
        }
        base.update(kwargs)
        base["storage_configured"] = all(base.values())
        return mock.Mock(**base)

    def test_falls_back_to_local_when_unconfigured(self) -> None:
        storage.get_object_store.cache_clear()
        with mock.patch("backend.app.settings.get_settings", return_value=self._settings()):
            self.assertIsInstance(storage.get_object_store(), storage.LocalObjectStore)

    def test_uses_r2_when_every_setting_is_present(self) -> None:
        storage.get_object_store.cache_clear()
        configured = self._settings(
            r2_account_id="acc", r2_access_key_id="k", r2_secret_access_key="s", r2_bucket="b"
        )
        with mock.patch("backend.app.settings.get_settings", return_value=configured):
            self.assertIsInstance(storage.get_object_store(), storage.R2ObjectStore)


class LocalObjectEndpointTests(unittest.TestCase):
    """The development serving endpoint: reachable locally, inert once R2 is configured."""

    def tearDown(self) -> None:
        storage.get_object_store.cache_clear()

    def test_serves_a_local_object_with_its_stored_content_type(self) -> None:
        from backend.app.routers import videos as videos_router

        root = Path(tempfile.mkdtemp())
        local = storage.LocalObjectStore(root)
        local.put("uploads/u1/v1/source", b"bytes", content_type="video/webm")
        with mock.patch.object(storage, "get_object_store", return_value=local):
            response = videos_router.get_local_object("uploads/u1/v1/source")
        self.assertEqual(response.media_type, "video/webm")

    def test_404s_for_a_missing_key(self) -> None:
        from fastapi import HTTPException

        from backend.app.routers import videos as videos_router

        local = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        with mock.patch.object(storage, "get_object_store", return_value=local):
            with self.assertRaises(HTTPException) as ctx:
                videos_router.get_local_object("uploads/u1/v1/nope")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_404s_for_an_unsafe_key(self) -> None:
        from fastapi import HTTPException

        from backend.app.routers import videos as videos_router

        local = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        with mock.patch.object(storage, "get_object_store", return_value=local):
            with self.assertRaises(HTTPException) as ctx:
                videos_router.get_local_object("uploads/../../etc/passwd")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_is_inert_when_r2_is_live(self) -> None:
        from fastapi import HTTPException

        from backend.app.routers import videos as videos_router

        r2 = storage.R2ObjectStore(account_id="a", access_key_id="k", secret_access_key="s", bucket="b")
        with mock.patch.object(storage, "get_object_store", return_value=r2):
            with self.assertRaises(HTTPException) as ctx:
                videos_router.get_local_object("uploads/u1/v1/source")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
