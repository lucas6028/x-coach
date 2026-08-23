"""Object storage for user-uploaded artifacts (raw video, pose JSON, thumbnail).

Two backends behind one narrow interface, so no caller ever learns which one is live:

* ``LocalObjectStore`` — plain files under ``data/runtime/objects/``. The default, and what CI
  and offline development run on: no credentials, no network.
* ``R2ObjectStore`` — Cloudflare R2 over the S3 API (boto3), used only when every ``R2_*``
  setting is present.

Key layout (see docs/superpowers/specs/2026-08-02-r2-object-storage-design.md)::

    uploads/{owner}/{video_id}/source      the raw upload; ContentType carries the format
    uploads/{owner}/{video_id}/pose.json   the pose JSON, when the analysis produced one
    uploads/{owner}/{video_id}/thumb.jpg   one browser-captured frame

``owner`` is the authenticated user's id, or the literal ``anon`` for a demo upload. The source
object deliberately carries NO file extension: R2 replays the ``ContentType`` set at put time and
``<video src>`` plays by content type, so an extension would only force the read path to discover
which suffix was used.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from backend.app import config

# Where LocalObjectStore keeps its objects. Under data/runtime/, which is gitignored.
OBJECTS_DIR = config.RUNTIME_DIR / "objects"

# How long a presigned playback URL stays valid. One hour: long enough for a coaching session,
# short enough that a leaked URL is not a durable capability.
DEFAULT_URL_TTL = 3600

# What R2 replays as ``Cache-Control`` on every stored object. Safe to make this aggressive
# because keys are WRITE-ONCE: ``upload_prefix`` embeds a fresh uuid per upload, and nothing in
# the codebase overwrites an existing key — a changed clip is a new video_id. Without it a
# re-watch, or a scrub back past what the browser dropped, re-fetches ranges over the network.
#
# Note the URL query string changes every time the URL is re-signed, which would ordinarily defeat
# the cache. The browser's MEDIA cache is keyed on the full URL, so the win here is within one
# signed URL's hour — exactly the window a coaching session spends scrubbing one clip.
DEFAULT_CACHE_CONTROL = "public, max-age=31536000, immutable"

# Suffixes mimetypes gets wrong or does not know, and the fallback for anything unrecognised.
# Guessing wrong here means a clip that will not play, so the map is explicit rather than clever.
_VIDEO_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}
_DEFAULT_VIDEO_CONTENT_TYPE = "video/mp4"

# A key is interpolated into a filesystem path by LocalObjectStore, so backslashes (a Windows
# separator) and NUL are as dangerous as a traversal segment.
_UNSAFE_KEY_CHARS = frozenset("\\\x00")


class StorageError(RuntimeError):
    """A put / presign / delete against the object store failed, or a key was unsafe."""


def _validate_key(key: str) -> str:
    """Return ``key`` if it is safe to interpolate into a path or an S3 key, else raise."""
    if not key or key.startswith("/") or ".." in key.split("/"):
        raise StorageError(f"Unsafe object key: {key!r}")
    if any(ch in _UNSAFE_KEY_CHARS for ch in key):
        raise StorageError(f"Unsafe object key: {key!r}")
    # A DRIVE-ANCHORED key ("C:/Windows/System32/x", or the drive-relative "D:x") escapes the
    # store entirely: pathlib's `/` DISCARDS the left operand when the right one carries a drive,
    # so `OBJECTS_DIR / key` returns the caller's absolute path rather than anything under the
    # store. `Path("C:/tmp/objects") / "C:/Windows/x"` is `WindowsPath("C:/Windows/x")` — verified,
    # not theoretical. The dev serving endpoint has no auth, so on Windows (this project's primary
    # OS) that is an unauthenticated arbitrary file read. A colon never appears in a legitimate
    # key, and rejecting it covers both forms on every platform — which matters because a POSIX
    # CI box can mint a key that only escapes once a Windows box resolves it.
    if ":" in key:
        raise StorageError(f"Unsafe object key: {key!r}")
    return key


def upload_prefix(owner: str, video_id: str) -> str:
    """The key prefix holding every artifact for one upload. Stored as ``videos.storage_key``."""
    return f"uploads/{owner}/{video_id}"


def video_content_type(suffix: str) -> str:
    """Map an upload's file suffix to the content type R2 should replay it with."""
    return _VIDEO_CONTENT_TYPES.get(suffix.lower(), _DEFAULT_VIDEO_CONTENT_TYPE)


class ObjectStore(Protocol):
    """What the rest of the backend is allowed to ask of object storage."""

    def put(self, key: str, data: bytes, *, content_type: str) -> None: ...

    def presigned_url(self, key: str, *, expires_in: int = DEFAULT_URL_TTL) -> str: ...

    def delete_prefix(self, prefix: str) -> None: ...

    # ``iter_keys`` / ``get`` exist for MAINTENANCE, not for serving. Nothing on a request path
    # reads an object back — playback goes through a presigned URL straight to R2, which is the
    # point of using R2 at all. They are here so an offline job can rewrite objects already in the
    # bucket (see ``faststart.backfill``); routing a video through the API process would undo the
    # streaming this whole change set exists to enable.
    def iter_keys(self, prefix: str) -> Iterator[str]: ...

    def get(self, key: str) -> tuple[bytes, str] | None: ...


class LocalObjectStore:
    """Filesystem-backed store for development and tests."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root if root is not None else OBJECTS_DIR

    def _path(self, key: str) -> Path:
        return self._root / _validate_key(key)

    def put(self, key: str, data: bytes, *, content_type: str) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        # Remember the content type alongside the object. Unlike R2 the filesystem stores no
        # metadata, and the source object has no extension to guess from — so without this the
        # dev serving endpoint could not tell a WebM recording from an MP4 upload.
        dest.with_name(dest.name + ".type").write_text(content_type, encoding="utf-8")

    def presigned_url(self, key: str, *, expires_in: int = DEFAULT_URL_TTL) -> str:
        """A URL for the development serving endpoint. Unsigned, and ``expires_in`` is ignored:
        that endpoint is inert whenever R2 is configured, so there is nothing to sign against."""
        return f"/api/local-object/{quote(_validate_key(key))}"

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self._path(prefix), ignore_errors=True)

    def iter_keys(self, prefix: str) -> Iterator[str]:
        """Every object key under ``prefix``, in sorted order so a run is reproducible.

        The ``.type`` sidecars this store writes alongside each object are metadata, not objects,
        and are skipped — yielding them would have a maintenance job try to remux a content-type
        string.
        """
        root = self._path(prefix)
        if not root.is_dir():
            return
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.endswith(".type"):
                continue
            yield path.relative_to(self._root).as_posix()

    def get(self, key: str) -> tuple[bytes, str] | None:
        found = self.open_object(key)
        if found is None:
            return None
        path, content_type = found
        return path.read_bytes(), content_type

    def open_object(self, key: str) -> tuple[Path, str] | None:
        """Local-only: ``(path, content_type)`` behind ``key``, for the dev serving endpoint."""
        path = self._path(key)
        if not path.is_file():
            return None
        meta = path.with_name(path.name + ".type")
        content_type = (
            meta.read_text(encoding="utf-8").strip() if meta.is_file() else "application/octet-stream"
        )
        return path, content_type


class R2ObjectStore:
    """Cloudflare R2 over the S3 API."""

    def __init__(
        self, *, account_id: str, access_key_id: str, secret_access_key: str, bucket: str
    ) -> None:
        self._bucket = bucket
        self._endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._client = None

    def _s3(self):
        """Build the boto3 client on first use. Deferred so importing this module stays light
        and the unit tests can inject a fake by assigning ``_client``."""
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
                region_name="auto",
                config=Config(signature_version="s3v4"),
            )
        return self._client

    def put(self, key: str, data: bytes, *, content_type: str) -> None:
        key = _validate_key(key)
        try:
            self._s3().put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                CacheControl=DEFAULT_CACHE_CONTROL,
            )
        except Exception as exc:  # noqa: BLE001 — botocore raises a wide family; callers see one type
            raise StorageError(f"Failed to store object {key!r}.") from exc

    def presigned_url(self, key: str, *, expires_in: int = DEFAULT_URL_TTL) -> str:
        key = _validate_key(key)
        try:
            return self._s3().generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Failed to sign a URL for {key!r}.") from exc

    def iter_keys(self, prefix: str) -> Iterator[str]:
        """Every object key under ``prefix``, paginated so a large bucket is never held at once."""
        prefix = _validate_key(prefix)
        try:
            paginator = self._s3().get_paginator("list_objects_v2")
            # The trailing slash, for the same reason ``delete_prefix`` needs it: S3 matches on the
            # raw string, so listing `.../upload_ab` would also return `.../upload_abc`.
            for page in paginator.paginate(Bucket=self._bucket, Prefix=f"{prefix}/"):
                for obj in page.get("Contents", []):
                    yield obj["Key"]
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Failed to list objects under {prefix!r}.") from exc

    def get(self, key: str) -> tuple[bytes, str] | None:
        """``(bytes, content_type)``, or ``None`` when the key does not exist.

        A missing key is not an error: a maintenance pass runs against a listing that may have
        moved under it (a user can delete an analysis mid-run), and treating that race as a
        failure would abort a job that is behaving correctly.
        """
        key = _validate_key(key)
        try:
            resp = self._s3().get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 — botocore's NoSuchKey is not importable directly
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in {
                "NoSuchKey",
                "404",
                "NotFound",
            }:
                return None
            raise StorageError(f"Failed to read object {key!r}.") from exc
        return resp["Body"].read(), resp.get("ContentType") or _DEFAULT_VIDEO_CONTENT_TYPE

    def delete_prefix(self, prefix: str) -> None:
        prefix = _validate_key(prefix)
        failures: list[dict] = []
        try:
            client = self._s3()
            paginator = client.get_paginator("list_objects_v2")
            # The TRAILING SLASH is load-bearing: without it, deleting `.../upload_ab` would also
            # reap `.../upload_abc`, because S3 prefixes match on raw string prefix, not path segments.
            for page in paginator.paginate(Bucket=self._bucket, Prefix=f"{prefix}/"):
                keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if keys:
                    resp = client.delete_objects(Bucket=self._bucket, Delete={"Objects": keys})
                    failures.extend(resp.get("Errors") or [])
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Failed to delete objects under {prefix!r}.") from exc
        # `delete_objects` reports PER-KEY failures in its response body and raises only for
        # whole-request errors, so discarding the return value would report a successful reap that
        # deleted nothing. Raised OUTSIDE the try so it is not caught and re-wrapped as its own cause.
        if failures:
            raise StorageError(
                f"Failed to delete {len(failures)} object(s) under {prefix!r}; "
                f"first: {failures[0].get('Key')} ({failures[0].get('Code')})"
            )


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    """The live object store: R2 when fully configured, the local filesystem otherwise.

    Cached, so the boto3 client is built once. Tests that change the settings must call
    ``get_object_store.cache_clear()``.
    """
    from backend.app.settings import get_settings  # deferred: settings imports config, not us

    settings = get_settings()
    if settings.storage_configured:
        return R2ObjectStore(
            account_id=settings.r2_account_id,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket=settings.r2_bucket,
        )
    return LocalObjectStore()
