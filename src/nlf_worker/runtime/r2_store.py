"""Cloudflare R2 over the S3 API -- ``ObjectStore`` and ``Fetcher`` for ``loop.run_one``.

Uses the worker's dedicated, bucket-scoped R2 token (plan "Decisions taken"). Results are
write-once under a per-attempt prefix, so every put carries the same immutable
``Cache-Control`` the backend uses (``backend/app/services/storage.py``); the gzip JSON also
carries ``Content-Encoding: gzip`` so a later browser fetch decodes it transparently.

R2 tokens scope to a bucket, not a prefix, so ``delete_prefix`` refuses anything that is not an
``nlf.v*`` result folder: a bug that passed the video's own prefix must not reap the user's
source clip. ``boto3`` is imported lazily in ``make_s3_client``, so tests can inject a fake.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

DEFAULT_CACHE_CONTROL = "public, max-age=31536000, immutable"


def make_s3_client(account_id: str, access_key_id: str, secret_access_key: str) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def _validate_key(key: str) -> str:
    if not key or key.startswith("/") or ".." in key.split("/") or "\\" in key or "\x00" in key:
        raise ValueError(f"unsafe object key: {key!r}")
    return key


class R2Store:
    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def put(self, key: str, data: bytes, *, content_type: str) -> None:
        extra = {"ContentEncoding": "gzip"} if key.endswith(".json.gz") else {}
        self._client.put_object(
            Bucket=self._bucket,
            Key=_validate_key(key),
            Body=data,
            ContentType=content_type,
            CacheControl=DEFAULT_CACHE_CONTROL,
            **extra,
        )

    def delete_prefix(self, prefix: str) -> None:
        prefix = _validate_key(prefix.rstrip("/"))
        if not prefix.rsplit("/", 1)[-1].startswith("nlf.v"):
            raise ValueError(f"refusing to delete outside an nlf result folder: {prefix!r}")
        failures: list[dict] = []
        paginator = self._client.get_paginator("list_objects_v2")
        # Trailing slash: S3 prefixes match raw strings, so `.../nlf.v1.a1` alone would also
        # match `.../nlf.v1.a10`.
        for page in paginator.paginate(Bucket=self._bucket, Prefix=f"{prefix}/"):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if keys:
                resp = self._client.delete_objects(Bucket=self._bucket, Delete={"Objects": keys})
                failures.extend(resp.get("Errors") or [])
        # delete_objects reports per-key failures in its body rather than raising.
        if failures:
            raise RuntimeError(
                f"failed to delete {len(failures)} object(s) under {prefix!r}; "
                f"first: {failures[0].get('Key')} ({failures[0].get('Code')})"
            )


class R2Fetcher:
    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def download(self, key: str, dest_dir: str) -> str:
        """Downloads ``key`` to a new temp file in ``dest_dir`` and returns its path; a partial
        file from a failed download is removed."""
        os.makedirs(dest_dir, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="nlf_src_", dir=dest_dir)
        os.close(fd)
        try:
            self._client.download_file(self._bucket, _validate_key(key), path)
        except BaseException:
            try:
                os.remove(path)
            except OSError:
                pass
            raise
        return path
