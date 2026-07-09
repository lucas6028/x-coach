"""Object storage for user videos on Cloudflare R2 (S3-compatible), via presigned URLs.

Uploads that belong to a signed-in user are pushed here so they survive the (ephemeral) app
container and can be streamed back through a short-lived presigned GET URL — the bucket stays
private. R2 is chosen for zero egress fees, since video streaming is egress-heavy.

Config lives in ``settings`` (``r2_*``); when it's blank ``is_configured()`` is False and the
callers fall back to the local runtime disk (dev / anonymous demo). ``boto3`` is imported lazily
inside ``_client`` so importing this module (and the routers that use it) stays light and the unit
tests can mock the client without ``boto3`` installed — mirroring ``store._user_client``.

The object key is derived from ``video_id`` alone (``uploads/<video_id>``, no suffix): the id is an
unguessable ``upload_<12 hex>`` slug, so this preserves the exact capability-by-id trust model the
local ``/api/video-file`` streaming already uses, and lets that endpoint rebuild the key without a
DB lookup. The original content type is stored on the object so the presigned GET plays correctly
regardless of the extension-less key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Map upload suffixes to the content type stored on the R2 object (so a presigned GET streams with
# the right ``Content-Type`` for the browser's ``<video>`` element). Mirrors the analyze allow-list.
_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


def is_configured() -> bool:
    """True when R2 is fully configured (else callers keep videos on local disk)."""
    from backend.app.settings import get_settings

    return get_settings().storage_configured


def object_key(video_id: str) -> str:
    """The R2 object key for an upload — derived from ``video_id`` alone (no suffix)."""
    return f"uploads/{video_id}"


def content_type_for(suffix: str) -> str:
    """Best-effort content type for an upload suffix (defaults to ``video/mp4``)."""
    return _CONTENT_TYPES.get(suffix.lower(), "video/mp4")


def _client() -> Any:
    """Build a boto3 S3 client pointed at the R2 endpoint (SigV4)."""
    import boto3  # deferred heavy import
    from botocore.config import Config

    from backend.app.settings import get_settings

    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_resolved_endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",  # R2 ignores region but boto3 requires one; SigV4 is required.
        config=Config(signature_version="s3v4"),
    )


def upload_video_file(video_id: str, path: Path, suffix: str) -> str:
    """Upload the local video at ``path`` to R2 and return its object key.

    Streams from disk (never holds the whole file in RAM) so the analyze path can ``del`` the
    request bytes before this runs. The caller may then drop the local temp copy.
    """
    from backend.app.settings import get_settings

    key = object_key(video_id)
    client = _client()
    with open(path, "rb") as fh:
        client.put_object(
            Bucket=get_settings().r2_bucket,
            Key=key,
            Body=fh,
            ContentType=content_type_for(suffix),
        )
    return key


def video_exists(video_id: str) -> bool:
    """True if an object for ``video_id`` is present in the bucket."""
    from botocore.exceptions import ClientError

    from backend.app.settings import get_settings

    client = _client()
    try:
        client.head_object(Bucket=get_settings().r2_bucket, Key=object_key(video_id))
        return True
    except ClientError:
        return False


def presigned_get_url(video_id: str) -> str:
    """A short-lived presigned GET URL for ``video_id``'s object (bucket stays private)."""
    from backend.app.settings import get_settings

    settings = get_settings()
    client = _client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.r2_bucket, "Key": object_key(video_id)},
        ExpiresIn=settings.r2_url_ttl_s,
    )


def delete_video(video_id: str) -> None:
    """Remove ``video_id``'s object from the bucket (best-effort; ignores a missing object)."""
    from botocore.exceptions import ClientError

    from backend.app.settings import get_settings

    client = _client()
    try:
        client.delete_object(Bucket=get_settings().r2_bucket, Key=object_key(video_id))
    except ClientError:
        pass
