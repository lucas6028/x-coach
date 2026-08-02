"""Library listing, precomputed analysis, pose overlay, and video URL endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app.auth import CurrentUser, get_current_user
from backend.app.services import analysis, library, storage, store

router = APIRouter(prefix="/api", tags=["videos"])

# How many uploads one batch URL request may cover. A history page is 50 rows by default; the
# cap keeps a crafted request from asking the DB for an unbounded `in_` list.
MAX_URL_BATCH = 200


class UploadUrlBatch(BaseModel):
    video_ids: list[str]


def _upload_urls(prefix: str) -> dict[str, str]:
    """Signed URLs for one upload's playable artifacts.

    The thumbnail URL is signed unconditionally — a clip uploaded before thumbnails existed, or
    one whose capture failed, simply 404s when the browser fetches it, and the UI falls back.
    Probing for existence first would cost a round trip per row to save an occasional 404.
    """
    obj = storage.get_object_store()
    return {
        "video_url": obj.presigned_url(f"{prefix}/source"),
        "thumbnail_url": obj.presigned_url(f"{prefix}/thumb.jpg"),
    }


@router.get("/videos")
def list_videos(limit: int = 50, offset: int = 0, fault: str | None = None) -> dict:
    """List precomputed library clips (clips containing faults first)."""
    return library.list_videos(limit=limit, offset=offset, fault=fault)


@router.get("/analysis/{video_id}")
def get_analysis(video_id: str) -> dict:
    """Return the precomputed analysis for a library video (retrieval enriched on demand)."""
    try:
        return library.load_analysis(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/pose/{video_id}")
def get_pose(video_id: str) -> dict:
    """Return the slim 33-landmark overlay block for a library video."""
    pose_path = library.pose_json_path(video_id)
    if pose_path is None:
        raise HTTPException(status_code=404, detail=f"No pose data for '{video_id}'.")
    return analysis.build_pose_block(pose_path)


@router.get("/video-file/{video_id}")
def get_video_file(video_id: str) -> FileResponse:
    """Stream a LIBRARY demo clip's mp4 (public, shared assets). Supports HTTP Range seeking.

    Uploads are deliberately NOT reachable here. This endpoint has no auth dependency, and its
    former fallback to ``library.uploaded_video_path`` therefore handed any caller who knew a
    ``video_id`` any user's upload. Uploads now go through ``/api/uploads/{video_id}/url``, which
    resolves the key as the caller so RLS enforces ownership. The fallback is gone rather than
    guarded, so there is no code path from here to a user's clip to re-open by accident.
    """
    path = library.video_path(video_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"No video file for '{video_id}'.")
    return FileResponse(path, media_type="video/mp4")


@router.get("/uploads/{video_id}/url")
def get_upload_urls(video_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Short-lived playback + thumbnail URLs for one of the CALLER'S uploads.

    404 covers both "no such upload" and "not yours": the storage key is read with the caller's
    own JWT, so RLS makes the two indistinguishable — the same shape ``delete_analysis`` uses.
    """
    prefix = store.get_storage_key(token=user.token, video_id=video_id)
    if prefix is None:
        raise HTTPException(status_code=404, detail=f"No upload '{video_id}'.")
    try:
        urls = _upload_urls(prefix)
    except storage.StorageError as exc:
        raise HTTPException(status_code=503, detail="Storage is unavailable.") from exc
    return {**urls, "expires_in": storage.DEFAULT_URL_TTL}


@router.post("/uploads/urls")
def get_upload_urls_batch(
    body: UploadUrlBatch, user: CurrentUser = Depends(get_current_user)
) -> dict:
    """The same URLs for many uploads at once, for a history page.

    One request and one DB round trip for a whole page, rather than N of each. Ids the caller
    does not own are absent from ``items`` rather than an error — a partial answer is the honest
    one when RLS has filtered the rest.
    """
    if len(body.video_ids) > MAX_URL_BATCH:
        raise HTTPException(
            status_code=400, detail=f"At most {MAX_URL_BATCH} video ids per request."
        )
    if not body.video_ids:
        return {"items": {}, "expires_in": storage.DEFAULT_URL_TTL}
    prefixes = store.get_storage_keys(token=user.token, video_ids=body.video_ids)
    try:
        items = {video_id: _upload_urls(prefix) for video_id, prefix in prefixes.items()}
    except storage.StorageError as exc:
        raise HTTPException(status_code=503, detail="Storage is unavailable.") from exc
    return {"items": items, "expires_in": storage.DEFAULT_URL_TTL}


@router.get("/local-object/{key:path}")
def get_local_object(key: str) -> FileResponse:
    """DEVELOPMENT ONLY: serve an object out of the local filesystem store.

    Inert in production: when R2 is configured, ``get_object_store()`` returns an
    ``R2ObjectStore`` and this endpoint 404s for every key. It exists so ``LocalObjectStore``
    can hand back a URL the browser can actually fetch, keeping the frontend contract identical
    in both modes. It carries no signature — reaching a key still requires having been given it
    by the ownership-checked ``/api/uploads/{video_id}/url``.
    """
    store_ = storage.get_object_store()
    if not isinstance(store_, storage.LocalObjectStore):
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        found = store_.open_object(key)
    except storage.StorageError as exc:
        raise HTTPException(status_code=404, detail="Not found.") from exc
    if found is None:
        raise HTTPException(status_code=404, detail="Not found.")
    path, content_type = found
    return FileResponse(path, media_type=content_type)
