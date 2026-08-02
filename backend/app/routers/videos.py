"""Library listing, precomputed analysis, pose overlay, and video streaming endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.services import analysis, library, storage

router = APIRouter(prefix="/api", tags=["videos"])


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
    """Stream the source mp4 (library clip or a prior upload). Supports HTTP Range seeking."""
    # Fall back to an uploaded file in the runtime dir. Both lookups validate ``video_id`` and
    # match exact names only — never a glob — so a wildcard id cannot reach another user's clip.
    path = library.video_path(video_id) or library.uploaded_video_path(video_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"No video file for '{video_id}'.")
    return FileResponse(path, media_type="video/mp4")


@router.get("/local-object/{key:path}")
def get_local_object(key: str) -> FileResponse:
    """DEVELOPMENT ONLY: serve an object out of the local filesystem store.

    Inert in production: when R2 is configured, ``get_object_store()`` returns an
    ``R2ObjectStore`` and this endpoint 404s for every key. It exists so ``LocalObjectStore``
    can hand back a URL the browser can actually fetch, keeping the frontend contract identical
    in both modes. It carries no signature — reaching a key still requires having been given it
    by the ownership-checked ``/api/uploads/{video_id}/url``.
    """
    # Named `store_`: Task 4 adds a `store` service import to this module, and the trailing
    # underscore keeps this local from shadowing it later.
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
