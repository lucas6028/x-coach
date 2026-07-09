"""Library listing, precomputed analysis, pose overlay, and video streaming endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from backend.app.services import analysis, library, object_store

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
def get_video_file(video_id: str):
    """Stream the source video (library clip or a prior upload). Supports HTTP Range seeking.

    Prefers a local file (library clips, and uploads still on disk in dev / anonymous demo). When the
    binary isn't local — the normal case in production after an upload was pushed to R2 and its temp
    copy dropped — it redirects to a short-lived presigned GET URL on the private bucket, which the
    browser's ``<video>`` element follows (R2 serves Range requests directly). Both local lookups
    validate ``video_id`` and match exact names only — never a glob — so a wildcard id cannot reach
    another user's clip; the R2 key is likewise derived from the exact id.
    """
    path = library.video_path(video_id) or library.uploaded_video_path(video_id)
    if path is not None and path.exists():
        return FileResponse(path, media_type="video/mp4")

    if object_store.is_configured() and library.is_safe_video_id(video_id):
        if object_store.video_exists(video_id):
            return RedirectResponse(object_store.presigned_get_url(video_id), status_code=307)

    raise HTTPException(status_code=404, detail=f"No video file for '{video_id}'.")
