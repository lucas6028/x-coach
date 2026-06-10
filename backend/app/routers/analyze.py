"""Live-upload analysis endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.services import analysis

router = APIRouter(prefix="/api", tags=["analyze"])

_ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    """Accept a squat video, extract pose, detect faults, and return the full analysis.

    The response matches the library analysis contract (metadata/view/quality/detections/
    retrievals/pose) so the frontend renders uploads and library clips identically.
    """
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    video_id, saved_path = analysis.save_upload(data, suffix=suffix)
    try:
        result = analysis.analyze_video_file(saved_path, video_id=video_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result
