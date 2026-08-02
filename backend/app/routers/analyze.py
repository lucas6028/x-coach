"""Live-upload analysis endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.app import config, settings
from backend.app.auth import CurrentUser, get_optional_user
from backend.app.services import analysis, storage, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])

# Bound how many uploads run the (blocking, CPU/RAM-heavy) pipeline at once. Excess requests
# await a slot here instead of spawning unbounded worker threads. This is a single-process
# stop-gap; durable queueing and backpressure move to Celery + Redis later.
_ANALYSIS_SEMAPHORE = asyncio.Semaphore(config.MAX_CONCURRENT_ANALYSES)

# A thumbnail is one downscaled JPEG frame (the browser caps its longest edge at 480px), so
# anything approaching this is not a thumbnail. Bounds what an upload can push into storage.
MAX_THUMBNAIL_BYTES = 512 * 1024


def _source_url(prefix: str) -> str | None:
    """A short-lived playback URL for the upload's source object, or None if signing failed.

    Never raises: the analysis has already been produced by the time this is called, so a
    signing problem degrades playback rather than discarding a completed result.
    """
    try:
        return storage.get_object_store().presigned_url(f"{prefix}/source")
    except storage.StorageError:
        logger.exception("Failed to sign a playback URL for %s", prefix)
        return None


async def _read_thumbnail(thumbnail: UploadFile | None) -> bytes | None:
    """Validate and read the optional browser-captured frame.

    A missing thumbnail is NOT an error — older clients and browsers where frame capture failed
    must still be able to analyze. Only a wrong type or an implausible size is rejected.
    """
    if thumbnail is None:
        return None
    content_type = (thumbnail.content_type or "").split(";")[0].strip().lower()
    if content_type != "image/jpeg":
        raise HTTPException(status_code=400, detail="Thumbnail must be image/jpeg.")
    # Cap the READ itself, not just the check after it: an unbounded ``.read()`` would fully
    # materialize an oversized part (up to whatever the client sends) before the length check
    # below ever gets a chance to reject it. Reading one byte past the limit is enough to still
    # detect "too large" without ever buffering more than that.
    data = await thumbnail.read(MAX_THUMBNAIL_BYTES + 1)
    if not data:
        return None
    if len(data) > MAX_THUMBNAIL_BYTES:
        raise HTTPException(status_code=400, detail="Thumbnail is too large.")
    return data


async def _stage_analyze_persist(
    file: UploadFile,
    *,
    suffix: str,
    thumb: bytes | None,
    user: CurrentUser | None,
    persist_log_message: str,
    run: Callable[[analysis.StagedUpload], dict[str, Any]],
) -> dict[str, Any]:
    """Stage an upload, run its analysis, persist derived artifacts/history, and sign a URL.

    Shared by ``analyze`` and ``analyze_pose``, which differ only in how the analysis itself is
    invoked (``run``) and in one log message. Everything else -- the fail-fast 503 BEFORE any
    CPU is spent, that only a successful analysis gets its derived artifacts kept, that the
    stage is discarded on every path, that persistence is best-effort, and that the presigned
    ``video_url`` is attached to the RESPONSE only after persistence returns (never into the
    JSONB ``result`` itself, which would otherwise carry an already-expiring URL into history) --
    is enforced ONCE here rather than kept in sync by hand across two endpoint copies.

    ``file`` (not pre-read bytes) is the parameter on purpose: reading it, checking for an empty
    upload, staging it, and dropping the raw-bytes reference all happen inside this single frame,
    so no caller ever holds its own copy of the video bytes alive for the (potentially queued,
    then CPU-bound) analysis phase -- the same memory property the original two-copy code had via
    its own ``del data``.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Anonymous demo uploads are still stored, under their own key prefix, so both paths behave
    # identically. A bucket lifecycle rule expires `uploads/anon/` — see the design doc.
    owner = user.id if user is not None else "anon"
    try:
        staged = await run_in_threadpool(analysis.stage_upload, data, suffix=suffix, owner=owner)
    except storage.StorageError as exc:
        logger.exception("Failed to store upload (owner=%s)", owner)
        raise HTTPException(
            status_code=503, detail="Storage is unavailable; please try again."
        ) from exc
    del data  # bytes are now stored and staged; don't pin the whole video in RAM while queued.

    try:
        async with _ANALYSIS_SEMAPHORE:
            result = await run_in_threadpool(run, staged)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        # Only a SUCCESSFUL analysis has derived artifacts worth keeping. In an ``else`` (not
        # inside the ``try``) so this holds even if ``store_artifacts`` ever stopped honoring its
        # own never-raises contract -- a raise here must map to 422 by construction, not by
        # accident of where the line sits.
        await run_in_threadpool(analysis.store_artifacts, staged, thumbnail=thumb)
    finally:
        await run_in_threadpool(analysis.discard_stage, staged)

    if user is not None:
        try:
            result["analysis_id"] = await run_in_threadpool(
                store.persist_analysis,
                token=user.token,
                user_id=user.id,
                video_id=staged.video_id,
                source="upload",
                result=result,
                storage_key=staged.prefix,
                filename=file.filename,
            )
        except Exception:  # noqa: BLE001 — never lose a completed analysis to a storage error
            logger.exception(persist_log_message, user.id, staged.video_id)
            result["analysis_id"] = None

    # AFTER the persist, deliberately: `result` is stored verbatim as JSONB, and a presigned URL
    # written into the history row would already be expired by the time anyone replayed it. The
    # replay path re-signs through GET /api/uploads/{video_id}/url instead.
    result["video_url"] = await run_in_threadpool(_source_url, staged.prefix)
    return result


def _validated_movement(movement: str) -> str:
    """Resolve a requested movement to its canonical name, or 400.

    Rejecting HERE -- before save_upload and before pose extraction -- means a bad request
    costs no compute. The registry lookup is case-insensitive (get_detector lowercases its
    key), so the canonical spelling is what comes back.

    An explicit empty/whitespace-only ``movement`` is rejected here rather than left to
    ``registry.get_detector``: that function's ``(movement or "Squat")`` fallback exists for
    callers that legitimately pass ``None`` (the library path), not for a client that sent
    ``movement=""`` on purpose. Silently mapping an explicit empty string to Squat would mask
    exactly the kind of bad request this endpoint is meant to catch.
    """
    from src.pose.movements import registry

    if not movement or not movement.strip():
        known = ", ".join(d.name for d in registry.list_detectors())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported movement '{movement}'. Analyzable movements: {known}.",
        )

    try:
        return registry.get_detector(movement).name
    except KeyError:
        known = ", ".join(d.name for d in registry.list_detectors())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported movement '{movement}'. Analyzable movements: {known}.",
        ) from None


def _validated_max_reps(max_reps: int | None) -> int | None:
    """Resolve an optional client-supplied rep cap. 0 means 'every rep'.

    Bounded so a client cannot ask for an unbounded amount of per-rep work on the shared
    analysis semaphore.
    """
    if max_reps is None:
        return config.DEFAULT_MAX_REPS
    if max_reps < 0 or max_reps > 20:
        raise HTTPException(status_code=400, detail="max_reps must be between 0 and 20.")
    return None if max_reps == 0 else max_reps


def _validate_pose_landmarks(payload: dict) -> None:
    """Reject structurally malformed landmark entries before they reach the pose pipeline.

    Landmark COUNT is not validated here — a ``landmarks`` list with fewer than 33
    well-formed entries is legitimate (the detector treats it as "no pose"). Only
    structural malformation (non-dict frames/landmarks, missing/non-numeric fields)
    is rejected. ``world_landmarks`` is intentionally not checked: the detector never
    reads it.
    """
    for frame in payload.get("frames", []):
        if not isinstance(frame, dict):
            raise HTTPException(status_code=400, detail="Malformed pose frame.")
        lms = frame.get("landmarks")
        if lms is None:
            continue
        if not isinstance(lms, list):
            raise HTTPException(status_code=400, detail="Malformed pose landmarks.")
        for lm in lms:
            if not isinstance(lm, dict) or not all(
                isinstance(lm.get(k), (int, float)) for k in ("x", "y", "z", "visibility")
            ):
                raise HTTPException(status_code=400, detail="Malformed pose landmarks.")


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    movement: str = Form(config.DEFAULT_ANALYSIS_MOVEMENT),
    max_reps: int | None = Form(None),
    thumbnail: UploadFile | None = File(None),
    user: CurrentUser | None = Depends(get_optional_user),
) -> dict:
    """Accept a video of a supported movement, extract pose, detect faults, and return the analysis.

    The pose pipeline is synchronous and CPU-bound, so the disk write and the analysis run in a
    worker thread via ``run_in_threadpool`` — this keeps the event loop responsive instead of
    freezing every other request for the duration of one analysis. A concurrency semaphore caps
    how many analyses run at once so concurrent uploads queue rather than exhaust the machine.

    Auth is optional: an anonymous caller gets the analysis but nothing is saved (public demo);
    an authenticated caller also has the result persisted to their history and gets an
    ``analysis_id`` back. Persistence is best-effort — a DB hiccup is logged but never discards
    a completed (and expensive) analysis.

    The response contract is otherwise unchanged: it matches the library analysis shape
    (metadata/view/quality/detections/retrievals/pose) so the frontend renders uploads and
    library clips identically.
    """
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    # ``allowed_upload_suffixes`` reads the admin overrides, which can do a synchronous Supabase
    # round-trip on a cold cache — run it in a threadpool so it never blocks the event loop.
    allowed = await run_in_threadpool(settings.allowed_upload_suffixes)
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    canonical_movement = await run_in_threadpool(_validated_movement, movement)
    resolved_max_reps = _validated_max_reps(max_reps)
    thumb = await _read_thumbnail(thumbnail)

    def _run(staged: analysis.StagedUpload) -> dict[str, Any]:
        return analysis.analyze_video_file(
            staged.video_path,
            video_id=staged.video_id,
            pose_json_path=staged.pose_path,
            movement=canonical_movement,
            max_reps=resolved_max_reps,
        )

    return await _stage_analyze_persist(
        file,
        suffix=suffix,
        thumb=thumb,
        user=user,
        persist_log_message="Failed to persist analysis (user=%s video=%s)",
        run=_run,
    )


@router.post("/analyze/pose")
async def analyze_pose(
    movement: str = Form(...),
    pose: str = Form(...),
    file: UploadFile = File(...),
    max_reps: int | None = Form(None),
    thumbnail: UploadFile | None = File(None),
    user: CurrentUser | None = Depends(get_optional_user),
) -> dict:
    """Analyze a client-extracted pose JSON (no server-side MediaPipe).

    The browser ran MediaPipe on the recorded/uploaded clip and posts the resulting pose JSON
    alongside the raw video (still stored for replay/overlay). Routing/persistence mirror
    ``/api/analyze`` so uploads and library clips still render identically.
    """
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    allowed = await run_in_threadpool(settings.allowed_upload_suffixes)
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    resolved_max_reps = _validated_max_reps(max_reps)

    try:
        payload = json.loads(pose)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Malformed pose JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise HTTPException(status_code=400, detail="Pose JSON must have a 'frames' list.")
    _validate_pose_landmarks(payload)

    thumb = await _read_thumbnail(thumbnail)

    def _run(staged: analysis.StagedUpload) -> dict[str, Any]:
        return analysis.analyze_pose_payload(
            payload,
            movement=movement,
            video_id=staged.video_id,
            pose_json_path=staged.pose_path,
            max_reps=resolved_max_reps,
        )

    return await _stage_analyze_persist(
        file,
        suffix=suffix,
        thumb=thumb,
        user=user,
        persist_log_message="Failed to persist pose analysis (user=%s video=%s)",
        run=_run,
    )
