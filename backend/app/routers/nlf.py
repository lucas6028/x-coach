"""NLF 3D view: a selected user asks for a turntable render of one of their analysed videos.

The heavy lifting (running NLF, rendering 24-angle turntable strips) happens OFF this backend, on
a worker PC that polls ``nlf_jobs`` with its own service_role key and writes the result — this
router only ever creates/reads the job row and presigns the finished strips. Three endpoints:

  * ``GET /api/nlf/status`` — whether the caller holds the 'nlf_user' role (UX gating, so the
    frontend only shows the "3D view" button to opted-in users).
  * ``POST /api/analyses/{analysis_id}/nlf`` — request (or re-request, if the prior attempt
    failed) a render for the analysis's video; 202 + the job's current state.
  * ``GET /api/analyses/{analysis_id}/nlf`` — poll that job's current state.

Both analysis-scoped routes resolve ``video_id`` through ``store.get_analysis`` (the caller's own
JWT, RLS-scoped) rather than trusting a client-supplied video id — the same ownership shape as
every other ``/api/analyses/{id}/...`` route.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.app.auth import CurrentUser, get_current_user
from backend.app.services import storage, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["nlf"])

# Every strip the worker writes for a video lives under ``{storage_key}/nlf.v1.*`` (see
# src/nlf_worker/result.py's ``strip_key``). Presigning is restricted to keys under this prefix so
# a buggy or compromised worker writing an arbitrary ``strip_key`` into ``meta`` can never turn
# this endpoint into an oracle that mints a URL to unrelated storage.
_NLF_STRIP_PREFIX = "nlf.v1."

# A worker checking in within this window counts as "online" for the status page; a `None`
# last-seen (never checked in) is always offline.
_WORKER_ONLINE_WINDOW = timedelta(minutes=10)


def _uid(raw: str) -> str:
    """Normalize an analysis id to canonical UUID form, or 404 — same guard as
    ``routers/analyses.py``'s DELETE route: a junk path param must not reach PostgREST, where it
    would raise ``22P02`` and surface as a 500."""
    try:
        return str(uuid.UUID(raw))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"No analysis '{raw}'.") from exc


def _worker_online(last_seen: str | None) -> bool:
    """Whether the worker's last check-in falls within ``_WORKER_ONLINE_WINDOW``; ``False`` for
    an absent (never checked in) timestamp or one that fails to parse."""
    if not last_seen:
        return False
    try:
        parsed = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed) <= _WORKER_ONLINE_WINDOW


def _build_key_frames(
    *, token: str, video_id: str, meta: dict[str, Any]
) -> list[dict[str, Any]]:
    """Presign each key frame's stored strip, translating ``meta.key_frames`` into the wire shape.

    Only a ``strip_key`` that actually lives under THIS job's video storage prefix
    (``{storage_key}/nlf.v1.*``) is presigned — the worker writes ``meta`` itself, so a buggy or
    compromised worker writing an arbitrary key must not turn into an oracle that presigns and
    hands back a URL into unrelated storage (see ``_NLF_STRIP_PREFIX``). An out-of-prefix entry is
    dropped and logged (only the expected prefix is logged, never the offending key itself). If
    the video's own storage key can't be resolved at all, there is nothing safe to check an entry
    against, so this returns an empty list rather than trusting any of them.

    May raise ``storage.StorageError`` (propagated to the caller, who turns it into a 503) — same
    failure mode as ``routers/videos.py``'s ``_upload_urls``.
    """
    storage_key = store.get_storage_key(token=token, video_id=video_id)
    if storage_key is None:
        return []
    allowed_prefix = f"{storage_key}/{_NLF_STRIP_PREFIX}"

    obj = storage.get_object_store()
    frames = []
    for kf in meta.get("key_frames") or []:
        strip_key = kf.get("strip_key")
        if not strip_key or not strip_key.startswith(allowed_prefix):
            logger.warning(
                "Dropping nlf key frame: strip_key is not under the expected prefix %r",
                allowed_prefix,
            )
            continue
        frames.append(
            {
                "t_s": kf.get("t_s"),
                "kind": kf.get("kind"),
                "fault_ids": kf.get("fault_ids") or [],
                "rep_index": kf.get("rep_index"),
                "strip_url": obj.presigned_url(strip_key),
            }
        )
    return frames


def _nlf_job_response(*, token: str, video_id: str, job: dict[str, Any]) -> dict[str, Any]:
    """Build the ``NlfJob`` wire shape from a raw ``nlf_jobs`` row.

    ``key_frames`` is only ever populated for a ``done`` job — the worker's ``meta`` is null (or
    stale from a prior attempt) for every other status, and presigning would be meaningless before
    the strips exist. No ``pose3d_url`` is ever returned; only the pre-rendered turntable strips.
    """
    meta = job.get("meta") or {}
    status = job.get("status")
    key_frames = (
        _build_key_frames(token=token, video_id=video_id, meta=meta) if status == "done" else []
    )
    last_seen = store.nlf_worker_last_seen(token=token)
    return {
        "status": status,
        "worker_online": _worker_online(last_seen),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "key_frames": key_frames,
        "angles": meta.get("angles"),
        "tile_px": meta.get("tile_px"),
        "expires_in": storage.DEFAULT_URL_TTL,
    }


def _owned_video_id(*, token: str, user_id: str, analysis_id: str) -> str:
    """Resolve ``analysis_id`` to its ``video_id``; 404 if it doesn't exist or isn't the caller's.

    ``store.get_analysis`` is RLS-scoped to "rows I can SELECT" — which, per the clinic
    migration's ``analyses_clinician_select`` policy, includes a LINKED CLINICIAN reading a
    PATIENT's analysis too. So "the row exists" does not by itself prove "it's mine"; the
    explicit ``row["user_id"] == caller`` check is what does — same reasoning, and the same
    check, as ``routers/checkins.py`` and ``routers/plans.py::update_item``.
    """
    row = store.get_analysis(token=token, analysis_id=analysis_id)
    if row is None or row.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail=f"No analysis '{analysis_id}'.")
    return row.get("video_id")


@router.get("/nlf/status")
def nlf_status(user: CurrentUser = Depends(get_current_user)) -> dict:
    """Whether the caller holds the 'nlf_user' role (frontend gating for the 3D-view button)."""
    return {"enabled": store.get_nlf_enabled(token=user.token, user_id=user.id)}


@router.post("/analyses/{analysis_id}/nlf", status_code=202)
def request_nlf(analysis_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Request (or re-request) a 3D turntable render for one of the caller's analyses.

    403 if the caller lacks the 'nlf_user' role (RPC 42501); 404 if the analysis is not the
    caller's (``store.get_analysis`` returns ``None``, or returns someone else's row — see
    ``_owned_video_id``) or its video row is missing (RPC P0002, which RLS makes indistinguishable
    from "never existed"). Re-requesting an already-``done`` or in-flight job is a no-op that
    returns its current state; a ``failed`` job is re-queued — both decided inside
    ``request_nlf_job`` itself, not here.
    """
    aid = _uid(analysis_id)
    video_id = _owned_video_id(token=user.token, user_id=user.id, analysis_id=aid)

    try:
        job = store.request_nlf_job(token=user.token, video_id=video_id)
    except Exception as exc:  # noqa: BLE001 — translate the RPC's postgrest error code only.
        code = getattr(exc, "code", None)
        if code == "42501":
            raise HTTPException(
                status_code=403, detail="NLF 3D view is not enabled for this account."
            ) from exc
        if code == "P0002":
            raise HTTPException(status_code=404, detail=f"No analysis '{aid}'.") from exc
        raise

    try:
        return _nlf_job_response(token=user.token, video_id=video_id, job=job)
    except storage.StorageError as exc:
        raise HTTPException(status_code=503, detail="Storage is unavailable.") from exc


@router.get("/analyses/{analysis_id}/nlf")
def get_nlf(analysis_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Poll the caller's NLF job for one analysis; 404 if the analysis isn't theirs or no job
    has been requested yet."""
    aid = _uid(analysis_id)
    video_id = _owned_video_id(token=user.token, user_id=user.id, analysis_id=aid)

    job = store.get_nlf_job(token=user.token, user_id=user.id, video_id=video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No NLF job for this analysis.")

    try:
        return _nlf_job_response(token=user.token, video_id=video_id, job=job)
    except storage.StorageError as exc:
        raise HTTPException(status_code=503, detail="Storage is unavailable.") from exc
