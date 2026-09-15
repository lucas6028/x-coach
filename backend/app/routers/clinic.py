"""Clinician-side endpoints ("/api/clinic"): invite a patient, read their plans/check-ins, assign
plans. The patient-side counterpart (accept an invite, list one's own clinicians) is
``routers/care.py``.

Every route here is gated by ``get_clinician_user`` (403 for a non-clinician) EXCEPT
``GET /status``, which uses ``get_current_user`` so a non-clinician gets a truthful
``{"is_clinician": false}`` instead of a 403 -- the same split as ``routers/admin.py``'s
``/status``.

RLS is the backstop, not this router: ``clinic_patients()`` and the ``is_linked()`` policies on
``training_plans``/``plan_items``/``analyses``/``session_checkins`` (see the migration) are what
actually scope a clinician's reads to patients who accepted THEIR invite. This router's 404s for
an unlinked patient id are a UX nicety layered on top -- ``services/clinic.find_patient`` is a
plain list-and-match over the same RPC RLS already scopes, not an independent security check.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from backend.app.auth import CurrentUser, get_clinician_user, get_current_user
from backend.app.routers.plans import CreatePlanBody, _build_plan_items
from backend.app.services import clinic
from backend.app.services import plans as plans_store
from backend.app.services import store

router = APIRouter(prefix="/api/clinic", tags=["clinic"])


def _uid(raw: str) -> str:
    """Normalize a path id to canonical UUID form, or 404 -- same guard as
    ``routers/plans.py::_plan_uuid``: a non-UUID would otherwise reach PostgREST and surface as a
    500 (``22P02``)."""
    try:
        return str(uuid.UUID(raw))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"No such id '{raw}'.") from exc


@router.get("/status")
def clinic_status(user: CurrentUser = Depends(get_current_user)) -> dict:
    """Report whether the signed-in caller holds the clinician role (UX gating for the frontend)."""
    return {"is_clinician": store.is_clinician(token=user.token, user_id=user.id)}


@router.post("/invites", status_code=201)
def create_invite(user: CurrentUser = Depends(get_clinician_user)) -> dict:
    """Generate a fresh 6-character invite code as a pending ``care_links`` row."""
    return clinic.create_invite(token=user.token, clinician_id=user.id)


@router.get("/invites")
def list_invites(user: CurrentUser = Depends(get_clinician_user)) -> dict:
    """The caller's own pending, unexpired invites, newest first."""
    return {"invites": clinic.list_invites(token=user.token, clinician_id=user.id)}


def _patient_summary(
    row: dict, plans: list[dict], items: list[dict], checkins: list[dict], now: datetime
) -> dict:
    """One patient's dashboard row: the display fields off ``row`` plus the four aggregations.

    ``plans``/``items``/``checkins`` must already be filtered to THIS patient -- callers slice the
    batched cross-patient reads before calling this, so the aggregation functions (which assume
    single-patient input, see ``services/clinic.py``) are never handed another patient's rows.
    """
    return {
        "link_id": row.get("link_id"),
        "patient_id": row.get("patient_id"),
        "display_name": row.get("display_name"),
        "email": row.get("email"),
        "accepted_at": row.get("accepted_at"),
        "last_checkin_at": clinic.last_checkin_at(checkins),
        "adherence_7d": clinic.adherence_7d(plans, items, checkins, now=now),
        "open_flags": clinic.open_flags(checkins),
        "inactive_7d": clinic.inactive_7d(plans, checkins, now=now),
    }


@router.get("/patients")
def list_patients(user: CurrentUser = Depends(get_clinician_user)) -> dict:
    """Every linked patient with their adherence/red-flag summary.

    Batched: one plans query, one items query and one check-ins query cover EVERY linked patient
    (``services.clinic.batch_patient_data``), not one round trip per patient per table.
    """
    patients = clinic.list_patients(token=user.token)
    if not patients:
        return {"patients": []}

    patient_ids = [p["patient_id"] for p in patients]
    plans, items, checkins = clinic.batch_patient_data(token=user.token, patient_ids=patient_ids)
    now = datetime.now(timezone.utc)

    summaries = []
    for row in patients:
        pid = row["patient_id"]
        p_plans = [p for p in plans if p.get("user_id") == pid]
        p_items = [i for i in items if i.get("user_id") == pid]
        p_checkins = [c for c in checkins if c.get("user_id") == pid]
        summaries.append(_patient_summary(row, p_plans, p_items, p_checkins, now))
    return {"patients": summaries}


@router.get("/patients/{patient_id}")
def get_patient(patient_id: str, user: CurrentUser = Depends(get_clinician_user)) -> dict:
    """One linked patient's full clinic view: summary, plans, recent check-ins, trend, open flags.

    404 unless ``patient_id`` is one of the caller's ``clinic_patients()`` -- see the module
    docstring on why that check is a UX nicety, not the security boundary.
    """
    pid = _uid(patient_id)
    row = clinic.find_patient(token=user.token, patient_id=pid)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No patient '{patient_id}'.")

    plans, items, checkins = clinic.batch_patient_data(token=user.token, patient_ids=[pid])
    now = datetime.now(timezone.utc)

    all_checkins = clinic.list_checkins(token=user.token, patient_id=pid)
    return {
        "patient": _patient_summary(row, plans, items, checkins, now),
        "plans": plans_store.list_plans(token=user.token, user_id=pid),
        "checkins": clinic.recent_checkins(all_checkins),
        "trend": clinic.trend(all_checkins, now=now),
        "open_flags": clinic.open_flag_rows(all_checkins),
    }


@router.post("/patients/{patient_id}/plans", status_code=201)
def create_patient_plan(
    patient_id: str,
    body: CreatePlanBody,
    user: CurrentUser = Depends(get_clinician_user),
) -> dict:
    """Assign a plan to a linked patient: same body/validation as ``POST /api/plans``, but the new
    plan is owned by the PATIENT (``user_id``) and stamped ``assigned_by`` = the caller.

    404 unless linked (checked BEFORE building the items, so an unknown-template 400 never fires
    for a patient the caller cannot even see).
    """
    pid = _uid(patient_id)
    if clinic.find_patient(token=user.token, patient_id=pid) is None:
        raise HTTPException(status_code=404, detail=f"No patient '{patient_id}'.")

    items = _build_plan_items(body)
    return plans_store.create_plan(
        token=user.token,
        user_id=pid,
        name=body.name.strip(),
        notes=body.notes,
        template_key=body.template_key,
        items=items,
        assigned_by=user.id,
    )


@router.delete("/links/{link_id}")
def revoke_link(link_id: str, user: CurrentUser = Depends(get_clinician_user)) -> dict:
    """Revoke a care link (an active link, or a still-pending invite this clinician issued)."""
    lid = _uid(link_id)
    result = clinic.revoke_link(token=user.token, link_id=lid)
    if "error" in result:
        raise HTTPException(status_code=404, detail="No such care link.")
    return result
