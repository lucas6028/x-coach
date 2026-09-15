"""Patient-side endpoints ("/api/checkins"): report pain/RPE/note after a session, and read one's
own check-in history (WP2). The clinician-side acknowledgement of a flagged check-in
(``PATCH /api/clinic/flags/{id}/ack``) lives in ``routers/clinic.py`` instead, since it is gated by
``get_clinician_user`` rather than the plain ``get_current_user`` every route here uses -- any
signed-in user may be a patient reporting on their own plan.

OWNERSHIP OF EVERY REFERENCED ROW IS CHECKED IN PYTHON, NOT LEFT TO RLS. ``store.get_analysis`` is
RLS-scoped to "rows I can SELECT", which since the clinic migration's ``analyses_clinician_select``
policy includes a LINKED CLINICIAN reading a PATIENT's analysis too -- so "the row exists" no
longer proves "it's mine", and without the explicit ``row["user_id"] == caller`` check a
clinician's own JWT could pin a patient's analysis (and its form score) onto the clinician's own
check-in. Same reasoning, and the same check, as ``routers/plans.py::update_item`` around lines
506-511 -- see that comment for the full story.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.auth import CurrentUser, get_current_user
from backend.app.services import checkins as checkins_store
from backend.app.services import plans as plans_store
from backend.app.services import redflags
from backend.app.services import scoring
from backend.app.services import store

router = APIRouter(prefix="/api", tags=["checkins"])

# GET /api/checkins?limit= bounds -- same shape as store.list_analyses's clamp.
_DEFAULT_LIST_LIMIT = 30
_MAX_LIST_LIMIT = 100


def _uid(raw: str, *, status_code: int, detail: str) -> str:
    """Normalize an id to canonical UUID form, or raise with the given status -- same guard as
    ``routers/plans.py::_plan_uuid`` and ``routers/clinic.py::_uid``, parameterized on status
    because this router needs BOTH a 404 (plan/item) and a 400 (analysis) version of it."""
    try:
        return str(uuid.UUID(raw))
    except ValueError as exc:
        raise HTTPException(status_code=status_code, detail=detail) from exc


class CreateCheckinBody(BaseModel):
    plan_id: str
    plan_item_id: str | None = None
    analysis_id: str | None = None
    pain_nrs: int = Field(..., ge=0, le=10)
    rpe: int | None = Field(None, ge=0, le=10)
    note: str | None = Field(None, max_length=500)


@router.post("/checkins", status_code=201)
def create_checkin(
    body: CreateCheckinBody, user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Store one session check-in: verify ownership of everything it references, compute the form
    score off the linked analysis (if any), run the deterministic red-flag rules against the
    caller's own recent history on that plan, and insert."""
    plan_id = _uid(body.plan_id, status_code=404, detail=f"No plan '{body.plan_id}'.")
    if not plans_store.plan_exists(token=user.token, plan_id=plan_id, user_id=user.id):
        raise HTTPException(status_code=404, detail=f"No plan '{body.plan_id}'.")

    plan_item_id: str | None = None
    if body.plan_item_id is not None:
        plan_item_id = _uid(
            body.plan_item_id, status_code=404, detail=f"No plan item '{body.plan_item_id}'."
        )
        item = plans_store.get_item(
            token=user.token, user_id=user.id, plan_id=plan_id, item_id=plan_item_id
        )
        if item is None:
            raise HTTPException(status_code=404, detail=f"No plan item '{body.plan_item_id}'.")

    analysis_id: str | None = None
    form_score: int | None = None
    if body.analysis_id is not None:
        analysis_id = _uid(
            body.analysis_id, status_code=400, detail=f"No analysis '{body.analysis_id}'."
        )
        analysis_row = store.get_analysis(token=user.token, analysis_id=analysis_id)
        if analysis_row is None or str(analysis_row.get("user_id")) != user.id:
            raise HTTPException(status_code=400, detail=f"No analysis '{body.analysis_id}'.")
        form_score = scoring.form_score_for_analysis(analysis_row.get("result"))

    history = checkins_store.recent_checkins(token=user.token, user_id=user.id, plan_id=plan_id)
    new_checkin = {"plan_id": plan_id, "pain_nrs": body.pain_nrs, "form_score": form_score}
    reasons = redflags.evaluate(new_checkin, history)

    return checkins_store.create_checkin(
        token=user.token,
        payload={
            "user_id": user.id,
            "plan_id": plan_id,
            "plan_item_id": plan_item_id,
            "analysis_id": analysis_id,
            "pain_nrs": body.pain_nrs,
            "rpe": body.rpe,
            "note": body.note,
            "form_score": form_score,
            "flagged": bool(reasons),
            "flag_reasons": reasons,
        },
    )


@router.get("/checkins")
def list_checkins(
    plan_id: str | None = None,
    limit: int = _DEFAULT_LIST_LIMIT,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """The caller's own check-ins, newest first, optionally scoped to one plan."""
    canonical_plan_id: str | None = None
    if plan_id is not None:
        canonical_plan_id = _uid(plan_id, status_code=404, detail=f"No plan '{plan_id}'.")
    bounded_limit = max(1, min(limit, _MAX_LIST_LIMIT))
    rows = checkins_store.list_checkins(
        token=user.token, user_id=user.id, plan_id=canonical_plan_id, limit=bounded_limit
    )
    return {"checkins": rows}
