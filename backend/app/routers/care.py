"""Patient-side endpoints ("/api/care"): accept a clinician's invite, list one's own clinicians,
unlink. The clinician-side counterpart (invite, read a patient's data, assign a plan) is
``routers/clinic.py``.

NOT clinician-gated -- every route uses plain ``get_current_user``, since any signed-in user is a
potential PATIENT (that is the whole point of the invite flow: a clinician invites someone who has
no role at all). The trust boundary is the database: ``accept_care_invite`` and
``revoke_care_link`` are SECURITY DEFINER functions that check the caller against ``auth.uid()``
themselves (see the migration's header note on why ``care_links`` has no UPDATE policy at all), so
this router does no ownership check of its own -- it only translates the RPC's ``{"error": ...}``
result into an HTTP status.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.auth import CurrentUser, get_current_user
from backend.app.services import clinic

router = APIRouter(prefix="/api/care", tags=["care"])


def _uid(raw: str) -> str:
    """Normalize a path id to canonical UUID form, or 404 -- see ``routers/clinic.py::_uid``."""
    try:
        return str(uuid.UUID(raw))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"No such id '{raw}'.") from exc


class AcceptInviteBody(BaseModel):
    code: str = Field(..., min_length=1, max_length=32)


@router.post("/accept")
def accept_invite(body: AcceptInviteBody, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Consume a clinician's invite code, creating (or reactivating) an active care link.

    ``accept_care_invite`` normalizes the code itself (uppercases, trims), so the raw string is
    forwarded as-is. ``"invalid"`` covers an unknown, already-used, revoked, or expired code alike
    -- the RPC deliberately does not distinguish them (see the migration), so neither does this.
    """
    result = clinic.accept_invite(token=user.token, code=body.code)
    error = result.get("error")
    if error == "invalid":
        raise HTTPException(status_code=404, detail="Invite code is invalid or expired.")
    if error == "self":
        raise HTTPException(status_code=400, detail="You cannot accept your own invite.")
    return result


@router.get("/clinicians")
def my_clinicians(user: CurrentUser = Depends(get_current_user)) -> dict:
    """The caller's linked (active) clinicians, display-mapped (synthetic LINE emails hidden)."""
    return {"clinicians": clinic.my_clinicians(token=user.token)}


@router.delete("/links/{link_id}")
def revoke_link(link_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """End a care link from the patient's side (works for either party, per the RPC)."""
    lid = _uid(link_id)
    result = clinic.revoke_link(token=user.token, link_id=lid)
    if "error" in result:
        raise HTTPException(status_code=404, detail="No such care link.")
    return result
