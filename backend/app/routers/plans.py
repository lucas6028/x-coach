"""Training-plan endpoints ("訓練菜單"): author a weekly routine over the movement catalog.

A plan is a REUSABLE TEMPLATE of relative day slots (Day 1..7), not a dated schedule -- see the
migration ``db/migrations/20260813000000_training_plans.sql`` for the data model and, in
particular, for what restarting a plan deliberately throws away.

Everything except the built-in template catalog requires a valid Supabase JWT: a plan is one
user's own data. ``GET /api/plans/templates`` is public for the same reason ``GET /api/movements``
is -- it is a static catalog describing what the feature offers, and gating it would leave a
signed-out visitor unable to see the thing they are being asked to sign in for.

The movement of every item is validated against ``src/pose/movements/catalog.py`` (all sixteen),
NOT against the detector registry (fourteen): a user may plan Jumping Jacks even though no video
of one can be analysed yet. Which items can be analysed is the frontend's business, decided from
``GET /api/movements``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.auth import CurrentUser, get_current_user
from backend.app.services import plans as plans_store
from backend.app.services import store

router = APIRouter(prefix="/api", tags=["plans"])

# How many day slots a plan spans. Matches the `day_index between 1 and 7` check in the migration;
# the two are one decision ("a plan covers at most a week") expressed at both layers.
MAX_DAY = 7


class TemplateItem(BaseModel):
    """One exercise of a built-in template.

    The bounds are the SAME as ``PlanItemBody``'s, and they are here rather than trusted because a
    template's rows are the one item payload that never passes through request validation: they are
    copied straight into the insert. Without them, a ``TEMPLATES`` edit putting an exercise on day 8
    would pass every test and then fail against the migration's ``day_index between 1 and 7`` check
    as a 500 on ``POST /api/plans``. With them it fails at import, which is where a typo in our own
    literals belongs.
    """

    day_index: int = Field(..., ge=1, le=MAX_DAY)
    movement: str = Field(..., min_length=1)
    sets: int = Field(..., ge=1, le=20)
    reps: int = Field(..., ge=1, le=200)


class PlanTemplate(BaseModel):
    """A built-in starting point. ``name``/``description`` are English fallbacks; the frontend
    renders its own localized strings off ``key`` and sends the user-visible name back on create,
    so a plan created in Chinese is stored in Chinese."""

    key: str
    name: str
    description: str
    items: list[TemplateItem]


def _t(day: int, movement: str, sets: int, reps: int) -> TemplateItem:
    return TemplateItem(day_index=day, movement=movement, sets=sets, reps=reps)


# The built-in templates. Every movement here is one the detector registry can actually analyse, so
# a user who starts from a template can run the whole thing through the studio -- the two catalog
# movements without a detector (Jumping Jacks, High Knee) are addable by hand but are not put in
# anyone's path by default.
TEMPLATES: list[PlanTemplate] = [
    PlanTemplate(
        key="full_body_starter",
        name="Full-body starter",
        description="Three sessions a week covering the whole body. A good first plan.",
        items=[
            _t(1, "Squat", 3, 10),
            _t(1, "Push-up", 3, 8),
            _t(1, "Row", 3, 10),
            _t(3, "Lunge", 3, 10),
            _t(3, "Overhead Press", 3, 8),
            _t(3, "Sit-up", 3, 12),
            _t(5, "Deadlift", 3, 8),
            _t(5, "Bicep Curl", 3, 12),
            _t(5, "Torso Twist", 3, 15),
        ],
    ),
    PlanTemplate(
        key="upper_body",
        name="Upper-body focus",
        description="Two pushing and pulling sessions for chest, back and shoulders.",
        items=[
            _t(1, "Push-up", 4, 8),
            _t(1, "Overhead Press", 4, 8),
            _t(1, "Band Pull Apart", 3, 15),
            _t(4, "Row", 4, 10),
            _t(4, "Bicep Curl", 3, 12),
            _t(4, "Arm Abduction", 3, 15),
        ],
    ),
    PlanTemplate(
        key="lower_body",
        name="Lower-body focus",
        description="Two leg sessions built around the squat and the hinge.",
        items=[
            _t(1, "Squat", 4, 10),
            _t(1, "Lunge", 3, 10),
            _t(1, "Shoulder Bridge", 3, 15),
            _t(4, "Deadlift", 4, 8),
            _t(4, "Leg Abduction", 3, 15),
            _t(4, "Squat", 2, 15),
        ],
    ),
    PlanTemplate(
        key="mobility",
        name="Mobility & rehab",
        description="Low-load shoulder and hip work, three short sessions a week.",
        items=[
            _t(1, "Arm Abduction", 3, 12),
            _t(1, "Arm VW", 3, 10),
            _t(3, "Band Pull Apart", 3, 15),
            _t(3, "Shoulder Bridge", 3, 12),
            _t(5, "Leg Abduction", 3, 12),
            _t(5, "Torso Twist", 3, 12),
        ],
    ),
    PlanTemplate(
        key="quick_core",
        name="Quick core session",
        description="One 15-minute session you can drop into any week.",
        items=[
            _t(1, "Sit-up", 3, 15),
            _t(1, "Torso Twist", 3, 20),
            _t(1, "Shoulder Bridge", 3, 15),
        ],
    ),
]

_TEMPLATES_BY_KEY = {tpl.key: tpl for tpl in TEMPLATES}


def _canonical_movement(movement: str) -> str:
    """Resolve ``movement`` to the catalog's spelling, or 400.

    Imported lazily for the same reason ``routers/movements.py`` defers the registry import: the
    API layer is tested without the heavy ML stack, and this module is imported at app startup.
    """
    from src.pose.movements.catalog import canonical_movement

    resolved = canonical_movement(movement)
    if resolved is None:
        raise HTTPException(status_code=400, detail=f"Unknown movement '{movement}'.")
    return resolved


def _plan_uuid(plan_id: str) -> str:
    """Normalize a path id to canonical UUID form, or 404.

    A non-UUID would reach PostgREST and surface as a 500 (``22P02 invalid input syntax for type
    uuid``); the same guard, and the same normalization, as ``analyses.delete_my_analysis``.
    """
    try:
        return str(uuid.UUID(plan_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"No plan '{plan_id}'.") from exc


def _now() -> str:
    """UTC now, ISO-8601 with an explicit offset — what PostgREST wants for a timestamptz."""
    return datetime.now(timezone.utc).isoformat()


class PlanItemBody(BaseModel):
    day_index: int = Field(..., ge=1, le=MAX_DAY)
    movement: str = Field(..., min_length=1)
    sets: int = Field(3, ge=1, le=20)
    reps: int = Field(10, ge=1, le=200)
    notes: str | None = Field(None, max_length=200)
    position: int | None = Field(None, ge=0)


class PlanItemPatch(BaseModel):
    """Every editable field of an item, all optional. Completion rides this same PATCH rather than
    getting its own endpoint: "tick this off" and "move this to Tuesday" are both partial updates of
    one row, and a second path would only duplicate the ownership checks."""

    day_index: int | None = Field(None, ge=1, le=MAX_DAY)
    movement: str | None = Field(None, min_length=1)
    sets: int | None = Field(None, ge=1, le=20)
    reps: int | None = Field(None, ge=1, le=200)
    notes: str | None = Field(None, max_length=200)
    position: int | None = Field(None, ge=0)
    # True stamps `completed_at`; False clears it AND the analysis link, so an untick returns the
    # item to the state a fresh run starts in.
    completed: bool | None = None
    # The analysis produced for this item, set by the studio once an upload persists. Only
    # meaningful together with `completed: true`.
    analysis_id: str | None = None


class CreatePlanBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    notes: str | None = Field(None, max_length=500)
    # Copy the items of this built-in template into the new plan. The copy is independent: editing
    # TEMPLATES later never touches a plan a user already owns.
    template_key: str | None = None
    # Explicit items, for "build my own". Ignored when `template_key` is given — a create is one or
    # the other, and silently merging the two would make the resulting plan hard to predict.
    items: list[PlanItemBody] = Field(default_factory=list)


class UpdatePlanBody(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=80)
    notes: str | None = Field(None, max_length=500)


# ---------------------------------------------------------------------------
# Templates. DECLARED BEFORE /plans/{plan_id}: FastAPI matches routes in declaration order, so the
# dynamic route would otherwise swallow "templates" as a plan id and answer 404.
# ---------------------------------------------------------------------------
@router.get("/plans/templates")
def list_templates() -> dict:
    """The built-in plan templates. Public — a static catalog, like ``GET /api/movements``."""
    return {"templates": [tpl.model_dump() for tpl in TEMPLATES]}


@router.get("/plans")
def list_plans(user: CurrentUser = Depends(get_current_user)) -> dict:
    """The caller's plans, newest first, each with its progress summary."""
    return {"plans": plans_store.list_plans(token=user.token, user_id=user.id)}


@router.post("/plans", status_code=201)
def create_plan(
    body: CreatePlanBody,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Create a plan, either empty, from explicit items, or copied from a built-in template."""
    items: list[dict[str, Any]]
    if body.template_key is not None:
        template = _TEMPLATES_BY_KEY.get(body.template_key)
        if template is None:
            raise HTTPException(
                status_code=400, detail=f"Unknown template '{body.template_key}'."
            )
        items = [
            {
                "day_index": it.day_index,
                # Canonicalized even though these are our own literals: it means a typo in
                # TEMPLATES fails loudly at request time instead of writing an unanalysable
                # movement name into a user's plan.
                "movement": _canonical_movement(it.movement),
                "sets": it.sets,
                "reps": it.reps,
            }
            for it in template.items
        ]
    else:
        items = [
            {
                "day_index": it.day_index,
                "movement": _canonical_movement(it.movement),
                "sets": it.sets,
                "reps": it.reps,
                "notes": it.notes,
                **({"position": it.position} if it.position is not None else {}),
            }
            for it in body.items
        ]

    return plans_store.create_plan(
        token=user.token,
        user_id=user.id,
        name=body.name.strip(),
        notes=body.notes,
        template_key=body.template_key,
        items=items,
    )


@router.get("/plans/{plan_id}")
def get_plan(plan_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """One plan with its full item list, or 404."""
    plan = plans_store.get_plan(
        token=user.token, plan_id=_plan_uuid(plan_id), user_id=user.id
    )
    if plan is None:
        raise HTTPException(status_code=404, detail=f"No plan '{plan_id}'.")
    return plan


@router.patch("/plans/{plan_id}")
def update_plan(
    plan_id: str,
    body: UpdatePlanBody,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Rename a plan or change its notes."""
    fields = body.model_dump(exclude_unset=True)
    if "name" in fields and fields["name"] is not None:
        fields["name"] = fields["name"].strip()
    row = plans_store.update_plan(
        token=user.token, plan_id=_plan_uuid(plan_id), user_id=user.id, fields=fields
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"No plan '{plan_id}'.")
    return row


@router.delete("/plans/{plan_id}")
def delete_plan(plan_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Delete a plan and (by cascade) its items."""
    if not plans_store.delete_plan(
        token=user.token, plan_id=_plan_uuid(plan_id), user_id=user.id
    ):
        raise HTTPException(status_code=404, detail=f"No plan '{plan_id}'.")
    return {"deleted": 1}


@router.post("/plans/{plan_id}/start")
def start_plan(plan_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Begin a run: stamp ``started_at`` and clear every item's tick and analysis link.

    Destructive by design and documented as such in the migration — the plan tracks the current run
    only, while the analyses it produced stay in 我的紀錄.
    """
    plan = plans_store.start_plan(
        token=user.token, plan_id=_plan_uuid(plan_id), user_id=user.id, now=_now()
    )
    if plan is None:
        raise HTTPException(status_code=404, detail=f"No plan '{plan_id}'.")
    return plan


@router.post("/plans/{plan_id}/items", status_code=201)
def add_item(
    plan_id: str,
    body: PlanItemBody,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Append one exercise to a day of a plan."""
    canonical_plan_id = _plan_uuid(plan_id)
    # Checked explicitly so a bad plan id answers 404 rather than 201-with-nothing-written: the
    # insert itself would happily create an item pointing at a plan the caller does not own, and be
    # rejected only by the foreign key, which surfaces as a 500.
    if not plans_store.plan_exists(
        token=user.token, plan_id=canonical_plan_id, user_id=user.id
    ):
        raise HTTPException(status_code=404, detail=f"No plan '{plan_id}'.")

    return plans_store.add_item(
        token=user.token,
        user_id=user.id,
        plan_id=canonical_plan_id,
        day_index=body.day_index,
        movement=_canonical_movement(body.movement),
        sets=body.sets,
        reps=body.reps,
        notes=body.notes,
        position=body.position,
    )


@router.patch("/plans/{plan_id}/items/{item_id}")
def update_item(
    plan_id: str,
    item_id: str,
    body: PlanItemPatch,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Edit one item: move it, change its sets/reps/notes, or tick it off (optionally with the
    analysis that proves it)."""
    canonical_plan_id = _plan_uuid(plan_id)
    canonical_item_id = _plan_uuid(item_id)

    fields = body.model_dump(exclude_unset=True)
    # `completed` and `analysis_id` are API-level names; the columns are `completed_at` and
    # `analysis_id`. Translate here so the store stays a thin column writer.
    completed = fields.pop("completed", None)
    analysis_id = fields.pop("analysis_id", None)

    # An explicit null for a NOT NULL column is dropped, not forwarded. Every field of
    # ``PlanItemPatch`` is optional and therefore typed ``X | None``, so a client sending
    # ``{"movement": null}`` produces ``fields["movement"] = None`` -- which PostgREST would reject
    # against a NOT NULL column and FastAPI would surface as a 500. ``notes`` is deliberately not in
    # this set: it is nullable, and null is how a note is cleared.
    for column in ("day_index", "movement", "sets", "reps", "position"):
        if column in fields and fields[column] is None:
            del fields[column]

    if "movement" in fields:
        fields["movement"] = _canonical_movement(fields["movement"])

    if completed is True:
        fields["completed_at"] = _now()
    elif completed is False:
        # An untick returns the item to a fresh-run state, link included: leaving a stale
        # `analysis_id` on an unticked item would light the 看報告 affordance on an item the user
        # just said they had not done.
        fields["completed_at"] = None
        fields["analysis_id"] = None

    if analysis_id is not None and completed is not False:
        # Verified against the caller's own analyses before it is stored. The column's foreign key
        # points at `analyses.id` and FK checks do not run under RLS, so without this a client
        # could pin someone else's analysis id onto its own plan item. It leaks nothing on its own
        # (reading that analysis still goes through the RLS-scoped GET), but a plan item claiming
        # an analysis its owner cannot open is a broken link we can cheaply refuse.
        resolved = str(uuid.UUID(analysis_id)) if _is_uuid(analysis_id) else None
        if resolved is None or store.get_analysis(token=user.token, analysis_id=resolved) is None:
            raise HTTPException(status_code=400, detail=f"No analysis '{analysis_id}'.")
        fields["analysis_id"] = resolved

    row = plans_store.update_item(
        token=user.token,
        user_id=user.id,
        plan_id=canonical_plan_id,
        item_id=canonical_item_id,
        fields=fields,
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"No plan item '{item_id}'.")
    return row


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


@router.delete("/plans/{plan_id}/items/{item_id}")
def delete_item(
    plan_id: str,
    item_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Remove one exercise from a plan."""
    if not plans_store.delete_item(
        token=user.token,
        user_id=user.id,
        plan_id=_plan_uuid(plan_id),
        item_id=_plan_uuid(item_id),
    ):
        raise HTTPException(status_code=404, detail=f"No plan item '{item_id}'.")
    return {"deleted": 1}
