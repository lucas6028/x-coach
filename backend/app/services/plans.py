"""Persistence service for training plans ("訓練菜單"): per-user routines over the movement catalog.

Same contract as ``store.py`` and for the same reason: every call runs through a Supabase client
authenticated with the CALLER'S OWN JWT, so PostgREST executes as that user and the
``training_plans`` / ``plan_items`` RLS policies scope every row to ``auth.uid() = user_id``. The
backend holds no service_role key, so a missing ``user_id`` predicate here is a bug, not a breach.

``_user_client`` is imported from ``store`` rather than redefined: one place builds the client, one
place is patched by the unit tests (``backend.app.services.plans._user_client``).
"""

from __future__ import annotations

from typing import Any

from backend.app.services.store import _user_client

# Every column of a plan, and of its items, that the API returns. Spelled out rather than "*" so a
# column added later (or an internal one) does not silently start appearing in API responses.
_PLAN_COLUMNS = "id, name, notes, template_key, started_at, created_at, updated_at"
_ITEM_COLUMNS = (
    "id, plan_id, day_index, position, movement, sets, reps, notes, completed_at, "
    "analysis_id, created_at"
)


def _sorted_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Items in the order the UI renders them: by day, then by position, then oldest first.

    Sorted in Python rather than trusting PostgREST's ``.order()`` alone because the items of a
    whole plan arrive in ONE query and are then split per day; a single stable key here keeps the
    day columns and the flat list agreeing. ``created_at`` breaks a position tie, so two items
    added to the same day without an explicit position keep their insertion order instead of
    swapping places between reads.
    """
    return sorted(
        items,
        key=lambda it: (it.get("day_index") or 0, it.get("position") or 0, it.get("created_at") or ""),
    )


def list_plans(*, token: str, user_id: str) -> list[dict[str, Any]]:
    """The caller's plans, newest first, each with a progress summary for the list card.

    Two queries, not one per plan: the plans, then every item belonging to them in a single
    ``in_`` fetch. The summary (``item_count``, ``completed_count``, ``day_count``, ``movements``)
    is computed here rather than by a PostgREST aggregate, matching ``get_storage_used``'s
    reasoning -- aggregates depend on ``db-aggregates-enabled``, which is not ours to guarantee.
    """
    client = _user_client(token)
    resp = (
        client.table("training_plans")
        .select(_PLAN_COLUMNS)
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    plans = resp.data or []
    if not plans:
        return []

    plan_ids = [str(p["id"]) for p in plans]
    items_resp = (
        client.table("plan_items")
        .select("plan_id, day_index, position, movement, completed_at, created_at")
        .in_("plan_id", plan_ids)
        .execute()
    )
    by_plan: dict[str, list[dict[str, Any]]] = {pid: [] for pid in plan_ids}
    for item in items_resp.data or []:
        by_plan.setdefault(str(item.get("plan_id")), []).append(item)

    summarized = []
    for plan in plans:
        items = _sorted_items(by_plan.get(str(plan["id"]), []))
        days = {it.get("day_index") for it in items if it.get("day_index")}
        summarized.append(
            {
                **plan,
                "item_count": len(items),
                "completed_count": sum(1 for it in items if it.get("completed_at")),
                # How many days this plan actually uses. Derived, never stored -- see the
                # migration's note on why there is no `day_count` column.
                "day_count": len(days),
                # The distinct movements in plan order, so the card can show what a plan trains
                # without the client fetching every item.
                "movements": list(dict.fromkeys(it["movement"] for it in items if it.get("movement"))),
            }
        )
    return summarized


def get_plan(*, token: str, plan_id: str, user_id: str) -> dict[str, Any] | None:
    """One of the caller's plans with its full item list, or ``None`` if there is no such plan.

    "Not yours" and "does not exist" are the same answer here for the same reason they are in
    ``store.get_analysis``: RLS scopes the read, so the two are indistinguishable by construction.
    """
    client = _user_client(token)
    resp = (
        client.table("training_plans")
        .select(_PLAN_COLUMNS)
        .eq("id", plan_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None

    items_resp = (
        client.table("plan_items").select(_ITEM_COLUMNS).eq("plan_id", plan_id).execute()
    )
    return {**rows[0], "items": _sorted_items(items_resp.data or [])}


def create_plan(
    *,
    token: str,
    user_id: str,
    name: str,
    notes: str | None = None,
    template_key: str | None = None,
    items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Insert a plan and, in one more round trip, its initial items; return the plan with items.

    ``items`` is how a built-in template lands as a real, independently editable plan: the
    template's rows are COPIED in at creation, so a later edit to the template in code never
    mutates a plan the user already owns (the migration's `template_key` note).

    The item insert is a single batched call. If it fails the plan row survives as an empty plan
    rather than being rolled back -- PostgREST has no cross-request transaction, and an empty plan
    the user can add to is a better failure than a 500 with an invisible orphan row.
    """
    client = _user_client(token)
    resp = (
        client.table("training_plans")
        .insert(
            {
                "user_id": user_id,
                "name": name,
                "notes": notes,
                "template_key": template_key,
            }
        )
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise RuntimeError("Plan insert returned no row.")
    plan = rows[0]
    plan_id = str(plan["id"])

    created_items: list[dict[str, Any]] = []
    if items:
        payload = [
            {
                "plan_id": plan_id,
                "user_id": user_id,
                "day_index": item["day_index"],
                "position": item.get("position", index),
                "movement": item["movement"],
                "sets": item.get("sets", 3),
                "reps": item.get("reps", 10),
                "notes": item.get("notes"),
            }
            for index, item in enumerate(items)
        ]
        items_resp = client.table("plan_items").insert(payload).execute()
        created_items = _sorted_items(items_resp.data or [])

    return {**plan, "items": created_items}


def update_plan(
    *,
    token: str,
    plan_id: str,
    user_id: str,
    fields: dict[str, Any],
) -> dict[str, Any] | None:
    """Patch a plan's editable fields; return the updated row, or ``None`` if it is not the caller's.

    ``fields`` is already filtered by the router to the columns a client may set (name, notes), so
    this never has to defend the column list itself. An empty patch is a read: PostgREST rejects an
    update with no payload, and answering "here is the unchanged plan" is what a no-op PATCH means.
    """
    if not fields:
        plan = get_plan(token=token, plan_id=plan_id, user_id=user_id)
        return {k: v for k, v in plan.items() if k != "items"} if plan else None

    client = _user_client(token)
    resp = (
        client.table("training_plans")
        .update(fields)
        .eq("id", plan_id)
        .eq("user_id", user_id)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def delete_plan(*, token: str, plan_id: str, user_id: str) -> bool:
    """Delete a plan; return whether a row was actually removed.

    The items go with it via ``plan_items.plan_id references ... on delete cascade`` -- there is no
    second delete here, and there must not be one: a manual item sweep that ran BEFORE the plan
    delete failed would strip a plan the user still owns.
    """
    client = _user_client(token)
    resp = (
        client.table("training_plans")
        .delete()
        .eq("id", plan_id)
        .eq("user_id", user_id)
        .execute()
    )
    return bool(resp.data or [])


def start_plan(*, token: str, plan_id: str, user_id: str, now: str) -> dict[str, Any] | None:
    """Begin a run of this plan: stamp ``started_at`` and clear every item's progress.

    Returns the plan with its (now unticked) items, or ``None`` if it is not the caller's.

    THE PLAN ROW IS STAMPED FIRST, and only then are the items cleared. The other order would, on a
    failure between the two calls, leave a plan whose progress had been wiped without any record
    that a new run began -- the user's ticks gone for nothing. This way a failure leaves a started
    plan carrying the previous run's ticks, which the user can see and clear again.

    ``now`` is passed in rather than read from the clock here so the router owns the timestamp and
    the tests can pin it.
    """
    client = _user_client(token)
    resp = (
        client.table("training_plans")
        .update({"started_at": now})
        .eq("id", plan_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not (resp.data or []):
        return None

    # Clearing `analysis_id` as well as `completed_at` is what makes the restart honest: an item
    # showing "✓ 已分析" must not keep pointing at last week's clip. The analysis itself is
    # untouched and stays in 我的紀錄 (migration note).
    client.table("plan_items").update({"completed_at": None, "analysis_id": None}).eq(
        "plan_id", plan_id
    ).eq("user_id", user_id).execute()

    return get_plan(token=token, plan_id=plan_id, user_id=user_id)


def plan_exists(*, token: str, plan_id: str, user_id: str) -> bool:
    """Whether the caller owns a plan with this id -- the ownership check for the item endpoints.

    Item writes filter on ``plan_id`` AND ``user_id`` anyway, so this is not the security boundary;
    it is what lets the router answer 404 for a bad plan id instead of silently succeeding with a
    write that matched nothing.
    """
    client = _user_client(token)
    resp = (
        client.table("training_plans")
        .select("id")
        .eq("id", plan_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return bool(resp.data or [])


def add_item(
    *,
    token: str,
    user_id: str,
    plan_id: str,
    day_index: int,
    movement: str,
    sets: int,
    reps: int,
    notes: str | None = None,
    position: int | None = None,
) -> dict[str, Any]:
    """Append one exercise to a plan's day; return the created row.

    ``position`` defaults to "after everything already in that day", computed from a count rather
    than from the client: two tabs adding to the same day would otherwise both send 0 and the day
    would order arbitrarily. A tie still resolves by ``created_at`` (see ``_sorted_items``).
    """
    client = _user_client(token)
    if position is None:
        existing = (
            client.table("plan_items")
            .select("position")
            .eq("plan_id", plan_id)
            .eq("user_id", user_id)
            .eq("day_index", day_index)
            .execute()
        )
        rows = existing.data or []
        position = max((int(r.get("position") or 0) for r in rows), default=-1) + 1

    resp = (
        client.table("plan_items")
        .insert(
            {
                "plan_id": plan_id,
                "user_id": user_id,
                "day_index": day_index,
                "position": position,
                "movement": movement,
                "sets": sets,
                "reps": reps,
                "notes": notes,
            }
        )
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise RuntimeError("Plan item insert returned no row.")
    return rows[0]


def update_item(
    *,
    token: str,
    user_id: str,
    plan_id: str,
    item_id: str,
    fields: dict[str, Any],
) -> dict[str, Any] | None:
    """Patch one item (day/position/sets/reps/notes, and its completion); return it, or ``None``.

    ``plan_id`` is in the predicate as well as ``item_id`` so an item id from a DIFFERENT plan of
    the same user's answers 404 rather than being edited through the wrong plan's URL -- RLS scopes
    by user, and would happily allow it.
    """
    if not fields:
        return get_item(token=token, user_id=user_id, plan_id=plan_id, item_id=item_id)

    client = _user_client(token)
    resp = (
        client.table("plan_items")
        .update(fields)
        .eq("id", item_id)
        .eq("plan_id", plan_id)
        .eq("user_id", user_id)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def get_item(
    *, token: str, user_id: str, plan_id: str, item_id: str
) -> dict[str, Any] | None:
    """One item of one of the caller's plans, or ``None``."""
    client = _user_client(token)
    resp = (
        client.table("plan_items")
        .select(_ITEM_COLUMNS)
        .eq("id", item_id)
        .eq("plan_id", plan_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def delete_item(*, token: str, user_id: str, plan_id: str, item_id: str) -> bool:
    """Delete one item from a plan; return whether a row was actually removed."""
    client = _user_client(token)
    resp = (
        client.table("plan_items")
        .delete()
        .eq("id", item_id)
        .eq("plan_id", plan_id)
        .eq("user_id", user_id)
        .execute()
    )
    return bool(resp.data or [])
