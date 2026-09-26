"""Lumen the plan agent: build and edit a training plan (訓練菜單) by conversation.

Reuses ``services/chat.py``'s tool-calling contract wholesale via ``chat_service._run_tool_loop``
(spec v3.3) — the same streaming, retraction, round/time-cap, and 4xx-without-tools-retry behaviour
the coaching chat already has, pointed at a completely different tool catalogue: six plan mutations
instead of three knowledge lookups. See ``specs/llm-chat-spec.md`` (the "plan agent" section) for
the frame contract this module promises the frontend.

Everything here runs as the CALLING USER, exactly like ``routers/plans.py``: every
``backend/app/services/plans.py`` call is threaded the caller's own ``token``/``user_id``, so RLS
scopes every row. The backend holds no service_role key — a missing predicate would be a bug, not a
breach, but there is no predicate to miss here since it is the store layer's job, not this one's.

ONE PLAN PER CONVERSATION. ``_PlanDispatcher.plan_id`` starts at whatever the request scoped (``None``
in "builder" mode) and is PINNED the moment ``create_plan`` succeeds, for the rest of the request —
so a later ``add_item`` in the SAME turn (the model creating a plan and immediately seeding it) acts
on the plan it just made, and a second ``create_plan`` in the same conversation is refused rather than
silently starting a second plan nobody asked for.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from backend.app.services import chat as chat_service
from backend.app.services import plans as plans_store

# Every movement string a tool call carries goes through this to reach the catalog's canonical
# spelling — same rule ``routers/plans.py::_canonical_movement`` enforces for the REST endpoints, so
# a plan built by Lumen and a plan built by hand end up with identically-spelled movement names.
# Imported lazily, inside the function that needs it, for the same reason ``routers/plans.py`` and
# ``routers/movements.py`` defer it: the API layer (and this module, imported at app startup via
# ``routers/plans.py``) is tested without the heavy ML stack.
def _canonicalize_movement(raw: Any) -> tuple[str | None, str | None]:
    """Resolve ``raw`` to the catalog's spelling, or an error naming the sixteen valid movements."""
    from src.pose.movements.catalog import CATALOG, canonical_movement

    resolved = canonical_movement(str(raw) if raw is not None else None)
    if resolved is None:
        return None, f"Unknown movement {raw!r}. Valid movements: {', '.join(CATALOG)}."
    return resolved, None


def _validate_int(value: Any, *, low: int, high: int, field: str) -> tuple[int | None, str | None]:
    """Coerce a MODEL-SUPPLIED number into ``[low, high]``, or an error naming the bound.

    Deliberately REJECTS rather than clamps (unlike ``chat_service._clamp_int``, used for the
    coaching tools' internal ``hops``/``top_k`` parameters): a plan's ``day_index``/``sets``/``reps``
    are literal, user-visible values the model is choosing on the user's behalf, so silently
    rewriting an out-of-range one would let Lumen claim a plan holds what it does not. Returning the
    bound as an error lets the model retry with a value that fits, the same way a REST 422 would.
    """
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: json.loads accepts bare Infinity as an extension, and int(float('inf'))
        # raises OverflowError, not ValueError — the same trap chat_service._clamp_int guards.
        return None, f"{field} must be a whole number between {low} and {high}."
    if n < low or n > high:
        return None, f"{field} must be between {low} and {high} (got {n})."
    return n, None


def _readable_muscle(key: str) -> str:
    """A muscle KEY (``muscles.MUSCLES`` spelling, e.g. ``"upperBack"``, ``"hipFlexors"``) as
    space-separated lowercase words, for prose the model reads (never for tool-result JSON, which
    stays keys-only — see ``_plan_coverage``)."""
    import re

    return re.sub(r"(?<!^)(?=[A-Z])", " ", key).lower()


def _movement_catalog_lines() -> str:
    """The sixteen catalog movements, grouped by body region, each with its primary muscles, for
    the system prompt — e.g. ``Squat — quads, glutes``. The muscle table lets Lumen reason about
    balance (pair pushing with pulling, don't stack the same primary group two days running)
    without waiting on a tool round-trip just to remember what a movement trains.
    """
    from src.pose.movements.catalog import CORE, FULL_BODY, LOWER_BODY, UPPER_BODY
    from src.pose.movements.muscles import MOVEMENT_MUSCLES

    def _entry(name: str) -> str:
        primary = MOVEMENT_MUSCLES.get(name, {}).get("primary", ())
        if not primary:
            return name
        return f"{name} — {', '.join(_readable_muscle(m) for m in primary)}"

    groups = [
        ("Lower body", LOWER_BODY),
        ("Upper body", UPPER_BODY),
        ("Core", CORE),
        # Jumping Jacks / High Knee: registered 2026-09-26 (one Beta rule each), so every group
        # here is analysable in the studio.
        ("Full body", FULL_BODY),
    ]
    lines = []
    for label, names in groups:
        lines.append(f"- {label}:")
        lines.extend(f"  - {_entry(name)}" for name in names)
    return "\n".join(lines)


def _plan_coverage(plan: dict[str, Any] | None) -> dict[str, Any]:
    """The shared-vocabulary muscle-coverage summary for ``plan``'s CURRENT items.

    Rides only the tool-result TEXT the model reads back (never the ``tool_done`` SSE frame's
    ``plan`` payload, which stays the plain plan the frontend already knows how to draw) — see the
    module's ``PLAN_TOOLS``/``_result`` machinery. Muscle KEYS only (``"quads"``, not "quadriceps"
    or a sentence), so the model can quote them back verbatim and so this stays compact against
    ``_MAX_RESULT_CHARS`` alongside the plan itself.

    ``by_day`` adds the PRIMARY muscles trained on each day that has at least one item, keyed by
    ``day_index`` — the piece the whole-plan ``primary``/``secondary``/``gaps`` triple can't answer
    on its own: "don't stack legs two days running" needs to know THIS day's primaries, not the
    week's.  A plan with no items yet (``create_plan`` was never called, or every item was removed)
    is not an error here — it answers the same empty-safe shape ``muscles.coverage(())`` does: empty
    ``primary``/``secondary``, the full gap checklist, and an empty ``by_day``.
    """
    from src.pose.movements.muscles import coverage as muscle_coverage

    items = (plan or {}).get("items") or []
    result = muscle_coverage(it.get("movement") for it in items if it.get("movement"))

    by_day: dict[int, list[str]] = {}
    days = sorted({it["day_index"] for it in items if it.get("day_index") is not None})
    for day in days:
        day_movements = (
            it.get("movement")
            for it in items
            if it.get("day_index") == day and it.get("movement")
        )
        by_day[day] = muscle_coverage(day_movements)["primary"]
    result["by_day"] = by_day
    return result


# WP4 rehab mode: appended to the system prompt (in the conversation's own language, the same way
# lang_line/scope_line already switch) whenever the plan scoped to this conversation was assigned by
# a therapist rather than built by the user themselves -- see `_plan_chat_stream_inner`, which is the
# only caller that ever passes `rehab=True`. Kept as one block rather than woven into RULES so the
# ordinary (non-rehab) prompt stays byte-identical to before this feature existed.
_REHAB_BLOCK_ZH = (
    "\n復健模式 — 這份菜單是治療師指派的：\n"
    "- 你是在協助一位使用治療師指派菜單的患者。\n"
    "- 絕不做出診斷或說出病名。\n"
    "- 絕不更改菜單的動作、組數、次數或安排的日子 — 只有治療師能調整；如果使用者要求變更，婉拒並說明原因。\n"
    "- 鼓勵使用者按表操課、注意動作品質。\n"
    "- 如果使用者提到疼痛、頭暈、腫脹或症狀惡化，請他立刻停止動作並聯繫治療師。\n"
    "- 你仍然可以說明動作怎麼做、回答一般問題。\n\n"
)
_REHAB_BLOCK_EN = (
    "\nREHAB MODE — this plan was assigned by the user's therapist:\n"
    "- You are supporting a patient on a plan their therapist assigned.\n"
    "- NEVER diagnose or name a medical condition.\n"
    "- NEVER change the plan's movements, sets, reps, or days — only the therapist may; refuse and "
    "say so if the user asks.\n"
    "- Encourage adherence and good form.\n"
    "- If the user mentions pain, dizziness, swelling, or worsening symptoms, tell them to stop and "
    "contact their therapist immediately.\n"
    "- You may still explain movements and answer general questions.\n\n"
)

# The refusal every write tool returns verbatim on an assigned plan (see _PlanDispatcher.dispatch) —
# one fixed string, not localized: the model reads it as a tool result, not user-facing prose, and
# is instructed (via the rehab block above) to explain the refusal itself in the user's language.
REHAB_WRITE_REFUSAL = "This plan was assigned by your therapist; ask them to change it."


def _system_prompt(*, lang: str, plan_scoped: bool, rehab: bool = False) -> str:
    """Lumen's system prompt: persona, honesty rules, and the movement catalog.

    ``plan_scoped`` tells the model whether it is in BUILDER mode (no plan yet — the common case for
    ``/plans/new``) or editing an existing one (``/plans/:id``), so it knows whether ``create_plan``
    is even on the table without having to call ``get_plan`` first to find out.

    ``rehab`` is WP4's gate: true only when the scoped plan's ``assigned_by`` is set (a therapist
    assigned it). It can only be true together with ``plan_scoped`` — builder mode has no plan yet,
    so nothing to have been assigned — but that invariant lives in the caller, not here.
    """
    lang_line = (
        "Reply in Traditional Chinese (繁體中文), the way a Taiwanese fitness app actually talks — "
        "warm, specific, plain language, never a stiff literal translation."
        if lang == "zh-Hant"
        else "Reply in English."
    )
    scope_line = (
        "A plan is already scoped to this conversation. Call get_plan before making any claim about "
        "what it currently contains — never guess or assume from earlier turns."
        if plan_scoped
        else "No plan exists yet in this conversation: you are in BUILDER mode."
    )
    rehab_block = ""
    if rehab:
        rehab_block = _REHAB_BLOCK_ZH if lang == "zh-Hant" else _REHAB_BLOCK_EN
    return (
        "You are Lumen, x-coach's AI training coach. You build and edit the user's training plan "
        "(訓練菜單) by calling the tools available to you. You NEVER claim a change you did not make "
        "through a tool, and you NEVER re-list the whole plan in prose unless asked — it is already "
        "shown next to this chat.\n\n"
        f"{lang_line}\n\n"
        f"{scope_line}\n"
        f"{rehab_block}\n"
        "RULES:\n"
        "- A plan is a REUSABLE TEMPLATE of relative day slots, Day 1 through Day 7 — not dated "
        "calendar days.\n"
        "- In builder mode: if the goal, days per week, experience, or any limits are unknown, ask "
        "at most 2-3 short questions in ONE message before creating anything. Once you have enough, "
        "create the WHOLE plan in a single create_plan call — never create an empty plan and add "
        "items one at a time.\n"
        "- Only one plan can be scoped to this conversation: a second create_plan call is refused — "
        "edit the existing plan instead.\n"
        "- Every movement below can be analysed in the studio.\n"
        "- sets: 1-20. reps: 1-200.\n"
        "- After a tool runs, confirm in 1-3 short lines exactly WHAT changed (day, movement, "
        "sets×reps) — never claim a change a tool did not confirm.\n"
        "- When the request is ambiguous (which day? replace this exercise or add another?), ask — "
        "never guess.\n"
        "- Balance the week: cover the major muscle groups across all days, pair pushing movements "
        "with pulling ones, and avoid stacking the same primary muscle group on consecutive days.\n"
        "- When the user asks what a plan trains, or what it's missing (\"這週哪裡沒練到\"), answer "
        "from the coverage a tool result just reported — never from memory or the catalog above; "
        "call get_plan first if you don't already have a fresh one this turn.\n\n"
        "MOVEMENT CATALOG (16 movements, with primary muscles):\n" + _movement_catalog_lines() + "\n"
    )


# The tool catalogue, in the OpenAI-compatible ``tools`` schema _run_tool_loop expects. Mirrors
# ``chat_service._TOOLS``'s shape; descriptions do real work here too, since they are the only place
# the model learns the builder-mode / one-plan-per-conversation rules apply to THESE calls specifically.
PLAN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_plan",
            "description": (
                "Read the plan currently scoped to this conversation, with its items and their "
                "ids. Only meaningful once a plan exists — call it before claiming what the plan "
                "contains, never assume from earlier turns."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_plan",
            "description": (
                "Create a brand-new plan with its initial items, all in ONE call. Allowed ONLY when "
                "no plan is scoped yet in this conversation — ask your 2-3 clarifying questions "
                "FIRST, then create the whole plan at once, never an empty plan you add to later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": 'A short plan name, e.g. "3-day full body".',
                    },
                    "notes": {"type": ["string", "null"]},
                    "items": {
                        "type": "array",
                        "description": "At least one exercise.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "day_index": {
                                    "type": "integer",
                                    "description": "1-7, a relative day slot in the week.",
                                },
                                "movement": {"type": "string"},
                                "sets": {
                                    "type": "integer",
                                    "description": "1-20. Defaults to 3 if omitted.",
                                },
                                "reps": {
                                    "type": "integer",
                                    "description": "1-200. Defaults to 10 if omitted.",
                                },
                                "notes": {"type": ["string", "null"]},
                            },
                            "required": ["day_index", "movement"],
                        },
                    },
                },
                "required": ["name", "items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Add one exercise to a day of the plan currently scoped to this conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day_index": {"type": "integer", "description": "1-7."},
                    "movement": {"type": "string"},
                    "sets": {"type": "integer", "description": "1-20. Defaults to 3."},
                    "reps": {"type": "integer", "description": "1-200. Defaults to 10."},
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["day_index", "movement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_item",
            "description": (
                "Edit one exercise of the scoped plan: move it to another day, change its "
                "sets/reps/notes, or swap the movement. Only the fields given are changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "From get_plan or a prior tool result."},
                    "day_index": {"type": ["integer", "null"]},
                    "movement": {"type": ["string", "null"]},
                    "sets": {"type": ["integer", "null"]},
                    "reps": {"type": ["integer", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_item",
            "description": "Remove one exercise from the scoped plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "From get_plan or a prior tool result."}
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Rename the scoped plan or change its notes. Only the fields given are changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
            },
        },
    },
]

# Same budget as the coaching tools (chat_service._MAX_TOOL_RESULT_CHARS): a tool result is pasted
# back into the next round's context verbatim, and a plan with many items could otherwise run long.
_MAX_RESULT_CHARS = chat_service._MAX_TOOL_RESULT_CHARS


def _plan_query_label(name: str, args: dict[str, Any]) -> str:
    """The human-readable subject of a plan tool call, for the ``tool`` SSE frame the UI displays.

    Yielded BEFORE the tool runs (chat_service._run_tool_loop's contract), so it can only come from
    the call's own arguments — a movement for the item tools, the plan's own name for the plan-level
    ones, nothing for the two that carry neither.
    """
    if name in ("add_item", "update_item"):
        return str(args.get("movement") or "")
    if name in ("create_plan", "update_plan"):
        return str(args.get("name") or "")
    return ""


class _PlanDispatcher:
    """Holds the mutable per-request state the six plan tools share, and dispatches calls to them.

    ``plan_id`` starts as whatever the request scoped and is PINNED the moment ``create_plan``
    succeeds — see the module docstring. A single instance is constructed once per request and its
    ``dispatch`` bound method is handed to ``chat_service._run_tool_loop`` as the ``dispatch``
    callable, so every tool call in the same turn (including a create followed by an add, in the
    SAME round-trip) shares the one pinned id.

    ``rehab`` is WP4's gate (see ``_system_prompt``): when true, every WRITE tool refuses instead of
    running — see ``_WRITE_TOOLS`` and ``dispatch``. A conversation can only ever be scoped to an
    ALREADY-assigned plan (a clinician's own assignment flow does not go through this dispatcher at
    all), so ``rehab`` never flips true mid-conversation; it is fixed once, at construction, exactly
    like ``token``/``user_id``.
    """

    # The write tools this gate covers. Deliberately NOT ``create_plan``: a rehab-mode conversation
    # is by definition already scoped to the assigned plan (``rehab`` can only be true when
    # ``plan_id`` is set — see the module's `_plan_chat_stream_inner`), so `_create_plan`'s own
    # "already exists" refusal already covers it; adding it here too would just be a second,
    # untested path to the same outcome. ``get_plan`` is deliberately absent: read-only tools must
    # keep working so Lumen can still explain what the plan contains.
    _WRITE_TOOLS = frozenset({"add_item", "update_item", "remove_item", "update_plan"})

    def __init__(
        self, *, token: str, user_id: str, plan_id: str | None, rehab: bool = False
    ) -> None:
        self.token = token
        self.user_id = user_id
        self.plan_id = plan_id
        self.rehab = rehab

    def dispatch(self, name: str, args: dict[str, Any]) -> chat_service._ToolResult:
        """Run one plan tool call. NEVER RAISES — see ``chat_service._run_tool`` for why this
        matters: a crash here would otherwise propagate straight out of ``_run_tool_loop`` (which
        deliberately does not guard the ``dispatch`` call) and kill the stream after the HTTP 200 is
        already committed. Any tool method may raise (a malformed args shape, a store call failing
        against a plan deleted mid-conversation); this is the one boundary that turns that into an
        ``{"error": ...}`` payload the model can read and recover from.

        THE REHAB REFUSAL RUNS BEFORE ANY HANDLER, for every write tool — see ``_WRITE_TOOLS``. It
        performs NO store call: the refusal is a pure gate, so there is nothing here to undo if the
        model retries. The refusal payload has the same shape as any other tool error (``frame_plan``
        omitted), which is what keeps the client from redrawing a preview that never changed.
        """
        try:
            if self.rehab and name in self._WRITE_TOOLS:
                return self._result({"error": REHAB_WRITE_REFUSAL})
            handler = {
                "get_plan": self._get_plan,
                "create_plan": self._create_plan,
                "add_item": self._add_item,
                "update_item": self._update_item,
                "remove_item": self._remove_item,
                "update_plan": self._update_plan,
            }.get(name)
            if handler is None:
                return self._result({"error": f"Unknown tool {name!r}."})
            return handler(args)
        except Exception as exc:  # noqa: BLE001 — see the docstring: this boundary must never raise.
            return self._result({"error": f"{name} failed: {type(exc).__name__}"})

    # -- result building -----------------------------------------------------------------------

    def _result(
        self, model_payload: dict[str, Any], *, frame_plan: dict[str, Any] | None = None
    ) -> chat_service._ToolResult:
        """Build a ``_ToolResult``: ``model_payload`` is what the model reads back (truncated to the
        shared budget); ``frame_plan``, when given, rides the ``tool_done`` SSE frame so the client
        can update its live preview from that frame alone. Read-only ``get_plan`` and every error
        pass ``frame_plan=None`` — nothing changed, so there is nothing new for the client to draw.
        """
        try:
            text = json.dumps(model_payload, ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001 — a hostile value must not crash the answer stream.
            text = json.dumps({"error": f"serialisation failed: {type(exc).__name__}"}, ensure_ascii=False)
        if len(text) > _MAX_RESULT_CHARS:
            text = text[:_MAX_RESULT_CHARS] + "…[truncated]"
        payload = {"plan": frame_plan} if frame_plan is not None else None
        return chat_service._ToolResult(text=text, sources=[], payload=payload)

    def _fresh_plan(self) -> dict[str, Any] | None:
        """Re-fetch the FULL plan (same shape as ``GET /api/plans/{id}``) after a mutation.

        Re-fetching rather than trusting a mutation's own return value is deliberate, not just for
        shape consistency: ``plans_store.create_plan``/``add_item`` return the raw insert row, which
        carries ``user_id`` — a field ``get_plan`` never selects (it projects through
        ``_PLAN_COLUMNS``/``_ITEM_COLUMNS``). Re-fetching is what keeps that out of the SSE payload
        this module ships straight to the browser.
        """
        if not self.plan_id:
            return None
        return plans_store.get_plan(token=self.token, plan_id=self.plan_id, user_id=self.user_id)

    # -- tools ----------------------------------------------------------------------------------

    def _get_plan(self, args: dict[str, Any]) -> chat_service._ToolResult:
        if not self.plan_id:
            return self._result({"error": "No plan is scoped in this conversation yet."})
        plan = self._fresh_plan()
        if plan is None:
            return self._result({"error": "The scoped plan no longer exists."})
        # read-only: no frame_plan, nothing changed for the client to draw. coverage still rides
        # the text so the model can answer a "what am I missing" question off this one call.
        return self._result({"plan": plan, "coverage": _plan_coverage(plan)})

    def _create_plan(self, args: dict[str, Any]) -> chat_service._ToolResult:
        if self.plan_id:
            return self._result(
                {"error": "A plan already exists in this conversation; edit it instead."}
            )

        name = str(args.get("name") or "").strip()[:80]
        if not name:
            return self._result({"error": "create_plan needs a non-empty name."})

        raw_notes = args.get("notes")
        notes = str(raw_notes)[:500] if raw_notes else None

        raw_items = args.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            return self._result({"error": "create_plan needs at least one item."})

        items: list[dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                return self._result(
                    {"error": "Each item must be an object with day_index and movement."}
                )
            validated, error = self._validate_item(raw_item)
            if error:
                return self._result({"error": error})
            items.append(validated)

        created = plans_store.create_plan(
            token=self.token,
            user_id=self.user_id,
            name=name,
            notes=notes,
            template_key=None,
            items=items,
        )
        self.plan_id = str(created["id"])  # PINNED for the rest of this request.
        plan = self._fresh_plan()
        return self._result({"plan": plan, "coverage": _plan_coverage(plan)}, frame_plan=plan)

    def _add_item(self, args: dict[str, Any]) -> chat_service._ToolResult:
        if not self.plan_id:
            return self._result({"error": "No plan exists yet — create one first."})
        item, error = self._validate_item(args)
        if error:
            return self._result({"error": error})
        plans_store.add_item(
            token=self.token,
            user_id=self.user_id,
            plan_id=self.plan_id,
            day_index=item["day_index"],
            movement=item["movement"],
            sets=item["sets"],
            reps=item["reps"],
            notes=item["notes"],
        )
        plan = self._fresh_plan()
        return self._result({"plan": plan, "coverage": _plan_coverage(plan)}, frame_plan=plan)

    def _update_item(self, args: dict[str, Any]) -> chat_service._ToolResult:
        if not self.plan_id:
            return self._result({"error": "No plan is scoped in this conversation yet."})
        item_id = str(args.get("item_id") or "").strip()
        if not item_id:
            return self._result({"error": "update_item needs item_id."})

        fields: dict[str, Any] = {}
        if args.get("day_index") is not None:
            value, error = _validate_int(args["day_index"], low=1, high=7, field="day_index")
            if error:
                return self._result({"error": error})
            fields["day_index"] = value
        if args.get("movement"):
            resolved, error = _canonicalize_movement(args["movement"])
            if error:
                return self._result({"error": error})
            fields["movement"] = resolved
        if args.get("sets") is not None:
            value, error = _validate_int(args["sets"], low=1, high=20, field="sets")
            if error:
                return self._result({"error": error})
            fields["sets"] = value
        if args.get("reps") is not None:
            value, error = _validate_int(args["reps"], low=1, high=200, field="reps")
            if error:
                return self._result({"error": error})
            fields["reps"] = value
        if "notes" in args:
            raw_notes = args["notes"]
            fields["notes"] = str(raw_notes)[:200] if raw_notes else None

        row = plans_store.update_item(
            token=self.token,
            user_id=self.user_id,
            plan_id=self.plan_id,
            item_id=item_id,
            fields=fields,
        )
        if row is None:
            return self._result({"error": f"No item {item_id!r} in this plan."})
        plan = self._fresh_plan()
        return self._result({"plan": plan, "coverage": _plan_coverage(plan)}, frame_plan=plan)

    def _remove_item(self, args: dict[str, Any]) -> chat_service._ToolResult:
        if not self.plan_id:
            return self._result({"error": "No plan is scoped in this conversation yet."})
        item_id = str(args.get("item_id") or "").strip()
        if not item_id:
            return self._result({"error": "remove_item needs item_id."})
        removed = plans_store.delete_item(
            token=self.token, user_id=self.user_id, plan_id=self.plan_id, item_id=item_id
        )
        if not removed:
            return self._result({"error": f"No item {item_id!r} in this plan."})
        plan = self._fresh_plan()
        return self._result({"plan": plan, "coverage": _plan_coverage(plan)}, frame_plan=plan)

    def _update_plan(self, args: dict[str, Any]) -> chat_service._ToolResult:
        if not self.plan_id:
            return self._result({"error": "No plan is scoped in this conversation yet."})
        fields: dict[str, Any] = {}
        if "name" in args and args["name"] is not None:
            name = str(args["name"]).strip()
            if not name:
                return self._result({"error": "update_plan's name cannot be empty."})
            fields["name"] = name[:80]
        if "notes" in args:
            raw_notes = args["notes"]
            fields["notes"] = str(raw_notes)[:500] if raw_notes else None

        row = plans_store.update_plan(
            token=self.token, plan_id=self.plan_id, user_id=self.user_id, fields=fields
        )
        if row is None:
            return self._result({"error": "The scoped plan no longer exists."})
        plan = self._fresh_plan()
        return self._result({"plan": plan, "coverage": _plan_coverage(plan)}, frame_plan=plan)

    # -- shared item validation ------------------------------------------------------------------

    def _validate_item(self, raw: dict[str, Any]) -> tuple[dict[str, Any], None] | tuple[None, str]:
        """Validate one item's fields (used by both ``create_plan`` and ``add_item``): day_index and
        movement are required, sets/reps take the same defaults ``PlanItemBody`` does (3/10) when
        omitted, and notes are truncated rather than rejected — free text is not safety-critical the
        way a range is."""
        day_index, error = _validate_int(raw.get("day_index"), low=1, high=7, field="day_index")
        if error:
            return None, error
        movement, error = _canonicalize_movement(raw.get("movement"))
        if error:
            return None, error
        sets, error = _validate_int(raw.get("sets", 3), low=1, high=20, field="sets")
        if error:
            return None, error
        reps, error = _validate_int(raw.get("reps", 10), low=1, high=200, field="reps")
        if error:
            return None, error
        raw_notes = raw.get("notes")
        notes = str(raw_notes)[:200] if raw_notes else None
        return {
            "day_index": day_index,
            "movement": movement,
            "sets": sets,
            "reps": reps,
            "notes": notes,
        }, None


def plan_chat_stream(
    *,
    messages: list[dict[str, str]],
    plan_id: str | None,
    token: str,
    user_id: str,
    model: str,
    lang: str,
) -> Iterator[str]:
    """Thin outer shell: the last line of defence for the SSE contract (mirrors ``chat.answer_stream``).

    THIS IS THE ONE FRAME IN THIS MODULE THAT MUST NEVER LET AN EXCEPTION ESCAPE — the HTTP 200 is
    already committed by the time ``routers/plans.py`` hands this generator to ``StreamingResponse``,
    so there is no status code left to change, only the choice between an honest in-band ``error``
    frame and a silently truncated response. ``_PlanDispatcher.dispatch`` already never raises (its
    own docstring), so this really is only a backstop for something failing OUTSIDE that boundary —
    building the system prompt, constructing the dispatcher — the same shape as ``answer_stream``.
    """
    try:
        yield from _plan_chat_stream_inner(
            messages=messages, plan_id=plan_id, token=token, user_id=user_id, model=model, lang=lang
        )
    except Exception as exc:  # noqa: BLE001 — see the docstring: this is the outermost frame.
        yield chat_service._sse("error", {"detail": type(exc).__name__})


def _plan_chat_stream_inner(
    *,
    messages: list[dict[str, str]],
    plan_id: str | None,
    token: str,
    user_id: str,
    model: str,
    lang: str,
) -> Iterator[str]:
    # WP4: a plan scoped to this conversation may have been assigned by a therapist. Resolved once,
    # up front — the system prompt needs to know before the model's first round, not after a tool
    # call reveals it — and reused for both the prompt's rehab block and the dispatcher's write gate,
    # so the two can never disagree about which mode this conversation is in. Never true in builder
    # mode: there is no plan yet for anyone to have assigned.
    rehab = plan_id is not None and plans_store.plan_is_assigned(
        token=token, plan_id=plan_id, user_id=user_id
    )
    dispatcher = _PlanDispatcher(token=token, user_id=user_id, plan_id=plan_id, rehab=rehab)
    system = _system_prompt(lang=lang, plan_scoped=plan_id is not None, rehab=rehab)
    yield from chat_service._run_tool_loop(
        messages=messages,
        system=system,
        model=model,
        tools=PLAN_TOOLS,
        dispatch=dispatcher.dispatch,
        query_label=_plan_query_label,
        # A plan exists at the end of the turn either because the request scoped one, or because
        # create_plan pinned one during this very turn — dispatcher.plan_id covers both.
        done_extra=lambda: {"plan_id": dispatcher.plan_id} if dispatcher.plan_id else {},
    )
