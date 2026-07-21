"""Admin endpoints for the x-coach admin panel.

``GET /api/admin/status`` (P1) is a status probe: any signed-in user may ask "am I an admin?" so
the frontend can decide whether to show the Admin nav link and page. It is gated by
``get_current_user`` (not ``get_admin_user``) precisely because a non-admin must get a truthful
``{"is_admin": false}`` rather than a 403.

``GET``/``PUT /api/admin/settings`` (P2) are the runtime-settings surface, both gated by
``get_admin_user`` (403 for a non-admin). GET returns the currently-effective knobs merged with the
env/constant defaults so the form can show "current vs default"; PUT validates and upserts the
provided knobs into ``app_settings`` (RLS ``is_admin`` re-enforces admin-only writes in Postgres),
then clears the resolution cache so the change takes effect on the next read. No secret is ever read
or written here — the LLM API key and Supabase credentials stay pure-env.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend.app import config, settings
from backend.app.auth import CurrentUser, get_admin_user, get_current_user
from backend.app.services import line_quota, runtime_config, store

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/status")
def admin_status(user: CurrentUser = Depends(get_current_user)) -> dict:
    """Report whether the signed-in caller holds the admin role (UX gating for the frontend)."""
    return {"is_admin": store.is_admin(token=user.token, user_id=user.id)}


def _effective_settings() -> dict[str, Any]:
    """The currently-effective knobs (env defaults with admin overrides merged), grouped for the form."""
    return {
        "llm": {
            "llm_models": settings.chat_models(),
            "llm_followup_model": settings.followup_chat_model(),
            "llm_base_url": settings.chat_base_url(),
            "chat_temperature": settings.chat_temperature(),
            "chat_timeout": settings.chat_timeout(),
            "followup_timeout": settings.followup_timeout(),
        },
        "rag_kg": {
            "rag_top_k": settings.rag_top_k_default(),
            "kg_hops": settings.kg_hops_default(),
            "kg_seeds": settings.kg_seeds_default(),
        },
        "analyze": {
            "allowed_upload_suffixes": list(settings.allowed_upload_suffixes()),
            # READ-ONLY display value: the semaphore is fixed at import from the env var (see
            # routers/analyze), so this is never overridable — it is surfaced purely so the admin UI
            # can show the effective ceiling. It is sourced from the env constant and never written.
            "max_concurrent_analyses": config.MAX_CONCURRENT_ANALYSES,
        },
    }


def _default_settings() -> dict[str, Any]:
    """The pure env/constant defaults (no overrides applied), so the form can show "reset to default"."""
    s = settings.get_settings()
    default_models = settings._models_from(s.llm_models) or [settings._FALLBACK_MODEL]
    return {
        "llm": {
            "llm_models": default_models,
            "llm_followup_model": s.llm_followup_model.strip() or default_models[0],
            "llm_base_url": s.llm_base_url,
            "chat_temperature": None,
            "chat_timeout": settings._DEFAULT_CHAT_TIMEOUT_S,
            "followup_timeout": settings._DEFAULT_FOLLOWUP_TIMEOUT_S,
        },
        "rag_kg": {
            "rag_top_k": settings._DEFAULT_RAG_TOP_K,
            "kg_hops": settings._DEFAULT_KG_HOPS,
            "kg_seeds": settings._DEFAULT_KG_SEEDS,
        },
        "analyze": {
            "allowed_upload_suffixes": list(settings._DEFAULT_UPLOAD_SUFFIXES),
            "max_concurrent_analyses": config.MAX_CONCURRENT_ANALYSES,
        },
    }


def _settings_payload() -> dict[str, Any]:
    """The GET/PUT response shape: effective values + defaults (secrets never appear here)."""
    return {"effective": _effective_settings(), "defaults": _default_settings()}


@router.get("/settings")
def get_admin_settings(user: CurrentUser = Depends(get_admin_user)) -> dict:
    """Return the effective runtime knobs merged with their env/constant defaults (admin-only)."""
    return _settings_payload()


class AdminSettingsUpdate(BaseModel):
    """A partial update — every knob is optional; only the provided ones are persisted.

    Ranges are validated here so an out-of-band value is rejected (422) before it reaches the DB. The
    field names match the ``app_settings`` override keys exactly, so ``model_dump(exclude_unset=True)``
    yields the upsert payload directly. No secret field exists on this model by design.
    """

    llm_models: list[str] | None = None
    llm_followup_model: str | None = None
    llm_base_url: str | None = None
    chat_temperature: float | None = Field(default=None, ge=0, le=2)
    chat_timeout: float | None = Field(default=None, gt=0, le=300)
    followup_timeout: float | None = Field(default=None, gt=0, le=300)
    rag_top_k: int | None = Field(default=None, ge=1, le=50)
    kg_hops: int | None = Field(default=None, ge=1, le=3)
    kg_seeds: int | None = Field(default=None, ge=1, le=20)
    # NOTE: ``max_concurrent_analyses`` is intentionally NOT a field here — the analyze semaphore is
    # fixed at import from the env var and an override never applied, so the PUT no longer accepts it.
    # It remains a READ-ONLY, env-sourced display value in the GET payload (see ``_effective_settings``).
    allowed_upload_suffixes: list[str] | None = None

    @field_validator("llm_models")
    @classmethod
    def _models_non_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        cleaned = [m.strip() for m in v if m and m.strip()]
        if not cleaned:
            raise ValueError("llm_models must contain at least one non-empty model id")
        return cleaned

    @field_validator("llm_base_url")
    @classmethod
    def _base_url_is_http(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("llm_base_url must be an http(s) URL")
        # Early 422 on the API path for an off-allowlist host — the read-time guard in
        # ``settings.chat_base_url`` is the authoritative backstop, but rejecting here gives the admin
        # a clear error instead of a silently-ignored override. The LLM API key is sent as a bearer
        # token to this host, so only allowlisted provider hosts may be set.
        if not settings._base_url_allowed(v):
            raise ValueError(
                "llm_base_url host is not allowlisted; set LLM_ALLOWED_BASE_HOSTS to permit it"
            )
        return v

    @field_validator("allowed_upload_suffixes")
    @classmethod
    def _suffixes_well_formed(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        cleaned = [s.strip().lower() for s in v if s and s.strip()]
        if not cleaned or not all(s.startswith(".") for s in cleaned):
            raise ValueError("allowed_upload_suffixes must be non-empty '.ext' strings")
        return cleaned


@router.put("/settings")
def put_admin_settings(
    body: AdminSettingsUpdate,
    user: CurrentUser = Depends(get_admin_user),
) -> dict:
    """Validate + upsert the provided knobs, invalidate the resolution cache, and return the new state.

    Only the fields the client actually sent are written (``exclude_unset``); the rest keep their prior
    value or env default. The write goes through the user's JWT, so the ``is_admin`` RLS policy is the
    Postgres-side backstop even though ``get_admin_user`` already gated the request.
    """
    items = body.model_dump(exclude_unset=True)
    store.upsert_app_settings(token=user.token, items=items)
    runtime_config.clear_cache()  # so the next getter read reflects the new overrides immediately.
    return _settings_payload()


# ---------------------------------------------------------------------------------------------------
# P3 — read-only user oversight, in-app role assignment, and the system-status dashboard.
# ---------------------------------------------------------------------------------------------------


@router.get("/users")
def list_admin_users(user: CurrentUser = Depends(get_admin_user)) -> dict:
    """Return every user with lightweight activity counts (admin-only; read via the definer function)."""
    return {"users": store.admin_list_users(token=user.token)}


class RoleUpdate(BaseModel):
    """Body for the role toggle: ``make_admin`` True grants the admin role, False revokes it."""

    make_admin: bool


@router.put("/users/{user_id}/role")
def set_admin_role(
    user_id: str,
    body: RoleUpdate,
    user: CurrentUser = Depends(get_admin_user),
) -> dict:
    """Grant/revoke another user's admin role. An admin may NOT revoke their OWN role (anti-lockout).

    The self-demote guard is a UX/safety backstop against locking the last admin out of the panel; the
    write itself is still RLS-gated on ``is_admin(auth.uid())`` in Postgres.
    """
    if user_id == user.id and not body.make_admin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot remove your own admin role.",
        )
    if not body.make_admin and store.count_admins(token=user.token) <= 1:
        # Anti-lockout: refuse a revoke that would leave the project with zero admins. This closes the
        # UI-driven path; a truly concurrent double-demote is a residual race acceptable at this scale.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove the last admin.",
        )
    store.set_user_role(token=user.token, user_id=user_id, make_admin=body.make_admin)
    return {"ok": True}


@router.get("/overview")
def admin_overview(user: CurrentUser = Depends(get_admin_user)) -> dict:
    """System-status dashboard: the same health flags as ``GET /api/health`` plus user/analysis totals.

    The health portion is replicated here (rather than imported from ``main``, which would be a circular
    import) — it is a small, secret-free dict. Totals are aggregated from the same admin overview the
    ``/users`` endpoint serves. No secret (API key / Supabase creds) is ever included.
    """
    s = settings.get_settings()
    users = store.admin_list_users(token=user.token)
    total_analyses = sum(int(u.get("analyses_count") or 0) for u in users)
    return {
        "auth_configured": s.auth_configured,
        "chat_configured": s.chat_configured,
        "chat_models": settings.chat_models(),
        "chat_default": settings.default_chat_model(),
        "stores": {
            "labeled_videos": config.VIDEOS_DIR.exists(),
            "detections": config.DETECTIONS_DIR.exists(),
            "kg_graph": config.KG_GRAPH_FILE.exists(),
            "rag_db": config.RAG_DB_DIR.exists(),
        },
        "total_users": len(users),
        "total_analyses": total_analyses,
    }


@router.get("/line/status")
def admin_line_status(user: CurrentUser = Depends(get_admin_user)) -> dict:
    """LINE connection status + this month's push-message quota (admin-only; read-only).

    Never returns a secret: the channel access token is used server-side (in ``line_quota``) to
    read LINE's quota endpoints, and the channel secret / service_role key are never touched.
    ``channel_id`` is the non-secret LINE Login channel id, surfaced only so an admin can confirm
    which channel is wired. When messaging isn't configured we skip the LINE call entirely; when it
    is configured but the read fails, ``quota`` is ``None`` and ``quota_error`` flags it.
    """
    s = settings.get_settings()
    quota = line_quota.fetch_quota() if s.line_messaging_configured else None
    return {
        "messaging_configured": s.line_messaging_configured,
        "login_configured": s.line_login_configured,
        "channel_id": s.line_channel_id,
        "quota": quota,
        "quota_error": "unreachable" if (s.line_messaging_configured and quota is None) else None,
    }
