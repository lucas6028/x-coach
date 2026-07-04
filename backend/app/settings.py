"""Environment-driven settings for the x-coach backend (secrets, Supabase config).

Kept separate from ``config.py`` (which holds repo-root *paths*) so the path layer stays
import-light and free of secrets. Values come from environment variables or a ``.env`` file
at the repo root (gitignored). Nothing here is committed.

Required for auth + persistence (P1):
    SUPABASE_URL          https://<project>.supabase.co
    SUPABASE_ANON_KEY     anon public key (client acts as the user; RLS enforced)

Access tokens are validated through the Supabase Auth API (see ``auth._verify``), so no JWT
signing secret is needed here — that works whether the project signs tokens with the legacy
HS256 secret or the newer asymmetric keys. If these vars are unset the server still runs:
auth-gated endpoints return 503 and ``/api/analyze`` falls back to anonymous "demo" mode
(analysis works, nothing is persisted).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.app import config


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(config.REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    supabase_url: str = ""
    supabase_anon_key: str = ""

    # LLM conversational-coaching layer, served via OpenRouter (OpenAI-compatible API).
    # The key is server-side only — it is never sent to the browser. Model ids use OpenRouter's
    # ``vendor/model`` namespace; the default is a cost-effective, capable general model that can
    # be overridden per-deployment without a code change.
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-v4-flash"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @property
    def auth_configured(self) -> bool:
        """True when the Supabase project URL and anon key are both present."""
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def chat_configured(self) -> bool:
        """True when an OpenRouter API key is present (the chat endpoint is otherwise 503)."""
        return bool(self.openrouter_api_key)


# Chat models a deployed user may pick from in Settings (OpenRouter ``vendor/model`` slugs). Kept in
# sync with ``frontend/src/lib/model.ts``. This is the security boundary for the *hosted* app: the
# client sends a chosen model per request and the backend only honours it if it's on this list — the
# browser must never be able to name an arbitrary (possibly far more expensive) model.
#
# It does NOT constrain the operator: whoever runs the server sets ``OPENROUTER_MODEL`` to ANY model
# they want, and that is always honoured (it's the default and is explicitly allowed below). A
# self-hoster who clones the repo therefore runs any model by setting one env var — no need to edit
# this list — while hosted users stay limited to the curated four.
ALLOWED_CHAT_MODELS: frozenset[str] = frozenset(
    {
        "deepseek/deepseek-v4-flash",
        "xiaomi/mimo-v2.5",
        "minimax/minimax-m3",
        "tencent/hy3-preview",
    }
)


def resolve_chat_model(requested: str | None) -> str:
    """Resolve the model to call OpenRouter with.

    ``requested`` (the client's per-chat pick) is honoured only if it's one of the curated
    ``ALLOWED_CHAT_MODELS`` or it equals the operator-configured default — so the browser can't name
    an arbitrary model, but the operator's ``OPENROUTER_MODEL`` (any model) always works. Everything
    else, including a missing request (the "server default" choice), uses the configured default.
    """
    default = get_settings().openrouter_model
    if requested and (requested in ALLOWED_CHAT_MODELS or requested == default):
        return requested
    return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
