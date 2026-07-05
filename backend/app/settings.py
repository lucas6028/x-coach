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
    # The comma-separated list of models a user may pick in Settings; the FIRST is the default (used
    # when the client picks nothing). Env-configurable, so a self-hoster changes the whole picker —
    # and the default — without editing code. Set it to a single custom slug to run any model.
    openrouter_models: str = (
        "deepseek/deepseek-v4-flash,xiaomi/mimo-v2.5,minimax/minimax-m3,tencent/hy3-preview"
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Follow-up chips are a separate, latency-sensitive call (see services/chat.suggest_followups):
    # pinned to a fast model independent of the answer model, so a slow/reasoning answer model doesn't
    # make the chips crawl. Env-overridable; blank it to reuse the default answer model instead.
    openrouter_followup_model: str = "openai/gpt-oss-120b"

    @property
    def auth_configured(self) -> bool:
        """True when the Supabase project URL and anon key are both present."""
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def chat_configured(self) -> bool:
        """True when an OpenRouter API key is present (the chat endpoint is otherwise 503)."""
        return bool(self.openrouter_api_key)


# Used only if ``OPENROUTER_MODELS`` is misconfigured to empty, so the picker is never empty.
_FALLBACK_MODEL = "deepseek/deepseek-v4-flash"


def chat_models() -> list[str]:
    """The selectable model ids, parsed from ``OPENROUTER_MODELS`` (first = default).

    Order is preserved and ids are deduped; a blank/empty setting falls back to a single built-in
    model so the picker is never empty. Display names are a frontend concern — this is purely the
    authoritative id list.
    """
    raw = [m.strip() for m in get_settings().openrouter_models.split(",") if m.strip()]
    return list(dict.fromkeys(raw)) or [_FALLBACK_MODEL]


def default_chat_model() -> str:
    """The model used when the client sends none — the first entry of ``OPENROUTER_MODELS``."""
    return chat_models()[0]


def followup_chat_model() -> str:
    """The model for follow-up suggestions — a fast one pinned server-side, independent of the answer
    model the user picked. Falls back to the default answer model if ``OPENROUTER_FOLLOWUP_MODEL`` is
    blanked (a self-hoster whose account lacks the pinned model). Not client-selectable by design."""
    return get_settings().openrouter_followup_model.strip() or default_chat_model()


def resolve_chat_model(requested: str | None) -> str:
    """Honour the client's ``requested`` model only if it's one of the offered models, else default.

    So the browser can't name an arbitrary (possibly far more expensive) model, while the operator
    controls both the picker and the default via ``OPENROUTER_MODELS`` (first = default).
    """
    models = chat_models()
    if requested and requested in set(models):
        return requested
    return models[0]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
