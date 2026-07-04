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
    # The model used by default (the "server default"). A self-hoster sets this to ANY OpenRouter
    # model and it just works — it is always offered and honoured, never restricted.
    openrouter_model: str = "deepseek/deepseek-v4-flash"
    # The comma-separated list of models a hosted user may pick in Settings. Env-configurable, so a
    # self-hoster changes the whole picker without editing code. The default above is always added.
    openrouter_models: str = (
        "deepseek/deepseek-v4-flash,xiaomi/mimo-v2.5,minimax/minimax-m3,tencent/hy3-preview"
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @property
    def auth_configured(self) -> bool:
        """True when the Supabase project URL and anon key are both present."""
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def chat_configured(self) -> bool:
        """True when an OpenRouter API key is present (the chat endpoint is otherwise 503)."""
        return bool(self.openrouter_api_key)


# Display labels for the curated slugs. Any other (self-hoster) slug is shown as its raw id — the
# backend is the single source of truth for the picker, so the frontend never hard-codes this.
_MODEL_LABELS: dict[str, str] = {
    "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
    "xiaomi/mimo-v2.5": "MiMo V2.5",
    "minimax/minimax-m3": "MiniMax M3",
    "tencent/hy3-preview": "Hy3 Preview",
}


def chat_models() -> list[dict[str, str]]:
    """The selectable chat models as ``[{"id", "label"}]``, parsed from ``OPENROUTER_MODELS``.

    The configured default (``OPENROUTER_MODEL``) is always included first, so a self-hoster who
    sets only that env var still gets a working, offered model. Order is preserved; ids are deduped.
    """
    s = get_settings()
    raw = [m.strip() for m in s.openrouter_models.split(",") if m.strip()]
    ordered = raw if s.openrouter_model in raw else [s.openrouter_model, *raw]
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for slug in ordered:
        if slug and slug not in seen:
            seen.add(slug)
            out.append({"id": slug, "label": _MODEL_LABELS.get(slug, slug)})
    return out


def default_chat_model() -> str:
    """The model used when the client sends none — the configured ``OPENROUTER_MODEL``."""
    return get_settings().openrouter_model


def resolve_chat_model(requested: str | None) -> str:
    """Honour the client's ``requested`` model only if it's one of the offered models, else default.

    The offered set always contains the configured default, so the browser can't name an arbitrary
    (possibly far more expensive) model, while the operator's ``OPENROUTER_MODEL`` always works.
    """
    if requested and requested in {m["id"] for m in chat_models()}:
        return requested
    return default_chat_model()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
