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

    # LLM conversational-coaching layer, served over an OpenAI-compatible chat-completions API. The
    # provider is deliberately NOT baked into the names: the transport speaks only the plain OpenAI
    # dialect, so these ``LLM_*`` vars point at any compatible endpoint — OpenRouter (the default
    # base URL) or NVIDIA NIM at ``integrate.api.nvidia.com/v1`` — by swapping base URL + key + model
    # ids, no code change (provider-specific request extras are gated on the base URL in
    # ``services/chat``). The key is server-side only — never sent to the browser. Model ids use the
    # provider's ``vendor/model`` namespace; the default is a cost-effective, capable general model,
    # overridable per-deployment.
    llm_api_key: str = ""
    # The comma-separated list of models a user may pick in Settings; the FIRST is the default (used
    # when the client picks nothing). Env-configurable, so a self-hoster changes the whole picker —
    # and the default — without editing code. Set it to a single custom slug to run any model.
    llm_models: str = (
        "deepseek/deepseek-v4-flash,xiaomi/mimo-v2.5,minimax/minimax-m3,tencent/hy3-preview"
    )
    llm_base_url: str = "https://openrouter.ai/api/v1"
    # Follow-up chips are a separate, latency-sensitive call (see services/chat.suggest_followups):
    # pinned to a fast model independent of the answer model, so a slow/reasoning answer model doesn't
    # make the chips crawl. Env-overridable; blank it to reuse the default answer model instead.
    llm_followup_model: str = "openai/gpt-oss-120b"

    @property
    def auth_configured(self) -> bool:
        """True when the Supabase project URL and anon key are both present."""
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def chat_configured(self) -> bool:
        """True when an LLM API key is present (the chat endpoint is otherwise 503)."""
        return bool(self.llm_api_key)


# Used only if ``LLM_MODELS`` is misconfigured to empty, so the picker is never empty.
_FALLBACK_MODEL = "deepseek/deepseek-v4-flash"

# Constant fallbacks for the runtime-tunable knobs. These preserve today's behaviour exactly when
# no admin override is present (the ``chat.py`` timeout constants and the analyze router's suffix set
# are duplicated here as the authoritative defaults for the override-first getters below).
_DEFAULT_CHAT_TIMEOUT_S = 60.0
_DEFAULT_FOLLOWUP_TIMEOUT_S = 15.0
_DEFAULT_RAG_TOP_K = 5
_DEFAULT_KG_HOPS = 1
_DEFAULT_KG_SEEDS = 5
_DEFAULT_UPLOAD_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _overrides() -> dict:
    """The admin runtime overrides (``{}`` when unconfigured / on error). Imported lazily to avoid an
    import cycle: ``runtime_config`` imports this module's ``get_settings``."""
    from backend.app.services import runtime_config

    return runtime_config.get_overrides()


def _models_from(value: object) -> list[str]:
    """Normalise a models override (a comma-string OR a list) into a deduped, ordered id list."""
    if isinstance(value, str):
        raw = [m.strip() for m in value.split(",") if m.strip()]
    elif isinstance(value, (list, tuple)):
        raw = [str(m).strip() for m in value if str(m).strip()]
    else:
        raw = []
    return list(dict.fromkeys(raw))


def chat_models() -> list[str]:
    """The selectable model ids (first = default): admin override ``llm_models`` first, else ``LLM_MODELS``.

    The override may be a comma-string or a JSON list. Order is preserved and ids are deduped; a
    blank/empty result falls back to a single built-in model so the picker is never empty. Display
    names are a frontend concern — this is purely the authoritative id list.
    """
    models = _models_from(_overrides().get("llm_models"))
    if not models:
        models = _models_from(get_settings().llm_models)
    return models or [_FALLBACK_MODEL]


def default_chat_model() -> str:
    """The model used when the client sends none — the first entry of the effective model list."""
    return chat_models()[0]


def followup_chat_model() -> str:
    """The model for follow-up suggestions — a fast one pinned server-side, independent of the answer
    model the user picked. Admin override ``llm_followup_model`` first, else ``LLM_FOLLOWUP_MODEL``;
    a blank value falls back to the default answer model. Not client-selectable by design."""
    override = _overrides().get("llm_followup_model")
    pinned = str(override).strip() if override is not None else get_settings().llm_followup_model.strip()
    return pinned or default_chat_model()


def chat_base_url() -> str:
    """The LLM provider base URL — admin override ``llm_base_url`` first, else ``LLM_BASE_URL``."""
    override = _overrides().get("llm_base_url")
    if isinstance(override, str) and override.strip():
        return override.strip()
    return get_settings().llm_base_url


def chat_temperature() -> float | None:
    """The sampling temperature to send, or ``None`` to omit it (the current default behaviour).

    Override key ``chat_temperature``. Coerced to float; a missing/blank/uncoercible override yields
    ``None`` so the completion body carries no ``temperature`` field (unchanged from today).
    """
    override = _overrides().get("chat_temperature")
    if override is None or override == "":
        return None
    try:
        return float(override)
    except (TypeError, ValueError):
        return None


def chat_timeout() -> float:
    """The answer-call round-trip budget in seconds — override ``chat_timeout``, else 60.0."""
    return _coerce_float(_overrides().get("chat_timeout"), _DEFAULT_CHAT_TIMEOUT_S)


def followup_timeout() -> float:
    """The follow-up-call round-trip budget in seconds — override ``followup_timeout``, else 15.0."""
    return _coerce_float(_overrides().get("followup_timeout"), _DEFAULT_FOLLOWUP_TIMEOUT_S)


def rag_top_k_default() -> int:
    """The default RAG result count — override ``rag_top_k``, else 5 (endpoint query params still win)."""
    return _coerce_int(_overrides().get("rag_top_k"), _DEFAULT_RAG_TOP_K)


def kg_hops_default() -> int:
    """The default KG traversal depth — override ``kg_hops``, else 1 (endpoint query params still win)."""
    return _coerce_int(_overrides().get("kg_hops"), _DEFAULT_KG_HOPS)


def kg_seeds_default() -> int:
    """The default KG seed-node count — override ``kg_seeds``, else 5."""
    return _coerce_int(_overrides().get("kg_seeds"), _DEFAULT_KG_SEEDS)


def allowed_upload_suffixes() -> tuple[str, ...]:
    """The accepted upload file suffixes — override ``allowed_upload_suffixes`` (a list), else the
    built-in set. A non-list / empty / malformed override falls back to the default."""
    override = _overrides().get("allowed_upload_suffixes")
    if isinstance(override, (list, tuple)):
        cleaned = tuple(
            str(s).strip().lower()
            for s in override
            if str(s).strip().startswith(".")
        )
        if cleaned:
            return cleaned
    return _DEFAULT_UPLOAD_SUFFIXES


def _coerce_float(value: object, default: float) -> float:
    """Best-effort float coercion; returns ``default`` on a missing/blank/uncoercible value."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int) -> int:
    """Best-effort int coercion; returns ``default`` on a missing/blank/uncoercible value."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_chat_model(requested: str | None) -> str:
    """Honour the client's ``requested`` model only if it's one of the offered models, else default.

    So the browser can't name an arbitrary (possibly far more expensive) model, while the operator
    controls both the picker and the default via ``LLM_MODELS`` (first = default).
    """
    models = chat_models()
    if requested and requested in set(models):
        return requested
    return models[0]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
