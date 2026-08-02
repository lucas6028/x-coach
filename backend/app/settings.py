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
from urllib.parse import urlparse

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

    # LINE Login bridge (LIFF): POST /api/auth/line verifies a LINE ID token against the
    # channel named here, then mints a Supabase session via the Admin API. The service_role
    # key exists ONLY for that mint (create user + generate one-shot link) — it is never
    # used for data access, which stays on the user's own JWT with RLS as the backstop.
    # Leave both unset to keep the endpoint disabled (503); web LINE login via Supabase's
    # custom OIDC provider works without either. See services/line_auth.
    line_channel_id: str = ""
    supabase_service_role_key: str = ""

    # Cloudflare R2 object storage for user uploads (raw video, pose JSON, thumbnail), reached
    # over the S3-compatible API. Leave any of these blank and the backend transparently uses the
    # local-filesystem store instead (backend/app/services/storage.py) — which is what CI and
    # offline development run on, so no credentials are needed to work on this codebase.
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""

    # LINE Messaging API bot (the official account chat room). A SEPARATE channel from the
    # Login channel above — its own secret (webhook signature) and access token (reply API).
    # Both channels must live under the SAME LINE provider: that is what makes the webhook's
    # source.userId identical to the Login ID token's ``sub`` (which line_auth stores in
    # user_metadata.line_sub), so the bot can find the account with no binding flow.
    # Leave unset to keep POST /api/line/webhook disabled (503). See services/line_bot.
    line_messaging_channel_secret: str = ""
    line_messaging_access_token: str = ""
    # LIFF app id, used only to build the "open x-coach" deep link in bot replies. Optional:
    # when blank the link line is omitted.
    line_liff_id: str = ""

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
    # Extra base-URL hosts an admin may point the LLM transport at, beyond the env-default host and the
    # built-in provider set (see ``_allowed_base_hosts``). Comma-separated hostnames. This is a
    # security boundary: the ``Authorization: Bearer <llm_api_key>`` header is only ever sent to an
    # allowlisted host, so an admin override (or a direct-DB write) can't exfiltrate the key to an
    # arbitrary endpoint or SSRF an internal one.
    llm_allowed_base_hosts: str = ""
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

    @property
    def storage_configured(self) -> bool:
        """True when R2 is fully configured; otherwise the local filesystem store is used."""
        return bool(
            self.r2_account_id
            and self.r2_access_key_id
            and self.r2_secret_access_key
            and self.r2_bucket
        )

    @property
    def line_login_configured(self) -> bool:
        """True when the LIFF→Supabase bridge can run: LINE channel id + service_role key,
        on top of the base Supabase config (the mint needs both anon and admin clients)."""
        return bool(self.line_channel_id and self.supabase_service_role_key and self.auth_configured)

    @property
    def line_messaging_configured(self) -> bool:
        """True when the bot can verify webhook signatures, reply, and read the summary.

        The service_role key is required because the webhook has no user JWT: it reads the
        summary through the ``line_training_summary`` SECURITY DEFINER function, which is
        granted to service_role only.
        """
        return bool(
            self.line_messaging_channel_secret
            and self.line_messaging_access_token
            and self.supabase_service_role_key
            and self.auth_configured
        )


# Used only if ``LLM_MODELS`` is misconfigured to empty, so the picker is never empty.
_FALLBACK_MODEL = "deepseek/deepseek-v4-flash"

# Provider hosts the LLM transport is always allowed to reach, in addition to the env-default base
# URL's host and any ``LLM_ALLOWED_BASE_HOSTS`` entry. The chat call sends the API key in an
# ``Authorization`` header, so the destination host is a security boundary — see ``_base_url_allowed``.
_BUILTIN_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {"openrouter.ai", "api.openai.com", "integrate.api.nvidia.com"}
)


def _allowed_base_hosts() -> set[str]:
    """The set of hostnames the LLM base URL may point at (lower-cased).

    The env-default ``llm_base_url`` host is always allowed (so the shipped default works), plus the
    built-in provider set, plus any comma-separated hostnames in ``LLM_ALLOWED_BASE_HOSTS``.
    """
    s = get_settings()
    hosts = {h.lower() for h in _BUILTIN_ALLOWED_HOSTS}
    # ``getattr`` defaults keep this robust when a test patches ``get_settings`` to a lightweight
    # stand-in that lacks these fields (several existing unit tests do exactly that).
    default_host = urlparse(getattr(s, "llm_base_url", "") or "").hostname
    if default_host:
        hosts.add(default_host.lower())
    for extra in (getattr(s, "llm_allowed_base_hosts", "") or "").split(","):
        extra = extra.strip().lower()
        if extra:
            hosts.add(extra)
    return hosts


def _base_url_allowed(url: object) -> bool:
    """True when ``url`` is an http(s) URL whose host is on the allowlist (``_allowed_base_hosts``).

    The authoritative guard for where the ``Authorization: Bearer <llm_api_key>`` request may go:
    an off-list host (an arbitrary external endpoint, ``localhost``, a cloud metadata IP, …) is
    rejected so the key can't be exfiltrated and internal endpoints can't be SSRF'd.
    """
    if not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    return host.lower() in _allowed_base_hosts()

# Constant fallbacks for the runtime-tunable knobs. These preserve today's behaviour exactly when
# no admin override is present (the ``chat.py`` timeout constants and the analyze router's suffix set
# are duplicated here as the authoritative defaults for the override-first getters below).
_DEFAULT_CHAT_TIMEOUT_S = 60.0
_DEFAULT_FOLLOWUP_TIMEOUT_S = 15.0
# Clamp bounds for the timeout getters: a request budget must be positive and is capped so an
# out-of-band write can't leave a request hanging indefinitely. Mirrors the PUT validator (``gt=0,
# le=300``); the positive floor is a small sane minimum since a sub-second budget is never useful.
_MIN_TIMEOUT_S = 1.0
_MAX_TIMEOUT_S = 300.0
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
    """The LLM provider base URL — admin override ``llm_base_url`` first, else ``LLM_BASE_URL``.

    The override is honoured ONLY when its host is allowlisted (``_base_url_allowed``); an off-list
    override falls back to the env default. This is the authoritative read-time guard — it protects
    the ``Authorization: Bearer <llm_api_key>`` request even against a direct-DB write that bypasses
    the PUT validator.
    """
    override = _overrides().get("llm_base_url")
    if isinstance(override, str) and override.strip() and _base_url_allowed(override.strip()):
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
        value = float(override)
    except (TypeError, ValueError):
        return None
    # Clamp into the same 0..2 range the PUT validator enforces, so a direct-DB write can't drive an
    # out-of-range sampling temperature into the completion body.
    return min(max(value, 0.0), 2.0)


def chat_timeout() -> float:
    """The answer-call round-trip budget in seconds — override ``chat_timeout``, else 60.0 (clamped 1..300)."""
    return _coerce_float(
        _overrides().get("chat_timeout"),
        _DEFAULT_CHAT_TIMEOUT_S,
        minimum=_MIN_TIMEOUT_S,
        maximum=_MAX_TIMEOUT_S,
    )


def followup_timeout() -> float:
    """The follow-up-call round-trip budget in seconds — override ``followup_timeout``, else 15.0 (clamped 1..300)."""
    return _coerce_float(
        _overrides().get("followup_timeout"),
        _DEFAULT_FOLLOWUP_TIMEOUT_S,
        minimum=_MIN_TIMEOUT_S,
        maximum=_MAX_TIMEOUT_S,
    )


def rag_top_k_default() -> int:
    """The default RAG result count — override ``rag_top_k``, else 5 (clamped 1..50; query params still win)."""
    return _coerce_int(_overrides().get("rag_top_k"), _DEFAULT_RAG_TOP_K, minimum=1, maximum=50)


def kg_hops_default() -> int:
    """The default KG traversal depth — override ``kg_hops``, else 1 (clamped 1..3; query params still win)."""
    return _coerce_int(_overrides().get("kg_hops"), _DEFAULT_KG_HOPS, minimum=1, maximum=3)


def kg_seeds_default() -> int:
    """The default KG seed-node count — override ``kg_seeds``, else 5 (clamped 1..20)."""
    return _coerce_int(_overrides().get("kg_seeds"), _DEFAULT_KG_SEEDS, minimum=1, maximum=20)


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


def _coerce_float(
    value: object,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Best-effort float coercion; returns ``default`` on a missing/blank/uncoercible value.

    When ``minimum``/``maximum`` are given, the coerced result is CLAMPED into that range (not
    rejected) so an out-of-band / direct-DB write can't drive an out-of-range value downstream.
    """
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and result < minimum:
        result = minimum
    if maximum is not None and result > maximum:
        result = maximum
    return result


def _coerce_int(
    value: object,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Best-effort int coercion; returns ``default`` on a missing/blank/uncoercible value.

    When ``minimum``/``maximum`` are given, the coerced result is CLAMPED into that range (not
    rejected) so an out-of-band / direct-DB write can't drive an out-of-range value downstream.
    """
    if value is None or value == "":
        return default
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and result < minimum:
        result = minimum
    if maximum is not None and result > maximum:
        result = maximum
    return result


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
