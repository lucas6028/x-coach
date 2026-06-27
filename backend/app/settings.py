"""Environment-driven settings for the x-coach backend (secrets, Supabase config).

Kept separate from ``config.py`` (which holds repo-root *paths*) so the path layer stays
import-light and free of secrets. Values come from environment variables or a ``.env`` file
at the repo root (gitignored). Nothing here is committed.

Required for auth + persistence (P1):
    SUPABASE_URL          https://<project>.supabase.co
    SUPABASE_ANON_KEY     anon public key (client acts as the user; RLS enforced)
    SUPABASE_JWT_SECRET   the project's JWT secret (HS256) — used to verify access tokens

If these are unset the server still runs: auth-gated endpoints return 503 and ``/api/analyze``
falls back to anonymous "demo" mode (analysis works, nothing is persisted).
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
    supabase_jwt_secret: str = ""

    # Supabase signs access tokens with HS256 and aud="authenticated".
    jwt_algorithm: str = "HS256"
    jwt_audience: str = "authenticated"

    @property
    def auth_configured(self) -> bool:
        """True only when all three Supabase secrets are present."""
        return bool(self.supabase_url and self.supabase_anon_key and self.supabase_jwt_secret)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
