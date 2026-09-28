"""Supabase job queue -- ``Db`` for ``loop.run_one``.

The worker signs in as its own Supabase user (``nlf_worker`` role) with the ANON key and a
password, never the service_role key (plan decision 3), and touches jobs only through the
``nlf_*`` SECURITY DEFINER RPCs in ``db/migrations/20260928000000_nlf_3d_view.sql``.

Access tokens expire after about an hour and a home PC sleeps, so before every RPC the session
is refreshed when it is within ``REFRESH_MARGIN_S`` of expiry (falling back to a fresh password
sign-in if the refresh fails), and an RPC rejected for an expired/invalid JWT is retried once
after a new sign-in. A 42501 (the account lacks the worker role) is NOT an auth error and
propagates: signing in again cannot fix it.

``supabase`` is imported lazily, so tests can inject a fake client without it installed.
Credentials are never logged or put into exception messages.
"""
from __future__ import annotations

import time
from typing import Any, Callable

REFRESH_MARGIN_S = 300

# PostgREST's JWT errors (invalid / missing / claims-or-expiry) and a bare HTTP 401.
_AUTH_ERROR_CODES = frozenset({"PGRST301", "PGRST302", "PGRST303", "401"})


def is_auth_error(exc: BaseException) -> bool:
    """Whether ``exc`` means the request's JWT was rejected (so a new sign-in may fix it)."""
    code = str(getattr(exc, "code", "") or "")
    if code in _AUTH_ERROR_CODES:
        return True
    if getattr(exc, "status", None) == 401:
        return True
    message = str(getattr(exc, "message", "") or "").lower()
    return "jwt" in message


def _rpc_bool(data: Any) -> bool:
    """``True`` only for an unambiguous true result: ``True`` or ``[True]``. Anything else is
    False -- but callers treat False as "the job is gone, delete the upload", so the shapes
    PostgREST actually returns for a boolean function are matched exactly."""
    if data is True:
        return True
    return isinstance(data, list) and len(data) == 1 and data[0] is True


class SupabaseDb:
    def __init__(
        self,
        url: str,
        anon_key: str,
        email: str,
        password: str,
        *,
        client: Any = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._url = url
        self._anon_key = anon_key
        self._email = email
        self._password = password
        self._client = client
        self._clock = clock
        self._session: Any = None

    def __repr__(self) -> str:
        return f"SupabaseDb(url={self._url!r})"

    def _get_client(self) -> Any:
        if self._client is None:
            from supabase import create_client

            self._client = create_client(self._url, self._anon_key)
        return self._client

    def _sign_in(self) -> None:
        resp = self._get_client().auth.sign_in_with_password(
            {"email": self._email, "password": self._password}
        )
        session = getattr(resp, "session", None)
        if session is None:
            raise RuntimeError("worker sign-in returned no session")
        self._session = session

    def _ensure_session(self) -> None:
        if self._session is None:
            self._sign_in()
            return
        expires_at = getattr(self._session, "expires_at", None)
        if expires_at is not None and expires_at - self._clock() > REFRESH_MARGIN_S:
            return
        try:
            session = getattr(self._get_client().auth.refresh_session(), "session", None)
        except Exception:  # noqa: BLE001 -- any refresh failure falls back to a password sign-in
            session = None
        if session is None:
            self._sign_in()
        else:
            self._session = session

    def _rpc(self, fn: str, params: dict[str, Any] | None = None) -> Any:
        self._ensure_session()
        try:
            return self._get_client().rpc(fn, params or {}).execute()
        except Exception as exc:
            if not is_auth_error(exc):
                raise
        # The first call was rejected at authentication, so it changed nothing; safe to repeat.
        self._sign_in()
        return self._get_client().rpc(fn, params or {}).execute()

    # --- Db protocol -------------------------------------------------------------------
    def heartbeat(self) -> None:
        self._rpc("nlf_worker_heartbeat")

    def claim(self) -> dict[str, Any] | None:
        data = self._rpc("nlf_claim_job").data
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict) or data.get("job_id") is None:
            return None
        return dict(data)

    def complete(self, job_id: str, result_key: str, meta: dict[str, Any]) -> bool:
        resp = self._rpc(
            "nlf_complete_job", {"p_job_id": job_id, "p_result_key": result_key, "p_meta": meta}
        )
        return _rpc_bool(resp.data)

    def fail(self, job_id: str, error: str) -> None:
        self._rpc("nlf_fail_job", {"p_job_id": job_id, "p_error": error})
