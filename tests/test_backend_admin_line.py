"""Unit tests for services/line_quota + GET /api/admin/line/status.

Mirrors tests/test_backend_line_webhook.py: unittest.TestCase, external HTTP (LINE's quota
endpoints) mocked at the httpx.get seam, get_settings patched to a lightweight stand-in, and
the FastAPI route exercised via TestClient with dependency_overrides + store.is_admin patched.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from backend.app.auth import CurrentUser, get_current_user
from backend.app.main import app
from backend.app.services import line_quota, store
from backend.app.settings import Settings


class _FakeResp:
    """A stand-in httpx.Response: .raise_for_status() optionally raises, .json() returns payload."""

    def __init__(self, payload, *, ok: bool = True) -> None:
        self._payload = payload
        self._ok = ok

    def raise_for_status(self) -> None:
        if not self._ok:
            raise httpx.HTTPStatusError("bad", request=mock.Mock(), response=mock.Mock())

    def json(self):
        return self._payload


def _stub_settings(token: str = "chan-token") -> types.SimpleNamespace:
    return types.SimpleNamespace(line_messaging_access_token=token)


class LineQuotaFetchTests(unittest.TestCase):
    def setUp(self) -> None:
        line_quota.clear_cache()
        self.addCleanup(line_quota.clear_cache)

    def _run(self, quota_payload, consumption_payload, *, token="chan-token"):
        responses = [_FakeResp(quota_payload), _FakeResp(consumption_payload)]
        with mock.patch.object(line_quota, "get_settings", return_value=_stub_settings(token)), \
             mock.patch.object(line_quota.httpx, "get", side_effect=responses) as g:
            return line_quota.fetch_quota(), g

    def test_limited_computes_remaining(self) -> None:
        result, _ = self._run({"type": "limited", "value": 200}, {"totalUsage": 12})
        self.assertEqual(result, {"type": "limited", "used": 12, "value": 200, "remaining": 188})

    def test_none_type_omits_value_and_remaining(self) -> None:
        result, _ = self._run({"type": "none"}, {"totalUsage": 5})
        self.assertEqual(result, {"type": "none", "used": 5})

    def test_remaining_never_negative(self) -> None:
        result, _ = self._run({"type": "limited", "value": 100}, {"totalUsage": 150})
        self.assertEqual(result["remaining"], 0)

    def test_missing_token_returns_none_without_calling_line(self) -> None:
        with mock.patch.object(line_quota, "get_settings", return_value=_stub_settings("")), \
             mock.patch.object(line_quota.httpx, "get") as g:
            self.assertIsNone(line_quota.fetch_quota())
        g.assert_not_called()

    def test_non_200_returns_none(self) -> None:
        responses = [_FakeResp({"type": "limited", "value": 200}, ok=False)]
        with mock.patch.object(line_quota, "get_settings", return_value=_stub_settings()), \
             mock.patch.object(line_quota.httpx, "get", side_effect=responses):
            self.assertIsNone(line_quota.fetch_quota())

    def test_malformed_consumption_returns_none(self) -> None:
        result, _ = self._run({"type": "limited", "value": 200}, {"totalUsage": "oops"})
        self.assertIsNone(result)

    def test_ttl_cache_hits_avoid_second_line_call(self) -> None:
        result1, g = self._run({"type": "limited", "value": 200}, {"totalUsage": 12})
        # Second call within TTL must NOT re-hit LINE (httpx.get called twice total for the first read).
        with mock.patch.object(line_quota.httpx, "get") as g2:
            result2 = line_quota.fetch_quota()
        self.assertEqual(result1, result2)
        g2.assert_not_called()

    def test_malformed_quota_value_returns_none(self) -> None:
        # Not in the brief: closes the `_safe_int(quota.get("value")) is None` branch
        # (a "limited" quota whose value isn't a well-formed int).
        result, _ = self._run({"type": "limited", "value": "oops"}, {"totalUsage": 5})
        self.assertIsNone(result)

    def test_non_dict_payload_returns_none(self) -> None:
        # Not in the brief: closes `_get`'s "unexpected LINE quota payload" branch — a 200
        # response whose JSON body isn't a dict (e.g. LINE returns a bare list/array).
        responses = [_FakeResp([1, 2, 3])]
        with mock.patch.object(line_quota, "get_settings", return_value=_stub_settings()), \
             mock.patch.object(line_quota.httpx, "get", side_effect=responses):
            self.assertIsNone(line_quota.fetch_quota())


def _settings(**overrides) -> Settings:
    """A real Settings whose LINE properties compute correctly (mirrors the webhook test helper)."""
    values = {
        "supabase_url": "https://proj.supabase.co",
        "supabase_anon_key": "anon-key",
        "supabase_service_role_key": "service-key",
        "line_channel_id": "2010629653",
        "line_messaging_channel_secret": "secret",
        "line_messaging_access_token": "token",
    }
    values.update(overrides)
    return Settings(**values)


class AdminLineStatusRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)
        line_quota.clear_cache()
        self.addCleanup(line_quota.clear_cache)

    def _get(self, *, quota, settings_obj):
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=settings_obj), \
             mock.patch.object(line_quota, "fetch_quota", return_value=quota):
            return self.client.get("/api/admin/line/status")

    def test_configured_with_limited_quota(self) -> None:
        quota = {"type": "limited", "used": 12, "value": 200, "remaining": 188}
        resp = self._get(quota=quota, settings_obj=_settings())
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["messaging_configured"])
        self.assertTrue(body["login_configured"])
        self.assertEqual(body["channel_id"], "2010629653")
        self.assertEqual(body["quota"], quota)
        self.assertIsNone(body["quota_error"])

    def test_configured_but_line_unreachable_sets_error(self) -> None:
        resp = self._get(quota=None, settings_obj=_settings())
        body = resp.json()
        self.assertIsNone(body["quota"])
        self.assertEqual(body["quota_error"], "unreachable")

    def test_not_configured_skips_line_and_has_no_error(self) -> None:
        settings_obj = _settings(line_messaging_access_token="")  # -> messaging_configured False
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=settings_obj), \
             mock.patch.object(line_quota, "fetch_quota") as fq:
            resp = self.client.get("/api/admin/line/status")
        body = resp.json()
        self.assertFalse(body["messaging_configured"])
        self.assertIsNone(body["quota"])
        self.assertIsNone(body["quota_error"])
        fq.assert_not_called()  # unconfigured => never call LINE

    def test_response_carries_no_secret(self) -> None:
        resp = self._get(quota={"type": "none", "used": 3}, settings_obj=_settings())
        blob = resp.text
        self.assertNotIn("token", blob)   # access token / channel secret never serialised
        self.assertNotIn("secret", blob)
        self.assertNotIn("service-key", blob)

    def test_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.get("/api/admin/line/status")
        self.assertEqual(resp.status_code, 403)

    def test_requires_auth(self) -> None:
        app.dependency_overrides.clear()  # drop override -> real dependency runs
        resp = self.client.get("/api/admin/line/status")
        self.assertEqual(resp.status_code, 401)
