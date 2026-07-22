"""Unit tests for services/line_admin + GET /api/admin/line/status.

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
from backend.app.services import line_admin, store
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


class LineAdminQuotaTests(unittest.TestCase):
    def setUp(self) -> None:
        line_admin.clear_cache()
        self.addCleanup(line_admin.clear_cache)

    def _run(self, quota_payload, consumption_payload, *, token="chan-token"):
        responses = [_FakeResp(quota_payload), _FakeResp(consumption_payload)]
        with mock.patch.object(line_admin, "get_settings", return_value=_stub_settings(token)), \
             mock.patch.object(line_admin.httpx, "get", side_effect=responses) as g:
            return line_admin.fetch_quota(), g

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
        with mock.patch.object(line_admin, "get_settings", return_value=_stub_settings("")), \
             mock.patch.object(line_admin.httpx, "get") as g:
            self.assertIsNone(line_admin.fetch_quota())
        g.assert_not_called()

    def test_non_200_returns_none(self) -> None:
        responses = [_FakeResp({"type": "limited", "value": 200}, ok=False)]
        with mock.patch.object(line_admin, "get_settings", return_value=_stub_settings()), \
             mock.patch.object(line_admin.httpx, "get", side_effect=responses):
            self.assertIsNone(line_admin.fetch_quota())

    def test_malformed_consumption_returns_none(self) -> None:
        result, _ = self._run({"type": "limited", "value": 200}, {"totalUsage": "oops"})
        self.assertIsNone(result)

    def test_ttl_cache_hits_avoid_second_line_call(self) -> None:
        result1, g = self._run({"type": "limited", "value": 200}, {"totalUsage": 12})
        # Second call within TTL must NOT re-hit LINE (httpx.get called twice total for the first read).
        with mock.patch.object(line_admin.httpx, "get") as g2:
            result2 = line_admin.fetch_quota()
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
        with mock.patch.object(line_admin, "get_settings", return_value=_stub_settings()), \
             mock.patch.object(line_admin.httpx, "get", side_effect=responses):
            self.assertIsNone(line_admin.fetch_quota())


class LineBotInfoWebhookDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        line_admin.clear_cache()
        self.addCleanup(line_admin.clear_cache)

    def _patch_get(self, responses):
        return mock.patch.object(line_admin.httpx, "get", side_effect=responses)

    def _settings_stub(self, token="chan-token"):
        return mock.patch.object(line_admin, "get_settings",
                                 return_value=types.SimpleNamespace(line_messaging_access_token=token))

    def test_bot_info_happy(self) -> None:
        payload = {"displayName": "x-coach", "basicId": "@xcoach", "premiumId": None,
                   "chatMode": "bot", "markAsReadMode": "auto"}
        with self._settings_stub(), self._patch_get([_FakeResp(payload)]):
            self.assertEqual(line_admin.fetch_bot_info(), {
                "display_name": "x-coach", "basic_id": "@xcoach", "premium_id": None,
                "chat_mode": "bot", "mark_as_read_mode": "auto"})

    def test_bot_info_no_token_returns_none(self) -> None:
        with self._settings_stub(token=""), self._patch_get([]) as g:
            self.assertIsNone(line_admin.fetch_bot_info())
        g.assert_not_called()

    def test_bot_info_non_200_returns_none(self) -> None:
        with self._settings_stub(), self._patch_get([_FakeResp({}, ok=False)]):
            self.assertIsNone(line_admin.fetch_bot_info())

    def test_webhook_happy(self) -> None:
        with self._settings_stub(), self._patch_get([_FakeResp({"endpoint": "https://x/api/line/webhook", "active": True})]):
            self.assertEqual(line_admin.fetch_webhook(), {"endpoint": "https://x/api/line/webhook", "active": True})

    def test_webhook_missing_endpoint_returns_none(self) -> None:
        with self._settings_stub(), self._patch_get([_FakeResp({"active": True})]):
            self.assertIsNone(line_admin.fetch_webhook())

    def test_delivery_ready_counts(self) -> None:
        with self._settings_stub(), \
             mock.patch.object(line_admin, "_yesterday_yyyymmdd", return_value="20260720"), \
             self._patch_get([_FakeResp({"status": "ready", "success": 12}),
                              _FakeResp({"status": "ready", "success": 3})]):
            self.assertEqual(line_admin.fetch_delivery(), {"date": "20260720", "reply": 12, "push": 3})

    def test_delivery_unready_yields_none_counts(self) -> None:
        with self._settings_stub(), \
             mock.patch.object(line_admin, "_yesterday_yyyymmdd", return_value="20260720"), \
             self._patch_get([_FakeResp({"status": "unready"}), _FakeResp({"status": "unready"})]):
            self.assertEqual(line_admin.fetch_delivery(), {"date": "20260720", "reply": None, "push": None})

    def test_readonly_cache_hits_avoid_second_call(self) -> None:
        with self._settings_stub(), self._patch_get([_FakeResp({"endpoint": "https://x", "active": False})]):
            first = line_admin.fetch_webhook()
        with mock.patch.object(line_admin.httpx, "get") as g2:
            second = line_admin.fetch_webhook()
        self.assertEqual(first, second)
        g2.assert_not_called()

    def test_webhook_no_token_returns_none(self) -> None:
        # Not in the brief: closes _fetch_webhook's "no token" branch (mirrors bot_info's).
        with self._settings_stub(token=""), self._patch_get([]) as g:
            self.assertIsNone(line_admin.fetch_webhook())
        g.assert_not_called()

    def test_webhook_read_failure_returns_none(self) -> None:
        # Not in the brief: closes _fetch_webhook's except-on-read-failure branch.
        with self._settings_stub(), self._patch_get([_FakeResp({}, ok=False)]):
            self.assertIsNone(line_admin.fetch_webhook())

    def test_delivery_no_token_returns_none(self) -> None:
        # Not in the brief: closes _fetch_delivery's "no token" branch.
        with self._settings_stub(token=""), self._patch_get([]) as g:
            self.assertIsNone(line_admin.fetch_delivery())
        g.assert_not_called()

    def test_delivery_read_failure_returns_none(self) -> None:
        # Not in the brief: closes _fetch_delivery's except-on-read-failure branch.
        with self._settings_stub(), \
             mock.patch.object(line_admin, "_yesterday_yyyymmdd", return_value="20260720"), \
             self._patch_get([_FakeResp({}, ok=False)]):
            self.assertIsNone(line_admin.fetch_delivery())


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
        line_admin.clear_cache()
        self.addCleanup(line_admin.clear_cache)

    def _get(self, *, quota, settings_obj):
        # bot_info/webhook/delivery are stubbed to None here (unexercised by these quota-focused
        # tests) so the route never falls through to a real, unmocked LINE call.
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=settings_obj), \
             mock.patch.object(line_admin, "fetch_quota", return_value=quota), \
             mock.patch.object(line_admin, "fetch_bot_info", return_value=None), \
             mock.patch.object(line_admin, "fetch_webhook", return_value=None), \
             mock.patch.object(line_admin, "fetch_delivery", return_value=None):
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
             mock.patch.object(line_admin, "fetch_quota") as fq:
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

    def test_status_includes_bot_info_webhook_delivery(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=_settings()), \
             mock.patch.object(line_admin, "fetch_quota", return_value={"type": "none", "used": 1}), \
             mock.patch.object(line_admin, "fetch_bot_info", return_value={"display_name": "x", "basic_id": "@x", "premium_id": None, "chat_mode": "bot", "mark_as_read_mode": "auto"}), \
             mock.patch.object(line_admin, "fetch_webhook", return_value={"endpoint": "https://x", "active": True}), \
             mock.patch.object(line_admin, "fetch_delivery", return_value={"date": "20260720", "reply": 4, "push": 0}):
            body = self.client.get("/api/admin/line/status").json()
        self.assertEqual(body["bot_info"]["chat_mode"], "bot")
        self.assertTrue(body["webhook"]["active"])
        self.assertEqual(body["delivery"]["reply"], 4)

    def test_status_not_configured_nulls_new_keys_and_skips_line(self) -> None:
        settings_obj = _settings(line_messaging_access_token="")
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=settings_obj), \
             mock.patch.object(line_admin, "fetch_bot_info") as bi, \
             mock.patch.object(line_admin, "fetch_webhook") as wh, \
             mock.patch.object(line_admin, "fetch_delivery") as dl:
            body = self.client.get("/api/admin/line/status").json()
        self.assertIsNone(body["bot_info"])
        self.assertIsNone(body["webhook"])
        self.assertIsNone(body["delivery"])
        bi.assert_not_called(); wh.assert_not_called(); dl.assert_not_called()


class LineWebhookTestTests(unittest.TestCase):
    def _settings_stub(self, token="chan-token"):
        return mock.patch.object(line_admin, "get_settings",
                                 return_value=types.SimpleNamespace(line_messaging_access_token=token))

    def test_success_result(self) -> None:
        payload = {"success": True, "statusCode": 200, "reason": "OK", "detail": "200"}
        with self._settings_stub(), mock.patch.object(line_admin.httpx, "post", return_value=_FakeResp(payload)):
            self.assertEqual(line_admin.test_webhook(),
                             {"success": True, "status_code": 200, "reason": "OK", "detail": "200"})

    def test_transport_failure_returns_none(self) -> None:
        with self._settings_stub(), mock.patch.object(line_admin.httpx, "post", return_value=_FakeResp({}, ok=False)):
            self.assertIsNone(line_admin.test_webhook())

    def test_no_token_returns_none(self) -> None:
        with self._settings_stub(token=""), mock.patch.object(line_admin.httpx, "post") as p:
            self.assertIsNone(line_admin.test_webhook())
        p.assert_not_called()


class AdminLineWebhookTestRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)

    def test_not_configured(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=_settings(line_messaging_access_token="")):
            body = self.client.post("/api/admin/line/webhook-test").json()
        self.assertEqual(body, {"result": None, "error": "not_configured"})

    def test_unreachable(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=_settings()), \
             mock.patch.object(line_admin, "test_webhook", return_value=None):
            body = self.client.post("/api/admin/line/webhook-test").json()
        self.assertEqual(body, {"result": None, "error": "unreachable"})

    def test_success(self) -> None:
        result = {"success": True, "status_code": 200, "reason": "OK", "detail": "200"}
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=_settings()), \
             mock.patch.object(line_admin, "test_webhook", return_value=result):
            body = self.client.post("/api/admin/line/webhook-test").json()
        self.assertEqual(body, {"result": result, "error": None})

    def test_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.post("/api/admin/line/webhook-test")
        self.assertEqual(resp.status_code, 403)

    def test_requires_auth(self) -> None:
        app.dependency_overrides.clear()
        resp = self.client.post("/api/admin/line/webhook-test")
        self.assertEqual(resp.status_code, 401)
