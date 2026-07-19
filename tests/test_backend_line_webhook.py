"""Unit tests for the LINE Messaging API bot (services/line_bot + routers/line_webhook).

Mirrors ``tests/test_backend_line_auth.py``: unittest.TestCase classes, the ``supabase``
package faked through ``sys.modules`` (it is not installed in CI), external HTTP (LINE's
reply endpoint) mocked at the ``httpx.post`` seam, and FastAPI routes exercised through
``TestClient`` with ``get_settings`` patched.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import sys
import types
import unittest
from unittest import mock

from backend.app.services import line_bot
from backend.app.settings import Settings


class MessagingConfiguredTests(unittest.TestCase):
    def _settings(self, **overrides) -> Settings:
        values = {
            "supabase_url": "https://proj.supabase.co",
            "supabase_anon_key": "anon-key",
            "supabase_service_role_key": "service-key",
            "line_messaging_channel_secret": "secret",
            "line_messaging_access_token": "token",
        }
        values.update(overrides)
        return Settings(**values)

    def test_configured_when_all_present(self) -> None:
        self.assertTrue(self._settings().line_messaging_configured)

    def test_not_configured_without_secret(self) -> None:
        self.assertFalse(self._settings(line_messaging_channel_secret="").line_messaging_configured)

    def test_not_configured_without_access_token(self) -> None:
        self.assertFalse(self._settings(line_messaging_access_token="").line_messaging_configured)

    def test_not_configured_without_service_role_key(self) -> None:
        self.assertFalse(self._settings(supabase_service_role_key="").line_messaging_configured)

    def test_not_configured_without_supabase(self) -> None:
        self.assertFalse(self._settings(supabase_url="").line_messaging_configured)

    def test_liff_id_defaults_to_empty(self) -> None:
        # Assert the declared default, not an instance: a real repo-root .env with
        # LINE_LIFF_ID set would otherwise make this pass/fail by machine.
        self.assertEqual(Settings.model_fields["line_liff_id"].default, "")


def _settings(**overrides) -> types.SimpleNamespace:
    """A lightweight Settings stand-in with the fields line_bot reads."""
    values = {
        "supabase_url": "https://proj.supabase.co",
        "supabase_anon_key": "anon-key",
        "supabase_service_role_key": "service-key",
        "line_messaging_channel_secret": "chan-secret",
        "line_messaging_access_token": "chan-token",
        "line_liff_id": "1234567890-Abcdefgh",
        "line_messaging_configured": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _sign(body: bytes, secret: str = "chan-secret") -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


class VerifySignatureTests(unittest.TestCase):
    def test_valid_signature_passes(self) -> None:
        body = b'{"events":[]}'
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertTrue(line_bot.verify_signature(body, _sign(body)))

    def test_tampered_body_fails(self) -> None:
        signature = _sign(b'{"events":[]}')
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertFalse(line_bot.verify_signature(b'{"events":[1]}', signature))

    def test_missing_signature_fails(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertFalse(line_bot.verify_signature(b"{}", None))

    def test_wrong_secret_fails(self) -> None:
        body = b"{}"
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertFalse(line_bot.verify_signature(body, _sign(body, "other-secret")))

    def test_empty_secret_fails(self) -> None:
        body = b"{}"
        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(line_messaging_channel_secret="")
        ):
            self.assertFalse(line_bot.verify_signature(body, _sign(body)))

    def test_non_ascii_signature_fails(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertFalse(line_bot.verify_signature(b"{}", "不是-base64-簽章"))


def _fake_supabase_module(client: mock.Mock) -> types.ModuleType:
    """A fake ``supabase`` package whose ``create_client`` returns ``client``."""
    module = types.ModuleType("supabase")

    def create_client(url: str, key: str):  # noqa: ARG001 — signature parity
        return client

    module.create_client = create_client  # type: ignore[attr-defined]
    return module


def _rpc_client(data) -> mock.Mock:
    client = mock.Mock()
    client.rpc.return_value.execute.return_value = types.SimpleNamespace(data=data)
    return client


_SUMMARY = {
    "total": 12,
    "latest": {"created_at": "2026-07-19T13:03:11.5+00:00", "view_type": "side", "fault_count": 3},
    "top_faults": [
        {"id": "knees_inward", "name": "Knees Inward / Knee Valgus", "count": 7},
        {"id": "shallow_depth", "name": "Shallow Depth", "count": 5},
    ],
}


class SummaryForLineUserTests(unittest.TestCase):
    def test_returns_rpc_payload(self) -> None:
        client = _rpc_client(dict(_SUMMARY))
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.dict(
            sys.modules, {"supabase": _fake_supabase_module(client)}
        ):
            result = line_bot.summary_for_line_user("Uabc123")
        self.assertEqual(result["total"], 12)
        client.rpc.assert_called_once_with(
            "line_training_summary", {"p_line_sub": "Uabc123"}
        )

    def test_null_payload_returns_none(self) -> None:
        client = _rpc_client(None)
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.dict(
            sys.modules, {"supabase": _fake_supabase_module(client)}
        ):
            self.assertIsNone(line_bot.summary_for_line_user("Uabc123"))

    def test_unexpected_payload_shape_returns_none(self) -> None:
        client = _rpc_client([1, 2, 3])
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.dict(
            sys.modules, {"supabase": _fake_supabase_module(client)}
        ):
            self.assertIsNone(line_bot.summary_for_line_user("Uabc123"))
