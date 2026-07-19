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
import json
import sys
import types
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routers import line_webhook
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


class FormatSummaryTests(unittest.TestCase):
    def _format(self, summary: dict, **setting_overrides) -> str:
        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(**setting_overrides)
        ):
            return line_bot.format_summary(summary)

    def test_full_summary_has_counts_faults_and_link(self) -> None:
        text = self._format(dict(_SUMMARY))
        self.assertIn("累積分析：12 次", text)
        # 2026-07-19T13:03Z is 21:03 in UTC+8.
        self.assertIn("2026-07-19 21:03", text)
        self.assertIn("側面", text)
        self.assertIn("3 個問題", text)
        self.assertIn("1. 膝蓋內夾 ×7", text)
        self.assertIn("2. 深度不足 ×5", text)
        self.assertIn("https://liff.line.me/1234567890-Abcdefgh", text)

    def test_unknown_fault_id_falls_back_to_english_name(self) -> None:
        summary = {
            "total": 1,
            "latest": None,
            "top_faults": [{"id": "brand_new_fault", "name": "Brand New Fault", "count": 2}],
        }
        self.assertIn("Brand New Fault ×2", self._format(summary))

    def test_no_faults_omits_the_fault_section(self) -> None:
        text = self._format({"total": 2, "latest": None, "top_faults": []})
        self.assertNotIn("最常出現的問題", text)
        self.assertIn("累積分析：2 次", text)

    def test_blank_liff_id_omits_the_link(self) -> None:
        text = self._format(dict(_SUMMARY), line_liff_id="")
        self.assertNotIn("liff.line.me", text)

    def test_unknown_view_type_and_bad_timestamp_degrade_gracefully(self) -> None:
        summary = {
            "total": 1,
            "latest": {"created_at": "not-a-date", "view_type": "weird", "fault_count": None},
            "top_faults": [],
        }
        text = self._format(summary)
        self.assertIn("未知時間", text)
        self.assertIn("未知", text)
        self.assertIn("0 個問題", text)

    def test_missing_keys_do_not_raise(self) -> None:
        self.assertIn("累積分析：0 次", self._format({}))

    def test_missing_created_at_and_naive_timestamp_are_handled(self) -> None:
        # Not in the brief: closes two branches _format_time otherwise leaves uncovered
        # (non-string `created_at`, and a naive timestamp with no offset).
        missing = {
            "total": 1,
            "latest": {"created_at": None, "view_type": "side", "fault_count": 1},
            "top_faults": [],
        }
        self.assertIn("未知時間", self._format(missing))

        naive = {
            "total": 1,
            "latest": {"created_at": "2026-07-19T13:03:00", "view_type": "side", "fault_count": 1},
            "top_faults": [],
        }
        # A naive timestamp is treated as UTC before converting to the UTC+8 display tz.
        self.assertIn("2026-07-19 21:03", self._format(naive))


class StaticMessageTests(unittest.TestCase):
    def test_unbound_message_points_at_line_sign_in(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            text = line_bot.unbound_message()
        self.assertIn("LINE 登入", text)
        self.assertIn("https://liff.line.me/1234567890-Abcdefgh", text)

    def test_empty_message_mentions_no_records(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertIn("還沒有分析紀錄", line_bot.empty_message())

    def test_help_message_lists_the_keyword(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertIn("摘要", line_bot.help_message())


def _text_event(text: str, user_id: str = "Uabc123", reply_token: str = "rt-1") -> dict:
    return {
        "type": "message",
        "replyToken": reply_token,
        "source": {"type": "user", "userId": user_id},
        "message": {"type": "text", "id": "m1", "text": text},
    }


class HandleEventsTests(unittest.TestCase):
    def _handle(self, payload: dict, summary_return=..., summary_side_effect=None) -> list[dict]:
        """Patch ``summary_for_line_user`` and run ``handle_events`` under it.

        A plain ``mock.patch.object(..., side_effect=X, return_value=Y)`` call is used
        rather than juggling ``**kwargs`` conditionally: passing ``return_value=None``
        alongside a real ``side_effect`` is harmless (Mock only consults ``return_value``
        when ``side_effect`` is unset/None), so there is no need to omit either kwarg — one
        straightforward call covers every case the tests below exercise.
        """
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.object(
            line_bot,
            "summary_for_line_user",
            side_effect=summary_side_effect,
            return_value=None if summary_return is ... else summary_return,
        ):
            return line_bot.handle_events(payload)

    def test_keyword_returns_the_summary(self) -> None:
        replies = self._handle({"events": [_text_event("我的訓練摘要")]}, summary_return=dict(_SUMMARY))
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["reply_token"], "rt-1")
        self.assertIn("累積分析：12 次", replies[0]["text"])

    def test_short_keyword_and_whitespace_and_case_are_normalised(self) -> None:
        replies = self._handle({"events": [_text_event("  Summary  ")]}, summary_return=dict(_SUMMARY))
        self.assertIn("你的訓練摘要", replies[0]["text"])

    def test_unknown_account_returns_the_sign_in_reply(self) -> None:
        replies = self._handle({"events": [_text_event("摘要")]}, summary_return=None)
        self.assertIn("還沒有找到你的 x-coach 帳號", replies[0]["text"])

    def test_zero_analyses_returns_the_empty_reply(self) -> None:
        replies = self._handle(
            {"events": [_text_event("摘要")]},
            summary_return={"total": 0, "latest": None, "top_faults": []},
        )
        self.assertIn("還沒有分析紀錄", replies[0]["text"])

    def test_unknown_text_returns_help(self) -> None:
        replies = self._handle({"events": [_text_event("你好")]}, summary_return=dict(_SUMMARY))
        self.assertIn("傳「摘要」", replies[0]["text"])

    def test_non_text_and_non_message_events_are_ignored(self) -> None:
        payload = {
            "events": [
                {"type": "follow", "replyToken": "rt", "source": {"userId": "U1"}},
                {
                    "type": "message",
                    "replyToken": "rt",
                    "source": {"userId": "U1"},
                    "message": {"type": "sticker", "id": "s1"},
                },
                "not-a-dict",
            ]
        }
        self.assertEqual(self._handle(payload, summary_return=dict(_SUMMARY)), [])

    def test_events_without_reply_token_or_user_id_are_skipped(self) -> None:
        payload = {
            "events": [
                _text_event("摘要", reply_token=""),
                {"type": "message", "replyToken": "rt", "source": {}, "message": {"type": "text", "text": "摘要"}},
            ]
        }
        self.assertEqual(self._handle(payload, summary_return=dict(_SUMMARY)), [])

    def test_missing_events_key_returns_nothing(self) -> None:
        self.assertEqual(self._handle({}, summary_return=dict(_SUMMARY)), [])

    def test_rpc_failure_falls_back_to_an_apology_and_keeps_other_events(self) -> None:
        payload = {"events": [_text_event("摘要", reply_token="rt-1"), _text_event("你好", reply_token="rt-2")]}
        replies = self._handle(payload, summary_side_effect=RuntimeError("db down"))
        self.assertEqual(len(replies), 2)
        self.assertIn("暫時查不到", replies[0]["text"])
        self.assertIn("傳「摘要」", replies[1]["text"])


class ReplyTests(unittest.TestCase):
    def test_posts_a_text_message_with_the_bearer_token(self) -> None:
        response = mock.Mock(status_code=200)
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.object(
            line_bot.httpx, "post", return_value=response
        ) as post:
            line_bot.reply("rt-1", "嗨")
        self.assertEqual(post.call_args[0][0], line_bot.LINE_REPLY_URL)
        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer chan-token")
        self.assertEqual(
            kwargs["json"],
            {"replyToken": "rt-1", "messages": [{"type": "text", "text": "嗨"}]},
        )

    def test_non_200_is_swallowed(self) -> None:
        response = mock.Mock(status_code=400)
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.object(
            line_bot.httpx, "post", return_value=response
        ):
            line_bot.reply("rt-1", "嗨")  # must not raise

    def test_network_error_is_swallowed(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.object(
            line_bot.httpx, "post", side_effect=httpx.ConnectError("boom")
        ):
            line_bot.reply("rt-1", "嗨")  # must not raise


class WebhookRouteTests(unittest.TestCase):
    def _post(self, body: dict, *, signature: str | None = ..., settings=None):
        raw = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if signature is ...:
            headers["X-Line-Signature"] = _sign(raw)
        elif signature is not None:
            headers["X-Line-Signature"] = signature
        with mock.patch.object(
            line_webhook, "get_settings", return_value=settings or _settings()
        ), mock.patch.object(line_bot, "get_settings", return_value=settings or _settings()):
            with TestClient(app) as client:
                return client.post("/api/line/webhook", content=raw, headers=headers)

    def test_unconfigured_is_503(self) -> None:
        response = self._post({"events": []}, settings=_settings(line_messaging_configured=False))
        self.assertEqual(response.status_code, 503)

    def test_bad_signature_is_400(self) -> None:
        response = self._post({"events": []}, signature="wrong")
        self.assertEqual(response.status_code, 400)

    def test_missing_signature_is_400(self) -> None:
        response = self._post({"events": []}, signature=None)
        self.assertEqual(response.status_code, 400)

    def test_valid_event_replies_and_returns_200(self) -> None:
        with mock.patch.object(
            line_bot, "handle_events", return_value=[{"reply_token": "rt-1", "text": "嗨"}]
        ), mock.patch.object(line_bot, "reply") as reply:
            response = self._post({"events": [_text_event("摘要")]})
        self.assertEqual(response.status_code, 200)
        reply.assert_called_once_with("rt-1", "嗨")

    def test_non_text_event_returns_200_without_replying(self) -> None:
        with mock.patch.object(line_bot, "reply") as reply:
            response = self._post({"events": [{"type": "follow", "replyToken": "rt"}]})
        self.assertEqual(response.status_code, 200)
        reply.assert_not_called()

    def test_malformed_json_after_valid_signature_is_still_200(self) -> None:
        raw = b"not json"
        headers = {"X-Line-Signature": _sign(raw), "Content-Type": "application/json"}
        with mock.patch.object(
            line_webhook, "get_settings", return_value=_settings()
        ), mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            with TestClient(app) as client:
                response = client.post("/api/line/webhook", content=raw, headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_reply_failure_is_still_200(self) -> None:
        with mock.patch.object(
            line_bot, "handle_events", return_value=[{"reply_token": "rt-1", "text": "嗨"}]
        ), mock.patch.object(line_bot, "reply", side_effect=RuntimeError("boom")):
            response = self._post({"events": [_text_event("摘要")]})
        self.assertEqual(response.status_code, 200)
