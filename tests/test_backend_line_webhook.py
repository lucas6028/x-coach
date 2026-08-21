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
        # The LLM conversation is opt-in; the keyword/summary tests run without it.
        "chat_configured": False,
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


class ServiceClientTests(unittest.TestCase):
    def test_rpc_client_gets_a_short_postgrest_timeout(self) -> None:
        module = types.ModuleType("supabase")
        module.ClientOptions = mock.Mock(name="ClientOptions")  # type: ignore[attr-defined]
        module.create_client = mock.Mock(name="create_client")  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"supabase": module}), mock.patch.object(
            line_bot, "get_settings", return_value=_settings()
        ):
            line_bot._service_client()
        module.ClientOptions.assert_called_once_with(postgrest_client_timeout=line_bot._RPC_TIMEOUT_S)
        module.create_client.assert_called_once_with(
            "https://proj.supabase.co", "service-key", options=module.ClientOptions.return_value
        )

    def test_without_client_options_falls_back_to_the_bare_client(self) -> None:
        client = mock.Mock()
        with mock.patch.dict(sys.modules, {"supabase": _fake_supabase_module(client)}), mock.patch.object(
            line_bot, "get_settings", return_value=_settings()
        ):
            self.assertIs(line_bot._service_client(), client)


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
            line_bot, "iter_replies", return_value=[{"reply_token": "rt-1", "text": "嗨"}]
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
            line_bot, "iter_replies", return_value=[{"reply_token": "rt-1", "text": "嗨"}]
        ), mock.patch.object(line_bot, "reply", side_effect=RuntimeError("boom")):
            response = self._post({"events": [_text_event("摘要")]})
        self.assertEqual(response.status_code, 200)

    def test_batched_events_are_answered_independently_and_in_parallel(self) -> None:
        # LINE batches events and each free-text one can spend seconds in the LLM. Event A's
        # answer is held until event B's reply has been SENT: only parallel, per-event handling
        # can satisfy that — serial handling (in either order) would deadlock and time out.
        import threading

        b_sent = threading.Event()
        order: list[str] = []

        def fake_replies(payload):
            (event,) = payload["events"]
            token = event["replyToken"]
            if token == "rt-a":
                self.assertTrue(b_sent.wait(timeout=5), "event A waited on B in vain: serial handling")
            yield {"reply_token": token, "text": event["message"]["text"]}

        def fake_reply(token, _text):
            order.append(token)
            if token == "rt-b":
                b_sent.set()

        with mock.patch.object(line_bot, "iter_replies", side_effect=fake_replies), mock.patch.object(
            line_bot, "reply", side_effect=fake_reply
        ):
            response = self._post(
                {
                    "events": [
                        _text_event("a", user_id="Ua", reply_token="rt-a"),
                        _text_event("b", user_id="Ub", reply_token="rt-b"),
                    ]
                }
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(order, ["rt-b", "rt-a"])

    def test_events_are_handled_after_the_ack_as_a_background_task(self) -> None:
        # The 200 must not wait for the LLM: the work is scheduled via BackgroundTasks (which runs
        # a sync callable in the threadpool) rather than awaited inline in the handler.
        with mock.patch.object(
            line_webhook.BackgroundTasks, "add_task", autospec=True
        ) as add_task, mock.patch.object(line_bot, "reply") as reply:
            response = self._post({"events": [_text_event("摘要")]})
        self.assertEqual(response.status_code, 200)
        add_task.assert_called_once()
        self.assertIs(add_task.call_args.args[1], line_webhook._process)
        reply.assert_not_called()  # nothing ran inline

    def test_same_chat_events_stay_in_order_on_one_lane(self) -> None:
        # Two messages from one user + one from another: the user's two are answered serially,
        # in delivery order (one iter_replies call, both events), the other user's in parallel.
        calls: list[list[str]] = []

        def fake_replies(payload):
            calls.append([e["replyToken"] for e in payload["events"]])
            return iter([])

        payload = {
            "events": [
                _text_event("第一句", user_id="Ua", reply_token="a1"),
                _text_event("嗨", user_id="Ub", reply_token="b1"),
                _text_event("那第二點呢？", user_id="Ua", reply_token="a2"),
            ]
        }
        with mock.patch.object(line_bot, "iter_replies", side_effect=fake_replies):
            line_webhook._process(json.dumps(payload).encode())
        self.assertEqual(sorted(calls), [["a1", "a2"], ["b1"]])

    def test_lanes_group_by_group_room_or_user_and_keep_order(self) -> None:
        group = _text_event("g", user_id="Ua", reply_token="g1")
        group["source"] = {"type": "group", "groupId": "Cg", "userId": "Ua"}
        events = [_text_event("x", user_id="Ua"), group, _text_event("y", user_id="Ua"), {"type": "follow"}]
        lanes = line_webhook._lanes_by_chat(events)
        self.assertEqual([[e.get("replyToken") for e in lane] for lane in lanes], [["rt-1", "rt-1"], ["g1"], [None]])

    def test_single_event_skips_the_thread_pool(self) -> None:
        with mock.patch.object(line_webhook, "ThreadPoolExecutor") as pool, mock.patch.object(
            line_bot, "iter_replies", return_value=[]
        ) as replies:
            line_webhook._process(json.dumps({"events": [_text_event("a"), _text_event("b")]}).encode())
        pool.assert_not_called()  # same chat → one lane → no pool needed
        replies.assert_called_once()
        self.assertEqual(len(replies.call_args.args[0]["events"]), 2)

    def test_non_dict_payload_is_swallowed(self) -> None:
        with mock.patch.object(line_bot, "iter_replies") as replies:
            line_webhook._process(b"[1, 2]")
            line_webhook._process(json.dumps({"events": "nope"}).encode())
        self.assertEqual(
            [c.args[0] for c in replies.call_args_list], [{"events": []}, {"events": []}]
        )

    def test_one_bad_reply_does_not_skip_the_rest_of_the_events(self) -> None:
        # line_bot.reply only swallows httpx.HTTPError itself; a non-httpx exception on the
        # first event's reply must not stop the router from replying to the second event.
        planned = [
            {"reply_token": "rt-1", "text": "first"},
            {"reply_token": "rt-2", "text": "second"},
        ]
        with mock.patch.object(
            line_bot, "iter_replies", return_value=planned
        ), mock.patch.object(
            line_bot, "reply", side_effect=[RuntimeError("boom"), None]
        ) as reply:
            # One event: since events are answered per-event, the fake stream above is what
            # ONE event yields; the loop over it must survive a failing reply.
            response = self._post({"events": [_text_event("摘要")]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            reply.call_args_list,
            [mock.call("rt-1", "first"), mock.call("rt-2", "second")],
        )


# ---------------------------------------------------------------------------
# LLM conversation (free text → the coach)
# ---------------------------------------------------------------------------


class ChatReplyTests(unittest.TestCase):
    """``chat_reply_for``: the LLM seam is ``chat_service._stream_completion`` (patched, never
    hit), the summary RPC is patched, and the typing-indicator POST is patched at ``httpx.post``."""

    def setUp(self) -> None:
        line_bot.clear_history()
        self.addCleanup(line_bot.clear_history)

    def _chat(
        self,
        text: str,
        *,
        user_id: str = "Uabc123",
        chunks=("好的，", "先從腳尖外開開始。"),
        llm_side_effect=None,
        summary_return=...,
        summary_side_effect=None,
        settings=None,
        now: float = 1_000.0,
    ):
        """Run ``chat_reply_for`` with every external seam patched; returns (reply, stream_mock, post_mock)."""
        with mock.patch.object(
            line_bot, "get_settings", return_value=settings or _settings(chat_configured=True)
        ), mock.patch.object(
            line_bot,
            "summary_for_line_user",
            side_effect=summary_side_effect,
            return_value=dict(_SUMMARY) if summary_return is ... else summary_return,
        ), mock.patch.object(
            line_bot.chat_service,
            "_stream_completion",
            side_effect=llm_side_effect,
            return_value=iter(chunks),
        ) as stream, mock.patch.object(
            line_bot, "default_chat_model", return_value="test/model"
        ), mock.patch.object(
            line_bot.httpx, "post", return_value=mock.Mock(status_code=202)
        ) as post, mock.patch.object(line_bot.time, "time", return_value=now):
            return line_bot.chat_reply_for(user_id, text), stream, post

    def _messages(self, stream) -> list[dict]:
        return stream.call_args.args[0]

    def test_unconfigured_falls_back_to_help(self) -> None:
        reply, stream, post = self._chat("膝蓋內夾怎麼辦", settings=_settings(chat_configured=False))
        self.assertIn("傳「摘要」", reply)
        stream.assert_not_called()
        post.assert_not_called()

    def test_answer_is_the_joined_stream_and_the_prompt_is_grounded(self) -> None:
        reply, stream, _ = self._chat("  膝蓋內夾怎麼辦  ")
        self.assertEqual(reply, "好的，先從腳尖外開開始。")
        messages = self._messages(stream)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "膝蓋內夾怎麼辦")
        system = messages[0]["content"]
        self.assertIn("total analyses = 12", system)
        self.assertIn("膝蓋內夾 x7", system)
        self.assertIn("側面 view", system)
        self.assertIn("2026-07-19 21:03", system)  # rendered in UTC+8 like the summary card
        self.assertIn("NO Markdown", system)
        self.assertIn("Traditional Chinese", system)
        # Model + the short stall timeout (the 25s wall clock is enforced separately), not the
        # web chat's 60s.
        self.assertEqual(stream.call_args.args[1], "test/model")
        self.assertEqual(stream.call_args.kwargs["timeout"], line_bot._CHAT_STALL_TIMEOUT_S)
        self.assertLess(line_bot._CHAT_STALL_TIMEOUT_S, line_bot._CHAT_TIMEOUT_S)

    def test_typing_indicator_is_requested_with_the_bot_token(self) -> None:
        _, _, post = self._chat("你好")
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], line_bot.LINE_LOADING_URL)
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer chan-token")
        self.assertEqual(post.call_args.kwargs["json"]["chatId"], "Uabc123")

    def test_typing_indicator_failures_never_block_the_answer(self) -> None:
        with mock.patch.object(line_bot.httpx, "post", side_effect=httpx.ConnectError("down")):
            with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
                line_bot.show_loading("Uabc123")  # must not raise
        with mock.patch.object(line_bot.httpx, "post", return_value=mock.Mock(status_code=400)):
            with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
                line_bot.show_loading("Uabc123")  # must not raise

    def test_unbound_user_is_told_to_sign_in_via_the_prompt(self) -> None:
        _, stream, _ = self._chat("我最近表現如何", summary_return=None)
        self.assertIn("not signed in", self._messages(stream)[0]["content"])

    def test_zero_analyses_user_prompt(self) -> None:
        _, stream, _ = self._chat("我最近表現如何", summary_return={"total": 0})
        self.assertIn("no analyses yet", self._messages(stream)[0]["content"])

    def test_summary_lookup_failure_still_chats_without_facts(self) -> None:
        reply, stream, _ = self._chat("你好", summary_side_effect=RuntimeError("db down"))
        self.assertEqual(reply, "好的，先從腳尖外開開始。")
        self.assertIn("not signed in", self._messages(stream)[0]["content"])

    def test_llm_failure_returns_the_apology_and_is_not_remembered(self) -> None:
        reply, _, _ = self._chat("你好", llm_side_effect=RuntimeError("502"))
        self.assertEqual(reply, line_bot._CHAT_UNAVAILABLE)
        _, stream, _ = self._chat("第二句")
        self.assertEqual([m["role"] for m in self._messages(stream)], ["system", "user"])

    def test_empty_completion_returns_the_apology(self) -> None:
        reply, _, _ = self._chat("你好", chunks=("  ", ""))
        self.assertEqual(reply, line_bot._CHAT_UNAVAILABLE)

    def test_generation_is_capped_with_max_tokens(self) -> None:
        _, stream, _ = self._chat("你好")
        self.assertEqual(
            stream.call_args.kwargs["extra_body"], {"max_tokens": line_bot._MAX_COMPLETION_TOKENS}
        )

    def test_wall_clock_deadline_stops_a_slow_stream_and_keeps_the_partial_answer(self) -> None:
        # httpx's timeout is per network operation, so a provider that keeps trickling tokens
        # never trips it; the monotonic deadline must cut the stream and reply with what arrived.
        consumed: list[str] = []

        def trickle():
            for piece in ("一", "二", "三", "四"):
                consumed.append(piece)
                yield piece

        ticks = iter([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
        with mock.patch.object(line_bot.time, "monotonic", side_effect=lambda: next(ticks)):
            reply, _, _ = self._chat("你好", chunks=trickle())
        self.assertEqual(reply, "一二三…")
        self.assertEqual(consumed, ["一", "二", "三"])  # the 4th chunk was never pulled
        # The partial answer still counts as this turn's history.
        self.assertEqual(line_bot._history["Uabc123"][1][-1]["content"], "一二三…")

    def test_overlong_answer_is_truncated_with_an_ellipsis(self) -> None:
        reply, _, _ = self._chat("你好", chunks=("字" * 3000,))
        self.assertEqual(len(reply), line_bot._MAX_REPLY_CHARS)
        self.assertTrue(reply.endswith("…"))

    def test_overlong_user_text_is_clipped(self) -> None:
        _, stream, _ = self._chat("問" * 5000)
        self.assertEqual(len(self._messages(stream)[-1]["content"]), line_bot._MAX_USER_TEXT_CHARS)

    def test_history_is_carried_into_the_next_turn_per_user(self) -> None:
        self._chat("第一句", chunks=("回一",))
        _, stream, _ = self._chat("第二句", chunks=("回二",))
        roles = [(m["role"], m["content"]) for m in self._messages(stream)[1:]]
        self.assertEqual(
            roles,
            [("user", "第一句"), ("assistant", "回一"), ("user", "第二句")],
        )
        # Another LINE user starts clean.
        _, other, _ = self._chat("嗨", user_id="Uother")
        self.assertEqual([m["role"] for m in self._messages(other)], ["system", "user"])

    def test_history_is_capped_to_the_trailing_window(self) -> None:
        for i in range(10):
            self._chat(f"問{i}", chunks=(f"答{i}",))
        _, stream, _ = self._chat("最後")
        history = self._messages(stream)[1:-1]
        self.assertEqual(len(history), line_bot._HISTORY_MAX_MESSAGES)
        self.assertEqual(history[0]["content"], f"問{10 - line_bot._HISTORY_MAX_MESSAGES // 2}")

    def test_history_expires_after_the_ttl(self) -> None:
        self._chat("第一句", now=1_000.0)
        _, stream, _ = self._chat("第二句", now=1_000.0 + line_bot._HISTORY_TTL_S + 1)
        self.assertEqual([m["role"] for m in self._messages(stream)], ["system", "user"])
        # The expired window is gone; only the fresh turn is stored now.
        self.assertEqual([m["content"] for m in line_bot._history["Uabc123"][1]][0], "第二句")

    def test_group_chat_never_loads_personal_data_and_keeps_its_own_history(self) -> None:
        # Private turn first, so there is something that must NOT leak into the group.
        self._chat("我最近練得如何", chunks=("私人回答",))
        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(chat_configured=True)
        ), mock.patch.object(line_bot, "summary_for_line_user") as summary, mock.patch.object(
            line_bot.chat_service, "_stream_completion", return_value=iter(("群組回答",))
        ) as stream, mock.patch.object(
            line_bot, "default_chat_model", return_value="m"
        ), mock.patch.object(line_bot.httpx, "post") as post, mock.patch.object(
            line_bot.time, "time", return_value=1_001.0  # same clock as the private turn above
        ):
            reply = line_bot.chat_reply_for("Uabc123", "深蹲要注意什麼", chat_id="Cgroup1")
        self.assertEqual(reply, "群組回答")
        summary.assert_not_called()  # the sender's records are never fetched for a group
        post.assert_not_called()  # typing indicator is 1:1-only
        messages = stream.call_args.args[0]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])  # no private history
        self.assertIn("group chat", messages[0]["content"])
        self.assertNotIn("total analyses", messages[0]["content"])
        # Stored under the group, and the user's private history is untouched.
        self.assertEqual([m["content"] for m in line_bot._history["Cgroup1"][1]], ["深蹲要注意什麼", "群組回答"])
        self.assertEqual([m["content"] for m in line_bot._history["Uabc123"][1]], ["我最近練得如何", "私人回答"])

    def test_turns_in_one_chat_are_serialized_across_requests(self) -> None:
        # A follow-up that arrives while the first answer is still streaming must wait for it
        # and then see it in history — even though each webhook request is its own task.
        # Patches are installed ONCE in the main thread (mock.patch is process-global; nested
        # per-thread patching would restore in the wrong order and leak a mock).
        import threading

        first_started = threading.Event()
        release_first = threading.Event()
        histories: dict[str, list[str]] = {}

        def fake_stream(messages, _model, **_kw):
            text = messages[-1]["content"]
            histories[text] = [m["content"] for m in messages[1:-1]]
            if text == "第一句":
                first_started.set()
                self.assertTrue(release_first.wait(timeout=5))
                yield "答一"
            else:
                yield "答二"

        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(chat_configured=True)
        ), mock.patch.object(line_bot, "summary_for_line_user", return_value=None), mock.patch.object(
            line_bot.chat_service, "_stream_completion", side_effect=fake_stream
        ), mock.patch.object(line_bot, "default_chat_model", return_value="m"), mock.patch.object(
            line_bot.httpx, "post", return_value=mock.Mock(status_code=202)
        ):
            t1 = threading.Thread(target=line_bot.chat_reply_for, args=("Uabc123", "第一句"))
            t1.start()
            self.assertTrue(first_started.wait(timeout=5))
            t2 = threading.Thread(target=line_bot.chat_reply_for, args=("Uabc123", "第二句"))
            t2.start()
            t2.join(timeout=0.5)
            self.assertTrue(t2.is_alive(), "the follow-up ran before the first turn finished")
            release_first.set()
            t1.join(timeout=5)
            t2.join(timeout=5)
        self.assertEqual(histories["第一句"], [])
        self.assertEqual(histories["第二句"], ["第一句", "答一"])

    def test_different_chats_do_not_block_each_other(self) -> None:
        import threading

        with line_bot._chat_turn_lock("Ua"):
            done = threading.Event()
            def other_chat():
                with line_bot._chat_turn_lock("Ub"):
                    done.set()

            other = threading.Thread(target=other_chat)
            other.start()
            self.assertTrue(done.wait(timeout=1), "a different chat was blocked")
            other.join(timeout=1)

    def test_chat_lanes_are_refcounted_and_vanish_when_idle(self) -> None:
        # The registry holds only chats with a turn in flight: no cap, no prune pass, and a lane
        # handed to a waiting caller can never be dropped from under it.
        import threading

        with mock.patch.dict(line_bot._chat_lanes, clear=True):
            self.assertEqual(line_bot._chat_lanes, {})
            with line_bot._chat_turn_lock("Ua"):
                lane = line_bot._chat_lanes["Ua"]
                self.assertEqual(lane.refs, 1)
                waiter_in = threading.Event()

                def waiter():
                    waiter_in.set()
                    with line_bot._chat_turn_lock("Ua"):
                        pass

                t = threading.Thread(target=waiter)
                t.start()
                self.assertTrue(waiter_in.wait(timeout=1))
                t.join(timeout=0.2)
                self.assertTrue(t.is_alive())  # blocked behind the same chat's running turn
                self.assertEqual(line_bot._chat_lanes["Ua"].refs, 2)  # holder + waiter
                self.assertIs(line_bot._chat_lanes["Ua"], lane)  # same lane object, not replaced
            t.join(timeout=2)
            self.assertFalse(t.is_alive())
            self.assertEqual(line_bot._chat_lanes, {})  # last turn out removes the entry

    def test_same_chat_turns_run_in_arrival_order_not_wakeup_order(self) -> None:
        # Three follow-ups queue up behind a running turn; they must run 1, 2, 3 — the order they
        # ARRIVED — regardless of which waiter the OS wakes first (a plain Lock gives no such
        # guarantee). Arrival is forced by waiting for each waiter to take its ticket.
        import threading

        ran: list[int] = []
        with mock.patch.dict(line_bot._chat_lanes, clear=True):
            with line_bot._chat_turn_lock("Ua"):
                threads = []
                for i in (1, 2, 3):
                    def turn(i=i):
                        with line_bot._chat_turn_lock("Ua"):
                            ran.append(i)

                    t = threading.Thread(target=turn)
                    t.start()
                    threads.append(t)
                    # wait until this waiter holds its ticket before starting the next one
                    for _ in range(200):
                        if line_bot._chat_lanes["Ua"].next_ticket == i + 1:
                            break
                        threading.Event().wait(0.005)
                    self.assertEqual(line_bot._chat_lanes["Ua"].next_ticket, i + 1)
            for t in threads:
                t.join(timeout=2)
        self.assertEqual(ran, [1, 2, 3])

    def test_concurrent_turns_from_one_user_are_both_kept(self) -> None:
        # Two messages in flight at once: both read the same (empty) history; the one finishing
        # LAST must not erase the one finishing first. ``_remember`` appends under the lock.
        line_bot._remember("Uabc123", [{"role": "user", "content": "甲"}, {"role": "assistant", "content": "A"}], 1_000.0)
        line_bot._remember("Uabc123", [{"role": "user", "content": "乙"}, {"role": "assistant", "content": "B"}], 1_001.0)
        self.assertEqual(
            [m["content"] for m in line_bot._history["Uabc123"][1]], ["甲", "A", "乙", "B"]
        )

    def test_other_users_expired_history_is_evicted_on_write(self) -> None:
        self._chat("嗨", user_id="Uold", now=1_000.0)
        self._chat("嗨", user_id="Unew", now=1_000.0 + line_bot._HISTORY_TTL_S + 1)
        self.assertEqual(set(line_bot._history), {"Unew"})


class HandleEventsRoutingTests(unittest.TestCase):
    """``handle_events`` routes: summary keywords → summary; help keywords → help; else → chat."""

    def _handle(self, text: str, *, chat_configured: bool = True) -> str:
        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(chat_configured=chat_configured)
        ), mock.patch.object(
            line_bot, "summary_for_line_user", return_value=dict(_SUMMARY)
        ), mock.patch.object(
            line_bot, "chat_reply_for", return_value="LLM 說話"
        ) as chat:
            replies = line_bot.handle_events({"events": [_text_event(text)]})
        self.chat = chat
        return replies[0]["text"]

    def test_free_text_goes_to_the_coach_with_the_original_casing(self) -> None:
        self.assertEqual(self._handle("  How do I fix Knee Valgus?  "), "LLM 說話")
        self.chat.assert_called_once_with("Uabc123", "How do I fix Knee Valgus?", chat_id="Uabc123")

    def test_help_keywords_skip_the_llm(self) -> None:
        for word in ("help", "幫助", "？", "?"):
            self.assertIn("傳「摘要」", self._handle(word))
            self.chat.assert_not_called()

    def test_help_mentions_free_questions_only_when_chat_is_configured(self) -> None:
        self.assertIn("直接問我", self._handle("help", chat_configured=True))
        self.assertNotIn("直接問我", self._handle("help", chat_configured=False))

    def test_summary_keyword_in_a_group_never_posts_personal_data(self) -> None:
        event = _text_event("摘要")
        event["source"] = {"type": "group", "groupId": "Cgroup1", "userId": "Uabc123"}
        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(chat_configured=True)
        ), mock.patch.object(line_bot, "summary_for_line_user") as summary, mock.patch.object(
            line_bot, "chat_reply_for"
        ) as chat:
            replies = line_bot.handle_events({"events": [event]})
        summary.assert_not_called()
        chat.assert_not_called()
        self.assertIn("私訊", replies[0]["text"])
        self.assertNotIn("累積分析", replies[0]["text"])

    def test_summary_keywords_still_skip_the_llm(self) -> None:
        self.assertIn("你的訓練摘要", self._handle("摘要"))
        self.chat.assert_not_called()

    def test_free_text_is_keyed_by_the_chat_it_was_sent_in(self) -> None:
        event = _text_event("教練你好")
        event["source"] = {"type": "group", "groupId": "Cgroup1", "userId": "Uabc123"}
        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(chat_configured=True)
        ), mock.patch.object(line_bot, "chat_reply_for", return_value="hi") as chat:
            line_bot.handle_events({"events": [event]})
        chat.assert_called_once_with("Uabc123", "教練你好", chat_id="Cgroup1")
        event["source"] = {"type": "room", "roomId": "Rroom1", "userId": "Uabc123"}
        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(chat_configured=True)
        ), mock.patch.object(line_bot, "chat_reply_for", return_value="hi") as chat:
            line_bot.handle_events({"events": [event]})
        chat.assert_called_once_with("Uabc123", "教練你好", chat_id="Rroom1")

    def test_a_redelivered_event_is_answered_unless_already_seen(self) -> None:
        line_bot.clear_history()
        self.addCleanup(line_bot.clear_history)
        event = _text_event("膝蓋內夾怎麼辦")
        event["webhookEventId"] = "01ABC"
        event["deliveryContext"] = {"isRedelivery": True}
        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(chat_configured=True)
        ), mock.patch.object(line_bot, "chat_reply_for", return_value="回") as chat:
            # First sighting: the token may still be live, so it is answered — redelivery or not.
            self.assertEqual(len(line_bot.handle_events({"events": [event]})), 1)
            # The same event again (a genuine duplicate delivery): skipped, LLM not re-run.
            self.assertEqual(line_bot.handle_events({"events": [event]}), [])
        chat.assert_called_once()

    def test_seen_event_ids_expire_and_the_store_is_bounded(self) -> None:
        line_bot.clear_history()
        self.addCleanup(line_bot.clear_history)
        self.assertTrue(line_bot._first_sighting("e1", 0.0))
        self.assertFalse(line_bot._first_sighting("e1", 1.0))
        self.assertTrue(line_bot._first_sighting("e1", line_bot._SEEN_EVENTS_TTL_S + 1.0))
        self.assertTrue(line_bot._first_sighting(None, 0.0))  # no id → always new
        self.assertTrue(line_bot._first_sighting("", 0.0))
        # Full store: expired ids are pruned; if still full, it is reset rather than grown.
        with mock.patch.object(line_bot, "_SEEN_EVENTS_MAX", 3):
            line_bot.clear_history()
            for i in range(3):
                line_bot._first_sighting(f"old{i}", 0.0)
            self.assertTrue(line_bot._first_sighting("new", line_bot._SEEN_EVENTS_TTL_S + 1.0))
            self.assertEqual(set(line_bot._seen_events), {"new"})
            for i in range(2):
                line_bot._first_sighting(f"fresh{i}", line_bot._SEEN_EVENTS_TTL_S + 2.0)
            self.assertTrue(line_bot._first_sighting("overflow", line_bot._SEEN_EVENTS_TTL_S + 3.0))
            self.assertEqual(set(line_bot._seen_events), {"overflow"})

    def test_empty_text_returns_help(self) -> None:
        self.assertIn("傳「摘要」", self._handle("   "))
        self.chat.assert_not_called()
