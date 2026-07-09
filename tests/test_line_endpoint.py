"""Tests for the LINE bot integration (``/api/line/webhook`` + ``/api/line/link-code``).

Nothing real is hit: the LINE Messaging API (``line_client.reply_message``), the Supabase
service-role store (``line_store._service_client``), and the LLM (``chat_service._stream_completion``)
are all patched at their seams — mirroring ``test_chat_endpoint.py``. The suite locks in the pieces
that make the feature correct and safe:

* **signature is the security boundary** — the webhook verifies the ``X-Line-Signature`` HMAC over the
  raw body and 400s a forgery before any processing; an unconfigured server just acks.
* **the linking flow** — a code redeems into a self-contained binding (code deleted, thread reset); an
  unknown/expired code is rejected; a bound user's questions are answered grounded in the snapshot.
* **groundedness carries over** — ``answer_once`` reuses the same grounded system prompt as the web
  chat and never sends a blank reply.
* **failures stay contained** — an LLM error replies a fallback without persisting the failed turn, and
  one bad event never aborts the batch.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import BackgroundTasks, HTTPException

from backend.app.auth import CurrentUser
from backend.app.routers import line as line_router
from backend.app.services import chat as chat_service
from backend.app.services import line_client, line_store
from backend.app.settings import Settings

_USER = CurrentUser(id="user-1", token="tok", email="u@example.com")

_CTX = {
    "video_id": "vid1",
    "view_type": "front",
    "fault_count": 1,
    "faults": [{"fault_name": "knees_inward", "corrections": ["drive knees out"]}],
}


class LineConfiguredTests(unittest.TestCase):
    """``line_configured`` gates the whole integration: every piece must be present or it's off."""

    _FULL = dict(
        line_channel_secret="s",
        line_channel_access_token="t",
        supabase_url="https://p.supabase.co",
        supabase_service_role_key="svc",
        llm_api_key="k",  # chat_configured
    )

    def test_true_when_everything_is_present(self) -> None:
        self.assertTrue(Settings(**self._FULL).line_configured)

    def test_false_when_any_piece_missing(self) -> None:
        for drop in self._FULL:
            cfg = {**self._FULL, drop: ""}
            self.assertFalse(Settings(**cfg).line_configured, msg=f"missing {drop}")


# --------------------------------------------------------------------- line_client: signature


class VerifySignatureTests(unittest.TestCase):
    def _sig(self, secret: str, body: bytes) -> str:
        return base64.b64encode(
            hmac.new(secret.encode(), body, hashlib.sha256).digest()
        ).decode()

    def test_valid_signature_passes(self) -> None:
        body = b'{"events":[]}'
        self.assertTrue(line_client.verify_signature("s3cr3t", body, self._sig("s3cr3t", body)))

    def test_wrong_secret_fails(self) -> None:
        body = b'{"events":[]}'
        self.assertFalse(line_client.verify_signature("other", body, self._sig("s3cr3t", body)))

    def test_tampered_body_fails(self) -> None:
        sig = self._sig("s3cr3t", b'{"events":[]}')
        self.assertFalse(line_client.verify_signature("s3cr3t", b'{"events":[1]}', sig))

    def test_missing_secret_or_signature_fails_closed(self) -> None:
        self.assertFalse(line_client.verify_signature("", b"x", "sig"))
        self.assertFalse(line_client.verify_signature("s", b"x", None))


class ReplyMessageTests(unittest.TestCase):
    def test_posts_reply_payload_with_bearer(self) -> None:
        captured: dict = {}

        def fake_post(url, headers, json, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return SimpleNamespace(raise_for_status=lambda: None)

        with mock.patch("httpx.post", fake_post):
            line_client.reply_message(access_token="tok-x", reply_token="rt", text="hi there")

        self.assertEqual(captured["url"], line_client._REPLY_URL)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer tok-x")
        self.assertEqual(captured["json"]["replyToken"], "rt")
        self.assertEqual(captured["json"]["messages"], [{"type": "text", "text": "hi there"}])

    def test_transport_failure_becomes_runtime_error(self) -> None:
        with mock.patch("httpx.post", side_effect=Exception("boom")):
            with self.assertRaises(RuntimeError):
                line_client.reply_message(access_token="t", reply_token="rt", text="x")

    def test_push_message_targets_the_user_id(self) -> None:
        captured: dict = {}

        def fake_post(url, headers, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return SimpleNamespace(raise_for_status=lambda: None)

        with mock.patch("httpx.post", fake_post):
            line_client.push_message(access_token="tok-x", to="L1", text="hello")

        self.assertEqual(captured["url"], line_client._PUSH_URL)
        self.assertEqual(captured["json"]["to"], "L1")
        self.assertEqual(captured["json"]["messages"], [{"type": "text", "text": "hello"}])


# --------------------------------------------------------------------- chat_service.answer_once


class AnswerOnceTests(unittest.TestCase):
    def test_joins_chunks_and_prepends_grounded_system_prompt(self) -> None:
        seen: dict = {}

        def fake_stream(messages, model):
            seen["messages"] = messages
            yield "Drive "
            yield "knees out."

        with mock.patch.object(chat_service, "_stream_completion", fake_stream):
            out = chat_service.answer_once(
                messages=[{"role": "user", "content": "why?"}], context=_CTX, model="m"
            )

        self.assertEqual(out, "Drive knees out.")
        self.assertEqual(seen["messages"][0]["role"], "system")
        self.assertIn("knees_inward", seen["messages"][0]["content"])  # grounded

    def test_empty_completion_raises(self) -> None:
        def fake_stream(messages, model):
            yield "   "

        with mock.patch.object(chat_service, "_stream_completion", fake_stream):
            with self.assertRaises(RuntimeError):
                chat_service.answer_once(messages=[{"role": "user", "content": "x"}], context=_CTX, model="m")

    def test_truncates_to_max_chars(self) -> None:
        def fake_stream(messages, model):
            yield "a" * 10_000

        with mock.patch.object(chat_service, "_stream_completion", fake_stream):
            out = chat_service.answer_once(
                messages=[{"role": "user", "content": "x"}], context=_CTX, model="m", max_chars=100
            )
        self.assertEqual(len(out), 100)


# --------------------------------------------------------------------- line_store (fake Supabase)


class _FakeQuery:
    """A fluent Supabase query stub: records writes/filters, returns the table's preset ``data``."""

    def __init__(self, table: "_FakeTable") -> None:
        self.table = table

    def select(self, *a, **k):
        return self

    def insert(self, payload):
        self.table.inserted.append(payload)
        return self

    def upsert(self, payload, **k):
        self.table.upserted.append((payload, k))
        return self

    def update(self, payload):
        self.table.updated.append(payload)
        return self

    def delete(self):
        self.table.deleted = True
        return self

    def eq(self, col, val):
        self.table.filters.append((col, val))
        return self

    def limit(self, n):
        return self

    def execute(self):
        return SimpleNamespace(data=list(self.table.data))


class _FakeTable:
    def __init__(self, data=None) -> None:
        self.data = data or []
        self.inserted: list = []
        self.upserted: list = []
        self.updated: list = []
        self.filters: list = []
        self.deleted = False


class _FakeClient:
    def __init__(self, **tables: _FakeTable) -> None:
        self.tables = tables

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self.tables[name])


class LineStoreTests(unittest.TestCase):
    def test_create_link_code_inserts_and_returns_a_code(self) -> None:
        codes = _FakeTable()
        client = _FakeClient(line_link_codes=codes)
        with mock.patch.object(line_store, "_service_client", return_value=client):
            code = line_store.create_link_code(user_id="u1", video_id="vid1", context=_CTX)

        self.assertEqual(len(code), line_store._CODE_LENGTH)
        self.assertTrue(all(ch in line_store._CODE_ALPHABET for ch in code))
        row = codes.inserted[0]
        self.assertEqual(row["user_id"], "u1")
        self.assertEqual(row["context"], _CTX)
        self.assertIn("expires_at", row)

    def test_redeem_valid_code_creates_binding_and_deletes_code(self) -> None:
        codes = _FakeTable(
            data=[
                {
                    "code": "ABC234",
                    "user_id": "u1",
                    "video_id": "vid1",
                    "context": _CTX,
                    "expires_at": "2999-01-01T00:00:00+00:00",  # far future
                }
            ]
        )
        bindings = _FakeTable()
        client = _FakeClient(line_link_codes=codes, line_bindings=bindings)
        with mock.patch.object(line_store, "_service_client", return_value=client):
            binding = line_store.redeem_link_code(line_user_id="L1", code="abc234")  # lower-case ok

        self.assertIsNotNone(binding)
        self.assertEqual(binding["user_id"], "u1")
        self.assertEqual(binding["context"], _CTX)
        self.assertEqual(binding["messages"], [])  # thread reset on (re)bind
        self.assertEqual(bindings.upserted[0][0]["line_user_id"], "L1")
        self.assertTrue(codes.deleted)  # one-time use

    def test_redeem_unknown_code_returns_none(self) -> None:
        client = _FakeClient(line_link_codes=_FakeTable(data=[]), line_bindings=_FakeTable())
        with mock.patch.object(line_store, "_service_client", return_value=client):
            self.assertIsNone(line_store.redeem_link_code(line_user_id="L1", code="NOPE22"))

    def test_redeem_expired_code_returns_none_and_deletes(self) -> None:
        codes = _FakeTable(
            data=[
                {
                    "code": "OLD234",
                    "user_id": "u1",
                    "video_id": "vid1",
                    "context": _CTX,
                    "expires_at": "2000-01-01T00:00:00+00:00",  # already past
                }
            ]
        )
        client = _FakeClient(line_link_codes=codes, line_bindings=_FakeTable())
        with mock.patch.object(line_store, "_service_client", return_value=client):
            self.assertIsNone(line_store.redeem_link_code(line_user_id="L1", code="OLD234"))
        self.assertTrue(codes.deleted)  # stale code tidied

    def test_get_binding_and_save_messages(self) -> None:
        bindings = _FakeTable(
            data=[{"user_id": "u1", "video_id": "vid1", "context": _CTX, "messages": []}]
        )
        client = _FakeClient(line_bindings=bindings)
        with mock.patch.object(line_store, "_service_client", return_value=client):
            got = line_store.get_binding(line_user_id="L1")
            self.assertEqual(got["user_id"], "u1")
            line_store.save_binding_messages(
                line_user_id="L1", messages=[{"role": "user", "content": "hi"}]
            )
        self.assertEqual(bindings.updated[0]["messages"], [{"role": "user", "content": "hi"}])

    def test_get_binding_none_when_unlinked(self) -> None:
        client = _FakeClient(line_bindings=_FakeTable(data=[]))
        with mock.patch.object(line_store, "_service_client", return_value=client):
            self.assertIsNone(line_store.get_binding(line_user_id="L1"))

    def test_redeem_blank_code_returns_none(self) -> None:
        with mock.patch.object(line_store, "_service_client") as client:
            self.assertIsNone(line_store.redeem_link_code(line_user_id="L1", code="   "))
        client.assert_not_called()  # short-circuits before any DB call

    def test_service_client_uses_the_service_role_key(self) -> None:
        # The store must authenticate with SUPABASE_SERVICE_ROLE_KEY (not the anon key) — it bypasses
        # RLS to reach the backend-owned line_* tables. Inject a fake supabase module so the deferred
        # import resolves without the package installed.
        captured: dict = {}

        def fake_create_client(url, key):
            captured["url"] = url
            captured["key"] = key
            return "client-obj"

        fake_supabase = types.ModuleType("supabase")
        fake_supabase.create_client = fake_create_client
        fake_settings = SimpleNamespace(
            supabase_url="https://proj.supabase.co", supabase_service_role_key="svc-role-key"
        )
        with mock.patch.dict("sys.modules", {"supabase": fake_supabase}), mock.patch(
            "backend.app.settings.get_settings", return_value=fake_settings
        ):
            client = line_store._service_client()

        self.assertEqual(client, "client-obj")
        self.assertEqual(captured["url"], "https://proj.supabase.co")
        self.assertEqual(captured["key"], "svc-role-key")

    def test_parse_ts_handles_non_string_garbage_and_naive(self) -> None:
        self.assertIsNone(line_store._parse_ts(None))  # non-string
        self.assertIsNone(line_store._parse_ts("not-a-timestamp"))  # unparseable
        aware = line_store._parse_ts("2020-01-01T00:00:00")  # naive -> assumed UTC
        self.assertIsNotNone(aware.tzinfo)


# --------------------------------------------------------------------- router: code extraction


class ExtractLinkCodeTests(unittest.TestCase):
    def test_keyword_prefix_forms(self) -> None:
        self.assertEqual(line_router._extract_link_code("連結 ABC234"), "ABC234")
        self.assertEqual(line_router._extract_link_code("綁定ABC234"), "ABC234")
        self.assertEqual(line_router._extract_link_code("link abc234"), "abc234")
        self.assertEqual(line_router._extract_link_code("BIND XYZ234"), "XYZ234")

    def test_bare_code_is_recognised(self) -> None:
        self.assertEqual(line_router._extract_link_code("ABC234"), "ABC234")

    def test_normal_question_is_not_a_code(self) -> None:
        self.assertIsNone(line_router._extract_link_code("為什麼我的膝蓋會內夾？"))
        self.assertIsNone(line_router._extract_link_code("How do I fix my depth?"))

    def test_bare_keyword_without_code_returns_none(self) -> None:
        self.assertIsNone(line_router._extract_link_code("連結"))


# --------------------------------------------------------------------- router: event handling


def _text_event(text: str, user_id: str = "L1", reply_token: str = "rt") -> dict:
    return {
        "type": "message",
        "replyToken": reply_token,
        "source": {"userId": user_id},
        "message": {"type": "text", "text": text},
    }


class HandleEventTests(unittest.TestCase):
    def setUp(self) -> None:
        # Every handler path ends in a reply; capture it and give _reply an access token.
        self.replies: list[tuple[str, str]] = []
        self._patches = [
            mock.patch.object(
                line_router,
                "get_settings",
                return_value=SimpleNamespace(line_channel_access_token="tok"),
            ),
            mock.patch.object(
                line_client,
                "reply_message",
                side_effect=lambda **k: self.replies.append((k["reply_token"], k["text"])),
            ),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def test_follow_event_sends_welcome(self) -> None:
        line_router._handle_event({"type": "follow", "replyToken": "rt", "source": {"userId": "L1"}})
        self.assertEqual(self.replies[0][1], line_router._MSG_WELCOME)

    def test_non_text_message_is_nudged(self) -> None:
        line_router._handle_event(
            {"type": "message", "replyToken": "rt", "source": {"userId": "L1"},
             "message": {"type": "sticker"}}
        )
        self.assertEqual(self.replies[0][1], line_router._MSG_NON_TEXT)

    def test_code_message_redeems_and_confirms(self) -> None:
        with mock.patch.object(line_store, "redeem_link_code", return_value={"user_id": "u1"}) as redeem:
            line_router._handle_event(_text_event("連結 ABC234"))
        redeem.assert_called_once()
        self.assertEqual(self.replies[0][1], line_router._MSG_LINK_OK)

    def test_invalid_code_reports_failure(self) -> None:
        with mock.patch.object(line_store, "redeem_link_code", return_value=None):
            line_router._handle_event(_text_event("ABC234"))
        self.assertEqual(self.replies[0][1], line_router._MSG_LINK_INVALID)

    def test_unbound_user_gets_link_help(self) -> None:
        with mock.patch.object(line_store, "get_binding", return_value=None):
            line_router._handle_event(_text_event("為什麼膝蓋會內夾？"))
        self.assertEqual(self.replies[0][1], line_router._MSG_LINK_HELP)

    def test_bound_user_question_is_answered_and_persisted(self) -> None:
        binding = {"user_id": "u1", "context": _CTX, "messages": []}
        saved: dict = {}
        with mock.patch.object(line_store, "get_binding", return_value=binding), \
            mock.patch.object(chat_service, "answer_once", return_value="Drive your knees out."), \
            mock.patch.object(line_router, "default_chat_model", return_value="m"), \
            mock.patch.object(
                line_store, "save_binding_messages",
                side_effect=lambda **k: saved.update(k),
            ):
            line_router._handle_event(_text_event("為什麼膝蓋會內夾？"))

        self.assertEqual(self.replies[0][1], "Drive your knees out.")
        # The full turn (user + assistant) is persisted for the next message's context.
        self.assertEqual(saved["messages"][-2], {"role": "user", "content": "為什麼膝蓋會內夾？"})
        self.assertEqual(saved["messages"][-1], {"role": "assistant", "content": "Drive your knees out."})

    def test_llm_error_replies_fallback_without_persisting(self) -> None:
        binding = {"user_id": "u1", "context": _CTX, "messages": []}
        with mock.patch.object(line_store, "get_binding", return_value=binding), \
            mock.patch.object(chat_service, "answer_once", side_effect=RuntimeError("down")), \
            mock.patch.object(line_router, "default_chat_model", return_value="m"), \
            mock.patch.object(line_store, "save_binding_messages") as save:
            line_router._handle_event(_text_event("why?"))

        self.assertEqual(self.replies[0][1], line_router._MSG_LLM_ERROR)
        save.assert_not_called()  # a failed turn is not stored

    def test_ignored_event_types_send_nothing(self) -> None:
        # An unfollow (no reply token, not a message) must be silently ignored.
        line_router._handle_event({"type": "unfollow", "source": {"userId": "L1"}})
        self.assertEqual(self.replies, [])

    def test_text_event_without_user_or_text_is_ignored(self) -> None:
        # A whitespace-only text (or a missing userId) is dropped before any store/LLM call.
        line_router._handle_event(_text_event("   "))
        self.assertEqual(self.replies, [])

    def test_reply_swallows_transport_failure(self) -> None:
        # A failed LINE send must not propagate out of a background task (which would 500 back at LINE).
        with mock.patch.object(line_client, "reply_message", side_effect=RuntimeError("down")):
            line_router._reply("rt", "hi")  # must not raise


class ProcessEventsTests(unittest.TestCase):
    def test_one_bad_event_does_not_abort_the_batch(self) -> None:
        handled: list = []

        def fake_handle(event):
            if event["type"] == "boom":
                raise ValueError("bad")
            handled.append(event["type"])

        with mock.patch.object(line_router, "_handle_event", side_effect=fake_handle):
            line_router._process_events([{"type": "boom"}, {"type": "ok"}])
        self.assertEqual(handled, ["ok"])  # the good event still ran


# --------------------------------------------------------------------- router: webhook + link-code


class _FakeRequest:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def body(self) -> bytes:
        return self._body


def _sign(secret: str, body: bytes) -> str:
    return base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()


class WebhookRouteTests(unittest.TestCase):
    def _run(self, request, bg, sig):
        return asyncio.run(line_router.webhook(request, bg, x_line_signature=sig))

    def test_unconfigured_acks_without_processing(self) -> None:
        bg = BackgroundTasks()
        with mock.patch.object(
            line_router, "get_settings", return_value=SimpleNamespace(line_configured=False)
        ):
            out = self._run(_FakeRequest(b'{"events":[]}'), bg, "sig")
        self.assertEqual(out, {})
        self.assertEqual(len(bg.tasks), 0)

    def test_bad_signature_is_rejected(self) -> None:
        bg = BackgroundTasks()
        body = b'{"events":[{}]}'
        with mock.patch.object(
            line_router,
            "get_settings",
            return_value=SimpleNamespace(line_configured=True, line_channel_secret="s3cr3t"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                self._run(_FakeRequest(body), bg, "wrong-signature")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_valid_signature_schedules_processing(self) -> None:
        bg = BackgroundTasks()
        body = json.dumps({"events": [_text_event("hi")]}).encode()
        with mock.patch.object(
            line_router,
            "get_settings",
            return_value=SimpleNamespace(line_configured=True, line_channel_secret="s3cr3t"),
        ):
            out = self._run(_FakeRequest(body), bg, _sign("s3cr3t", body))
        self.assertEqual(out, {})
        self.assertEqual(len(bg.tasks), 1)  # events queued for background handling
        self.assertIs(bg.tasks[0].func, line_router._process_events)

    def test_valid_signature_but_malformed_body_acks_without_scheduling(self) -> None:
        # A body that signs correctly but isn't a JSON object (here a bare int) is swallowed: nothing
        # to dispatch, and we still ack 200 so LINE doesn't retry.
        bg = BackgroundTasks()
        body = b"123"  # json.loads -> int -> .get(...) raises AttributeError
        with mock.patch.object(
            line_router,
            "get_settings",
            return_value=SimpleNamespace(line_configured=True, line_channel_secret="s3cr3t"),
        ):
            out = self._run(_FakeRequest(body), bg, _sign("s3cr3t", body))
        self.assertEqual(out, {})
        self.assertEqual(len(bg.tasks), 0)

    def test_valid_signature_but_no_events_schedules_nothing(self) -> None:
        bg = BackgroundTasks()
        body = json.dumps({"events": []}).encode()
        with mock.patch.object(
            line_router,
            "get_settings",
            return_value=SimpleNamespace(line_configured=True, line_channel_secret="s3cr3t"),
        ):
            out = self._run(_FakeRequest(body), bg, _sign("s3cr3t", body))
        self.assertEqual(out, {})
        self.assertEqual(len(bg.tasks), 0)


class LinkCodeRouteTests(unittest.TestCase):
    def test_503_when_not_configured(self) -> None:
        body = line_router.LinkCodeRequest(context=_CTX)
        with mock.patch.object(
            line_router, "get_settings", return_value=SimpleNamespace(line_configured=False)
        ):
            with self.assertRaises(HTTPException) as ctx:
                line_router.create_link_code(body, user=_USER)
        self.assertEqual(ctx.exception.status_code, 503)

    def test_mints_code_for_authenticated_user(self) -> None:
        body = line_router.LinkCodeRequest(context=_CTX)
        captured: dict = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return "ABC234"

        with mock.patch.object(
            line_router, "get_settings", return_value=SimpleNamespace(line_configured=True)
        ), mock.patch.object(line_store, "create_link_code", side_effect=fake_create):
            resp = line_router.create_link_code(body, user=_USER)

        self.assertEqual(resp.code, "ABC234")
        self.assertEqual(captured["user_id"], "user-1")  # bound to the caller
        self.assertEqual(captured["video_id"], "vid1")  # pulled from the context blob


if __name__ == "__main__":
    unittest.main()
