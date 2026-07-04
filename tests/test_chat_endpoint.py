"""Tests for the grounded conversational-coaching endpoint (``POST /api/chat``).

The network (OpenRouter) is never hit: service tests patch ``httpx.stream`` or the
``_stream_completion`` seam, and router tests call the coroutine directly with a stub user —
mirroring ``test_analyze_endpoint.py``. They lock in the things that matter for this feature:

* **groundedness** — the server-built system prompt carries the analysis facts (fault names +
  retrieved corrections) and the honesty constraint that forbids inventing anything,
* the **request contract** — chat is gated + configured (503 without a key) and the last turn must
  be the user's (422 otherwise), both enforced *before* the stream opens, and
* the **stream contract** (v2) — the endpoint emits SSE ``delta``/``done`` frames on success and an
  in-band ``error`` frame on any mid-stream failure or empty completion (never a post-stream HTTP
  code, since the 200 is already committed once the stream starts).
"""

from __future__ import annotations

import asyncio
import types
import unittest
from unittest import mock

from fastapi import HTTPException

from backend.app import settings as app_settings
from backend.app.auth import CurrentUser
from backend.app.routers import chat as chat_router
from backend.app.services import chat as chat_service

_USER = CurrentUser(id="user-1", token="tok", email="u@example.com")

# A representative fault-grounding context (one detected fault + its retrieved knowledge).
_FAULT_CTX = {
    "video_id": "vid1",
    "view_type": "front",
    "view_confidence": 0.91,
    "fault_count": 1,
    "quality": {"lower_body_visibility_mean": 0.88},
    "faults": [
        {
            "fault_name": "knees_inward",
            "phase": "descent",
            "severity": 0.8,
            "start_time": 1.2,
            "end_time": 1.8,
            "evidence": "knee_valgus_ratio 0.82",
            "causes": ["weak glute medius"],
            "risks": ["ACL strain"],
            "corrections": ["drive knees out over toes"],
            "rag_snippet": "cue: band around knees",
        }
    ],
}
_CLEAN_CTX = {"view_type": "side", "fault_count": 0, "quality": {}, "faults": []}


def _body(messages, context) -> chat_router.ChatRequest:
    return chat_router.ChatRequest(messages=messages, context=context)


async def _collect(resp) -> str:
    """Drain a StreamingResponse's body into one string (frames are str before encoding)."""
    out: list[str] = []
    async for chunk in resp.body_iterator:
        out.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(out)


# --------------------------------------------------------------------- service: prompt


class SystemPromptTests(unittest.TestCase):
    def test_embeds_fault_name_and_retrieved_corrections(self) -> None:
        prompt = chat_service._build_system_prompt(_FAULT_CTX)
        self.assertIn("knees_inward", prompt)
        self.assertIn("drive knees out over toes", prompt)  # the corrective cue
        self.assertIn("weak glute medius", prompt)  # the cause
        self.assertIn("ACL strain", prompt)  # the risk
        self.assertIn("knee_valgus_ratio 0.82", prompt)  # the evidence
        self.assertIn("band around knees", prompt)  # the rag snippet
        self.assertIn("front", prompt)  # camera view
        self.assertIn("0.88", prompt)  # lower-body visibility

    def test_carries_the_honesty_constraint(self) -> None:
        prompt = chat_service._build_system_prompt(_FAULT_CTX)
        self.assertIn("ONLY", prompt)
        self.assertIn("Do NOT invent", prompt)

    def test_permits_light_markdown_without_relaxing_grounding(self) -> None:
        # v2: the coach may format with light markdown, but the grounding/honesty rules must remain.
        prompt = chat_service._build_system_prompt(_FAULT_CTX)
        self.assertIn("markdown", prompt.lower())
        self.assertIn("ONLY", prompt)  # grounding constraint intact
        self.assertIn("Do NOT invent", prompt)  # honesty constraint intact

    def test_clean_rep_has_no_fault_list_but_flags_clean(self) -> None:
        prompt = chat_service._build_system_prompt(_CLEAN_CTX)
        self.assertIn("CLEAN REP", prompt)
        self.assertNotIn("DETECTED FAULTS", prompt)
        self.assertIn("Faults detected: 0", prompt)

    def test_fmt_list_handles_empty_and_values(self) -> None:
        self.assertEqual(chat_service._fmt_list([]), "—")
        self.assertEqual(chat_service._fmt_list(["a", "", "b"]), "a, b")

    def test_fault_without_evidence_or_snippet_still_renders(self) -> None:
        ctx = {
            "fault_count": 1,
            "faults": [{"fault_name": "shallow_depth", "corrections": ["sit lower"]}],
        }
        prompt = chat_service._build_system_prompt(ctx)
        self.assertIn("shallow_depth", prompt)
        self.assertIn("sit lower", prompt)
        self.assertNotIn("evidence:", prompt)  # no evidence line when the fault carries none
        self.assertNotIn("reference:", prompt)  # no rag-snippet line either


# --------------------------------------------------------------------- service: answer_stream


class AnswerStreamTests(unittest.TestCase):
    def test_yields_deltas_then_done_and_prepends_system_prompt(self) -> None:
        seen: dict[str, object] = {}

        def fake_stream(messages, model):
            seen["messages"] = messages
            seen["model"] = model
            yield "Drive "
            yield "knees out"

        history = [{"role": "user", "content": "why did my knees cave?"}]
        with mock.patch.object(chat_service, "_stream_completion", fake_stream):
            frames = "".join(
                chat_service.answer_stream(messages=history, context=_FAULT_CTX, model="vendor/m-x")
            )

        self.assertIn("event: delta", frames)
        self.assertIn("Drive ", frames)
        self.assertIn("knees out", frames)
        self.assertIn("event: done", frames)
        self.assertIn("vendor/m-x", frames)  # the actually-used model rides the done frame
        self.assertNotIn("event: error", frames)
        # The chosen model is passed straight to the transport.
        self.assertEqual(seen["model"], "vendor/m-x")
        # The grounded system prompt is prepended; the user history follows untouched.
        sent = seen["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("knees_inward", sent[0]["content"])
        self.assertEqual(sent[1:], history)

    def test_midstream_failure_yields_error_frame_and_no_done(self) -> None:
        def fake_stream(messages, model):
            yield "partial answer"
            raise RuntimeError("OpenRouter request failed: connection reset")

        with mock.patch.object(chat_service, "_stream_completion", fake_stream):
            frames = "".join(
                chat_service.answer_stream(
                    messages=[{"role": "user", "content": "hi"}], context=_FAULT_CTX, model="m"
                )
            )

        self.assertIn("partial answer", frames)  # deltas already flushed are kept
        self.assertIn("event: error", frames)
        self.assertIn("connection reset", frames)
        self.assertNotIn("event: done", frames)

    def test_empty_completion_yields_error_not_done(self) -> None:
        # The v1 empty-completion invariant survives into streaming: a blank accumulation must emit
        # an error, never a done, so the client never keeps an empty assistant turn.
        def fake_stream(messages, model):
            yield "   "  # whitespace only -> strips to empty

        with mock.patch.object(chat_service, "_stream_completion", fake_stream):
            frames = "".join(
                chat_service.answer_stream(
                    messages=[{"role": "user", "content": "hi"}], context=_CLEAN_CTX, model="m"
                )
            )

        self.assertIn("event: error", frames)
        self.assertIn("empty", frames.lower())
        self.assertNotIn("event: done", frames)


# --------------------------------------------------------- service: OpenRouter SSE transport


class StreamCompletionTests(unittest.TestCase):
    def _settings(self):
        return types.SimpleNamespace(
            openrouter_api_key="sk-or-test",
            openrouter_base_url="https://openrouter.ai/api/v1",
        )

    def test_parses_content_deltas_from_openai_sse(self) -> None:
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.iter_lines.return_value = iter(
            [
                ": keep-alive ping",  # SSE comment -> skip
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                "data: not-json",  # malformed -> skip
                'data: {"choices":[{"delta":{}}]}',  # no content -> skip
                'data: {"choices":[{"delta":{"content":" there"}}]}',
                "data: [DONE]",
                'data: {"choices":[{"delta":{"content":"unreached"}}]}',  # after [DONE] -> not read
            ]
        )
        cm = mock.MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with mock.patch.object(
            chat_service, "get_settings", return_value=self._settings()
        ), mock.patch("httpx.stream", return_value=cm) as stream:
            chunks = list(
                chat_service._stream_completion(
                    [{"role": "user", "content": "hi"}], "minimax/minimax-m3"
                )
            )

        self.assertEqual(chunks, ["Hello", " there"])
        _, kwargs = stream.call_args
        self.assertIs(kwargs["json"]["stream"], True)
        self.assertEqual(kwargs["json"]["model"], "minimax/minimax-m3")  # the passed model is sent
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-or-test")

    def test_stream_ending_without_done_terminator_exhausts_cleanly(self) -> None:
        # Some upstreams just close the connection instead of sending a final ``data: [DONE]``; the
        # loop must exit by exhausting iter_lines, yielding everything seen so far.
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.iter_lines.return_value = iter(
            ['data: {"choices":[{"delta":{"content":"Hi"}}]}']  # no [DONE]
        )
        cm = mock.MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with mock.patch.object(
            chat_service, "get_settings", return_value=self._settings()
        ), mock.patch("httpx.stream", return_value=cm):
            chunks = list(chat_service._stream_completion([{"role": "user", "content": "hi"}], "m"))
        self.assertEqual(chunks, ["Hi"])

    def test_transport_failure_becomes_runtime_error(self) -> None:
        with mock.patch.object(
            chat_service, "get_settings", return_value=self._settings()
        ), mock.patch("httpx.stream", side_effect=Exception("connection reset")):
            with self.assertRaises(RuntimeError):
                list(chat_service._stream_completion([{"role": "user", "content": "hi"}], "m"))


# --------------------------------------------------------------------- router contract


class ChatRouterTests(unittest.TestCase):
    @staticmethod
    def _run(body):
        return asyncio.run(chat_router.chat(body, user=_USER))

    def test_503_when_chat_not_configured(self) -> None:
        with mock.patch.object(
            chat_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=False)
        ):
            with self.assertRaises(HTTPException) as ctx:
                self._run(_body([{"role": "user", "content": "hi"}], _FAULT_CTX))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_422_when_last_message_is_not_user(self) -> None:
        body = _body(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            _FAULT_CTX,
        )
        with mock.patch.object(
            chat_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ):
            with self.assertRaises(HTTPException) as ctx:
                self._run(body)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_returns_event_stream_delegating_to_service(self) -> None:
        captured: dict = {}

        def fake_answer_stream(*, messages, context, model):
            captured["messages"] = messages
            captured["context"] = context
            captured["model"] = model
            yield 'event: delta\ndata: {"text": "drive knees out"}\n\n'
            yield 'event: done\ndata: {"model": "minimax/minimax-m3"}\n\n'

        body = chat_router.ChatRequest(
            messages=[{"role": "user", "content": "fix my knees?"}],
            context=_FAULT_CTX,
            model="minimax/minimax-m3",  # an allow-listed selection
        )
        with mock.patch.object(
            chat_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ), mock.patch.object(chat_service, "answer_stream", fake_answer_stream):
            resp = self._run(body)
            out = asyncio.run(_collect(resp))

        self.assertEqual(resp.media_type, "text/event-stream")
        self.assertIn("event: delta", out)
        self.assertIn("drive knees out", out)
        self.assertIn("event: done", out)
        # The service received the conversation + grounding blob + the resolved (allow-listed) model.
        self.assertEqual(captured["messages"][-1]["content"], "fix my knees?")
        self.assertEqual(captured["context"]["faults"][0]["fault_name"], "knees_inward")
        self.assertEqual(captured["model"], "minimax/minimax-m3")


def _fake_models(models: str):
    return types.SimpleNamespace(openrouter_models=models)


class ChatModelsCatalogTests(unittest.TestCase):
    def test_parses_env_list_and_labels_known_slugs(self) -> None:
        s = _fake_models("deepseek/deepseek-v4-flash, xiaomi/mimo-v2.5 ,minimax/minimax-m3")  # spaces ok
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            cat = app_settings.chat_models()
        self.assertEqual(
            [m["id"] for m in cat],
            ["deepseek/deepseek-v4-flash", "xiaomi/mimo-v2.5", "minimax/minimax-m3"],
        )
        self.assertEqual(cat[0]["label"], "DeepSeek V4 Flash")  # curated label

    def test_dedupes_and_labels_unknown_slug_as_its_id(self) -> None:
        s = _fake_models("custom/model,custom/model,minimax/minimax-m3")
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            cat = app_settings.chat_models()
        self.assertEqual([m["id"] for m in cat], ["custom/model", "minimax/minimax-m3"])
        self.assertEqual(cat[0]["label"], "custom/model")  # unknown slug -> raw id as label

    def test_blank_list_falls_back_to_one_built_in_model(self) -> None:
        s = _fake_models("  , , ")  # misconfigured to empty
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            cat = app_settings.chat_models()
        self.assertEqual(len(cat), 1)  # never empty
        self.assertTrue(cat[0]["id"])


class ResolveChatModelTests(unittest.TestCase):
    def test_offered_model_is_returned(self) -> None:
        s = _fake_models("deepseek/deepseek-v4-flash,minimax/minimax-m3")
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            self.assertEqual(
                app_settings.resolve_chat_model("minimax/minimax-m3"), "minimax/minimax-m3"
            )

    def test_unknown_or_missing_falls_back_to_the_first_model(self) -> None:
        s = _fake_models("deepseek/deepseek-v4-flash,minimax/minimax-m3")
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            self.assertEqual(app_settings.default_chat_model(), "deepseek/deepseek-v4-flash")
            self.assertEqual(
                app_settings.resolve_chat_model("evil/expensive-model"),
                "deepseek/deepseek-v4-flash",
            )
            self.assertEqual(app_settings.resolve_chat_model(None), "deepseek/deepseek-v4-flash")

    def test_self_hoster_single_custom_model(self) -> None:
        # Set OPENROUTER_MODELS to one custom slug -> it's the whole picker AND the default.
        s = _fake_models("some/self-hosted-model")
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            self.assertEqual(app_settings.default_chat_model(), "some/self-hosted-model")
            self.assertEqual(
                app_settings.resolve_chat_model("some/self-hosted-model"), "some/self-hosted-model"
            )
            self.assertEqual(app_settings.resolve_chat_model(None), "some/self-hosted-model")


if __name__ == "__main__":
    unittest.main()
