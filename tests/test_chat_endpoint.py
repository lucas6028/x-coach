"""Tests for the grounded conversational-coaching endpoint (``POST /api/chat``).

The network (OpenRouter) is never hit: service tests patch ``httpx.post`` or the ``_chat_completion``
seam, and router tests call the coroutine directly with a stub user — mirroring
``test_analyze_endpoint.py``. They lock in the two things that matter for this feature:

* **groundedness** — the server-built system prompt carries the analysis facts (fault names +
  retrieved corrections) and the honesty constraint that forbids inventing anything, and
* the **request contract** — chat is gated + configured (503 without a key), the last turn must
  be the user's (422 otherwise), and an upstream failure surfaces as a clean 502.
"""

from __future__ import annotations

import asyncio
import types
import unittest
from unittest import mock

from fastapi import HTTPException

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


# --------------------------------------------------------------------- service: answer


class AnswerTests(unittest.TestCase):
    def test_prepends_system_prompt_then_history(self) -> None:
        seen: dict[str, list] = {}

        def fake_completion(messages):
            seen["messages"] = messages
            return "keep your chest up"

        history = [
            {"role": "user", "content": "why did my knees cave?"},
        ]
        with mock.patch.object(chat_service, "_chat_completion", fake_completion), mock.patch.object(
            chat_service, "get_settings", return_value=types.SimpleNamespace(openrouter_model="m-x")
        ):
            out = chat_service.answer(messages=history, context=_FAULT_CTX)

        self.assertEqual(out, {"reply": "keep your chest up", "model": "m-x"})
        sent = seen["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("knees_inward", sent[0]["content"])
        self.assertEqual(sent[1:], history)  # user history follows the system prompt, untouched


# --------------------------------------------------------- service: OpenRouter transport


class ChatCompletionTests(unittest.TestCase):
    def _settings(self):
        return types.SimpleNamespace(
            openrouter_api_key="sk-or-test",
            openrouter_model="anthropic/claude-sonnet-5",
            openrouter_base_url="https://openrouter.ai/api/v1",
        )

    def test_parses_reply_from_openai_shape(self) -> None:
        resp = mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "hi there"}}]}
        with mock.patch.object(chat_service, "get_settings", return_value=self._settings()), mock.patch(
            "httpx.post", return_value=resp
        ) as post:
            reply = chat_service._chat_completion([{"role": "user", "content": "hi"}])
        self.assertEqual(reply, "hi there")
        # The model + auth header were sent to the completions URL.
        _, kwargs = post.call_args
        self.assertEqual(kwargs["json"]["model"], "anthropic/claude-sonnet-5")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-or-test")

    def test_network_failure_becomes_runtime_error(self) -> None:
        with mock.patch.object(chat_service, "get_settings", return_value=self._settings()), mock.patch(
            "httpx.post", side_effect=Exception("connection reset")
        ):
            with self.assertRaises(RuntimeError):
                chat_service._chat_completion([{"role": "user", "content": "hi"}])


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

    def test_delegates_to_service_and_returns_reply(self) -> None:
        captured: dict = {}

        def fake_answer(*, messages, context):
            captured["messages"] = messages
            captured["context"] = context
            return {"reply": "drive knees out", "model": "m"}

        with mock.patch.object(
            chat_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ), mock.patch.object(chat_service, "answer", fake_answer):
            out = self._run(_body([{"role": "user", "content": "fix my knees?"}], _FAULT_CTX))

        self.assertEqual(out["reply"], "drive knees out")
        self.assertEqual(captured["messages"][-1]["content"], "fix my knees?")
        self.assertEqual(captured["context"]["faults"][0]["fault_name"], "knees_inward")

    def test_service_runtime_error_maps_to_502(self) -> None:
        def boom(*, messages, context):
            raise RuntimeError("OpenRouter request failed: timeout")

        with mock.patch.object(
            chat_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ), mock.patch.object(chat_service, "answer", boom):
            with self.assertRaises(HTTPException) as ctx:
                self._run(_body([{"role": "user", "content": "hi"}], _FAULT_CTX))
        self.assertEqual(ctx.exception.status_code, 502)


if __name__ == "__main__":
    unittest.main()
