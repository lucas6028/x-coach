"""Tests for the grounded conversational-coaching endpoint (``POST /api/chat``).

The network (the LLM provider) is never hit: service tests patch ``httpx.stream`` or the
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
import json
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
# A GENUINELY clean rep: no faults AND the clip was measurable. `valid_frame_ratio` is load-bearing
# here, not decoration -- an empty fault list alone no longer earns the CLEAN REP instruction.
_CLEAN_CTX = {
    "view_type": "side",
    "fault_count": 0,
    "quality": {"valid_frame_ratio": 0.94, "valid_frames": 282, "total_frames": 300},
    "faults": [],
}

# The SAME empty fault list, produced instead by a clip no frame of which could be measured (e.g.
# framed above the ankles, which fails the push-up detector's validity gate on every frame).
_UNMEASURED_CTX = {
    "view_type": "side",
    "fault_count": 0,
    "quality": {"valid_frame_ratio": 0.0, "valid_frames": 0, "total_frames": 300},
    "faults": [],
}


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
        # _CLEAN_CTX carries no `movement`, so `_build_system_prompt` falls back to the pipeline
        # default (Squat) -- see TestChatPromptMovement for the movement-scoped wording itself.
        prompt = chat_service._build_system_prompt(_CLEAN_CTX)
        self.assertIn("CLEAN Squat REP", prompt)
        self.assertNotIn("DETECTED FAULTS", prompt)
        self.assertIn("Faults detected: 0", prompt)

    def test_unmeasured_clip_gets_no_clean_rep_instruction(self) -> None:
        """An empty fault list means BOTH "no faults" and "nothing was measurable". Telling the
        model to congratulate the user in the second case fabricates a verdict about form that was
        never analysed -- the same claim the prompt's honesty constraint forbids everywhere else."""
        prompt = chat_service._build_system_prompt(_UNMEASURED_CTX)
        self.assertNotIn("CLEAN REP", prompt)
        self.assertIn("NOT MEASURED", prompt)
        self.assertIn("Do NOT congratulate", prompt)
        self.assertIn("Faults detected: 0", prompt)
        self.assertNotIn("DETECTED FAULTS", prompt)

    def test_measurable_frame_ratio_is_surfaced_to_the_model(self) -> None:
        """Shipped by `buildChatContext` and accepted by `ChatContext.quality` all along, but never
        rendered -- so the model could not tell a clean rep from an unmeasured clip even in
        principle."""
        self.assertIn("Measurable frames: 94%", chat_service._build_system_prompt(_CLEAN_CTX))
        self.assertIn("Measurable frames: 0%", chat_service._build_system_prompt(_UNMEASURED_CTX))
        # Absent from the payload => no line at all (and see the next test for the verdict).
        self.assertNotIn("Measurable frames", chat_service._build_system_prompt(_FAULT_CTX))

    def test_absent_valid_frame_ratio_is_treated_as_unmeasured(self) -> None:
        """"The payload did not say" must not resolve to "everything is fine". The analyze pipeline
        always emits `valid_frame_ratio`, so its absence is missing information, not evidence of a
        clean rep. Matches `wasMeasured`'s `?? 0` in frontend/src/lib/quality.ts."""
        prompt = chat_service._build_system_prompt(
            {"view_type": "side", "fault_count": 0, "quality": {}, "faults": []}
        )
        self.assertNotIn("CLEAN REP", prompt)
        self.assertIn("NOT MEASURED", prompt)

    def test_faults_win_over_the_measurability_branch(self) -> None:
        """A clip with detected faults was measurable by construction, so neither no-fault branch
        may fire even if the quality dict is missing or degenerate."""
        ctx = dict(_FAULT_CTX, quality={"valid_frame_ratio": 0.0})
        prompt = chat_service._build_system_prompt(ctx)
        self.assertNotIn("CLEAN REP", prompt)
        self.assertNotIn("NOT MEASURED", prompt)
        self.assertIn("DETECTED FAULTS", prompt)

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

        def fake_turn(messages, model, *, timeout=None, tools=None):
            seen["messages"] = messages
            seen["model"] = model
            yield "Drive "
            yield "knees out"
            yield chat_service._Turn(
                text="Drive knees out", tool_calls=[], finish_reason="stop"
            )

        history = [{"role": "user", "content": "why did my knees cave?"}]
        with mock.patch.object(chat_service, "_stream_turn", fake_turn):
            frames = "".join(
                chat_service.answer_stream(messages=history, context=_FAULT_CTX, model="vendor/m-x")
            )

        self.assertIn("event: delta", frames)
        self.assertIn("Drive ", frames)
        self.assertIn("knees out", frames)
        self.assertIn("event: done", frames)
        self.assertIn("vendor/m-x", frames)  # the actually-used model rides the done frame
        self.assertNotIn("event: error", frames)
        self.assertEqual(seen["model"], "vendor/m-x")
        sent = seen["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("knees_inward", sent[0]["content"])
        self.assertEqual(sent[1:], history)  # user history still follows the system turn untouched

    def test_midstream_failure_yields_error_frame_and_no_done(self) -> None:
        def fake_turn(messages, model, *, timeout=None, tools=None):
            yield "partial answer"
            raise chat_service._LLMError("LLM request failed: connection reset")

        with mock.patch.object(chat_service, "_stream_turn", fake_turn):
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
        # The v1 empty-completion invariant survives into the tool loop: a blank accumulation must
        # emit an error, never a done, so the client never keeps an empty assistant turn.
        def fake_turn(messages, model, *, timeout=None, tools=None):
            yield "   "  # whitespace only -> strips to empty
            yield chat_service._Turn(text="   ", tool_calls=[], finish_reason="stop")

        with mock.patch.object(chat_service, "_stream_turn", fake_turn):
            frames = "".join(
                chat_service.answer_stream(
                    messages=[{"role": "user", "content": "hi"}], context=_CLEAN_CTX, model="m"
                )
            )

        self.assertIn("event: error", frames)
        self.assertIn("empty", frames.lower())
        self.assertNotIn("event: done", frames)

    def test_answer_stream_carries_no_followups(self) -> None:
        # Follow-ups are a SEPARATE endpoint (suggest_followups); the answer stream must never carry
        # a followups frame, tool loop or not.
        def fake_turn(messages, model, *, timeout=None, tools=None):
            yield "Drive knees out."
            yield chat_service._Turn(
                text="Drive knees out.", tool_calls=[], finish_reason="stop"
            )

        with mock.patch.object(chat_service, "_stream_turn", fake_turn):
            frames = "".join(
                chat_service.answer_stream(
                    messages=[{"role": "user", "content": "why?"}], context=_FAULT_CTX, model="m"
                )
            )

        self.assertIn("Drive knees out.", frames)
        self.assertNotIn("event: followups", frames)
        self.assertIn("event: done", frames)


class ToolLoopTests(unittest.TestCase):
    """The answer loop: tool rounds, retraction, round/time caps, and tools-unsupported fallback."""

    CONTEXT = {"movement": "Squat", "faults": [], "quality": {"valid_frame_ratio": 0.9}}

    def _turns(self, *rounds):
        """Build a _stream_turn stub that replays one canned round per call.

        Each ``round`` is ``(text_deltas, tool_calls)``; the stub yields the deltas then a _Turn.
        """
        calls = []

        def fake(messages, model, *, timeout=None, tools=None):
            calls.append({"messages": [dict(m) for m in messages], "tools": tools})
            deltas, tool_calls = rounds[len(calls) - 1]
            for d in deltas:
                yield d
            yield chat_service._Turn(
                text="".join(deltas),
                tool_calls=list(tool_calls),
                finish_reason="tool_calls" if tool_calls else "stop",
            )

        return fake, calls

    @staticmethod
    def _events(frames):
        """Parse rendered SSE frames into [(event, data), ...]."""
        out = []
        for frame in "".join(frames).split("\n\n"):
            if not frame.strip():
                continue
            lines = frame.split("\n")
            event = lines[0][len("event:") :].strip()
            data = json.loads(lines[1][len("data:") :].strip())
            out.append((event, data))
        return out

    def _run(self, fake_turn, dispatch=None):
        with mock.patch.object(chat_service, "_stream_turn", fake_turn), mock.patch.object(
            chat_service,
            "_dispatch_tool",
            dispatch or (lambda n, a, c: chat_service._ToolResult(text='{"ok": true}', sources=[])),
        ), mock.patch.object(chat_service, "chat_timeout", return_value=60.0):
            return self._events(
                list(
                    chat_service.answer_stream(
                        messages=[{"role": "user", "content": "hi"}],
                        context=self.CONTEXT,
                        model="m",
                    )
                )
            )

    def test_a_plain_turn_streams_deltas_with_no_tool_or_reset_frames(self) -> None:
        # Regression lock: the common path (no tool call) must stream token-by-token exactly as it
        # does today -- not be buffered and flushed at the end.
        fake, calls = self._turns((["Hel", "lo"], []))
        events = self._run(fake)
        self.assertEqual(
            events, [("delta", {"text": "Hel"}), ("delta", {"text": "lo"}), ("done", {"model": "m"})]
        )
        self.assertIsNotNone(calls[0]["tools"])  # tools were offered on round 1

    def test_a_tool_round_emits_a_tool_frame_then_the_answer(self) -> None:
        dispatched = []

        def dispatch(name, args, ctx):
            dispatched.append((name, args, ctx))
            return chat_service._ToolResult(text='{"ok": true}', sources=[])

        fake, calls = self._turns(
            ([], [{"id": "c1", "name": "kg_query", "arguments": '{"query": "valgus"}'}]),
            (["Answer"], []),
        )
        events = self._run(fake, dispatch=dispatch)
        self.assertEqual(
            events,
            [
                ("tool", {"id": 0, "name": "kg_query", "query": "valgus"}),
                ("tool_done", {"id": 0}),
                ("delta", {"text": "Answer"}),
                ("done", {"model": "m"}),
            ],
        )
        # Round 2 saw the assistant tool_calls turn plus the tool result appended.
        roles = [m["role"] for m in calls[1]["messages"]]
        self.assertEqual(roles[-2:], ["assistant", "tool"])
        self.assertEqual(calls[1]["messages"][-1]["tool_call_id"], "c1")
        # _dispatch_tool must receive the REQUEST's context (where ChatContext.detail lives), not
        # the accumulated conversation -- get_analysis exists solely to read it, and this is the
        # one new wire this task adds with no other lock on it.
        self.assertEqual(len(dispatched), 1)
        name, args, ctx = dispatched[0]
        self.assertEqual(name, "kg_query")
        self.assertEqual(args, {"query": "valgus"})
        self.assertIs(ctx, self.CONTEXT)

    def test_narration_plus_a_tool_call_is_retracted_with_a_reset_frame(self) -> None:
        fake, _ = self._turns(
            (["Let me check."], [{"id": "c1", "name": "rag_search", "arguments": '{"query": "x"}'}]),
            (["Real answer"], []),
        )
        events = self._run(fake)
        self.assertEqual(
            events,
            [
                ("delta", {"text": "Let me check."}),
                ("reset", {}),
                ("tool", {"id": 0, "name": "rag_search", "query": "x"}),
                ("tool_done", {"id": 0}),
                ("delta", {"text": "Real answer"}),
                ("done", {"model": "m"}),
            ],
        )

    def test_a_tool_round_with_no_narration_emits_no_reset(self) -> None:
        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "kg_query", "arguments": "{}"}]),
            (["A"], []),
        )
        self.assertNotIn("reset", [e for e, _ in self._run(fake)])

    def test_the_loop_stops_after_three_tool_rounds_and_forces_a_toolless_answer(self) -> None:
        call = [{"id": "c", "name": "kg_query", "arguments": "{}"}]
        fake, calls = self._turns(
            ([], call), ([], call), ([], call), (["Forced"], [])
        )
        events = self._run(fake)
        self.assertEqual(events[-2:], [("delta", {"text": "Forced"}), ("done", {"model": "m"})])
        self.assertEqual(len(calls), 4)
        self.assertIsNotNone(calls[2]["tools"])  # round 3 still offers tools
        self.assertIsNone(calls[3]["tools"])  # the forced final round does not

    def test_unsolicited_tool_calls_on_the_forced_final_round_still_produce_an_answer(self) -> None:
        # The forced round 4 offers no tools (offer_tools=False), but nothing stops a model from
        # emitting tool_calls anyway. The ``not offer_tools`` disjunct in the break condition exists
        # exactly for this: without it, a round with BOTH text and tool_calls would fall through to
        # the tool-dispatch code, find the range exhausted, and exit with no answer at all. Round 4
        # deliberately narrates too ("Here", not "") -- an all-empty round 4 would produce an empty
        # answer and read as the guard FAILING (an error frame) rather than holding.
        call = [{"id": "c", "name": "kg_query", "arguments": "{}"}]
        fake, calls = self._turns(([], call), ([], call), ([], call), (["Here"], call))
        events = self._run(fake)
        self.assertEqual(events[-2:], [("delta", {"text": "Here"}), ("done", {"model": "m"})])
        self.assertEqual(len(calls), 4)
        self.assertNotIn("reset", [e for e, _ in events])  # round 4 never reaches the reset check

    def test_an_exhausted_time_budget_stops_the_loop(self) -> None:
        # chat_timeout() of 0 means there is no budget left before the first round even starts.
        fake, _ = self._turns((["never"], []))
        with mock.patch.object(chat_service, "_stream_turn", fake), mock.patch.object(
            chat_service, "chat_timeout", return_value=0.0
        ):
            events = self._events(
                list(
                    chat_service.answer_stream(
                        messages=[{"role": "user", "content": "hi"}],
                        context=self.CONTEXT,
                        model="m",
                    )
                )
            )
        self.assertEqual([e for e, _ in events], ["error"])

    def test_a_4xx_before_any_delta_retries_once_without_tools(self) -> None:
        attempts = []

        def fake(messages, model, *, timeout=None, tools=None):
            attempts.append(tools)
            if len(attempts) == 1:
                raise chat_service._LLMError("no tool support", status=400)
            yield "Plain"
            yield chat_service._Turn(text="Plain", tool_calls=[], finish_reason="stop")

        events = self._run(fake)
        self.assertEqual(events, [("delta", {"text": "Plain"}), ("done", {"model": "m"})])
        self.assertIsNotNone(attempts[0])  # first attempt offered tools
        self.assertIsNone(attempts[1])  # the retry did not

    def test_a_4xx_retry_that_also_fails_is_a_terminal_error(self) -> None:
        # The retry exists because the first attempt 400'd on ``tools``; the second, tools-free
        # request can still fail on its own -- a genuine bad request, the provider going down
        # between the two calls, or the 1.0s floor timing out. It must not be retried again.
        attempts = []

        def fake(messages, model, *, timeout=None, tools=None):
            attempts.append(tools)
            raise chat_service._LLMError("still rejected", status=400)
            yield  # pragma: no cover — makes this a generator

        events = self._run(fake)
        self.assertEqual([e for e, _ in events], ["error"])
        self.assertEqual(len(attempts), 2)  # the original attempt plus the one retry, no more

    def test_a_non_llm_exception_during_the_retry_is_also_a_terminal_error(self) -> None:
        # The reassembly-crash exposure (see test_a_non_llm_exception_mid_round_...) applies
        # equally to the retry's own _stream_turn call, not just the first one -- the retry's inner
        # handler must be widened to Exception too, not left catching only _LLMError, or a crash
        # here would propagate straight out of the outer except _LLMError block uncaught.
        attempts = []

        def fake(messages, model, *, timeout=None, tools=None):
            attempts.append(tools)
            if len(attempts) == 1:
                raise chat_service._LLMError("no tool support", status=400)
            raise TypeError("bad chunk shape")
            yield  # pragma: no cover — makes this a generator

        events = self._run(fake)
        self.assertEqual([e for e, _ in events], ["error"])
        self.assertEqual(len(attempts), 2)

    def test_a_4xx_after_a_delta_is_a_terminal_error_with_no_retry(self) -> None:
        attempts = []

        def fake(messages, model, *, timeout=None, tools=None):
            attempts.append(tools)
            yield "partial"
            raise chat_service._LLMError("died mid-stream", status=400)

        events = self._run(fake)
        self.assertEqual([e for e, _ in events], ["delta", "error"])
        self.assertEqual(len(attempts), 1)  # committed output -> no retry

    def test_a_transport_failure_is_never_retried(self) -> None:
        attempts = []

        def fake(messages, model, *, timeout=None, tools=None):
            attempts.append(tools)
            raise chat_service._LLMError("connection reset")  # status=None
            yield  # pragma: no cover — makes this a generator

        events = self._run(fake)
        self.assertEqual([e for e, _ in events], ["error"])
        self.assertEqual(len(attempts), 1)

    def test_a_non_llm_exception_mid_round_yields_one_error_frame_not_a_crash(self) -> None:
        # _stream_turn's chunk-reassembly loop is NOT itself exception-wrapped (unlike
        # _stream_raw_chunks' transport, which always raises _LLMError): a chunk that is valid JSON
        # but the wrong SHAPE (a "tool_calls": 5, a non-string "arguments", ...) crashes with a
        # plain TypeError/AttributeError/OverflowError. answer_stream is the only frame in the
        # stack that knows the HTTP 200 is already committed, so it must convert that crash into an
        # in-band error frame -- not let it escape and kill the stream. This test's own success is
        # the proof: if the exception weren't caught, this test itself would raise, not fail.
        def fake(messages, model, *, timeout=None, tools=None):
            yield "partial"
            raise TypeError("'int' object is not iterable")

        events = self._run(fake)
        self.assertEqual([e for e, _ in events], ["delta", "error"])

    def test_an_empty_completion_is_still_an_error(self) -> None:
        fake, _ = self._turns(([], []))
        events = self._run(fake)
        self.assertEqual([e for e, _ in events], ["error"])

    def test_the_tool_grounding_rule_is_in_the_system_prompt(self) -> None:
        fake, calls = self._turns((["ok"], []))
        self._run(fake)
        system = calls[0]["messages"][0]["content"]
        self.assertEqual(calls[0]["messages"][0]["role"], "system")
        self.assertIn("REFERENCE MATERIAL", system)
        self.assertIn("ANALYSIS FACTS", system)  # the v2 prompt is still there, unchanged
        # FIX 7 (whole-branch review): a lookup on an unmeasurable clip must not be laundered into
        # coaching feedback about this rep just because it's technically framed as "general reference".
        self.assertIn("NO retrieved reference material", system)

    def test_a_dispatch_tool_crash_yields_exactly_one_error_frame_and_no_done(self) -> None:
        # FIX 3 (whole-branch review): _dispatch_tool is called OUTSIDE _stream_turn's own
        # try/except, in the loop body of _answer_stream_inner. A crash there (not one of the
        # failure modes _dispatch_tool's own try already contains) must still surface as a single
        # in-band error frame through the answer_stream outer wrapper -- never an unhandled
        # exception that kills the stream after the HTTP 200 is already committed.
        fake, _ = self._turns(
            (["let me check"], [{"id": "1", "name": "kg_query", "arguments": '{"query":"x"}'}])
        )

        def crashing_dispatch(name, args, context):
            raise RuntimeError("boom")

        events = self._run(fake, dispatch=crashing_dispatch)
        event_names = [e for e, _ in events]
        self.assertEqual(event_names.count("error"), 1)
        self.assertNotIn("done", event_names)

    def test_the_tool_done_frame_carries_sources(self) -> None:
        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "rag_search", "arguments": '{"query": "ankle"}'}]),
            (["Answer"], []),
        )
        result = chat_service._ToolResult(
            text='{"ok": true}',
            sources=[{"label": "Wikipedia: Squat (exercise)", "kind": "encyclopedia"}],
        )
        events = self._run(fake, dispatch=lambda n, a, c: result)
        self.assertEqual(
            [d for e, d in events if e == "tool"],
            [{"id": 0, "name": "rag_search", "query": "ankle"}],
        )
        self.assertEqual(
            [d for e, d in events if e == "tool_done"],
            [
                {
                    "id": 0,
                    "sources": [{"label": "Wikipedia: Squat (exercise)", "kind": "encyclopedia"}],
                }
            ],
        )

    def test_the_tool_done_frame_omits_sources_when_there_are_none(self) -> None:
        # get_analysis has no outside source to credit; the key is absent, not an empty array, so a
        # client can tell "nothing to cite" from "cited nothing".
        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "get_analysis", "arguments": '{"include": "all"}'}]),
            (["Answer"], []),
        )
        result = chat_service._ToolResult(text='{"ok": true}', sources=[])
        events = self._run(fake, dispatch=lambda n, a, c: result)
        self.assertEqual([d for e, d in events if e == "tool_done"], [{"id": 0}])

    def test_the_tool_message_content_is_the_result_text(self) -> None:
        # Regression lock: the model must receive `.text`, never a repr of the _ToolResult.
        fake, calls = self._turns(
            ([], [{"id": "c1", "name": "kg_query", "arguments": "{}"}]),
            (["A"], []),
        )
        result = chat_service._ToolResult(text='{"marker": 1}', sources=[{"label": "L", "kind": "concept"}])
        self._run(fake, dispatch=lambda n, a, c: result)
        self.assertEqual(calls[1]["messages"][-1]["content"], '{"marker": 1}')

    def test_the_tool_frame_is_yielded_before_the_tool_runs(self) -> None:
        # The whole point of v3.2: v3.1 dispatched first so the frame could carry sources, which
        # left the tray silent for the entire retrieval (past two minutes on a cold RAG process).
        # Asserting frame content is not enough -- only the ORDER proves the user sees the lookup
        # named while it is still running, so the dispatch stub records the frames emitted so far.
        seen_at_dispatch: list[list[str]] = []
        emitted: list[str] = []

        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "rag_search", "arguments": '{"query": "ankle"}'}]),
            (["Answer"], []),
        )

        def dispatch(name, args, ctx):
            seen_at_dispatch.append(list(emitted))
            return chat_service._ToolResult(text='{"ok": true}', sources=[])

        with mock.patch.object(chat_service, "_stream_turn", fake), mock.patch.object(
            chat_service, "_dispatch_tool", dispatch
        ), mock.patch.object(chat_service, "chat_timeout", return_value=60.0):
            for frame in chat_service.answer_stream(
                messages=[{"role": "user", "content": "hi"}], context=self.CONTEXT, model="m"
            ):
                emitted.append(frame.split("\n")[0][len("event:") :].strip())

        self.assertEqual(seen_at_dispatch, [["tool"]])

    def test_tool_done_is_emitted_even_when_the_tool_yields_no_sources(self) -> None:
        # get_analysis never has an outside source to credit. If tool_done were conditional on
        # sources, its row would sit on screen as "still running" for the rest of the session.
        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "get_analysis", "arguments": '{"include": "all"}'}]),
            (["Answer"], []),
        )
        result = chat_service._ToolResult(text='{"ok": true}', sources=[])
        events = self._run(fake, dispatch=lambda n, a, c: result)
        self.assertEqual(
            events,
            [
                ("tool", {"id": 0, "name": "get_analysis", "query": ""}),
                ("tool_done", {"id": 0}),
                ("delta", {"text": "Answer"}),
                ("done", {"model": "m"}),
            ],
        )

    def test_tool_ids_are_unique_across_rounds(self) -> None:
        # enumerate(turn.tool_calls) restarts at 0 every round, so a per-round index would collide
        # the moment two rounds each call a tool -- which is the NORMAL shape of a multi-round
        # conversation, not an edge case. Two rounds, two calls in the first, one in the second.
        fake, _ = self._turns(
            (
                [],
                [
                    {"id": "c1", "name": "kg_query", "arguments": '{"query": "a"}'},
                    {"id": "c2", "name": "rag_search", "arguments": '{"query": "b"}'},
                ],
            ),
            ([], [{"id": "c3", "name": "kg_query", "arguments": '{"query": "c"}'}]),
            (["Answer"], []),
        )
        events = self._run(fake)
        starts = [d["id"] for e, d in events if e == "tool"]
        dones = [d["id"] for e, d in events if e == "tool_done"]
        self.assertEqual(starts, [0, 1, 2])
        self.assertEqual(dones, [0, 1, 2])


# --------------------------------------------------------------- service: follow-up parsing


class ParseFollowupsTests(unittest.TestCase):
    def test_plain_json_array(self) -> None:
        self.assertEqual(chat_service._parse_followups('["a?", "b?"]'), ["a?", "b?"])

    def test_tolerates_code_fences_and_prose(self) -> None:
        raw = 'Sure! Here are two:\n```json\n["a?", "b?"]\n```'
        self.assertEqual(chat_service._parse_followups(raw), ["a?", "b?"])

    def test_non_list_payload_yields_empty(self) -> None:
        self.assertEqual(chat_service._parse_followups('{"q": "a?"}'), [])

    def test_unparseable_payload_yields_empty(self) -> None:
        self.assertEqual(chat_service._parse_followups("no json here at all"), [])

    def test_drops_blanks_and_truncates_to_two(self) -> None:
        self.assertEqual(
            chat_service._parse_followups('["a?", "", "  ", "b?", "c?"]'), ["a?", "b?"]
        )


class SuggestFollowupsTests(unittest.TestCase):
    def test_returns_grounded_questions_and_uses_tight_timeout(self) -> None:
        captured: dict[str, object] = {}

        def fake_stream(messages, model, timeout=None, extra_body=None):
            captured["messages"] = messages
            captured["timeout"] = timeout
            captured["extra_body"] = extra_body
            yield '["How do I '
            yield 'fix my depth?", "Why does valgus matter?"]'

        with mock.patch.object(chat_service, "_stream_completion", fake_stream), mock.patch.object(
            chat_service,
            "get_settings",
            return_value=types.SimpleNamespace(llm_base_url="https://openrouter.ai/api/v1"),
        ), mock.patch.object(
            chat_service, "chat_base_url", return_value="https://openrouter.ai/api/v1"
        ):
            qs = chat_service.suggest_followups(
                messages=[
                    {"role": "user", "content": "why?"},
                    {"role": "assistant", "content": "Drive knees out."},
                ],
                context=_FAULT_CTX,
                model="m",
            )

        self.assertEqual(qs, ["How do I fix my depth?", "Why does valgus matter?"])
        # Follow-up requests carry the low-latency provider routing (the fix for the chip-latency
        # variance), passed through to the transport as extra request body — on OpenRouter.
        self.assertEqual(captured["extra_body"], chat_service._FOLLOWUP_ROUTING)
        # Groundedness survives into the follow-up prompt: the analysis facts + honesty rule precede
        # the follow-up task, so a suggestion can't reference a fault outside the analysis.
        system = captured["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("knees_inward", system["content"])
        self.assertIn("Do NOT invent", system["content"])
        self.assertIn("FOLLOW-UP TASK", system["content"])
        # The assistant answer rides in the middle; a trailing user nudge closes the array so the
        # request isn't left open on the assistant turn.
        self.assertEqual(
            captured["messages"][-1], {"role": "user", "content": chat_service._FOLLOWUP_NUDGE}
        )
        # The tight follow-up budget is used, not the full answer timeout.
        self.assertEqual(captured["timeout"], chat_service._FOLLOWUP_TIMEOUT_S)

    def test_non_openrouter_base_url_sends_no_provider_routing(self) -> None:
        # Against any OpenAI-compatible peer that isn't OpenRouter (e.g. NVIDIA NIM), the OpenRouter-
        # only ``provider`` routing body must be omitted so a stricter peer can't 400 on it.
        captured: dict[str, object] = {}

        def fake_stream(messages, model, timeout=None, extra_body=None):
            captured["extra_body"] = extra_body
            yield '["a?", "b?"]'

        with mock.patch.object(chat_service, "_stream_completion", fake_stream), mock.patch.object(
            chat_service,
            "get_settings",
            return_value=types.SimpleNamespace(
                llm_base_url="https://integrate.api.nvidia.com/v1"
            ),
        ), mock.patch.object(
            chat_service, "chat_base_url", return_value="https://integrate.api.nvidia.com/v1"
        ):
            qs = chat_service.suggest_followups(
                messages=[{"role": "user", "content": "why?"}], context=_FAULT_CTX, model="m"
            )

        self.assertEqual(qs, ["a?", "b?"])  # parsing still works
        self.assertIsNone(captured["extra_body"])  # no OpenRouter-only routing body

    def test_malformed_reply_yields_no_suggestions(self) -> None:
        def fake_stream(messages, model, timeout=None, extra_body=None):
            yield "sorry, I can't do that"

        with mock.patch.object(chat_service, "_stream_completion", fake_stream):
            qs = chat_service.suggest_followups(
                messages=[{"role": "user", "content": "why?"}], context=_FAULT_CTX, model="m"
            )
        self.assertEqual(qs, [])

    def test_transport_failure_yields_no_suggestions(self) -> None:
        def fake_stream(messages, model, timeout=None, extra_body=None):
            raise RuntimeError("LLM request failed: reset")
            yield ""  # pragma: no cover — unreachable, keeps this a generator

        with mock.patch.object(chat_service, "_stream_completion", fake_stream):
            qs = chat_service.suggest_followups(
                messages=[{"role": "user", "content": "why?"}], context=_FAULT_CTX, model="m"
            )
        self.assertEqual(qs, [])


# ----------------------------------------------------- service: provider (OpenAI-compatible) SSE


class StreamCompletionTests(unittest.TestCase):
    def _settings(self):
        return types.SimpleNamespace(
            llm_api_key="sk-or-test",
            llm_base_url="https://openrouter.ai/api/v1",
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
        ), mock.patch.object(
            chat_service, "chat_base_url", return_value="https://openrouter.ai/api/v1"
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
        self.assertEqual(kwargs["headers"]["X-Title"], "x-coach")  # OpenRouter attribution on-path

    def test_non_openrouter_base_url_omits_attribution_headers(self) -> None:
        # Against a non-OpenRouter OpenAI-compatible peer (NVIDIA NIM), the OpenRouter attribution
        # headers are dropped; only Authorization + Content-Type are sent.
        settings = types.SimpleNamespace(
            llm_api_key="nvapi-test",
            llm_base_url="https://integrate.api.nvidia.com/v1",
        )
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.iter_lines.return_value = iter(["data: [DONE]"])
        cm = mock.MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with mock.patch.object(
            chat_service, "get_settings", return_value=settings
        ), mock.patch.object(
            chat_service, "chat_base_url", return_value="https://integrate.api.nvidia.com/v1"
        ), mock.patch("httpx.stream", return_value=cm) as stream:
            list(chat_service._stream_completion([{"role": "user", "content": "hi"}], "meta/llama-3.3-70b-instruct"))

        _, kwargs = stream.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer nvapi-test")
        self.assertNotIn("HTTP-Referer", kwargs["headers"])
        self.assertNotIn("X-Title", kwargs["headers"])
        # The request URL points at the configured (NIM) base.
        args, _ = stream.call_args
        self.assertEqual(args[1], "https://integrate.api.nvidia.com/v1/chat/completions")

    def test_extra_body_merges_into_the_request(self) -> None:
        # The follow-up call passes provider-routing preferences via extra_body; they must land in the
        # JSON request alongside model/messages/stream.
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.iter_lines.return_value = iter(["data: [DONE]"])
        cm = mock.MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with mock.patch.object(
            chat_service, "get_settings", return_value=self._settings()
        ), mock.patch("httpx.stream", return_value=cm) as stream:
            list(
                chat_service._stream_completion(
                    [{"role": "user", "content": "hi"}], "m", extra_body={"provider": {"sort": "latency"}}
                )
            )
        _, kwargs = stream.call_args
        self.assertEqual(kwargs["json"]["provider"], {"sort": "latency"})
        self.assertIs(kwargs["json"]["stream"], True)  # base fields still present

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

    def test_raw_chunks_yield_parsed_dicts_and_skip_unparseable(self) -> None:
        # The raw layer owns JSON-level tolerance ONLY: a truncated/garbage payload is skipped, but a
        # well-formed chunk is yielded whole even when it carries no ``content`` -- it may hold a
        # ``tool_calls`` fragment, and silently dropping it would corrupt the caller's reassembly.
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.iter_lines.return_value = iter(
            [
                ": keep-alive ping",  # SSE comment -> skip
                'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                "data: {broken",  # truncated JSON -> skip
                'data: {"choices":[{"delta":{}}]}',  # no content, but still a real chunk -> KEPT
                "data: [DONE]",
                'data: {"choices":[{"delta":{"content":"unreached"}}]}',
            ]
        )
        cm = mock.MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with mock.patch.object(
            chat_service, "get_settings", return_value=self._settings()
        ), mock.patch("httpx.stream", return_value=cm):
            chunks = list(
                chat_service._stream_raw_chunks([{"role": "user", "content": "hi"}], "m")
            )
        self.assertEqual(
            chunks,
            [
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {}}]},
            ],
        )

    def test_raw_chunks_http_status_error_carries_the_status(self) -> None:
        # A 4xx must be distinguishable from a transport failure: the tool loop retries without
        # ``tools`` on a 4xx (the model rejects the field) but not on a dead provider.
        import httpx

        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(400, request=request)
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400", request=request, response=response
        )
        cm = mock.MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with mock.patch.object(
            chat_service, "get_settings", return_value=self._settings()
        ), mock.patch("httpx.stream", return_value=cm):
            with self.assertRaises(chat_service._LLMError) as ctx:
                list(chat_service._stream_raw_chunks([{"role": "user", "content": "hi"}], "m"))
        self.assertEqual(ctx.exception.status, 400)
        self.assertIsInstance(ctx.exception, RuntimeError)  # existing callers still catch it

    def test_completion_shell_skips_chunks_with_no_usable_choice(self) -> None:
        # After the split the shell's own ``except (KeyError, IndexError, TypeError)`` has nothing
        # driving it from the SSE-level tests: ``data: not-json`` is now swallowed by the raw layer,
        # and a well-formed empty delta returns None from ``.get("content")`` without ever raising.
        # Drive it directly, or the branch goes uncovered under the 95% gate.
        with mock.patch.object(
            chat_service,
            "_stream_raw_chunks",
            return_value=iter(
                [
                    {"choices": []},  # IndexError
                    {"usage": {"total_tokens": 3}},  # KeyError
                    {"choices": [{"delta": None}]},  # AttributeError (None has no .get)
                    {"choices": [{"delta": {"content": "ok"}}]},
                ]
            ),
        ):
            self.assertEqual(
                list(chat_service._stream_completion([{"role": "user", "content": "hi"}], "m")),
                ["ok"],
            )

    def test_raw_chunks_transport_failure_has_no_status(self) -> None:
        with mock.patch.object(
            chat_service, "get_settings", return_value=self._settings()
        ), mock.patch("httpx.stream", side_effect=Exception("connection reset")):
            with self.assertRaises(chat_service._LLMError) as ctx:
                list(chat_service._stream_raw_chunks([{"role": "user", "content": "hi"}], "m"))
        self.assertIsNone(ctx.exception.status)


class StreamTurnTests(unittest.TestCase):
    """One model round: text passthrough plus reassembly of fragmented streamed tool calls."""

    def _settings(self):
        return types.SimpleNamespace(
            llm_api_key="sk-or-test",
            llm_base_url="https://openrouter.ai/api/v1",
        )

    def _run(self, chunks, **kwargs):
        """Drive _stream_turn over a canned chunk list; return (text_deltas, turn)."""
        with mock.patch.object(chat_service, "_stream_raw_chunks", return_value=iter(chunks)):
            items = list(chat_service._stream_turn([{"role": "user", "content": "hi"}], "m", **kwargs))
        turn = items[-1]
        self.assertIsInstance(turn, chat_service._Turn)
        return [i for i in items[:-1]], turn

    def test_text_only_round_yields_deltas_then_a_turn(self) -> None:
        deltas, turn = self._run(
            [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
        )
        self.assertEqual(deltas, ["Hel", "lo"])
        self.assertEqual(turn.text, "Hello")
        self.assertEqual(turn.tool_calls, [])
        self.assertEqual(turn.finish_reason, "stop")

    def test_tool_call_arguments_split_across_chunks_are_reassembled(self) -> None:
        # ``id``/``name`` arrive only on the first fragment; ``arguments`` streams in pieces.
        _, turn = self._run(
            [
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call_a", "function": {"name": "kg_query", "arguments": ""}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": '{"que'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": 'ry": "knee valgus"}'}}
                ]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        )
        self.assertEqual(
            turn.tool_calls,
            [{"id": "call_a", "name": "kg_query", "arguments": '{"query": "knee valgus"}'}],
        )
        self.assertEqual(turn.finish_reason, "tool_calls")

    def test_two_interleaved_tool_calls_are_keyed_by_index_not_arrival(self) -> None:
        # Fragments for index 1 arrive before index 0 finishes; accumulation must key on ``index``.
        _, turn = self._run(
            [
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "a", "function": {"name": "kg_query", "arguments": '{"q'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 1, "id": "b", "function": {"name": "rag_search", "arguments": '{"x'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": 'uery": "a"}'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 1, "function": {"arguments": '": "b"}'}}
                ]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        )
        self.assertEqual(
            turn.tool_calls,
            [
                {"id": "a", "name": "kg_query", "arguments": '{"query": "a"}'},
                {"id": "b", "name": "rag_search", "arguments": '{"x": "b"}'},
            ],
        )

    def test_narration_and_a_tool_call_in_one_round_are_both_captured(self) -> None:
        deltas, turn = self._run(
            [
                {"choices": [{"delta": {"content": "Let me look that up."}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "c", "function": {"name": "rag_search", "arguments": "{}"}}
                ]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        )
        self.assertEqual(deltas, ["Let me look that up."])
        self.assertEqual(turn.text, "Let me look that up.")
        self.assertEqual(len(turn.tool_calls), 1)

    def test_nameless_slot_is_dropped_as_a_stray_fragment(self) -> None:
        # A fragment can carry an ``index``/``id``/``arguments`` but never a ``function.name`` (e.g.
        # a truncated stream). The slot never became a real call and must not be dispatched.
        _, turn = self._run(
            [
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 5, "id": "x", "function": {"arguments": "{}"}}
                ]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        )
        self.assertEqual(turn.tool_calls, [])

    def test_mixed_string_and_int_index_are_coerced_to_the_same_slot(self) -> None:
        # ``index`` is provider-supplied and not guaranteed to be an int; a numeric-string index for
        # the same logical call must still land in the same accumulator slot as a later int index,
        # and the round must complete rather than raise out of ``sorted(acc)``.
        _, turn = self._run(
            [
                {"choices": [{"delta": {"tool_calls": [
                    {"index": "0", "id": "a", "function": {"name": "kg_query", "arguments": '{"q'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": 'uery": "x"}'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 1, "id": "b", "function": {"name": "rag_search", "arguments": "{}"}}
                ]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        )
        self.assertEqual(
            turn.tool_calls,
            [
                {"id": "a", "name": "kg_query", "arguments": '{"query": "x"}'},
                {"id": "b", "name": "rag_search", "arguments": "{}"},
            ],
        )

    def test_unparseable_index_falls_back_to_slot_zero(self) -> None:
        # A provider-supplied ``index`` that isn't coercible to int (not a numeric string, e.g.) must
        # not crash the round via ``sorted(acc)``; it folds into slot 0 instead.
        _, turn = self._run(
            [
                {"choices": [{"delta": {"tool_calls": [
                    {"index": "not-a-number", "id": "z", "function": {"name": "kg_query", "arguments": "{}"}}
                ]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        )
        self.assertEqual(
            turn.tool_calls, [{"id": "z", "name": "kg_query", "arguments": "{}"}]
        )

    def test_malformed_chunk_shapes_are_tolerated(self) -> None:
        # Shape tolerance lives HERE, not in the raw layer: usage-only frames, a non-dict choice, a
        # non-dict delta, and a non-dict tool_calls entry must all be skipped without aborting the
        # round.
        deltas, turn = self._run(
            [
                {"usage": {"total_tokens": 7}},  # no ``choices``
                {"choices": []},  # empty ``choices``
                {"choices": ["not-a-dict"]},  # non-dict choice
                {"choices": [{"delta": None}]},  # non-dict delta
                {"choices": [{"delta": {"tool_calls": ["not-a-dict"]}}]},
                {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
            ]
        )
        self.assertEqual(deltas, ["ok"])
        self.assertEqual(turn.tool_calls, [])

    def test_tools_are_sent_as_extra_body_only_when_offered(self) -> None:
        with mock.patch.object(
            chat_service, "_stream_raw_chunks", return_value=iter([])
        ) as raw:
            list(chat_service._stream_turn([], "m", tools=[{"type": "function"}]))
        self.assertEqual(
            raw.call_args.kwargs["extra_body"],
            {"tools": [{"type": "function"}], "tool_choice": "auto"},
        )
        with mock.patch.object(
            chat_service, "_stream_raw_chunks", return_value=iter([])
        ) as raw:
            list(chat_service._stream_turn([], "m", tools=None))
        self.assertIsNone(raw.call_args.kwargs["extra_body"])


class ToolDispatchTests(unittest.TestCase):
    """The three tools: argument clamping, routing, and total failure containment."""

    DETAIL = {
        "metadata": {"fps": 30.0, "total_frames": 120},
        "quality": {"valid_frame_ratio": 0.9},
        "view": {"view_type": "side"},
        "detections": [
            {
                "fault_id": "f1",
                "fault_name": "Insufficient Depth",
                "phase": "bottom",
                "severity": 0.7,
                "confidence": 0.9,
                "observability": "clear",
                "start_time": 1.0,
                "end_time": 2.0,
                "start_frame": 30,
                "end_frame": 60,
                "peak_frame": 45,
                "evidence": {"hip_knee_delta_deg": 12.5},
                "rep_index": 2,
                "occurred_reps": [2],
                "rep_count": 1,
            }
        ],
        "retrievals": [
            {"fault_id": "f1", "context": {"results": [{"text": "squat depth passage"}]}},
            {"fault_id": "other", "context": {"results": [{"text": "unrelated"}]}},
        ],
    }

    def _ctx(self):
        return {"movement": "Squat", "detail": self.DETAIL}

    def test_clamp_int_bounds_coerces_and_falls_back(self) -> None:
        self.assertEqual(chat_service._clamp_int(9, low=1, high=2, default=1), 2)
        self.assertEqual(chat_service._clamp_int(-4, low=1, high=8, default=5), 1)
        self.assertEqual(chat_service._clamp_int("3", low=1, high=8, default=5), 3)
        self.assertEqual(chat_service._clamp_int("lots", low=1, high=8, default=5), 5)
        self.assertEqual(chat_service._clamp_int(None, low=1, high=8, default=5), 5)

    def test_clamp_int_falls_back_on_a_non_finite_float(self) -> None:
        # A genuine defect this task's brief warned against: json.loads accepts bare Infinity as
        # an extension, so a model can hand this function an actual float('inf'), and
        # int(float('inf')) raises OverflowError -- not TypeError/ValueError, the tuple the brief
        # shipped verbatim. Added beyond the brief's own tests to give the fix real coverage.
        self.assertEqual(chat_service._clamp_int(float("inf"), low=1, high=8, default=5), 5)
        self.assertEqual(chat_service._clamp_int(float("-inf"), low=1, high=8, default=5), 5)

    def test_parse_tool_args_tolerates_missing_and_malformed_json(self) -> None:
        self.assertEqual(chat_service._parse_tool_args('{"a": 1}'), {"a": 1})
        self.assertEqual(chat_service._parse_tool_args(""), {})
        self.assertEqual(chat_service._parse_tool_args("{not json"), {})
        self.assertEqual(chat_service._parse_tool_args("[1,2]"), {})  # not an object

    def test_get_analysis_without_a_fault_name_returns_clip_level_material(self) -> None:
        out = json.loads(
            chat_service._dispatch_tool("get_analysis", {"include": "all"}, self._ctx()).text
        )
        self.assertEqual(out["quality"], {"valid_frame_ratio": 0.9})
        self.assertEqual(out["view"], {"view_type": "side"})
        self.assertEqual([d["fault_name"] for d in out["detections"]], ["Insufficient Depth"])
        self.assertNotIn("evidence", out["detections"][0])  # summary only, not the full dict

    def test_get_analysis_evidence_carries_the_rep_attribution(self) -> None:
        # Per-rep identity is what makes "第 2 rep 膝蓋幾度" answerable; it exists on the detection
        # but not in the compact blob the prompt is built from, so this tool is the only path to it.
        out = json.loads(
            chat_service._dispatch_tool(
                "get_analysis",
                {"fault_name": "Insufficient Depth", "include": "evidence"},
                self._ctx(),
            ).text
        )
        self.assertEqual(out["measured"]["rep_index"], 2)
        self.assertEqual(out["measured"]["occurred_reps"], [2])
        self.assertEqual(out["measured"]["rep_count"], 1)

    def test_get_analysis_evidence_returns_measurements_and_no_knowledge(self) -> None:
        out = json.loads(
            chat_service._dispatch_tool(
                "get_analysis",
                {"fault_name": "insufficient depth", "include": "evidence"},  # case-insensitive
                self._ctx(),
            ).text
        )
        self.assertEqual(out["evidence"], {"hip_knee_delta_deg": 12.5})
        self.assertEqual(out["measured"]["peak_frame"], 45)
        self.assertNotIn("knowledge", out)

    def test_get_analysis_knowledge_returns_only_that_faults_retrievals(self) -> None:
        out = json.loads(
            chat_service._dispatch_tool(
                "get_analysis",
                {"fault_name": "Insufficient Depth", "include": "knowledge"},
                self._ctx(),
            ).text
        )
        self.assertEqual(out["knowledge"], [{"results": [{"text": "squat depth passage"}]}])
        self.assertNotIn("evidence", out)

    def test_get_analysis_unknown_fault_names_what_was_detected(self) -> None:
        out = json.loads(
            chat_service._dispatch_tool(
                "get_analysis", {"fault_name": "Butt Wink", "include": "all"}, self._ctx()
            ).text
        )
        self.assertIn("error", out)
        self.assertEqual(out["detected_faults"], ["Insufficient Depth"])

    def test_kg_query_clamps_hops_and_forces_the_thread_movement(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "graph_context", return_value={"ok": True}
        ) as gc:
            chat_service._dispatch_tool("kg_query", {"query": "valgus", "hops": 99}, self._ctx()).text
        self.assertEqual(gc.call_args.args[0], "valgus")
        self.assertEqual(gc.call_args.kwargs["hops"], 2)  # clamped from 99
        self.assertEqual(gc.call_args.kwargs["movement"], "Squat")

    def test_rag_search_clamps_top_k(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "rag_snippets", return_value={"results": []}
        ) as rs:
            chat_service._dispatch_tool("rag_search", {"query": "ankle", "top_k": 500}, self._ctx()).text
        self.assertEqual(rs.call_args.kwargs["top_k"], 8)  # clamped from 500

    def test_unknown_tool_name_is_an_error_payload_not_an_exception(self) -> None:
        out = json.loads(chat_service._dispatch_tool("launch_missiles", {}, self._ctx()).text)
        self.assertIn("error", out)

    def test_a_raising_tool_becomes_an_error_payload(self) -> None:
        # A missing KG file must not kill an otherwise-fine answer stream.
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "graph_context", side_effect=FileNotFoundError("no graphml")
        ):
            raw = chat_service._dispatch_tool("kg_query", {"query": "x"}, self._ctx()).text
        # kg_query results (error or not) carry the REFERENCE ONLY marker ahead of the JSON payload
        # -- see test_reference_only_prefix_*  below -- so it must be stripped before parsing here.
        out = json.loads(raw[len(chat_service._REFERENCE_ONLY_PREFIX) :])
        self.assertIn("error", out)
        self.assertIn("no graphml", out["error"])

    def test_a_huge_tool_result_is_truncated(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "rag_snippets", return_value={"results": ["x" * 50_000]}
        ):
            out = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx()).text
        self.assertLessEqual(len(out), chat_service._MAX_TOOL_RESULT_CHARS + 32)
        self.assertTrue(out.endswith("…[truncated]"))

    def test_a_result_with_a_non_string_dict_key_does_not_raise(self) -> None:
        # json.dumps' `default` hook is never consulted for dict KEYS -- only values -- so a
        # non-string/int/float/bool/None key raises TypeError regardless of `default=str`. This
        # must be caught the same as a tool call raising, not escape as an unhandled exception.
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "rag_snippets", return_value={("a", "b"): "value"}
        ):
            raw = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx()).text
        out = json.loads(raw[len(chat_service._REFERENCE_ONLY_PREFIX) :])
        self.assertIn("error", out)
        self.assertIn("rag_search failed", out["error"])

    def test_a_circular_result_does_not_raise(self) -> None:
        # json's own cycle detector raises ValueError before `default` is ever reached, so
        # `default=str` cannot rescue a self-referencing structure either.
        from backend.app.services import knowledge as knowledge_service

        circular: dict[str, Any] = {}
        circular["self"] = circular
        with mock.patch.object(knowledge_service, "graph_context", return_value=circular):
            raw = chat_service._dispatch_tool("kg_query", {"query": "x"}, self._ctx()).text
        out = json.loads(raw[len(chat_service._REFERENCE_ONLY_PREFIX) :])
        self.assertIn("error", out)
        self.assertIn("kg_query failed", out["error"])

    def test_a_value_whose_str_raises_does_not_raise(self) -> None:
        # `default=str` calls str(obj) on an otherwise-unserialisable value; if that object's own
        # __str__ raises, json.dumps propagates whatever it raised -- this is the third of the
        # three `default=str` failure modes, distinct from the key and cycle cases above.
        from backend.app.services import knowledge as knowledge_service

        class _Explodes:
            def __str__(self) -> str:
                raise RuntimeError("boom")

        with mock.patch.object(
            knowledge_service, "rag_snippets", return_value={"bad": _Explodes()}
        ):
            raw = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx()).text
        out = json.loads(raw[len(chat_service._REFERENCE_ONLY_PREFIX) :])
        self.assertIn("error", out)
        self.assertIn("rag_search failed", out["error"])

    def test_a_raising_tool_whose_exception_arg_has_a_broken_str_does_not_raise(self) -> None:
        # _run_tool's own except block builds "{name} failed: {exc}", and f"{exc}" calls
        # str(exc) -- BaseException.__str__ returns str(args[0]) for a single-arg exception, so an
        # exception constructed AROUND an object whose own __str__ raises (a ValueError wrapping a
        # bad value, say) makes that interpolation raise too, from inside the except block that was
        # supposed to be the safety net. Pins the nested try added to _run_tool in review.
        from backend.app.services import knowledge as knowledge_service

        class _Explodes:
            def __str__(self) -> str:
                raise RuntimeError("boom")

        with mock.patch.object(
            knowledge_service, "graph_context", side_effect=ValueError(_Explodes())
        ):
            raw = chat_service._dispatch_tool("kg_query", {"query": "x"}, self._ctx()).text
        out = json.loads(raw[len(chat_service._REFERENCE_ONLY_PREFIX) :])
        self.assertIn("error", out)
        # Degrades to the exception's CLASS NAME (not the ordinary "{name} failed: {exc}" message)
        # only in this one case -- the ordinary path is pinned by the tests just above.
        self.assertIn("kg_query failed", out["error"])
        self.assertIn("ValueError", out["error"])

    def test_reference_only_prefix_marks_knowledge_tools_but_not_get_analysis(self) -> None:
        # FIX 2 (whole-branch review): the honesty rule for retrieved knowledge lives on a single
        # system-prompt sentence today (_TOOL_GROUNDING_RULE); this puts a marker on the data itself
        # so it survives even if a model pays less attention to instructions than to tool results.
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(knowledge_service, "graph_context", return_value={"ok": True}):
            kg_out = chat_service._dispatch_tool("kg_query", {"query": "x"}, self._ctx()).text
        with mock.patch.object(knowledge_service, "rag_snippets", return_value={"results": []}):
            rag_out = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx()).text
        analysis_out = chat_service._dispatch_tool("get_analysis", {"include": "all"}, self._ctx()).text

        self.assertTrue(kg_out.startswith(chat_service._REFERENCE_ONLY_PREFIX))
        self.assertTrue(rag_out.startswith(chat_service._REFERENCE_ONLY_PREFIX))
        # get_analysis IS an observation about this video (_TOOL_GROUNDING_RULE says so explicitly)
        # -- prefixing it here would contradict that claim.
        self.assertFalse(analysis_out.startswith("REFERENCE ONLY"))
        self.assertNotIn("REFERENCE ONLY", analysis_out)

    def test_rag_sources_use_reference_and_source_type(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        payload = {
            "query": "ankle",
            "results": [
                {
                    "text": "…",
                    "metadata": {
                        "reference": "Wikipedia: Squat (exercise)",
                        "source_type": "encyclopedia",
                        "source": r"data\rag\docs\squat_wiki.txt",
                    },
                }
            ],
        }
        with mock.patch.object(knowledge_service, "rag_snippets", return_value=payload):
            out = chat_service._dispatch_tool("rag_search", {"query": "ankle"}, self._ctx())
        self.assertEqual(
            out.sources, [{"label": "Wikipedia: Squat (exercise)", "kind": "encyclopedia"}]
        )

    def test_rag_sources_never_leak_the_server_path(self) -> None:
        # metadata.source is a server filesystem path. It must not reach the client in ANY field,
        # including as the fallback label — the fallback is the BASENAME only.
        from backend.app.services import knowledge as knowledge_service

        payload = {
            "results": [
                {"text": "…", "metadata": {"source": r"data\rag\docs\squat_wiki.txt"}},
                {"text": "…", "metadata": {"source": "/srv/x-coach/data/rag/docs/ohp.pdf"}},
            ]
        }
        with mock.patch.object(knowledge_service, "rag_snippets", return_value=payload):
            out = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx())
        labels = [s["label"] for s in out.sources]
        self.assertEqual(labels, ["squat_wiki.txt", "ohp.pdf"])
        blob = json.dumps(out.sources)
        self.assertNotIn("data", blob.replace("squat_wiki", "").replace("ohp", ""))
        self.assertNotIn("srv", blob)

    def test_rag_sources_are_deduped_and_capped(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        results = [
            {"metadata": {"reference": f"Doc {i // 3}", "source_type": "paper"}} for i in range(24)
        ]
        with mock.patch.object(
            knowledge_service, "rag_snippets", return_value={"results": results}
        ):
            out = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx())
        labels = [s["label"] for s in out.sources]
        self.assertEqual(len(labels), chat_service._MAX_TOOL_SOURCES)
        self.assertEqual(len(set(labels)), chat_service._MAX_TOOL_SOURCES)  # deduped
        self.assertEqual(labels[0], "Doc 0")  # first-seen order preserved

    def test_kg_sources_are_concepts_not_citations(self) -> None:
        # A KG node has no source field anywhere, so its `kind` is the literal "concept" — that is
        # what the renderer keys off to keep graph nodes out of the citation slot.
        from backend.app.services import knowledge as knowledge_service

        payload = {
            "matched_nodes": ["Squat:Insufficient Depth"],
            "subgraph": {
                "nodes": [
                    {"node_id": "Depth", "name": "Depth", "label": "QualityDimension"},
                    {"node_id": "Ankle", "name": "Ankle Mobility", "label": "Concept"},
                ],
                "edges": [],
            },
        }
        with mock.patch.object(knowledge_service, "graph_context", return_value=payload):
            out = chat_service._dispatch_tool("kg_query", {"query": "depth"}, self._ctx())
        self.assertEqual({s["kind"] for s in out.sources}, {"concept"})
        labels = [s["label"] for s in out.sources]
        self.assertIn("Insufficient Depth", labels)  # the "Squat:" prefix is stripped
        self.assertNotIn("Squat:Insufficient Depth", labels)
        self.assertIn("Ankle Mobility", labels)
        self.assertNotIn("QualityDimension", labels)  # internal taxonomy is not a source

    def test_kg_sources_drop_labels_that_are_blank_after_stripping(self) -> None:
        # Review fix: a matched node id of "Squat:" (nothing after the movement prefix) used to
        # survive filtering because the filter checked the PRE-split id, not the label actually
        # emitted -- producing a blank chip that still consumed one of the five dedupe slots. A
        # subgraph node with a blank/whitespace name is the same bug on the other branch.
        from backend.app.services import knowledge as knowledge_service

        payload = {
            "matched_nodes": ["Squat:", "Squat:Insufficient Depth"],
            "subgraph": {
                "nodes": [
                    {"node_id": "Blank", "name": "   ", "label": "Concept"},
                    {"node_id": "Ankle", "name": "Ankle Mobility", "label": "Concept"},
                ],
                "edges": [],
            },
        }
        with mock.patch.object(knowledge_service, "graph_context", return_value=payload):
            out = chat_service._dispatch_tool("kg_query", {"query": "depth"}, self._ctx())
        labels = [s["label"] for s in out.sources]
        self.assertNotIn("", labels)
        self.assertEqual(labels, ["Insufficient Depth", "Ankle Mobility"])

    def test_get_analysis_reports_no_sources(self) -> None:
        out = chat_service._dispatch_tool("get_analysis", {"include": "all"}, self._ctx())
        self.assertEqual(out.sources, [])

    def test_a_failing_tool_reports_no_sources(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "graph_context", side_effect=FileNotFoundError("no graphml")
        ):
            out = chat_service._dispatch_tool("kg_query", {"query": "x"}, self._ctx())
        self.assertEqual(out.sources, [])
        self.assertIn("no graphml", out.text)

    def test_sources_survive_truncation_of_a_huge_result(self) -> None:
        # THE LOAD-BEARING TEST. Sources are derived from the RAW result, before truncation — a hit
        # big enough to be cut is exactly the one whose provenance matters most.
        from backend.app.services import knowledge as knowledge_service

        payload = {
            "results": [
                {
                    "text": "x" * 60_000,
                    "metadata": {"reference": "Big Review 2026", "source_type": "paper"},
                }
            ]
        }
        with mock.patch.object(knowledge_service, "rag_snippets", return_value=payload):
            out = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx())
        self.assertTrue(out.text.endswith("…[truncated]"))
        self.assertEqual(out.sources, [{"label": "Big Review 2026", "kind": "paper"}])

    def test_tool_sources_is_total_over_garbage(self) -> None:
        # _tool_sources runs inside the never-raises path, so it must survive any shape a tool or a
        # future provider could hand it.
        for junk in (None, 42, "text", [], {"results": "not-a-list"}, {"results": [None, 7]},
                     {"results": [{"metadata": "not-a-dict"}]},
                     {"subgraph": None}, {"subgraph": {"nodes": "nope"}}, {"subgraph": {"nodes": [42]}},
                     {"matched_nodes": 5}):
            for name in ("rag_search", "kg_query", "get_analysis"):
                self.assertIsInstance(chat_service._tool_sources(name, junk), list)

    def test_dispatch_survives_tool_sources_itself_raising(self) -> None:
        # _tool_sources is total against every SHAPE a tool result can take (see the test above),
        # but it still calls str() on values it does not own. _dispatch_tool wraps the call anyway,
        # on the same "never trust a helper to be as total as it claims" principle as the json.dumps
        # guard just below it -- provenance is a nice-to-have and must never sink a good answer.
        with mock.patch.object(
            chat_service, "_tool_sources", side_effect=RuntimeError("boom")
        ):
            out = chat_service._dispatch_tool("get_analysis", {"include": "all"}, self._ctx())
        self.assertEqual(out.sources, [])
        self.assertIn("quality", out.text)  # the tool itself still succeeded

    def test_query_label_picks_the_right_argument_per_tool(self) -> None:
        self.assertEqual(
            chat_service._tool_query_label("kg_query", {"query": "knee valgus"}), "knee valgus"
        )
        self.assertEqual(
            chat_service._tool_query_label("get_analysis", {"fault_name": "Depth"}), "Depth"
        )
        self.assertEqual(chat_service._tool_query_label("get_analysis", {}), "")

    def test_tool_schemas_are_the_three_expected_functions(self) -> None:
        names = [t["function"]["name"] for t in chat_service._TOOLS]
        self.assertEqual(names, ["get_analysis", "kg_query", "rag_search"])
        for tool in chat_service._TOOLS:
            self.assertEqual(tool["type"], "function")
            self.assertIn("description", tool["function"])
            self.assertIn("properties", tool["function"]["parameters"])


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
        ), mock.patch.object(
            # Pin the allow-list so resolution is hermetic (independent of the deployment's real
            # LLM_MODELS): the client's "minimax/minimax-m3" is offered, so it passes through.
            app_settings, "get_settings", return_value=_fake_models("minimax/minimax-m3,openai/gpt-oss-120b")
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

    @staticmethod
    def _run_followups(body):
        return asyncio.run(chat_router.chat_followups(body, user=_USER))

    def test_followups_503_when_chat_not_configured(self) -> None:
        with mock.patch.object(
            chat_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=False)
        ):
            with self.assertRaises(HTTPException) as ctx:
                self._run_followups(_body([{"role": "user", "content": "hi"}], _FAULT_CTX))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_followups_returns_questions_using_the_pinned_fast_model(self) -> None:
        # The endpoint accepts a thread ending on the assistant turn (no last-must-be-user check) and
        # uses the server-pinned fast follow-up model — NOT the client's answer model.
        captured: dict = {}

        def fake_suggest(*, messages, context, model):
            captured["messages"] = messages
            captured["model"] = model
            return ["Widen my stance?", "Go lower next rep?"]

        body = chat_router.ChatRequest(
            messages=[
                {"role": "user", "content": "why did my knees cave?"},
                {"role": "assistant", "content": "Drive your knees out."},
            ],
            context=_FAULT_CTX,
            model="minimax/minimax-m3",  # the (slow) answer model — must be ignored for followups
        )
        with mock.patch.object(
            chat_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ), mock.patch.object(
            chat_router, "followup_chat_model", return_value="openai/gpt-oss-120b"
        ), mock.patch.object(chat_service, "suggest_followups", fake_suggest):
            resp = self._run_followups(body)

        self.assertEqual(resp.questions, ["Widen my stance?", "Go lower next rep?"])
        self.assertEqual(captured["messages"][-1]["role"], "assistant")  # thread ends on the answer
        self.assertEqual(captured["model"], "openai/gpt-oss-120b")  # pinned, not "minimax/minimax-m3"

    def test_chat_context_accepts_a_detail_blob(self) -> None:
        from backend.app.routers.chat import ChatContext

        ctx = ChatContext(fault_count=0, detail={"detections": [], "retrievals": []})
        self.assertEqual(ctx.model_dump()["detail"], {"detections": [], "retrievals": []})
        self.assertIsNone(ChatContext(fault_count=0).model_dump()["detail"])  # optional


def _fake_models(models: str, followup: str = "openai/gpt-oss-120b"):
    return types.SimpleNamespace(llm_models=models, llm_followup_model=followup)


class FollowupModelTests(unittest.TestCase):
    def test_returns_the_pinned_fast_model(self) -> None:
        s = _fake_models("deepseek/deepseek-v4-flash", followup="openai/gpt-oss-120b")
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            self.assertEqual(app_settings.followup_chat_model(), "openai/gpt-oss-120b")

    def test_blank_falls_back_to_the_default_answer_model(self) -> None:
        # A self-hoster who blanks LLM_FOLLOWUP_MODEL reuses the default answer model.
        s = _fake_models("deepseek/deepseek-v4-flash,minimax/minimax-m3", followup="  ")
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            self.assertEqual(app_settings.followup_chat_model(), "deepseek/deepseek-v4-flash")


class ChatModelsCatalogTests(unittest.TestCase):
    def test_parses_env_list_preserving_order(self) -> None:
        s = _fake_models("deepseek/deepseek-v4-flash, xiaomi/mimo-v2.5 ,minimax/minimax-m3")  # spaces ok
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            self.assertEqual(
                app_settings.chat_models(),
                ["deepseek/deepseek-v4-flash", "xiaomi/mimo-v2.5", "minimax/minimax-m3"],
            )

    def test_dedupes_preserving_order(self) -> None:
        s = _fake_models("custom/model,custom/model,minimax/minimax-m3")
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            self.assertEqual(app_settings.chat_models(), ["custom/model", "minimax/minimax-m3"])

    def test_blank_list_falls_back_to_one_built_in_model(self) -> None:
        s = _fake_models("  , , ")  # misconfigured to empty
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            cat = app_settings.chat_models()
        self.assertEqual(len(cat), 1)  # never empty
        self.assertTrue(cat[0])


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
        # Set LLM_MODELS to one custom slug -> it's the whole picker AND the default.
        s = _fake_models("some/self-hosted-model")
        with mock.patch.object(app_settings, "get_settings", return_value=s):
            self.assertEqual(app_settings.default_chat_model(), "some/self-hosted-model")
            self.assertEqual(
                app_settings.resolve_chat_model("some/self-hosted-model"), "some/self-hosted-model"
            )
            self.assertEqual(app_settings.resolve_chat_model(None), "some/self-hosted-model")


class TestChatPromptMovement(unittest.TestCase):
    def _prompt(self, **context) -> str:
        from backend.app.services.chat import _build_system_prompt

        base = {"quality": {"valid_frame_ratio": 0.9}, "faults": [], "fault_count": 0}
        base.update(context)
        return _build_system_prompt(base)

    def test_preamble_names_the_movement(self) -> None:
        prompt = self._prompt(movement="Push-up")
        self.assertIn("Push-up coach", prompt)
        self.assertNotIn("squat coach", prompt.lower())

    def test_clean_rep_branch_names_the_movement(self) -> None:
        """The spec's section 9 mitigation: a measurable clip measured by the WRONG rules is
        now reachable, so the clean verdict must be scoped to the movement the user asserted
        rather than stated bare."""
        prompt = self._prompt(movement="Overhead Press")
        self.assertIn("CLEAN Overhead Press REP", prompt)

    def test_unmeasured_branch_is_unchanged_by_movement(self) -> None:
        """An unmeasured clip must still refuse to congratulate, whatever the movement."""
        prompt = self._prompt(movement="Push-up", quality={"valid_frame_ratio": 0.0})
        self.assertIn("NOT MEASURED", prompt)
        self.assertNotIn("CLEAN", prompt)

    def test_defaults_to_squat_for_an_older_client(self) -> None:
        """A client that predates ChatContext.movement must still get a coherent prompt."""
        self.assertIn("Squat coach", self._prompt())

    def test_followup_instruction_names_the_movement(self) -> None:
        from backend.app.services.chat import _followup_instruction

        self.assertIn("THIS Push-up", _followup_instruction("Push-up"))


if __name__ == "__main__":
    unittest.main()
