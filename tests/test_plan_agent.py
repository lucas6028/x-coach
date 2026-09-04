"""Tests for the Lumen plan agent (``POST /api/plans/chat``).

Three layers, matching how ``test_chat_endpoint.py`` and ``test_plans_store.py`` split theirs:

* **helpers** — ``_validate_int``/``_canonicalize_movement``/``_plan_query_label``/``_system_prompt``,
  pure functions tested directly with no I/O.
* **dispatcher** — ``_PlanDispatcher`` exercised against the SAME in-memory PostgREST stand-in
  ``test_plans_store.py`` uses (``_FakeDb``), so "second create refused", "unknown movement",
  "out-of-range", and "missing item" all run against REAL store calls, not a mock that would pass
  regardless of what the dispatcher actually sent.
* **loop + router** — ``plan_chat_stream`` with ``chat_service._stream_turn`` faked (mirrors
  ``ToolLoopTests`` in ``test_chat_endpoint.py``), proving the frame shapes the frontend depends on
  (``tool_done.plan``, ``done.plan_id``, the language switch) and the router's pre-flight contract
  (401/503/422/404) mirroring ``ChatRouterTests``.

The refactor this feature required — extracting ``chat_service._run_tool_loop`` out of the
answer-only loop — is verified by NOT touching ``test_chat_endpoint.py``: it stays byte-identical
and its 114 cases still pass unchanged, which is the actual proof the extraction was behaviour
preserving.
"""

from __future__ import annotations

import asyncio
import json
import types
import unittest
import uuid
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app import settings as app_settings
from backend.app.auth import CurrentUser, get_current_user
from backend.app.main import app
from backend.app.routers import plans as plans_router
from backend.app.services import chat as chat_service
from backend.app.services import plan_agent
from backend.app.services import plans as plans_store
from tests.test_plans_store import _FakeDb

_USER = CurrentUser(id="u1", token="tok", email="u@example.com")


# --------------------------------------------------------------------------------- helper functions


class ValidateIntTests(unittest.TestCase):
    def test_within_range_passes_through(self) -> None:
        self.assertEqual(plan_agent._validate_int(5, low=1, high=7, field="day_index"), (5, None))

    def test_below_range_is_an_error_naming_the_bound(self) -> None:
        value, error = plan_agent._validate_int(0, low=1, high=7, field="day_index")
        self.assertIsNone(value)
        self.assertIn("between 1 and 7", error)

    def test_above_range_is_an_error_naming_the_bound(self) -> None:
        value, error = plan_agent._validate_int(21, low=1, high=20, field="sets")
        self.assertIsNone(value)
        self.assertIn("between 1 and 20", error)

    def test_non_numeric_is_an_error(self) -> None:
        value, error = plan_agent._validate_int("lots", low=1, high=20, field="sets")
        self.assertIsNone(value)
        self.assertIn("whole number", error)

    def test_none_is_an_error(self) -> None:
        value, error = plan_agent._validate_int(None, low=1, high=20, field="sets")
        self.assertIsNone(value)
        self.assertIn("whole number", error)

    def test_non_finite_float_is_an_error_not_a_crash(self) -> None:
        # json.loads accepts bare Infinity as an extension, and int(float('inf')) raises
        # OverflowError -- the same trap chat_service._clamp_int guards against.
        value, error = plan_agent._validate_int(float("inf"), low=1, high=20, field="sets")
        self.assertIsNone(value)
        self.assertIn("whole number", error)


class CanonicalizeMovementTests(unittest.TestCase):
    def test_resolves_case_and_whitespace(self) -> None:
        resolved, error = plan_agent._canonicalize_movement("  push-UP ")
        self.assertEqual(resolved, "Push-up")
        self.assertIsNone(error)

    def test_unknown_movement_lists_the_sixteen(self) -> None:
        resolved, error = plan_agent._canonicalize_movement("Burpee")
        self.assertIsNone(resolved)
        self.assertIn("Unknown movement", error)
        self.assertIn("Squat", error)
        self.assertIn("Jumping Jacks", error)  # unanalysable movements are still nameable

    def test_none_input_is_unknown(self) -> None:
        resolved, error = plan_agent._canonicalize_movement(None)
        self.assertIsNone(resolved)
        self.assertIn("Unknown movement", error)


class QueryLabelTests(unittest.TestCase):
    def test_item_tools_use_the_movement(self) -> None:
        self.assertEqual(plan_agent._plan_query_label("add_item", {"movement": "Squat"}), "Squat")
        self.assertEqual(plan_agent._plan_query_label("update_item", {"movement": "Row"}), "Row")

    def test_plan_level_tools_use_the_name(self) -> None:
        self.assertEqual(plan_agent._plan_query_label("create_plan", {"name": "My week"}), "My week")
        self.assertEqual(plan_agent._plan_query_label("update_plan", {"name": "Renamed"}), "Renamed")

    def test_get_and_remove_carry_no_label(self) -> None:
        self.assertEqual(plan_agent._plan_query_label("get_plan", {}), "")
        self.assertEqual(plan_agent._plan_query_label("remove_item", {"item_id": "x"}), "")


class SystemPromptTests(unittest.TestCase):
    def test_zh_hant_reads_like_a_local_app_not_a_translation(self) -> None:
        prompt = plan_agent._system_prompt(lang="zh-Hant", plan_scoped=False)
        self.assertIn("繁體中文", prompt)

    def test_english_lang_line(self) -> None:
        prompt = plan_agent._system_prompt(lang="en", plan_scoped=False)
        self.assertIn("Reply in English.", prompt)

    def test_builder_mode_when_unscoped(self) -> None:
        prompt = plan_agent._system_prompt(lang="en", plan_scoped=False)
        self.assertIn("BUILDER mode", prompt)

    def test_scoped_mode_tells_the_model_to_read_first(self) -> None:
        prompt = plan_agent._system_prompt(lang="en", plan_scoped=True)
        self.assertNotIn("BUILDER mode", prompt)
        self.assertIn("Call get_plan", prompt)

    def test_carries_the_full_movement_catalog(self) -> None:
        prompt = plan_agent._system_prompt(lang="en", plan_scoped=False)
        self.assertIn("Squat", prompt)
        self.assertIn("Overhead Press", prompt)
        self.assertIn("Jumping Jacks", prompt)
        self.assertIn("not yet analysable", prompt)


class PlanToolsShapeTests(unittest.TestCase):
    def test_exactly_the_six_tools(self) -> None:
        names = {t["function"]["name"] for t in plan_agent.PLAN_TOOLS}
        self.assertEqual(
            names, {"get_plan", "create_plan", "add_item", "update_item", "remove_item", "update_plan"}
        )


# ------------------------------------------------------------------------------------- dispatcher


class _DispatcherTestCase(unittest.TestCase):
    """Shared setup: a fresh in-memory PostgREST stand-in per test, as user ``u1``."""

    def setUp(self) -> None:
        self.db = _FakeDb()
        patcher = mock.patch.object(plans_store, "_user_client", return_value=self.db)
        self.addCleanup(patcher.stop)
        patcher.start()

    def _dispatcher(self, plan_id: str | None = None) -> plan_agent._PlanDispatcher:
        return plan_agent._PlanDispatcher(token="tok", user_id="u1", plan_id=plan_id)

    def _create(self, dispatcher: plan_agent._PlanDispatcher, **overrides) -> dict:
        args = {"name": "My week", "items": [{"day_index": 1, "movement": "Squat"}]}
        args.update(overrides)
        out = dispatcher.dispatch("create_plan", args)
        return json.loads(out.text)


class CreatePlanTests(_DispatcherTestCase):
    def test_creates_and_pins_the_plan_id(self) -> None:
        d = self._dispatcher()
        self.assertIsNone(d.plan_id)
        out = d.dispatch("create_plan", {"name": "W", "items": [{"day_index": 1, "movement": "Squat"}]})
        self.assertIsNotNone(d.plan_id)
        self.assertIsNotNone(out.payload)
        self.assertEqual(out.payload["plan"]["id"], d.plan_id)
        self.assertEqual(len(out.payload["plan"]["items"]), 1)

    def test_default_sets_and_reps(self) -> None:
        d = self._dispatcher()
        out = self._create(d, items=[{"day_index": 1, "movement": "Squat"}])
        item = out["plan"]["items"][0]
        self.assertEqual((item["sets"], item["reps"]), (3, 10))

    def test_a_second_create_in_the_same_conversation_is_refused(self) -> None:
        d = self._dispatcher()
        self._create(d)
        first_id = d.plan_id
        out = d.dispatch("create_plan", {"name": "Another", "items": [{"day_index": 1, "movement": "Row"}]})
        self.assertIsNone(out.payload)  # refused: nothing changed, no frame plan
        body = json.loads(out.text)
        self.assertIn("already exists", body["error"])
        self.assertEqual(d.plan_id, first_id)  # unchanged

    def test_create_when_a_plan_id_was_already_scoped_is_also_refused(self) -> None:
        d = self._dispatcher(plan_id="pre-existing")
        out = d.dispatch("create_plan", {"name": "X", "items": [{"day_index": 1, "movement": "Squat"}]})
        self.assertIn("already exists", json.loads(out.text)["error"])

    def test_blank_name_is_rejected(self) -> None:
        d = self._dispatcher()
        out = d.dispatch("create_plan", {"name": "   ", "items": [{"day_index": 1, "movement": "Squat"}]})
        self.assertIn("name", json.loads(out.text)["error"])
        self.assertIsNone(d.plan_id)

    def test_no_items_is_rejected(self) -> None:
        d = self._dispatcher()
        out = d.dispatch("create_plan", {"name": "W", "items": []})
        self.assertIn("at least one item", json.loads(out.text)["error"])

    def test_items_not_a_list_is_rejected(self) -> None:
        d = self._dispatcher()
        out = d.dispatch("create_plan", {"name": "W", "items": "Squat"})
        self.assertIn("at least one item", json.loads(out.text)["error"])

    def test_a_non_object_item_is_rejected(self) -> None:
        d = self._dispatcher()
        out = d.dispatch("create_plan", {"name": "W", "items": ["Squat"]})
        self.assertIn("object", json.loads(out.text)["error"])

    def test_an_invalid_day_index_in_one_item_aborts_the_whole_create(self) -> None:
        d = self._dispatcher()
        out = d.dispatch(
            "create_plan", {"name": "W", "items": [{"day_index": 9, "movement": "Squat"}]}
        )
        self.assertIn("between 1 and 7", json.loads(out.text)["error"])
        self.assertIsNone(d.plan_id)  # nothing was created

    def test_an_unknown_movement_in_one_item_aborts_the_whole_create(self) -> None:
        d = self._dispatcher()
        out = d.dispatch(
            "create_plan", {"name": "W", "items": [{"day_index": 1, "movement": "Burpee"}]}
        )
        self.assertIn("Unknown movement", json.loads(out.text)["error"])
        self.assertIsNone(d.plan_id)

    def test_notes_are_truncated_not_rejected(self) -> None:
        d = self._dispatcher()
        out = self._create(d, notes="x" * 600)
        self.assertEqual(len(out["plan"]["notes"]), 500)


class AddItemTests(_DispatcherTestCase):
    def test_requires_a_scoped_plan(self) -> None:
        d = self._dispatcher()
        out = d.dispatch("add_item", {"day_index": 1, "movement": "Squat"})
        self.assertIn("create one first", json.loads(out.text)["error"])

    def test_adds_to_the_scoped_plan(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("add_item", {"day_index": 2, "movement": "Row", "sets": 4, "reps": 8})
        self.assertIsNotNone(out.payload)
        self.assertEqual(len(out.payload["plan"]["items"]), 2)

    def test_unknown_movement_names_all_sixteen(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("add_item", {"day_index": 1, "movement": "Burpee"})
        self.assertIsNone(out.payload)
        self.assertIn("Unknown movement", json.loads(out.text)["error"])

    def test_out_of_range_sets_is_rejected(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("add_item", {"day_index": 1, "movement": "Squat", "sets": 99})
        self.assertIn("between 1 and 20", json.loads(out.text)["error"])

    def test_out_of_range_reps_is_rejected(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("add_item", {"day_index": 1, "movement": "Squat", "reps": 999})
        self.assertIn("between 1 and 200", json.loads(out.text)["error"])

    def test_out_of_range_day_index_is_rejected(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("add_item", {"day_index": 0, "movement": "Squat"})
        self.assertIn("between 1 and 7", json.loads(out.text)["error"])


class UpdateItemTests(_DispatcherTestCase):
    def _item_id(self, d: plan_agent._PlanDispatcher) -> str:
        created = self._create(d)
        return created["plan"]["items"][0]["id"]

    def test_requires_a_scoped_plan(self) -> None:
        d = self._dispatcher()
        out = d.dispatch("update_item", {"item_id": "x", "sets": 5})
        self.assertIn("No plan is scoped", json.loads(out.text)["error"])

    def test_blank_item_id_is_rejected(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("update_item", {"item_id": "", "sets": 5})
        self.assertIn("needs item_id", json.loads(out.text)["error"])

    def test_missing_item_is_an_error_payload(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("update_item", {"item_id": str(uuid.uuid4()), "sets": 5})
        self.assertIsNone(out.payload)
        self.assertIn("No item", json.loads(out.text)["error"])

    def test_rejects_a_bad_day_index(self) -> None:
        d = self._dispatcher()
        item_id = self._item_id(d)
        out = d.dispatch("update_item", {"item_id": item_id, "day_index": 8})
        self.assertIn("between 1 and 7", json.loads(out.text)["error"])

    def test_rejects_an_unknown_movement(self) -> None:
        d = self._dispatcher()
        item_id = self._item_id(d)
        out = d.dispatch("update_item", {"item_id": item_id, "movement": "Burpee"})
        self.assertIn("Unknown movement", json.loads(out.text)["error"])

    def test_rejects_a_bad_sets_value(self) -> None:
        d = self._dispatcher()
        item_id = self._item_id(d)
        out = d.dispatch("update_item", {"item_id": item_id, "sets": 0})
        self.assertIn("between 1 and 20", json.loads(out.text)["error"])

    def test_rejects_a_bad_reps_value(self) -> None:
        d = self._dispatcher()
        item_id = self._item_id(d)
        out = d.dispatch("update_item", {"item_id": item_id, "reps": 0})
        self.assertIn("between 1 and 200", json.loads(out.text)["error"])

    def test_edits_multiple_fields_and_clears_notes(self) -> None:
        d = self._dispatcher()
        item_id = self._item_id(d)
        out = d.dispatch(
            "update_item",
            {
                "item_id": item_id,
                "day_index": 3,
                "movement": "row",
                "sets": 5,
                "reps": 6,
                "notes": None,
            },
        )
        item = out.payload["plan"]["items"][0]
        self.assertEqual(
            (item["day_index"], item["movement"], item["sets"], item["reps"], item["notes"]),
            (3, "Row", 5, 6, None),
        )

    def test_no_fields_reads_the_item_back_unchanged(self) -> None:
        d = self._dispatcher()
        item_id = self._item_id(d)
        out = d.dispatch("update_item", {"item_id": item_id})
        self.assertIsNotNone(out.payload)
        self.assertEqual(out.payload["plan"]["items"][0]["id"], item_id)


class RemoveItemTests(_DispatcherTestCase):
    def test_requires_a_scoped_plan(self) -> None:
        d = self._dispatcher()
        out = d.dispatch("remove_item", {"item_id": "x"})
        self.assertIn("No plan is scoped", json.loads(out.text)["error"])

    def test_blank_item_id_is_rejected(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("remove_item", {"item_id": ""})
        self.assertIn("needs item_id", json.loads(out.text)["error"])

    def test_missing_item_is_an_error_payload(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("remove_item", {"item_id": str(uuid.uuid4())})
        self.assertIsNone(out.payload)
        self.assertIn("No item", json.loads(out.text)["error"])

    def test_removes_the_item(self) -> None:
        d = self._dispatcher()
        created = self._create(d, items=[{"day_index": 1, "movement": "Squat"}, {"day_index": 2, "movement": "Row"}])
        item_id = created["plan"]["items"][0]["id"]
        out = d.dispatch("remove_item", {"item_id": item_id})
        self.assertEqual(len(out.payload["plan"]["items"]), 1)


class UpdatePlanTests(_DispatcherTestCase):
    def test_requires_a_scoped_plan(self) -> None:
        d = self._dispatcher()
        out = d.dispatch("update_plan", {"name": "New name"})
        self.assertIn("No plan is scoped", json.loads(out.text)["error"])

    def test_blank_name_is_rejected(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("update_plan", {"name": "   "})
        self.assertIn("cannot be empty", json.loads(out.text)["error"])

    def test_renames_and_clears_notes(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("update_plan", {"name": "Renamed", "notes": None})
        self.assertEqual(out.payload["plan"]["name"], "Renamed")
        self.assertIsNone(out.payload["plan"]["notes"])

    def test_no_fields_reads_the_plan_back_unchanged(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("update_plan", {})
        self.assertEqual(out.payload["plan"]["name"], "My week")

    def test_a_plan_deleted_mid_conversation_is_an_error(self) -> None:
        d = self._dispatcher()
        self._create(d)
        with mock.patch.object(plans_store, "update_plan", return_value=None):
            out = d.dispatch("update_plan", {"name": "X"})
        self.assertIn("no longer exists", json.loads(out.text)["error"])


class GetPlanTests(_DispatcherTestCase):
    def test_requires_a_scoped_plan(self) -> None:
        d = self._dispatcher()
        out = d.dispatch("get_plan", {})
        self.assertIsNone(out.payload)  # read-only: never carries a frame plan
        self.assertIn("No plan is scoped", json.loads(out.text)["error"])

    def test_reads_the_scoped_plan_but_omits_the_frame_plan(self) -> None:
        d = self._dispatcher()
        self._create(d)
        out = d.dispatch("get_plan", {})
        self.assertIsNone(out.payload)  # read-only tools omit `plan` from the tool_done frame
        body = json.loads(out.text)
        self.assertEqual(body["plan"]["name"], "My week")

    def test_a_plan_deleted_mid_conversation_is_an_error(self) -> None:
        d = self._dispatcher(plan_id="ghost")
        out = d.dispatch("get_plan", {})
        self.assertIn("no longer exists", json.loads(out.text)["error"])


class DispatchBoundaryTests(_DispatcherTestCase):
    """The `dispatch` catch-all: never raises, whatever the handler does."""

    def test_unknown_tool_name_is_an_error_payload(self) -> None:
        d = self._dispatcher()
        out = d.dispatch("launch_missiles", {})
        self.assertIn("Unknown tool", json.loads(out.text)["error"])

    def test_a_raising_store_call_becomes_an_error_payload(self) -> None:
        d = self._dispatcher()
        self._create(d)
        with mock.patch.object(plans_store, "add_item", side_effect=RuntimeError("boom")):
            out = d.dispatch("add_item", {"day_index": 1, "movement": "Squat"})
        body = json.loads(out.text)
        self.assertIn("add_item failed", body["error"])
        self.assertIn("RuntimeError", body["error"])

    def test_result_serialisation_failure_does_not_raise(self) -> None:
        class _Explodes:
            def __str__(self) -> str:
                raise RuntimeError("boom")

        d = self._dispatcher()
        out = d._result({"bad": _Explodes()})
        body = json.loads(out.text)
        self.assertIn("serialisation failed", body["error"])

    def test_a_huge_result_is_truncated(self) -> None:
        d = self._dispatcher()
        out = d._result({"big": "x" * 50_000})
        self.assertLessEqual(len(out.text), plan_agent._MAX_RESULT_CHARS + 32)
        self.assertTrue(out.text.endswith("…[truncated]"))

    def test_fresh_plan_is_none_with_no_scoped_plan_id(self) -> None:
        # Defensive-only in the live tool paths (every caller already checks `plan_id` first), but
        # pinned directly so it is not a partial branch under the coverage gate.
        d = self._dispatcher()
        self.assertIsNone(d._fresh_plan())


# ------------------------------------------------------------------------------------ loop + frames


class LoopFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _FakeDb()
        patcher = mock.patch.object(plans_store, "_user_client", return_value=self.db)
        self.addCleanup(patcher.stop)
        patcher.start()

    @staticmethod
    def _turns(*rounds):
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
        out = []
        for frame in "".join(frames).split("\n\n"):
            if not frame.strip():
                continue
            lines = frame.split("\n")
            event = lines[0][len("event:") :].strip()
            data = json.loads(lines[1][len("data:") :].strip())
            out.append((event, data))
        return out

    def _run(self, fake_turn, *, plan_id=None, lang="zh-Hant", model="m"):
        with mock.patch.object(chat_service, "_stream_turn", fake_turn), mock.patch.object(
            chat_service, "chat_timeout", return_value=60.0
        ):
            return self._events(
                list(
                    plan_agent.plan_chat_stream(
                        messages=[{"role": "user", "content": "hi"}],
                        plan_id=plan_id,
                        token="tok",
                        user_id="u1",
                        model=model,
                        lang=lang,
                    )
                )
            )

    def test_builder_create_then_add_item_in_the_same_turn(self) -> None:
        create_args = json.dumps(
            {"name": "My week", "items": [{"day_index": 1, "movement": "Squat"}]}
        )
        add_args = json.dumps({"day_index": 2, "movement": "Row"})
        fake, calls = self._turns(
            ([], [{"id": "c1", "name": "create_plan", "arguments": create_args}]),
            ([], [{"id": "c2", "name": "add_item", "arguments": add_args}]),
            (["All set."], []),
        )
        events = self._run(fake, plan_id=None)
        tool_dones = [d for e, d in events if e == "tool_done"]
        self.assertEqual(len(tool_dones), 2)
        self.assertEqual(len(tool_dones[0]["plan"]["items"]), 1)
        self.assertEqual(len(tool_dones[1]["plan"]["items"]), 2)
        pinned_id = tool_dones[0]["plan"]["id"]
        self.assertEqual(tool_dones[1]["plan"]["id"], pinned_id)
        done = next(d for e, d in events if e == "done")
        self.assertEqual(done["plan_id"], pinned_id)
        self.assertEqual(done["model"], "m")
        # _fresh_plan re-fetches through get_plan (which projects _PLAN_COLUMNS/_ITEM_COLUMNS)
        # rather than trusting the raw insert row, specifically so user_id never rides this SSE
        # frame to the browser -- pinned as a fact, not just the docstring's claim.
        self.assertNotIn("user_id", tool_dones[0]["plan"])
        self.assertNotIn("user_id", tool_dones[0]["plan"]["items"][0])

    def test_second_create_in_the_same_turn_is_refused(self) -> None:
        create_args = json.dumps(
            {"name": "My week", "items": [{"day_index": 1, "movement": "Squat"}]}
        )
        fake, calls = self._turns(
            ([], [{"id": "c1", "name": "create_plan", "arguments": create_args}]),
            ([], [{"id": "c2", "name": "create_plan", "arguments": create_args}]),
            (["Done."], []),
        )
        events = self._run(fake, plan_id=None)
        tool_dones = [d for e, d in events if e == "tool_done"]
        self.assertIn("plan", tool_dones[0])
        self.assertNotIn("plan", tool_dones[1])
        # The refusal's tool result was appended before the final (3rd) round started.
        self.assertIn("already exists", calls[2]["messages"][-1]["content"])

    def test_done_omits_plan_id_when_no_plan_was_ever_scoped(self) -> None:
        fake, _ = self._turns((["Sure, tell me more."], []))
        events = self._run(fake, plan_id=None)
        done = next(d for e, d in events if e == "done")
        self.assertNotIn("plan_id", done)

    def test_editing_a_pre_scoped_plan_ships_a_fresh_plan_on_tool_done(self) -> None:
        plan = plans_store.create_plan(token="tok", user_id="u1", name="W", items=[])
        add_args = json.dumps({"day_index": 1, "movement": "Squat"})
        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "add_item", "arguments": add_args}]),
            (["Added it."], []),
        )
        events = self._run(fake, plan_id=plan["id"])
        tool_dones = [d for e, d in events if e == "tool_done"]
        self.assertEqual(tool_dones[0]["plan"]["id"], plan["id"])
        done = next(d for e, d in events if e == "done")
        self.assertEqual(done["plan_id"], plan["id"])

    def test_lang_switches_the_prompt_language(self) -> None:
        fake, calls = self._turns((["ok"], []))
        self._run(fake, plan_id=None, lang="en")
        self.assertIn("Reply in English.", calls[0]["messages"][0]["content"])

        fake, calls = self._turns((["ok"], []))
        self._run(fake, plan_id=None, lang="zh-Hant")
        self.assertIn("繁體中文", calls[0]["messages"][0]["content"])

    def test_outer_shell_never_raises(self) -> None:
        with mock.patch.object(plan_agent, "_system_prompt", side_effect=RuntimeError("boom")):
            frames = "".join(
                plan_agent.plan_chat_stream(
                    messages=[{"role": "user", "content": "hi"}],
                    plan_id=None,
                    token="tok",
                    user_id="u1",
                    model="m",
                    lang="zh-Hant",
                )
            )
        self.assertIn("event: error", frames)
        self.assertIn("RuntimeError", frames)
        self.assertNotIn("event: done", frames)


# ------------------------------------------------------------------------------------------ router


def _body(messages, *, plan_id=None, model=None, lang="zh-Hant") -> plans_router.PlanChatRequest:
    return plans_router.PlanChatRequest(messages=messages, plan_id=plan_id, model=model, lang=lang)


async def _collect(resp) -> str:
    out: list[str] = []
    async for chunk in resp.body_iterator:
        out.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(out)


def _fake_models(models: str):
    return types.SimpleNamespace(llm_models=models, llm_followup_model="openai/gpt-oss-120b")


class PlanChatRouterTests(unittest.TestCase):
    @staticmethod
    def _run(body):
        return asyncio.run(plans_router.plan_chat(body, user=_USER))

    def test_503_when_chat_not_configured(self) -> None:
        with mock.patch.object(
            plans_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=False)
        ):
            with self.assertRaises(HTTPException) as ctx:
                self._run(_body([{"role": "user", "content": "hi"}]))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_422_when_last_message_is_not_user(self) -> None:
        body = _body(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        with mock.patch.object(
            plans_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ):
            with self.assertRaises(HTTPException) as ctx:
                self._run(body)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_404_when_plan_id_is_not_a_uuid(self) -> None:
        body = _body([{"role": "user", "content": "hi"}], plan_id="not-a-uuid")
        with mock.patch.object(
            plans_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ):
            with self.assertRaises(HTTPException) as ctx:
                self._run(body)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_404_when_plan_id_is_not_owned_by_the_caller(self) -> None:
        body = _body([{"role": "user", "content": "hi"}], plan_id=str(uuid.uuid4()))
        with mock.patch.object(
            plans_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ), mock.patch.object(plans_store, "plan_exists", return_value=False):
            with self.assertRaises(HTTPException) as ctx:
                self._run(body)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_returns_event_stream_delegating_to_the_plan_agent(self) -> None:
        captured: dict = {}

        def fake_stream(*, messages, plan_id, token, user_id, model, lang):
            captured["messages"] = messages
            captured["plan_id"] = plan_id
            captured["token"] = token
            captured["user_id"] = user_id
            captured["model"] = model
            captured["lang"] = lang
            yield 'event: delta\ndata: {"text": "hi"}\n\n'
            yield 'event: done\ndata: {"model": "minimax/minimax-m3"}\n\n'

        body = _body(
            [{"role": "user", "content": "plan me a week"}],
            model="minimax/minimax-m3",
            lang="en",
        )
        with mock.patch.object(
            plans_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ), mock.patch.object(
            app_settings, "get_settings", return_value=_fake_models("minimax/minimax-m3,openai/gpt-oss-120b")
        ), mock.patch.object(plan_agent, "plan_chat_stream", fake_stream):
            resp = self._run(body)
            out = asyncio.run(_collect(resp))

        self.assertEqual(resp.media_type, "text/event-stream")
        self.assertIn("event: done", out)
        self.assertEqual(captured["model"], "minimax/minimax-m3")
        self.assertEqual(captured["lang"], "en")
        self.assertIsNone(captured["plan_id"])
        self.assertEqual(captured["token"], "tok")
        self.assertEqual(captured["user_id"], "u1")

    def test_scoped_plan_id_is_normalised_to_canonical_form_and_forwarded(self) -> None:
        # Uppercased/differently-hyphenated input must come out canonical (lowercase) -- the only
        # way to tell this path actually runs through `_plan_uuid` rather than passing the raw
        # string straight through.
        canonical_id = str(uuid.uuid4())
        shouted_id = canonical_id.upper()
        captured: dict = {}

        def fake_stream(*, messages, plan_id, token, user_id, model, lang):
            captured["plan_id"] = plan_id
            yield 'event: done\ndata: {"model": "m"}\n\n'

        body = _body([{"role": "user", "content": "add a day"}], plan_id=shouted_id)
        with mock.patch.object(
            plans_router, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ), mock.patch.object(
            app_settings, "get_settings", return_value=_fake_models("m")
        ), mock.patch.object(
            plans_store, "plan_exists", return_value=True
        ), mock.patch.object(plan_agent, "plan_chat_stream", fake_stream):
            resp = self._run(body)
            asyncio.run(_collect(resp))

        self.assertEqual(captured["plan_id"], canonical_id)


class PlanChatAuthTests(unittest.TestCase):
    """/api/plans/chat is one user's own data and must 401 without a session."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.assertEqual(
            app.dependency_overrides, {}, "a leaked dependency override would hide a 401 regression"
        )

    def test_requires_auth(self) -> None:
        resp = self.client.post(
            "/api/plans/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(resp.status_code, 401)
