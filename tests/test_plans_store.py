"""The training-plan persistence service, exercised against an in-memory stand-in for PostgREST.

The fake below is a small table store that actually APPLIES the filters it is given, rather than a
mock that returns a canned list. That distinction is the point of this file: nearly every bug this
service can have is a missing or wrong predicate (an item edited through the wrong plan's URL, a
delete that matches every row), and a mock returning fixed data is green either way.
"""

from __future__ import annotations

import unittest
import uuid
from typing import Any
from unittest import mock

from backend.app.services import plans as plans_store


class _Resp:
    def __init__(self, data: list[dict[str, Any]], count: int | None = None) -> None:
        self.data = data
        self.count = count


class _Query:
    """One chained PostgREST query. Filters accumulate; ``execute`` applies them."""

    def __init__(self, db: "_FakeDb", table: str, op: str, payload: Any = None) -> None:
        self.db = db
        self.table = table
        self.op = op
        self.payload = payload
        self.columns: list[str] | None = None
        self.filters: list[tuple[str, str, Any]] = []
        self.order_col: str | None = None
        self.order_desc = False
        self.limit_n: int | None = None

    def eq(self, column: str, value: Any) -> "_Query":
        self.filters.append((column, "eq", value))
        return self

    def in_(self, column: str, values: list[Any]) -> "_Query":
        self.filters.append((column, "in", list(values)))
        return self

    def order(self, column: str, desc: bool = False) -> "_Query":
        self.order_col, self.order_desc = column, desc
        return self

    def limit(self, n: int) -> "_Query":
        self.limit_n = n
        return self

    def _matches(self, row: dict[str, Any]) -> bool:
        for column, kind, value in self.filters:
            actual = row.get(column)
            if kind == "eq" and actual != value:
                return False
            if kind == "in" and actual not in value:
                return False
        return True

    def _project(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only the requested columns, the way PostgREST would.

        Projection is not cosmetic here: it means a store function that reads a column it never
        asked for gets a KeyError-shaped failure in these tests instead of silently working against
        the fake and breaking against the real API.
        """
        if self.columns is None:
            return [dict(row) for row in rows]
        return [{c: row.get(c) for c in self.columns} for row in rows]

    def execute(self) -> _Resp:
        table = self.db.tables.setdefault(self.table, [])
        if self.op == "insert":
            payload = self.payload if isinstance(self.payload, list) else [self.payload]
            created = []
            for values in payload:
                row = {"id": self.db.next_id(), "created_at": self.db.next_time(), **values}
                row.setdefault("completed_at", None)
                row.setdefault("analysis_id", None)
                table.append(row)
                created.append(dict(row))
            return _Resp(created)

        matched = [row for row in table if self._matches(row)]

        if self.op == "update":
            for row in matched:
                row.update(self.payload)
            return _Resp([dict(row) for row in matched])

        if self.op == "delete":
            self.db.tables[self.table] = [row for row in table if not self._matches(row)]
            return _Resp([dict(row) for row in matched])

        # select
        if self.order_col:
            matched = sorted(
                matched, key=lambda r: r.get(self.order_col) or "", reverse=self.order_desc
            )
        if self.limit_n is not None:
            matched = matched[: self.limit_n]
        return _Resp(self._project(matched), count=len(matched))


class _Table:
    def __init__(self, db: "_FakeDb", name: str) -> None:
        self.db, self.name = db, name

    def select(self, columns: str, count: str | None = None) -> _Query:
        query = _Query(self.db, self.name, "select")
        query.columns = [c.strip() for c in columns.split(",")]
        return query

    def insert(self, payload: Any) -> _Query:
        return _Query(self.db, self.name, "insert", payload)

    def update(self, payload: dict[str, Any]) -> _Query:
        return _Query(self.db, self.name, "update", payload)

    def delete(self) -> _Query:
        return _Query(self.db, self.name, "delete")


class _FakeDb:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self._ids = 0
        self._clock = 0

    def next_id(self) -> str:
        # Real UUIDs, not readable slugs: the router normalizes every path id through
        # ``uuid.UUID`` before it reaches the store, so a fake handing out "id-001" would make the
        # router suite unable to address any row it had just created.
        self._ids += 1
        return str(uuid.UUID(int=self._ids))

    def next_time(self) -> str:
        self._clock += 1
        return f"2026-08-13T00:00:{self._clock:02d}+00:00"

    def table(self, name: str) -> _Table:
        return _Table(self, name)


class _PlansTestCase(unittest.TestCase):
    """Shared setup: every store call runs against one in-memory db as user ``u1``."""

    def setUp(self) -> None:
        self.db = _FakeDb()
        patcher = mock.patch.object(plans_store, "_user_client", return_value=self.db)
        self.addCleanup(patcher.stop)
        patcher.start()

    def _plan(self, name: str = "Upper week", user: str = "u1", **kwargs: Any) -> dict[str, Any]:
        return plans_store.create_plan(token="tok", user_id=user, name=name, **kwargs)


class CreateAndReadTests(_PlansTestCase):
    def test_create_returns_the_plan_with_its_items(self) -> None:
        plan = self._plan(
            items=[
                {"day_index": 1, "movement": "Squat", "sets": 3, "reps": 10},
                {"day_index": 1, "movement": "Row", "sets": 4, "reps": 8},
            ]
        )
        self.assertEqual(plan["name"], "Upper week")
        self.assertEqual([i["movement"] for i in plan["items"]], ["Squat", "Row"])

    def test_created_items_default_position_to_their_order(self) -> None:
        # Template rows arrive without an explicit position; they must not all land on 0 and then
        # order arbitrarily.
        plan = self._plan(
            items=[
                {"day_index": 1, "movement": "Squat"},
                {"day_index": 1, "movement": "Row"},
                {"day_index": 1, "movement": "Lunge"},
            ]
        )
        self.assertEqual([i["position"] for i in plan["items"]], [0, 1, 2])

    def test_create_with_no_items_yields_an_empty_plan(self) -> None:
        plan = self._plan()
        self.assertEqual(plan["items"], [])

    def test_create_raises_when_the_insert_returns_nothing(self) -> None:
        # A silent empty insert would otherwise surface as a KeyError deep in the router.
        with mock.patch.object(_Query, "execute", return_value=_Resp([])):
            with self.assertRaises(RuntimeError):
                self._plan()

    def test_get_plan_returns_items_sorted_by_day_then_position(self) -> None:
        plan = self._plan(
            items=[
                {"day_index": 3, "movement": "Deadlift", "position": 0},
                {"day_index": 1, "movement": "Row", "position": 1},
                {"day_index": 1, "movement": "Squat", "position": 0},
            ]
        )
        loaded = plans_store.get_plan(token="tok", plan_id=plan["id"], user_id="u1")
        self.assertEqual(
            [(i["day_index"], i["movement"]) for i in loaded["items"]],
            [(1, "Squat"), (1, "Row"), (3, "Deadlift")],
        )

    def test_get_plan_is_none_for_another_users_plan(self) -> None:
        plan = self._plan(user="u2")
        self.assertIsNone(plans_store.get_plan(token="tok", plan_id=plan["id"], user_id="u1"))

    def test_get_plan_is_none_for_a_missing_id(self) -> None:
        self.assertIsNone(plans_store.get_plan(token="tok", plan_id="nope", user_id="u1"))


class ListPlansTests(_PlansTestCase):
    def test_empty_when_the_user_has_no_plans(self) -> None:
        self._plan(user="u2")
        self.assertEqual(plans_store.list_plans(token="tok", user_id="u1"), [])

    def test_summarizes_progress_per_plan(self) -> None:
        plan = self._plan(
            items=[
                {"day_index": 1, "movement": "Squat"},
                {"day_index": 1, "movement": "Row"},
                {"day_index": 4, "movement": "Squat"},
            ]
        )
        item_id = plan["items"][0]["id"]
        plans_store.update_item(
            token="tok",
            user_id="u1",
            plan_id=plan["id"],
            item_id=item_id,
            fields={"completed_at": "2026-08-13T00:00:00+00:00"},
        )

        (summary,) = plans_store.list_plans(token="tok", user_id="u1")
        self.assertEqual(summary["item_count"], 3)
        self.assertEqual(summary["completed_count"], 1)
        # Two DISTINCT days (1 and 4), not three items and not the highest day index.
        self.assertEqual(summary["day_count"], 2)
        # Distinct movements in plan order — Squat appears on two days but once in the summary.
        self.assertEqual(summary["movements"], ["Squat", "Row"])

    def test_does_not_count_another_users_items(self) -> None:
        mine = self._plan(items=[{"day_index": 1, "movement": "Squat"}])
        self._plan(user="u2", items=[{"day_index": 1, "movement": "Row"}])
        (summary,) = plans_store.list_plans(token="tok", user_id="u1")
        self.assertEqual(summary["id"], mine["id"])
        self.assertEqual(summary["item_count"], 1)


class UpdateAndDeletePlanTests(_PlansTestCase):
    def test_update_patches_only_the_given_fields(self) -> None:
        plan = self._plan()
        row = plans_store.update_plan(
            token="tok", plan_id=plan["id"], user_id="u1", fields={"name": "Renamed"}
        )
        self.assertEqual(row["name"], "Renamed")

    def test_empty_patch_reads_the_plan_back_without_its_items(self) -> None:
        # PostgREST rejects an update with no payload, so a no-op PATCH is served as a read.
        plan = self._plan(items=[{"day_index": 1, "movement": "Squat"}])
        row = plans_store.update_plan(token="tok", plan_id=plan["id"], user_id="u1", fields={})
        self.assertEqual(row["name"], "Upper week")
        self.assertNotIn("items", row)

    def test_empty_patch_of_a_missing_plan_is_none(self) -> None:
        self.assertIsNone(
            plans_store.update_plan(token="tok", plan_id="nope", user_id="u1", fields={})
        )

    def test_update_of_another_users_plan_is_none(self) -> None:
        plan = self._plan(user="u2")
        self.assertIsNone(
            plans_store.update_plan(
                token="tok", plan_id=plan["id"], user_id="u1", fields={"name": "Mine now"}
            )
        )

    def test_delete_removes_the_plan(self) -> None:
        plan = self._plan()
        self.assertTrue(plans_store.delete_plan(token="tok", plan_id=plan["id"], user_id="u1"))
        self.assertIsNone(plans_store.get_plan(token="tok", plan_id=plan["id"], user_id="u1"))

    def test_delete_of_another_users_plan_reports_false(self) -> None:
        plan = self._plan(user="u2")
        self.assertFalse(plans_store.delete_plan(token="tok", plan_id=plan["id"], user_id="u1"))
        # And leaves it standing.
        self.assertIsNotNone(plans_store.get_plan(token="tok", plan_id=plan["id"], user_id="u2"))

    def test_plan_exists_is_scoped_to_the_owner(self) -> None:
        plan = self._plan(user="u2")
        self.assertTrue(plans_store.plan_exists(token="tok", plan_id=plan["id"], user_id="u2"))
        self.assertFalse(plans_store.plan_exists(token="tok", plan_id=plan["id"], user_id="u1"))


class StartPlanTests(_PlansTestCase):
    def _started_plan(self) -> dict[str, Any]:
        plan = self._plan(
            items=[
                {"day_index": 1, "movement": "Squat"},
                {"day_index": 2, "movement": "Row"},
            ]
        )
        for item in plan["items"]:
            plans_store.update_item(
                token="tok",
                user_id="u1",
                plan_id=plan["id"],
                item_id=item["id"],
                fields={"completed_at": "2026-08-01T00:00:00+00:00", "analysis_id": "an-1"},
            )
        return plan

    def test_start_stamps_started_at_and_clears_every_tick(self) -> None:
        plan = self._started_plan()
        started = plans_store.start_plan(
            token="tok", plan_id=plan["id"], user_id="u1", now="2026-08-13T09:00:00+00:00"
        )
        self.assertEqual(started["started_at"], "2026-08-13T09:00:00+00:00")
        self.assertEqual([i["completed_at"] for i in started["items"]], [None, None])

    def test_start_also_clears_the_analysis_links(self) -> None:
        # The restart is only honest if the link goes too: an unticked item must not keep offering
        # 看報告 for last week's clip.
        plan = self._started_plan()
        started = plans_store.start_plan(
            token="tok", plan_id=plan["id"], user_id="u1", now="2026-08-13T09:00:00+00:00"
        )
        self.assertEqual([i["analysis_id"] for i in started["items"]], [None, None])

    def test_start_of_a_missing_plan_is_none_and_clears_nothing(self) -> None:
        plan = self._started_plan()
        self.assertIsNone(
            plans_store.start_plan(
                token="tok", plan_id="nope", user_id="u1", now="2026-08-13T09:00:00+00:00"
            )
        )
        # The plan row is stamped BEFORE the items are cleared precisely so this holds: a start
        # that never matched a plan must not have wiped anybody's progress.
        loaded = plans_store.get_plan(token="tok", plan_id=plan["id"], user_id="u1")
        self.assertTrue(all(i["completed_at"] for i in loaded["items"]))

    def test_start_does_not_touch_another_plans_items(self) -> None:
        other = self._plan(name="Other", items=[{"day_index": 1, "movement": "Lunge"}])
        plans_store.update_item(
            token="tok",
            user_id="u1",
            plan_id=other["id"],
            item_id=other["items"][0]["id"],
            fields={"completed_at": "2026-08-01T00:00:00+00:00"},
        )
        target = self._started_plan()
        plans_store.start_plan(
            token="tok", plan_id=target["id"], user_id="u1", now="2026-08-13T09:00:00+00:00"
        )
        loaded = plans_store.get_plan(token="tok", plan_id=other["id"], user_id="u1")
        self.assertIsNotNone(loaded["items"][0]["completed_at"])


class ItemTests(_PlansTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.plan = self._plan()

    def _add(self, **kwargs: Any) -> dict[str, Any]:
        defaults = {
            "day_index": 1,
            "movement": "Squat",
            "sets": 3,
            "reps": 10,
        }
        defaults.update(kwargs)
        return plans_store.add_item(
            token="tok", user_id="u1", plan_id=self.plan["id"], **defaults
        )

    def test_add_item_appends_after_the_days_existing_items(self) -> None:
        self._add(movement="Squat")
        self._add(movement="Row")
        third = self._add(movement="Lunge")
        self.assertEqual(third["position"], 2)

    def test_position_is_computed_per_day_not_per_plan(self) -> None:
        self._add(day_index=1)
        self._add(day_index=1)
        first_of_day_two = self._add(day_index=2)
        self.assertEqual(first_of_day_two["position"], 0)

    def test_an_explicit_position_is_honoured(self) -> None:
        self._add()
        pinned = self._add(position=0)
        self.assertEqual(pinned["position"], 0)

    def test_add_item_raises_when_the_insert_returns_nothing(self) -> None:
        with mock.patch.object(_Query, "execute", return_value=_Resp([])):
            with self.assertRaises(RuntimeError):
                self._add()

    def test_update_item_patches_the_row(self) -> None:
        item = self._add()
        row = plans_store.update_item(
            token="tok",
            user_id="u1",
            plan_id=self.plan["id"],
            item_id=item["id"],
            fields={"sets": 5, "day_index": 2},
        )
        self.assertEqual((row["sets"], row["day_index"]), (5, 2))

    def test_empty_item_patch_reads_the_item_back(self) -> None:
        item = self._add()
        row = plans_store.update_item(
            token="tok", user_id="u1", plan_id=self.plan["id"], item_id=item["id"], fields={}
        )
        self.assertEqual(row["id"], item["id"])

    def test_item_reached_through_the_wrong_plan_is_none(self) -> None:
        # RLS scopes by USER, so the same user's other plan would happily match on item id alone —
        # the plan_id predicate is what makes this 404 instead of an edit through the wrong URL.
        item = self._add()
        other = self._plan(name="Other")
        self.assertIsNone(
            plans_store.update_item(
                token="tok",
                user_id="u1",
                plan_id=other["id"],
                item_id=item["id"],
                fields={"sets": 9},
            )
        )
        self.assertIsNone(
            plans_store.get_item(
                token="tok", user_id="u1", plan_id=other["id"], item_id=item["id"]
            )
        )

    def test_get_item_returns_the_row(self) -> None:
        item = self._add()
        row = plans_store.get_item(
            token="tok", user_id="u1", plan_id=self.plan["id"], item_id=item["id"]
        )
        self.assertEqual(row["movement"], "Squat")

    def test_delete_item_removes_only_that_item(self) -> None:
        keep = self._add(movement="Squat")
        drop = self._add(movement="Row")
        self.assertTrue(
            plans_store.delete_item(
                token="tok", user_id="u1", plan_id=self.plan["id"], item_id=drop["id"]
            )
        )
        loaded = plans_store.get_plan(token="tok", plan_id=self.plan["id"], user_id="u1")
        self.assertEqual([i["id"] for i in loaded["items"]], [keep["id"]])

    def test_delete_of_a_missing_item_reports_false(self) -> None:
        self.assertFalse(
            plans_store.delete_item(
                token="tok", user_id="u1", plan_id=self.plan["id"], item_id="nope"
            )
        )
