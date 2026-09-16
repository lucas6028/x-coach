"""The clinic persistence + pure-aggregation service, exercised two ways:

  * I/O functions that filter a table (``create_invite``, ``list_invites``, ``batch_patient_data``,
    ``list_checkins``) run against an in-memory stand-in for PostgREST that actually APPLIES the
    filters it is given -- the ``_FakeDb`` pattern from ``tests/test_plans_store.py``, copied here
    and extended with ``upsert`` (composite-key semantics), ``rpc``, ``gte`` and ``in_`` support, per
    the task instructions.
  * Thin RPC wrappers (``accept_invite``, ``revoke_link``, ``list_patients``, ``my_clinicians``) are
    exercised against a plain ``mock.Mock`` client, the same style ``tests/test_backend.py`` uses
    for ``store.admin_list_users``/``count_admins`` -- there is no filtering logic to prove there,
    only "the right RPC name/params were sent and ``.data`` came back (or defaulted)".
  * The aggregation rules (``last_checkin_at``, ``adherence_7d``, ``open_flags``, ``inactive_7d``,
    plus the small views) are PURE and take plain dict/list fixtures directly -- no fake DB at all.
"""

from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest import mock

from backend.app.services import clinic
from backend.app.services import store as store_module


# ---------------------------------------------------------------------------------------------
# A filtering fake, copied from tests/test_plans_store.py::_FakeDb and extended with `upsert`
# (composite on_conflict), `rpc`, `gte`, and `in_` -- see that file's docstring for why a fake that
# actually applies its filters matters more than a mock returning canned data.
# ---------------------------------------------------------------------------------------------


class _Resp:
    def __init__(self, data: Any, count: int | None = None) -> None:
        self.data = data
        self.count = count


class _Query:
    def __init__(self, db: "_FakeDb", table: str, op: str, payload: Any = None, on_conflict: str | None = None) -> None:
        self.db = db
        self.table = table
        self.op = op
        self.payload = payload
        self.on_conflict = on_conflict
        self.columns: list[str] | None = None
        self.filters: list[tuple[str, str, Any]] = []
        self.order_col: str | None = None
        self.order_desc = False
        self.limit_n: int | None = None

    def eq(self, column: str, value: Any) -> "_Query":
        self.filters.append((column, "eq", value))
        return self

    def gte(self, column: str, value: Any) -> "_Query":
        self.filters.append((column, "gte", value))
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

    def range(self, start: int, end: int) -> "_Query":
        # PostgREST's inclusive bounds, as store.list_analyses pages with them.
        self.range_bounds = (start, end)
        return self

    def _matches(self, row: dict[str, Any]) -> bool:
        for column, kind, value in self.filters:
            actual = row.get(column)
            if kind == "eq" and actual != value:
                return False
            if kind == "gte" and not (actual is not None and actual >= value):
                return False
            if kind == "in" and actual not in value:
                return False
        return True

    def _project(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                table.append(row)
                created.append(dict(row))
            return _Resp(created)

        if self.op == "upsert":
            # Composite on_conflict: an existing row matching EVERY on_conflict column is updated
            # in place; otherwise a new row is inserted. This is what makes the (user_id, role)
            # composite key behave correctly -- an on_conflict of "user_id" alone (the pre-clinic
            # shape) would wrongly match/clobber a DIFFERENT role row for the same user.
            payload = self.payload if isinstance(self.payload, list) else [self.payload]
            conflict_cols = (self.on_conflict or "").split(",") if self.on_conflict else []
            results = []
            for values in payload:
                existing = next(
                    (
                        row
                        for row in table
                        if conflict_cols and all(row.get(c) == values.get(c) for c in conflict_cols)
                    ),
                    None,
                )
                if existing is not None:
                    existing.update(values)
                    results.append(dict(existing))
                else:
                    row = {"id": self.db.next_id(), "created_at": self.db.next_time(), **values}
                    table.append(row)
                    results.append(dict(row))
            return _Resp(results)

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
            matched = sorted(matched, key=lambda r: r.get(self.order_col) or "", reverse=self.order_desc)
        total = len(matched)
        bounds = getattr(self, "range_bounds", None)
        if bounds is not None:
            matched = matched[bounds[0] : bounds[1] + 1]
        if self.limit_n is not None:
            matched = matched[: self.limit_n]
        return _Resp(self._project(matched), count=total)


class _Table:
    def __init__(self, db: "_FakeDb", name: str) -> None:
        self.db, self.name = db, name

    def select(self, columns: str, count: str | None = None) -> _Query:
        query = _Query(self.db, self.name, "select")
        query.columns = [c.strip() for c in columns.split(",")]
        return query

    def insert(self, payload: Any) -> _Query:
        return _Query(self.db, self.name, "insert", payload)

    def upsert(self, payload: Any, on_conflict: str | None = None) -> _Query:
        return _Query(self.db, self.name, "upsert", payload, on_conflict=on_conflict)

    def update(self, payload: dict[str, Any]) -> _Query:
        return _Query(self.db, self.name, "update", payload)

    def delete(self) -> _Query:
        return _Query(self.db, self.name, "delete")


class _RpcCall:
    def __init__(self, resp: _Resp) -> None:
        self._resp = resp

    def execute(self) -> _Resp:
        return self._resp


class _FakeDb:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self._ids = 0
        self._clock = 0
        # {rpc_name: _Resp}. RPCs are not simulated (that is Postgres-side logic covered by
        # db/checks/clinic_rls_check.sql, out of this file's scope) -- a test presets what the RPC
        # would have returned.
        self.rpc_responses: dict[str, _Resp] = {}
        self.rpc_calls: list[tuple[str, Any]] = []

    def next_id(self) -> str:
        self._ids += 1
        return str(uuid.UUID(int=self._ids))

    def next_time(self) -> str:
        self._clock += 1
        return f"2026-09-01T00:00:{self._clock:02d}+00:00"

    def table(self, name: str) -> _Table:
        return _Table(self, name)

    def rpc(self, name: str, params: Any = None) -> _RpcCall:
        self.rpc_calls.append((name, params))
        return _RpcCall(self.rpc_responses.get(name, _Resp(None)))


class _ClinicStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _FakeDb()
        patcher = mock.patch.object(clinic, "_user_client", return_value=self.db)
        self.addCleanup(patcher.stop)
        patcher.start()


class ListAnalysesOwnerFilterTests(unittest.TestCase):
    """The history list stays the caller's own once clinicians can read patients' analyses.

    Against the filtering fake, the table holds a clinician's own analysis AND a linked patient's
    -- exactly what the clinician's JWT can see through RLS after the clinic migration. Only the
    explicit user_id predicate keeps the patient's row out of the clinician's 我的紀錄.
    """

    def test_lists_only_the_callers_rows_when_the_table_holds_two_users(self) -> None:
        db = _FakeDb()
        db.tables["analyses"] = [
            {"id": "a1", "user_id": "clin", "video_id": "own_clip", "created_at": "2026-09-02"},
            {"id": "a2", "user_id": "patient", "video_id": "patient_clip", "created_at": "2026-09-03"},
            {"id": "a3", "user_id": "clin", "video_id": "own_clip_2", "created_at": "2026-09-04"},
        ]
        with mock.patch.object(store_module, "_user_client", return_value=db):
            result = store_module.list_analyses(token="tok", user_id="clin")

        self.assertEqual([row["video_id"] for row in result["items"]], ["own_clip_2", "own_clip"])
        self.assertEqual(result["total"], 2)


# ---------------------------------------------------------------------------------------------
# create_invite: retry on 23505, and gives up after the attempt cap.
# ---------------------------------------------------------------------------------------------


class _Unique(Exception):
    code = "23505"


class InviteCreationTests(unittest.TestCase):
    def test_creates_a_pending_row_with_a_six_char_code(self) -> None:
        client = mock.Mock()
        client.table.return_value.insert.return_value.execute.return_value = _Resp(
            data=[{"id": "l1", "clinician_id": "c1", "invite_code": "ABCDEF", "status": "pending"}]
        )
        with mock.patch.object(clinic, "_user_client", return_value=client):
            row = clinic.create_invite(token="t", clinician_id="c1")
        self.assertEqual(row["id"], "l1")
        inserted = client.table.return_value.insert.call_args[0][0]
        self.assertEqual(inserted["clinician_id"], "c1")
        self.assertEqual(inserted["status"], "pending")
        self.assertEqual(len(inserted["invite_code"]), 6)
        self.assertTrue(set(inserted["invite_code"]) <= set(clinic._INVITE_ALPHABET))

    def test_retries_on_unique_violation_then_succeeds(self) -> None:
        client = mock.Mock()
        insert_exec = client.table.return_value.insert.return_value.execute
        insert_exec.side_effect = [
            _Unique(),
            _Unique(),
            _Resp(data=[{"id": "l1", "invite_code": "ABCDEF", "status": "pending"}]),
        ]
        with mock.patch.object(clinic, "_user_client", return_value=client):
            row = clinic.create_invite(token="t", clinician_id="c1")
        self.assertEqual(row["id"], "l1")
        self.assertEqual(insert_exec.call_count, 3)

    def test_gives_up_after_the_attempt_cap(self) -> None:
        client = mock.Mock()
        insert_exec = client.table.return_value.insert.return_value.execute
        insert_exec.side_effect = _Unique()
        with mock.patch.object(clinic, "_user_client", return_value=client):
            with self.assertRaises(RuntimeError):
                clinic.create_invite(token="t", clinician_id="c1")
        self.assertEqual(insert_exec.call_count, clinic._MAX_INVITE_ATTEMPTS)

    def test_a_non_collision_exception_is_not_retried(self) -> None:
        client = mock.Mock()
        insert_exec = client.table.return_value.insert.return_value.execute
        insert_exec.side_effect = RuntimeError("boom")
        with mock.patch.object(clinic, "_user_client", return_value=client):
            with self.assertRaises(RuntimeError):
                clinic.create_invite(token="t", clinician_id="c1")
        self.assertEqual(insert_exec.call_count, 1)

    def test_raises_if_the_insert_returns_no_row(self) -> None:
        client = mock.Mock()
        client.table.return_value.insert.return_value.execute.return_value = _Resp(data=[])
        with mock.patch.object(clinic, "_user_client", return_value=client):
            with self.assertRaises(RuntimeError):
                clinic.create_invite(token="t", clinician_id="c1")


class ListInvitesTests(_ClinicStoreTestCase):
    def test_lists_only_the_callers_pending_unexpired_invites(self) -> None:
        far_future = "2099-01-01T00:00:00+00:00"
        self.db.table("care_links").insert(
            {"clinician_id": "c1", "invite_code": "AAA111", "status": "pending", "expires_at": far_future}
        ).execute()
        self.db.table("care_links").insert(
            # Someone else's invite -- must not appear.
            {"clinician_id": "c2", "invite_code": "BBB222", "status": "pending", "expires_at": far_future}
        ).execute()
        self.db.table("care_links").insert(
            # Already accepted -- not "pending" any more.
            {"clinician_id": "c1", "invite_code": "CCC333", "status": "active", "expires_at": far_future}
        ).execute()
        self.db.table("care_links").insert(
            # Expired.
            {"clinician_id": "c1", "invite_code": "DDD444", "status": "pending", "expires_at": "2020-01-01T00:00:00+00:00"}
        ).execute()

        invites = clinic.list_invites(token="t", clinician_id="c1")
        self.assertEqual([i["invite_code"] for i in invites], ["AAA111"])

    def test_empty_when_none_pending(self) -> None:
        self.assertEqual(clinic.list_invites(token="t", clinician_id="c1"), [])


# ---------------------------------------------------------------------------------------------
# Thin RPC wrappers.
# ---------------------------------------------------------------------------------------------


class RpcWrapperTests(unittest.TestCase):
    def _client(self, data: Any) -> mock.Mock:
        client = mock.Mock()
        client.rpc.return_value.execute.return_value = _Resp(data=data)
        return client

    def test_accept_invite_calls_the_rpc_with_the_code(self) -> None:
        client = self._client({"link": {"id": "l1", "status": "active"}})
        with mock.patch.object(clinic, "_user_client", return_value=client):
            result = clinic.accept_invite(token="t", code="ABC123")
        client.rpc.assert_called_once_with("accept_care_invite", {"p_code": "ABC123"})
        self.assertEqual(result["link"]["id"], "l1")

    def test_accept_invite_defaults_to_empty_dict_when_data_is_none(self) -> None:
        client = self._client(None)
        with mock.patch.object(clinic, "_user_client", return_value=client):
            self.assertEqual(clinic.accept_invite(token="t", code="X"), {})

    def test_revoke_link_calls_the_rpc_with_the_link_id(self) -> None:
        client = self._client({"link": {"id": "l1", "status": "revoked"}})
        with mock.patch.object(clinic, "_user_client", return_value=client):
            result = clinic.revoke_link(token="t", link_id="l1")
        client.rpc.assert_called_once_with("revoke_care_link", {"p_link": "l1"})
        self.assertEqual(result["link"]["status"], "revoked")

    def test_list_patients_calls_the_rpc_and_applies_display(self) -> None:
        client = self._client(
            [{"link_id": "lk1", "patient_id": "p1", "email": "line_x@line.invalid", "display_name": None, "accepted_at": "t"}]
        )
        with mock.patch.object(clinic, "_user_client", return_value=client):
            rows = clinic.list_patients(token="t")
        client.rpc.assert_called_once_with("clinic_patients")
        self.assertIsNone(rows[0]["email"])

    def test_list_patients_empty_when_data_is_none(self) -> None:
        client = self._client(None)
        with mock.patch.object(clinic, "_user_client", return_value=client):
            self.assertEqual(clinic.list_patients(token="t"), [])

    def test_my_clinicians_calls_the_rpc_and_applies_display(self) -> None:
        client = self._client(
            [{"link_id": "lk1", "clinician_id": "c1", "email": "a@x.com", "display_name": None, "accepted_at": "t"}]
        )
        with mock.patch.object(clinic, "_user_client", return_value=client):
            rows = clinic.my_clinicians(token="t")
        client.rpc.assert_called_once_with("my_clinicians")
        self.assertEqual(rows[0]["display_name"], "a@x.com")

    def test_ack_checkin_flag_calls_the_rpc_with_the_checkin_id(self) -> None:
        client = self._client({"checkin": {"id": "c1", "acknowledged_at": "t"}})
        with mock.patch.object(clinic, "_user_client", return_value=client):
            result = clinic.ack_checkin_flag(token="t", checkin_id="c1")
        client.rpc.assert_called_once_with("ack_checkin_flag", {"p_checkin": "c1"})
        self.assertEqual(result["checkin"]["id"], "c1")

    def test_ack_checkin_flag_defaults_to_empty_dict_when_data_is_none(self) -> None:
        client = self._client(None)
        with mock.patch.object(clinic, "_user_client", return_value=client):
            self.assertEqual(clinic.ack_checkin_flag(token="t", checkin_id="c1"), {})

    def test_find_patient_matches_by_patient_id(self) -> None:
        client = self._client(
            [{"link_id": "lk1", "patient_id": "p1", "email": "a@x.com", "display_name": None, "accepted_at": "t"}]
        )
        with mock.patch.object(clinic, "_user_client", return_value=client):
            found = clinic.find_patient(token="t", patient_id="p1")
            missing = clinic.find_patient(token="t", patient_id="nope")
        self.assertEqual(found["link_id"], "lk1")
        self.assertIsNone(missing)


# ---------------------------------------------------------------------------------------------
# Display helper.
# ---------------------------------------------------------------------------------------------


class DisplayHelperTests(unittest.TestCase):
    def test_synthetic_line_email_is_hidden_but_display_name_kept(self) -> None:
        row = clinic._apply_display({"email": "line_abc@line.invalid", "display_name": "Coach A"})
        self.assertIsNone(row["email"])
        self.assertEqual(row["display_name"], "Coach A")

    def test_synthetic_email_with_no_display_name_stays_null(self) -> None:
        # NOT falling back to the (synthetic, meaningless) email.
        row = clinic._apply_display({"email": "line_abc@line.invalid", "display_name": None})
        self.assertIsNone(row["email"])
        self.assertIsNone(row["display_name"])

    def test_real_email_with_no_display_name_falls_back_to_the_email(self) -> None:
        row = clinic._apply_display({"email": "a@x.com", "display_name": None})
        self.assertEqual(row["email"], "a@x.com")
        self.assertEqual(row["display_name"], "a@x.com")

    def test_real_email_with_a_display_name_is_untouched(self) -> None:
        row = clinic._apply_display({"email": "a@x.com", "display_name": "Real Name"})
        self.assertEqual(row["email"], "a@x.com")
        self.assertEqual(row["display_name"], "Real Name")


# ---------------------------------------------------------------------------------------------
# Batched reads.
# ---------------------------------------------------------------------------------------------


class BatchPatientDataTests(_ClinicStoreTestCase):
    def test_empty_patient_ids_short_circuits_with_no_queries(self) -> None:
        with mock.patch.object(clinic, "_user_client") as uc:
            result = clinic.batch_patient_data(token="t", patient_ids=[])
        self.assertEqual(result, ([], [], []))
        uc.assert_not_called()

    def test_scopes_each_table_to_the_given_patient_ids(self) -> None:
        self.db.table("training_plans").insert({"user_id": "p1", "assigned_by": "c1"}).execute()
        self.db.table("training_plans").insert({"user_id": "p2", "assigned_by": "c1"}).execute()
        self.db.table("training_plans").insert({"user_id": "other", "assigned_by": "c1"}).execute()
        self.db.table("plan_items").insert({"user_id": "p1", "plan_id": "x", "day_index": 1}).execute()
        self.db.table("session_checkins").insert({"user_id": "p1", "plan_id": "x", "flagged": False}).execute()

        plans, items, checkins = clinic.batch_patient_data(token="t", patient_ids=["p1", "p2"])
        self.assertEqual({p["user_id"] for p in plans}, {"p1", "p2"})
        self.assertEqual(len(items), 1)
        self.assertEqual(len(checkins), 1)


class ListCheckinsTests(_ClinicStoreTestCase):
    def test_returns_only_the_given_patients_checkins_newest_first(self) -> None:
        self.db.table("session_checkins").insert(
            {"user_id": "p1", "created_at": "2026-09-01T00:00:00+00:00"}
        ).execute()
        self.db.table("session_checkins").insert(
            {"user_id": "p1", "created_at": "2026-09-05T00:00:00+00:00"}
        ).execute()
        self.db.table("session_checkins").insert(
            {"user_id": "other", "created_at": "2026-09-09T00:00:00+00:00"}
        ).execute()

        rows = clinic.list_checkins(token="t", patient_id="p1")
        self.assertEqual([r["created_at"] for r in rows], [
            "2026-09-05T00:00:00+00:00", "2026-09-01T00:00:00+00:00",
        ])


# ---------------------------------------------------------------------------------------------
# Pure aggregation. now = 2026-09-15T12:00:00Z throughout, so the window is
# (2026-09-08T12:00:00Z, 2026-09-15T12:00:00Z].
# ---------------------------------------------------------------------------------------------

_NOW = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)


def _plan(id_: str = "plan-1", assigned_by: str | None = "clinician-1", created_at: str = "2026-09-01T00:00:00+00:00") -> dict[str, Any]:
    return {"id": id_, "assigned_by": assigned_by, "created_at": created_at}


def _item(plan_id: str = "plan-1", day_index: int | None = 1) -> dict[str, Any]:
    return {"plan_id": plan_id, "day_index": day_index}


def _checkin(plan_id: str = "plan-1", created_at: str = "2026-09-15T00:00:00+00:00", **kw: Any) -> dict[str, Any]:
    row = {"plan_id": plan_id, "created_at": created_at, "flagged": False, "acknowledged_at": None}
    row.update(kw)
    return row


class LastCheckinAtTests(unittest.TestCase):
    def test_none_when_no_checkins(self) -> None:
        self.assertIsNone(clinic.last_checkin_at([]))

    def test_returns_the_newest_created_at(self) -> None:
        checkins = [_checkin(created_at="2026-09-01T00:00:00+00:00"), _checkin(created_at="2026-09-10T00:00:00+00:00")]
        self.assertEqual(clinic.last_checkin_at(checkins), "2026-09-10T00:00:00+00:00")


class AdherenceTests(unittest.TestCase):
    def test_none_when_no_assigned_plan(self) -> None:
        # Only a self-authored plan (assigned_by null) -- does not count as a prescription.
        plans = [_plan(assigned_by=None)]
        self.assertIsNone(clinic.adherence_7d(plans, [], [], now=_NOW))

    def test_none_when_the_assigned_plan_has_no_items(self) -> None:
        self.assertIsNone(clinic.adherence_7d([_plan()], [], [], now=_NOW))

    def test_full_adherence_when_every_planned_day_has_a_checkin(self) -> None:
        plans = [_plan()]
        items = [_item(day_index=1), _item(day_index=2)]
        checkins = [
            _checkin(created_at="2026-09-15T01:00:00+00:00"),  # Taipei 09-15
            _checkin(created_at="2026-09-14T01:00:00+00:00"),  # Taipei 09-14
        ]
        self.assertEqual(clinic.adherence_7d(plans, items, checkins, now=_NOW), 1.0)

    def test_partial_adherence(self) -> None:
        plans = [_plan()]
        items = [_item(day_index=1), _item(day_index=2)]
        checkins = [_checkin(created_at="2026-09-15T01:00:00+00:00")]
        self.assertEqual(clinic.adherence_7d(plans, items, checkins, now=_NOW), 0.5)

    def test_multiple_checkins_same_taipei_day_count_once(self) -> None:
        plans = [_plan()]
        items = [_item(day_index=1), _item(day_index=2)]
        checkins = [
            _checkin(created_at="2026-09-15T01:00:00+00:00"),
            _checkin(created_at="2026-09-15T02:00:00+00:00"),  # same Taipei date as above
        ]
        self.assertEqual(clinic.adherence_7d(plans, items, checkins, now=_NOW), 0.5)

    def test_taipei_bucketing_splits_checkins_on_the_same_utc_date(self) -> None:
        # 23:59 and 00:01 UTC on the same UTC date land on two DIFFERENT Taipei (UTC+8) calendar
        # dates -- this is what proves the bucketing is Taipei, not UTC or naive.
        plans = [_plan()]
        items = [_item(day_index=1), _item(day_index=2)]
        checkins = [
            _checkin(created_at="2026-09-13T23:59:00+00:00"),  # Taipei 2026-09-14 07:59
            _checkin(created_at="2026-09-14T16:01:00+00:00"),  # Taipei 2026-09-15 00:01
        ]
        self.assertEqual(clinic.adherence_7d(plans, items, checkins, now=_NOW), 1.0)

    def test_checkins_on_another_plan_do_not_count(self) -> None:
        plans = [_plan(id_="plan-1"), _plan(id_="plan-0", created_at="2026-08-01T00:00:00+00:00")]
        items = [_item(plan_id="plan-1", day_index=1)]
        checkins = [_checkin(plan_id="plan-0", created_at="2026-09-15T01:00:00+00:00")]
        self.assertEqual(clinic.adherence_7d(plans, items, checkins, now=_NOW), 0.0)

    def test_window_start_is_excluded_and_now_is_included(self) -> None:
        plans = [_plan()]
        items = [_item(day_index=1)]
        window_start = _NOW - timedelta(days=7)
        checkins = [_checkin(created_at=window_start.isoformat())]
        self.assertEqual(clinic.adherence_7d(plans, items, checkins, now=_NOW), 0.0)

        checkins = [_checkin(created_at=_NOW.isoformat())]
        self.assertEqual(clinic.adherence_7d(plans, items, checkins, now=_NOW), 1.0)

    def test_a_checkin_missing_created_at_is_skipped_not_fatal(self) -> None:
        plans = [_plan()]
        items = [_item(day_index=1)]
        checkins = [_checkin(created_at=None), _checkin(created_at="2026-09-15T00:00:00+00:00")]
        self.assertEqual(clinic.adherence_7d(plans, items, checkins, now=_NOW), 1.0)

    def test_a_naive_created_at_is_treated_as_utc(self) -> None:
        # PostgREST always sends tz-aware timestamps; a hand-built fixture without an offset must
        # not crash the comparison against a tz-aware `now`.
        plans = [_plan()]
        items = [_item(day_index=1)]
        checkins = [_checkin(created_at="2026-09-15T00:00:00")]
        self.assertEqual(clinic.adherence_7d(plans, items, checkins, now=_NOW), 1.0)

    def test_extra_days_of_checkins_clamp_to_one(self) -> None:
        # planned=1 but 2 distinct done-days somehow (shouldn't normally happen since a day without
        # a planned slot still counts toward `done` under this definition) -- value never exceeds 1.
        plans = [_plan()]
        items = [_item(day_index=1)]
        checkins = [
            _checkin(created_at="2026-09-15T01:00:00+00:00"),
            _checkin(created_at="2026-09-14T01:00:00+00:00"),
        ]
        self.assertEqual(clinic.adherence_7d(plans, items, checkins, now=_NOW), 1.0)


class OpenFlagsTests(unittest.TestCase):
    def test_counts_only_flagged_and_unacknowledged(self) -> None:
        checkins = [
            _checkin(flagged=True, acknowledged_at=None),
            _checkin(flagged=True, acknowledged_at="2026-09-10T00:00:00+00:00"),
            _checkin(flagged=False, acknowledged_at=None),
        ]
        self.assertEqual(clinic.open_flags(checkins), 1)

    def test_zero_when_empty(self) -> None:
        self.assertEqual(clinic.open_flags([]), 0)


class InactiveTests(unittest.TestCase):
    def test_false_when_no_assigned_plan(self) -> None:
        self.assertFalse(clinic.inactive_7d([_plan(assigned_by=None)], [], now=_NOW))

    def test_false_when_the_plan_has_no_created_at(self) -> None:
        plan = _plan()
        del plan["created_at"]
        self.assertFalse(clinic.inactive_7d([plan], [], now=_NOW))

    def test_a_checkin_missing_created_at_is_skipped_not_fatal(self) -> None:
        plan = _plan(created_at="2026-08-01T00:00:00+00:00")
        checkins = [_checkin(created_at=None)]
        self.assertTrue(clinic.inactive_7d([plan], checkins, now=_NOW))

    def test_false_when_the_plan_is_newer_than_seven_days(self) -> None:
        plan = _plan(created_at="2026-09-10T00:00:00+00:00")  # 5 days ago
        self.assertFalse(clinic.inactive_7d([plan], [], now=_NOW))

    def test_true_when_old_enough_and_no_recent_checkin(self) -> None:
        plan = _plan(created_at="2026-08-01T00:00:00+00:00")
        self.assertTrue(clinic.inactive_7d([plan], [], now=_NOW))

    def test_false_when_a_recent_checkin_exists_on_that_plan(self) -> None:
        plan = _plan(created_at="2026-08-01T00:00:00+00:00")
        checkins = [_checkin(created_at="2026-09-15T00:00:00+00:00")]
        self.assertFalse(clinic.inactive_7d([plan], checkins, now=_NOW))

    def test_a_checkin_on_a_different_plan_does_not_rescue_it(self) -> None:
        plan = _plan(id_="plan-1", created_at="2026-08-01T00:00:00+00:00")
        checkins = [_checkin(plan_id="plan-0", created_at="2026-09-15T00:00:00+00:00")]
        self.assertTrue(clinic.inactive_7d([plan], checkins, now=_NOW))

    def test_created_exactly_seven_days_ago_counts_as_old_enough(self) -> None:
        window_start = _NOW - timedelta(days=7)
        plan = _plan(created_at=window_start.isoformat())
        self.assertTrue(clinic.inactive_7d([plan], [], now=_NOW))

    def test_a_checkin_exactly_at_the_window_start_does_not_rescue_it(self) -> None:
        plan = _plan(created_at="2026-08-01T00:00:00+00:00")
        window_start = _NOW - timedelta(days=7)
        checkins = [_checkin(created_at=window_start.isoformat())]
        self.assertTrue(clinic.inactive_7d([plan], checkins, now=_NOW))

    def test_a_checkin_exactly_at_now_does_rescue_it(self) -> None:
        plan = _plan(created_at="2026-08-01T00:00:00+00:00")
        checkins = [_checkin(created_at=_NOW.isoformat())]
        self.assertFalse(clinic.inactive_7d([plan], checkins, now=_NOW))


class DetailViewTests(unittest.TestCase):
    def test_recent_checkins_takes_the_first_n(self) -> None:
        checkins = [_checkin(created_at=str(i)) for i in range(40)]
        self.assertEqual(len(clinic.recent_checkins(checkins)), 30)
        self.assertEqual(clinic.recent_checkins(checkins, limit=5), checkins[:5])

    def test_trend_filters_to_the_window_and_sorts_oldest_first(self) -> None:
        checkins = [
            _checkin(created_at="2026-09-14T00:00:00+00:00", form_score=80, pain_nrs=2, flagged=True),
            _checkin(created_at="2026-07-01T00:00:00+00:00", form_score=10, pain_nrs=9),  # too old
            _checkin(created_at="2026-09-01T00:00:00+00:00", form_score=60, pain_nrs=4),
        ]
        rows = clinic.trend(checkins, now=_NOW, days=30)
        self.assertEqual([r["created_at"] for r in rows], [
            "2026-09-01T00:00:00+00:00", "2026-09-14T00:00:00+00:00",
        ])
        self.assertEqual(rows[0]["form_score"], 60)
        self.assertEqual(rows[0]["pain_nrs"], 4)
        self.assertEqual(rows[0]["flagged"], False)
        self.assertEqual(rows[1]["flagged"], True)

    def test_trend_flagged_mirrors_the_checkins_own_flag_regardless_of_acknowledgement(self) -> None:
        # `flagged` is the check-in's own column, verbatim -- NOT `open_flags`' "flagged and not yet
        # acknowledged" -- so an acknowledged flag still reads as flagged on the trend chart.
        checkins = [
            _checkin(created_at="2026-09-10T00:00:00+00:00", flagged=True, acknowledged_at=None),
            _checkin(
                created_at="2026-09-11T00:00:00+00:00",
                flagged=True,
                acknowledged_at="2026-09-12T00:00:00+00:00",
            ),
            _checkin(created_at="2026-09-12T00:00:00+00:00", flagged=False),
        ]
        rows = clinic.trend(checkins, now=_NOW, days=30)
        self.assertEqual([r["flagged"] for r in rows], [True, True, False])

    def test_open_flag_rows_matches_open_flags_count(self) -> None:
        checkins = [
            _checkin(flagged=True, acknowledged_at=None),
            _checkin(flagged=True, acknowledged_at="x"),
            _checkin(flagged=False),
        ]
        rows = clinic.open_flag_rows(checkins)
        self.assertEqual(len(rows), clinic.open_flags(checkins))
        self.assertEqual(len(rows), 1)


# ---------------------------------------------------------------------------------------------
# store.py's composite-key role writes, proven against the FILTERING fake (a mock that ignores
# its own .eq() args would pass this even with a broken on_conflict/delete predicate).
# ---------------------------------------------------------------------------------------------


class RoleCompositeKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _FakeDb()
        patcher = mock.patch.object(store_module, "_user_client", return_value=self.db)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_granting_both_roles_yields_two_independent_rows(self) -> None:
        store_module.set_user_role(token="t", user_id="u1", make_admin=True)
        store_module.set_clinician_role(token="t", user_id="u1", make_clinician=True)
        roles = {r["role"] for r in self.db.tables["user_roles"]}
        self.assertEqual(roles, {"admin", "clinician"})

    def test_revoking_clinician_leaves_the_admin_row_intact(self) -> None:
        store_module.set_user_role(token="t", user_id="u1", make_admin=True)
        store_module.set_clinician_role(token="t", user_id="u1", make_clinician=True)
        store_module.set_clinician_role(token="t", user_id="u1", make_clinician=False)
        roles = [r["role"] for r in self.db.tables["user_roles"] if r["user_id"] == "u1"]
        self.assertEqual(roles, ["admin"])

    def test_revoking_admin_leaves_the_clinician_row_intact(self) -> None:
        store_module.set_user_role(token="t", user_id="u1", make_admin=True)
        store_module.set_clinician_role(token="t", user_id="u1", make_clinician=True)
        store_module.set_user_role(token="t", user_id="u1", make_admin=False)
        roles = [r["role"] for r in self.db.tables["user_roles"] if r["user_id"] == "u1"]
        self.assertEqual(roles, ["clinician"])

    def test_granting_admin_twice_does_not_duplicate_the_row(self) -> None:
        store_module.set_user_role(token="t", user_id="u1", make_admin=True)
        store_module.set_user_role(token="t", user_id="u1", make_admin=True)
        rows = [r for r in self.db.tables["user_roles"] if r["user_id"] == "u1"]
        self.assertEqual(len(rows), 1)

    def test_is_clinician_reflects_the_role_row(self) -> None:
        store_module.set_clinician_role(token="t", user_id="u1", make_clinician=True)
        self.assertTrue(store_module.is_clinician(token="t", user_id="u1"))
        self.assertFalse(store_module.is_clinician(token="t", user_id="u2"))


if __name__ == "__main__":
    unittest.main()
