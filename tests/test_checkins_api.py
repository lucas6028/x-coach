"""HTTP + persistence tests for WP2's check-in loop: ``POST/GET /api/checkins`` (patient side,
``routers/checkins.py``) and ``PATCH /api/clinic/flags/{id}/ack`` (clinician side, added to
``routers/clinic.py``).

Two tiers, matching ``tests/test_clinic_api.py``'s split:
  * Router tests run the real app (``TestClient``) with ``services.checkins``/``services.plans``/
    ``services.store``/``services.clinic`` mocked -- gating, 404/400 translation, request/response
    shape, and that ownership checks read ``row["user_id"]`` rather than trusting a successful
    ``get_analysis`` (RLS-scoped, and since the clinic migration a linked clinician can read a
    PATIENT's analysis through it too -- see ``routers/checkins.py``'s module docstring).
  * ``ListCheckinsFilteringTests`` runs against a REAL shared ``_FakeDb`` (the filtering fake from
    ``tests/test_plans_store.py``, same one ``tests/test_clinic_api.py`` reuses for its hand-off
    test) because the property under test -- "GET only ever returns the CALLER's rows, even though
    a clinician's RLS grant would let more through" -- needs a fake that actually applies `.eq()`,
    not a mock that returns canned data regardless of the filter it was given.
  * ``CheckinsServiceUnitTests`` exercises ``services/checkins.py`` directly (a plain ``mock.Mock``
    client, same style as ``services/clinic.py``'s RPC-wrapper tests), for the failure path
    (insert returns no row) and the plan-filter branch the router tests don't happen to reach.
"""

from __future__ import annotations

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from backend.app.auth import CurrentUser, get_current_user
from backend.app.main import app
from backend.app.services import checkins as checkins_store
from backend.app.services import clinic
from backend.app.services import plans as plans_store
from backend.app.services import store
from tests.test_plans_store import _FakeDb

PLAN_ID = "11111111-1111-1111-1111-111111111111"
ITEM_ID = "22222222-2222-2222-2222-222222222222"
ANALYSIS_ID = "33333333-3333-3333-3333-333333333333"
CHECKIN_ID = "44444444-4444-4444-4444-444444444444"

USER = CurrentUser(id="user-1", token="utok")
OTHER_USER_ID = "user-2"
CLINICIAN = CurrentUser(id="clinician-1", token="ctok")


class _PatientTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: USER
        self.addCleanup(app.dependency_overrides.clear)


# ---------------------------------------------------------------------------------------------
# POST /api/checkins -- ownership refusals (criteria a-d of the task).
# ---------------------------------------------------------------------------------------------


class CreateCheckinOwnershipTests(_PatientTestCase):
    def test_non_uuid_plan_id_is_404(self) -> None:
        resp = self.client.post("/api/checkins", json={"plan_id": "not-a-uuid", "pain_nrs": 2})
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_another_users_plan_is_404(self) -> None:
        with mock.patch.object(plans_store, "plan_exists", return_value=False) as pe:
            resp = self.client.post(
                "/api/checkins", json={"plan_id": PLAN_ID, "pain_nrs": 2}
            )
        self.assertEqual(resp.status_code, 404, resp.text)
        pe.assert_called_once_with(token="utok", plan_id=PLAN_ID, user_id="user-1")

    def test_non_uuid_item_id_is_404(self) -> None:
        with mock.patch.object(plans_store, "plan_exists", return_value=True):
            resp = self.client.post(
                "/api/checkins",
                json={"plan_id": PLAN_ID, "plan_item_id": "not-a-uuid", "pain_nrs": 2},
            )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_item_from_a_different_plan_is_404(self) -> None:
        with mock.patch.object(plans_store, "plan_exists", return_value=True), \
             mock.patch.object(plans_store, "get_item", return_value=None) as gi:
            resp = self.client.post(
                "/api/checkins",
                json={"plan_id": PLAN_ID, "plan_item_id": ITEM_ID, "pain_nrs": 2},
            )
        self.assertEqual(resp.status_code, 404, resp.text)
        gi.assert_called_once_with(
            token="utok", user_id="user-1", plan_id=PLAN_ID, item_id=ITEM_ID
        )

    def test_non_uuid_analysis_id_is_400(self) -> None:
        with mock.patch.object(plans_store, "plan_exists", return_value=True):
            resp = self.client.post(
                "/api/checkins",
                json={"plan_id": PLAN_ID, "analysis_id": "not-a-uuid", "pain_nrs": 2},
            )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_unknown_analysis_is_400(self) -> None:
        with mock.patch.object(plans_store, "plan_exists", return_value=True), \
             mock.patch.object(store, "get_analysis", return_value=None):
            resp = self.client.post(
                "/api/checkins",
                json={"plan_id": PLAN_ID, "analysis_id": ANALYSIS_ID, "pain_nrs": 2},
            )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_another_users_analysis_is_400_even_when_get_analysis_returns_the_row(self) -> None:
        # Simulates the exact hazard the docstring calls out: since the clinic migration, a
        # LINKED CLINICIAN'S own JWT can read get_analysis on a PATIENT's row too, so the row
        # existing is no longer proof of ownership -- only row["user_id"] == caller is.
        clinician_visible_row = {"user_id": OTHER_USER_ID, "result": {"detections": []}}
        with mock.patch.object(plans_store, "plan_exists", return_value=True), \
             mock.patch.object(store, "get_analysis", return_value=clinician_visible_row):
            resp = self.client.post(
                "/api/checkins",
                json={"plan_id": PLAN_ID, "analysis_id": ANALYSIS_ID, "pain_nrs": 2},
            )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_pain_nrs_out_of_range_is_422(self) -> None:
        resp = self.client.post("/api/checkins", json={"plan_id": PLAN_ID, "pain_nrs": 11})
        self.assertEqual(resp.status_code, 422)

    def test_missing_pain_nrs_is_422(self) -> None:
        resp = self.client.post("/api/checkins", json={"plan_id": PLAN_ID})
        self.assertEqual(resp.status_code, 422)


# ---------------------------------------------------------------------------------------------
# POST /api/checkins -- form_score is computed from the analysis and flags are stored.
# ---------------------------------------------------------------------------------------------


class CreateCheckinScoringAndFlagTests(_PatientTestCase):
    def test_form_score_is_computed_and_stored_from_the_linked_analysis(self) -> None:
        analysis_row = {
            "user_id": "user-1",
            "result": {
                "detections": [{"severity": 0.5}],
                "quality": {"valid_frame_ratio": 1.0},
            },
        }
        captured: dict = {}

        def fake_create(*, token: str, payload: dict) -> dict:
            captured.update(payload)
            return {**payload, "id": "c1", "created_at": "2026-09-15T00:00:00+00:00"}

        with mock.patch.object(plans_store, "plan_exists", return_value=True), \
             mock.patch.object(store, "get_analysis", return_value=analysis_row), \
             mock.patch.object(checkins_store, "recent_checkins", return_value=[]), \
             mock.patch.object(checkins_store, "create_checkin", side_effect=fake_create):
            resp = self.client.post(
                "/api/checkins",
                json={"plan_id": PLAN_ID, "analysis_id": ANALYSIS_ID, "pain_nrs": 2},
            )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(captured["form_score"], 88)
        self.assertEqual(resp.json()["form_score"], 88)
        self.assertFalse(captured["flagged"])
        self.assertEqual(captured["flag_reasons"], [])

    def test_no_analysis_id_means_no_form_score(self) -> None:
        captured: dict = {}

        def fake_create(*, token: str, payload: dict) -> dict:
            captured.update(payload)
            return {**payload, "id": "c1"}

        with mock.patch.object(plans_store, "plan_exists", return_value=True), \
             mock.patch.object(checkins_store, "recent_checkins", return_value=[]), \
             mock.patch.object(checkins_store, "create_checkin", side_effect=fake_create):
            resp = self.client.post(
                "/api/checkins", json={"plan_id": PLAN_ID, "pain_nrs": 2}
            )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertIsNone(captured["form_score"])

    def test_a_flag_from_history_on_the_same_plan_is_stored(self) -> None:
        # Proves the history plan_id predicate reaches redflags.evaluate correctly: a prior
        # check-in on the SAME plan with a much lower pain score makes pain_rise fire.
        history = [
            {
                "plan_id": PLAN_ID,
                "pain_nrs": 1,
                "form_score": None,
                "created_at": "2026-09-14T00:00:00+00:00",
            }
        ]
        captured: dict = {}

        def fake_create(*, token: str, payload: dict) -> dict:
            captured.update(payload)
            return {**payload, "id": "c1"}

        with mock.patch.object(plans_store, "plan_exists", return_value=True), \
             mock.patch.object(checkins_store, "recent_checkins", return_value=history), \
             mock.patch.object(checkins_store, "create_checkin", side_effect=fake_create):
            resp = self.client.post(
                "/api/checkins", json={"plan_id": PLAN_ID, "pain_nrs": 8}
            )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertTrue(captured["flagged"])
        self.assertEqual(captured["flag_reasons"], ["pain_high", "pain_rise"])
        self.assertEqual(resp.json()["flag_reasons"], ["pain_high", "pain_rise"])

    def test_a_checkin_on_a_different_plan_in_history_does_not_leak_into_pain_rise(self) -> None:
        history = [
            {
                "plan_id": "99999999-9999-9999-9999-999999999999",
                "pain_nrs": 0,
                "form_score": None,
                "created_at": "2026-09-14T00:00:00+00:00",
            }
        ]
        captured: dict = {}

        def fake_create(*, token: str, payload: dict) -> dict:
            captured.update(payload)
            return {**payload, "id": "c1"}

        with mock.patch.object(plans_store, "plan_exists", return_value=True), \
             mock.patch.object(checkins_store, "recent_checkins", return_value=history), \
             mock.patch.object(checkins_store, "create_checkin", side_effect=fake_create):
            resp = self.client.post(
                "/api/checkins", json={"plan_id": PLAN_ID, "pain_nrs": 5}
            )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertFalse(captured["flagged"])
        self.assertEqual(captured["flag_reasons"], [])


# ---------------------------------------------------------------------------------------------
# GET /api/checkins -- only the caller's own rows, against a REAL filtering fake.
# ---------------------------------------------------------------------------------------------


class ListCheckinsFilteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.db = _FakeDb()
        patcher = mock.patch.object(checkins_store, "_user_client", return_value=self.db)
        self.addCleanup(patcher.stop)
        patcher.start()
        self.addCleanup(app.dependency_overrides.clear)

    def test_lists_only_the_callers_rows_when_the_table_holds_two_users(self) -> None:
        self.db.table("session_checkins").insert(
            {"user_id": "user-1", "plan_id": PLAN_ID, "pain_nrs": 2, "flagged": False, "flag_reasons": []}
        ).execute()
        self.db.table("session_checkins").insert(
            {"user_id": "user-2", "plan_id": PLAN_ID, "pain_nrs": 9, "flagged": True, "flag_reasons": ["pain_high"]}
        ).execute()
        self.db.table("session_checkins").insert(
            {"user_id": "user-1", "plan_id": PLAN_ID, "pain_nrs": 3, "flagged": False, "flag_reasons": []}
        ).execute()

        app.dependency_overrides[get_current_user] = lambda: USER
        resp = self.client.get("/api/checkins")
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = resp.json()["checkins"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["user_id"] == "user-1" for r in rows))

    def test_plan_filter_narrows_further(self) -> None:
        other_plan = "55555555-5555-5555-5555-555555555555"
        self.db.table("session_checkins").insert(
            {"user_id": "user-1", "plan_id": PLAN_ID, "pain_nrs": 2, "flagged": False, "flag_reasons": []}
        ).execute()
        self.db.table("session_checkins").insert(
            {"user_id": "user-1", "plan_id": other_plan, "pain_nrs": 2, "flagged": False, "flag_reasons": []}
        ).execute()

        app.dependency_overrides[get_current_user] = lambda: USER
        resp = self.client.get(f"/api/checkins?plan_id={PLAN_ID}")
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = resp.json()["checkins"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["plan_id"], PLAN_ID)

    def test_non_uuid_plan_filter_is_404(self) -> None:
        app.dependency_overrides[get_current_user] = lambda: USER
        resp = self.client.get("/api/checkins?plan_id=not-a-uuid")
        self.assertEqual(resp.status_code, 404)

    def test_limit_is_clamped_to_the_max(self) -> None:
        for _ in range(3):
            self.db.table("session_checkins").insert(
                {"user_id": "user-1", "plan_id": PLAN_ID, "pain_nrs": 2, "flagged": False, "flag_reasons": []}
            ).execute()
        app.dependency_overrides[get_current_user] = lambda: USER
        resp = self.client.get("/api/checkins?limit=1000")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(len(resp.json()["checkins"]), 3)


# ---------------------------------------------------------------------------------------------
# Auth is required at all.
# ---------------------------------------------------------------------------------------------


class AuthRequiredTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_post_requires_auth(self) -> None:
        resp = self.client.post("/api/checkins", json={"plan_id": PLAN_ID, "pain_nrs": 2})
        self.assertEqual(resp.status_code, 401)

    def test_get_requires_auth(self) -> None:
        resp = self.client.get("/api/checkins")
        self.assertEqual(resp.status_code, 401)


# ---------------------------------------------------------------------------------------------
# PATCH /api/clinic/flags/{id}/ack.
# ---------------------------------------------------------------------------------------------


class AckCheckinFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CLINICIAN
        self.addCleanup(app.dependency_overrides.clear)

    def test_success_returns_the_checkin(self) -> None:
        row = {"id": CHECKIN_ID, "flagged": True, "acknowledged_at": "2026-09-15T00:00:00+00:00"}
        with mock.patch.object(store, "is_clinician", return_value=True), \
             mock.patch.object(clinic, "ack_checkin_flag", return_value={"checkin": row}) as ack:
            resp = self.client.patch(f"/api/clinic/flags/{CHECKIN_ID}/ack")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["checkin"]["id"], CHECKIN_ID)
        ack.assert_called_once_with(token="ctok", checkin_id=CHECKIN_ID)

    def test_not_found_is_404(self) -> None:
        with mock.patch.object(store, "is_clinician", return_value=True), \
             mock.patch.object(clinic, "ack_checkin_flag", return_value={"error": "not_found"}):
            resp = self.client.patch(f"/api/clinic/flags/{CHECKIN_ID}/ack")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_not_flagged_is_409(self) -> None:
        with mock.patch.object(store, "is_clinician", return_value=True), \
             mock.patch.object(clinic, "ack_checkin_flag", return_value={"error": "not_flagged"}):
            resp = self.client.patch(f"/api/clinic/flags/{CHECKIN_ID}/ack")
        self.assertEqual(resp.status_code, 409, resp.text)

    def test_non_uuid_checkin_id_is_404(self) -> None:
        with mock.patch.object(store, "is_clinician", return_value=True):
            resp = self.client.patch("/api/clinic/flags/not-a-uuid/ack")
        self.assertEqual(resp.status_code, 404)

    def test_non_clinician_is_403(self) -> None:
        with mock.patch.object(store, "is_clinician", return_value=False):
            resp = self.client.patch(f"/api/clinic/flags/{CHECKIN_ID}/ack")
        self.assertEqual(resp.status_code, 403)

    def test_requires_auth_at_all(self) -> None:
        app.dependency_overrides.clear()
        resp = self.client.patch(f"/api/clinic/flags/{CHECKIN_ID}/ack")
        self.assertEqual(resp.status_code, 401)


# ---------------------------------------------------------------------------------------------
# services/checkins.py directly -- the failure path and branches the router tests above don't
# happen to reach.
# ---------------------------------------------------------------------------------------------


class _Resp:
    def __init__(self, data) -> None:
        self.data = data


class CheckinsServiceUnitTests(unittest.TestCase):
    def test_create_checkin_raises_if_the_insert_returns_no_row(self) -> None:
        client = mock.Mock()
        client.table.return_value.insert.return_value.execute.return_value = _Resp(data=[])
        with mock.patch.object(checkins_store, "_user_client", return_value=client):
            with self.assertRaises(RuntimeError):
                checkins_store.create_checkin(token="t", payload={"user_id": "u1"})

    def test_create_checkin_returns_the_inserted_row(self) -> None:
        client = mock.Mock()
        client.table.return_value.insert.return_value.execute.return_value = _Resp(
            data=[{"id": "c1", "user_id": "u1"}]
        )
        with mock.patch.object(checkins_store, "_user_client", return_value=client):
            row = checkins_store.create_checkin(token="t", payload={"user_id": "u1"})
        self.assertEqual(row["id"], "c1")

    def test_recent_checkins_scopes_to_user_and_plan(self) -> None:
        db = _FakeDb()
        db.table("session_checkins").insert(
            {"user_id": "u1", "plan_id": PLAN_ID, "pain_nrs": 1}
        ).execute()
        db.table("session_checkins").insert(
            {"user_id": "u1", "plan_id": "other-plan", "pain_nrs": 1}
        ).execute()
        db.table("session_checkins").insert(
            {"user_id": "u2", "plan_id": PLAN_ID, "pain_nrs": 1}
        ).execute()
        with mock.patch.object(checkins_store, "_user_client", return_value=db):
            rows = checkins_store.recent_checkins(token="t", user_id="u1", plan_id=PLAN_ID)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["user_id"], "u1")

    def test_list_checkins_without_a_plan_filter_returns_everything_of_the_callers(self) -> None:
        db = _FakeDb()
        db.table("session_checkins").insert(
            {"user_id": "u1", "plan_id": PLAN_ID, "pain_nrs": 1}
        ).execute()
        db.table("session_checkins").insert(
            {"user_id": "u1", "plan_id": "other-plan", "pain_nrs": 1}
        ).execute()
        with mock.patch.object(checkins_store, "_user_client", return_value=db):
            rows = checkins_store.list_checkins(token="t", user_id="u1")
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
