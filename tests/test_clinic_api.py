"""HTTP endpoints for ``/api/clinic`` (clinician side) and ``/api/care`` (patient side).

Business logic (retry, display mapping, the four pure aggregations) is unit-tested directly in
``tests/test_clinic_store.py``; this file proves the ROUTER layer -- gating, 404/400 translation,
request/response shape -- over the real app with ``services.clinic``/``services.store`` mocked,
the same two-tier split ``test_plans_api.py``/``test_backend.py`` use. One test (the plan
hand-off) runs against a REAL shared ``_FakeDb`` instead, because it specifically needs to prove a
plan written by one caller (the clinician) is read back by a DIFFERENT caller (the patient).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from fastapi.testclient import TestClient

from backend.app.auth import CurrentUser, get_current_user
from backend.app.main import app
from backend.app.services import clinic
from backend.app.services import plans as plans_store
from backend.app.services import store
from tests.test_plans_store import _FakeDb

PATIENT_ID = "22222222-2222-2222-2222-222222222222"
LINK_ID = "33333333-3333-3333-3333-333333333333"
CLINICIAN = CurrentUser(id="clinician-1", token="ctok")
PATIENT = CurrentUser(id=PATIENT_ID, token="ptok")


# ---------------------------------------------------------------------------------------------
# Gating: every /api/clinic/* route requires the clinician role EXCEPT /status.
# ---------------------------------------------------------------------------------------------


class ClinicGatingTests(unittest.TestCase):
    # (method, path, json-body-or-None). Bodies are minimal-valid so a 403 is never masked by a
    # 422 that would fire first if the dependency resolved body validation before the auth guard.
    GATED_ROUTES = [
        ("POST", "/api/clinic/invites", None),
        ("GET", "/api/clinic/invites", None),
        ("GET", "/api/clinic/patients", None),
        ("GET", f"/api/clinic/patients/{PATIENT_ID}", None),
        ("POST", f"/api/clinic/patients/{PATIENT_ID}/plans", {"name": "P"}),
        ("DELETE", f"/api/clinic/links/{LINK_ID}", None),
    ]

    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CLINICIAN
        self.addCleanup(app.dependency_overrides.clear)

    def _call(self, method: str, path: str, body):
        if method == "GET":
            return self.client.get(path)
        if method == "POST":
            return self.client.post(path, json=body or {})
        if method == "DELETE":
            return self.client.delete(path)
        raise AssertionError(f"unhandled method {method}")

    def test_non_clinician_is_forbidden_on_every_gated_route(self) -> None:
        with mock.patch.object(store, "is_clinician", return_value=False):
            for method, path, body in self.GATED_ROUTES:
                with self.subTest(route=f"{method} {path}"):
                    resp = self._call(method, path, body)
                    self.assertEqual(resp.status_code, 403, resp.text)

    def test_status_is_not_clinician_gated(self) -> None:
        # A non-clinician gets a truthful false, not a 403 -- the frontend uses this to decide
        # whether to show the clinic nav at all.
        with mock.patch.object(store, "is_clinician", return_value=False):
            resp = self.client.get("/api/clinic/status")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"is_clinician": False})

    def test_status_true_for_a_clinician(self) -> None:
        with mock.patch.object(store, "is_clinician", return_value=True):
            resp = self.client.get("/api/clinic/status")
        self.assertEqual(resp.json(), {"is_clinician": True})

    def test_clinic_routes_require_auth_at_all(self) -> None:
        app.dependency_overrides.clear()
        self.assertEqual(self.client.get("/api/clinic/status").status_code, 401)
        self.assertEqual(self.client.get("/api/clinic/patients").status_code, 401)


class _ClinicianTestCase(unittest.TestCase):
    """Every route below is behind the clinician gate; assume it passes unless a test says otherwise."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CLINICIAN
        self.addCleanup(app.dependency_overrides.clear)
        patcher = mock.patch.object(store, "is_clinician", return_value=True)
        self.addCleanup(patcher.stop)
        patcher.start()


# ---------------------------------------------------------------------------------------------
# Invites.
# ---------------------------------------------------------------------------------------------


class InviteRouterTests(_ClinicianTestCase):
    def test_create_invite_returns_the_new_row(self) -> None:
        row = {"id": "l1", "invite_code": "ABCDEF", "status": "pending"}
        with mock.patch.object(clinic, "create_invite", return_value=row) as ci:
            resp = self.client.post("/api/clinic/invites")
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json(), row)
        ci.assert_called_once_with(token="ctok", clinician_id="clinician-1")

    def test_list_invites_wraps_in_a_key(self) -> None:
        rows = [{"id": "l1", "invite_code": "ABCDEF"}]
        with mock.patch.object(clinic, "list_invites", return_value=rows) as li:
            resp = self.client.get("/api/clinic/invites")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"invites": rows})
        li.assert_called_once_with(token="ctok", clinician_id="clinician-1")


# ---------------------------------------------------------------------------------------------
# GET /patients — list, with the aggregation functions run FOR REAL over mocked I/O.
# ---------------------------------------------------------------------------------------------


class PatientListRouterTests(_ClinicianTestCase):
    def test_empty_when_no_linked_patients(self) -> None:
        with mock.patch.object(clinic, "list_patients", return_value=[]) as lp:
            resp = self.client.get("/api/clinic/patients")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"patients": []})
        lp.assert_called_once_with(token="ctok")

    def test_returns_one_summary_row_per_patient(self) -> None:
        patients = [
            {
                "link_id": "lk1",
                "patient_id": "p1",
                "email": None,
                "display_name": "Alice",
                "accepted_at": "2026-09-01T00:00:00+00:00",
            },
            {
                "link_id": "lk2",
                "patient_id": "p2",
                "email": "bob@x.com",
                "display_name": "bob@x.com",
                "accepted_at": "2026-09-02T00:00:00+00:00",
            },
        ]
        plans = [{"id": "plan-1", "user_id": "p1", "assigned_by": "clinician-1", "created_at": "2026-09-01T00:00:00+00:00"}]
        items = [{"plan_id": "plan-1", "user_id": "p1", "day_index": 1}]
        checkins = [
            {
                "user_id": "p1",
                "plan_id": "plan-1",
                "created_at": "2026-09-14T00:00:00+00:00",
                "flagged": True,
                "acknowledged_at": None,
            }
        ]
        with mock.patch.object(clinic, "list_patients", return_value=patients), \
             mock.patch.object(clinic, "batch_patient_data", return_value=(plans, items, checkins)):
            resp = self.client.get("/api/clinic/patients")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()["patients"]
        self.assertEqual(len(body), 2)

        alice = next(p for p in body if p["patient_id"] == "p1")
        self.assertEqual(alice["display_name"], "Alice")
        self.assertEqual(alice["open_flags"], 1)
        self.assertEqual(alice["last_checkin_at"], "2026-09-14T00:00:00+00:00")

        bob = next(p for p in body if p["patient_id"] == "p2")
        self.assertIsNone(bob["adherence_7d"])  # no plans at all for p2
        self.assertEqual(bob["open_flags"], 0)
        self.assertFalse(bob["inactive_7d"])


# ---------------------------------------------------------------------------------------------
# GET /patients/{id} — detail, and the 404 boundary (criterion b).
# ---------------------------------------------------------------------------------------------


class PatientDetailRouterTests(_ClinicianTestCase):
    def test_404_for_a_patient_not_in_clinic_patients(self) -> None:
        with mock.patch.object(clinic, "find_patient", return_value=None) as fp:
            resp = self.client.get(f"/api/clinic/patients/{PATIENT_ID}")
        self.assertEqual(resp.status_code, 404, resp.text)
        fp.assert_called_once_with(token="ctok", patient_id=PATIENT_ID)

    def test_404_for_a_non_uuid_patient_id(self) -> None:
        resp = self.client.get("/api/clinic/patients/not-a-uuid")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_returns_summary_plans_checkins_trend_and_open_flags(self) -> None:
        # The router calls the real clock (datetime.now(timezone.utc)), so the fixture's check-in
        # is dated RELATIVE to it (1 day ago) rather than a hardcoded date -- otherwise this test
        # would start failing the day "now" drifts more than 30 days past a hardcoded fixture date.
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        row = {"link_id": "lk1", "patient_id": PATIENT_ID, "email": None, "display_name": "Alice", "accepted_at": "t"}
        all_checkins = [
            {
                "id": "c1",
                "user_id": PATIENT_ID,
                "plan_id": "plan-1",
                "created_at": recent,
                "form_score": 80,
                "pain_nrs": 2,
                "flagged": True,
                "acknowledged_at": None,
            }
        ]
        with mock.patch.object(clinic, "find_patient", return_value=row), \
             mock.patch.object(clinic, "batch_patient_data", return_value=([], [], all_checkins)), \
             mock.patch.object(clinic, "list_checkins", return_value=all_checkins), \
             mock.patch.object(plans_store, "list_plans", return_value=[{"id": "plan-1", "assigned_by": "clinician-1"}]) as lp:
            resp = self.client.get(f"/api/clinic/patients/{PATIENT_ID}")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["patient"]["display_name"], "Alice")
        self.assertEqual(body["plans"], [{"id": "plan-1", "assigned_by": "clinician-1"}])
        self.assertEqual(len(body["checkins"]), 1)
        self.assertEqual(len(body["trend"]), 1)
        self.assertEqual(body["trend"][0]["form_score"], 80)
        self.assertEqual(len(body["open_flags"]), 1)
        lp.assert_called_once_with(token="ctok", user_id=PATIENT_ID)


# ---------------------------------------------------------------------------------------------
# POST /patients/{id}/plans — 404 boundary + the extracted item-building logic.
# ---------------------------------------------------------------------------------------------


class AssignPlanRouterTests(_ClinicianTestCase):
    def test_404_for_a_patient_not_in_clinic_patients(self) -> None:
        with mock.patch.object(clinic, "find_patient", return_value=None):
            resp = self.client.post(
                f"/api/clinic/patients/{PATIENT_ID}/plans", json={"name": "Knee rehab"}
            )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_unknown_template_key_is_400(self) -> None:
        with mock.patch.object(clinic, "find_patient", return_value={"patient_id": PATIENT_ID}):
            resp = self.client.post(
                f"/api/clinic/patients/{PATIENT_ID}/plans",
                json={"name": "X", "template_key": "does_not_exist"},
            )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_creates_the_plan_owned_by_the_patient_and_assigned_by_the_clinician(self) -> None:
        with mock.patch.object(clinic, "find_patient", return_value={"patient_id": PATIENT_ID}), \
             mock.patch.object(plans_store, "create_plan", return_value={"id": "plan-1"}) as cp:
            resp = self.client.post(
                f"/api/clinic/patients/{PATIENT_ID}/plans",
                json={
                    "name": "Knee rehab",
                    "items": [{"day_index": 1, "movement": "Squat", "sets": 3, "reps": 10}],
                },
            )
        self.assertEqual(resp.status_code, 201, resp.text)
        cp.assert_called_once()
        kwargs = cp.call_args.kwargs
        self.assertEqual(kwargs["token"], "ctok")
        self.assertEqual(kwargs["user_id"], PATIENT_ID)
        self.assertEqual(kwargs["assigned_by"], "clinician-1")
        self.assertEqual(kwargs["items"][0]["movement"], "Squat")


class PlanAssignmentHandoffTests(unittest.TestCase):
    """Criterion (c): a plan a clinician assigns shows up in the PATIENT's own GET /api/plans,
    with assigned_by set -- proven against one real _FakeDb shared by both callers, not mocks."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.db = _FakeDb()
        patcher = mock.patch.object(plans_store, "_user_client", return_value=self.db)
        self.addCleanup(patcher.stop)
        patcher.start()
        self.addCleanup(app.dependency_overrides.clear)

    def test_assigned_plan_appears_in_the_patients_plan_list(self) -> None:
        app.dependency_overrides[get_current_user] = lambda: CLINICIAN
        with mock.patch.object(store, "is_clinician", return_value=True), \
             mock.patch.object(clinic, "find_patient", return_value={"patient_id": PATIENT_ID}):
            create_resp = self.client.post(
                f"/api/clinic/patients/{PATIENT_ID}/plans",
                json={
                    "name": "Knee rehab",
                    "items": [{"day_index": 1, "movement": "Squat", "sets": 3, "reps": 10}],
                },
            )
        self.assertEqual(create_resp.status_code, 201, create_resp.text)
        self.assertEqual(create_resp.json()["assigned_by"], "clinician-1")

        app.dependency_overrides[get_current_user] = lambda: PATIENT
        list_resp = self.client.get("/api/plans")
        self.assertEqual(list_resp.status_code, 200, list_resp.text)
        plans = list_resp.json()["plans"]
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["assigned_by"], "clinician-1")
        self.assertEqual(plans[0]["id"], create_resp.json()["id"])


# ---------------------------------------------------------------------------------------------
# DELETE /links/{id}.
# ---------------------------------------------------------------------------------------------


class RevokeLinkRouterTests(_ClinicianTestCase):
    def test_not_found_becomes_404(self) -> None:
        with mock.patch.object(clinic, "revoke_link", return_value={"error": "not_found"}) as rl:
            resp = self.client.delete(f"/api/clinic/links/{LINK_ID}")
        self.assertEqual(resp.status_code, 404, resp.text)
        rl.assert_called_once_with(token="ctok", link_id=LINK_ID)

    def test_success_returns_the_link(self) -> None:
        with mock.patch.object(clinic, "revoke_link", return_value={"link": {"id": LINK_ID, "status": "revoked"}}):
            resp = self.client.delete(f"/api/clinic/links/{LINK_ID}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["link"]["status"], "revoked")

    def test_non_uuid_link_id_is_404(self) -> None:
        resp = self.client.delete("/api/clinic/links/not-a-uuid")
        self.assertEqual(resp.status_code, 404)


# ---------------------------------------------------------------------------------------------
# /api/care — the patient side. Not clinician-gated: any signed-in user.
# ---------------------------------------------------------------------------------------------


class CareAcceptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: PATIENT
        self.addCleanup(app.dependency_overrides.clear)

    def test_invalid_code_is_404(self) -> None:
        with mock.patch.object(clinic, "accept_invite", return_value={"error": "invalid"}):
            resp = self.client.post("/api/care/accept", json={"code": "ABCDEF"})
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_self_invite_is_400(self) -> None:
        with mock.patch.object(clinic, "accept_invite", return_value={"error": "self"}):
            resp = self.client.post("/api/care/accept", json={"code": "ABCDEF"})
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_success_returns_the_link(self) -> None:
        with mock.patch.object(
            clinic, "accept_invite", return_value={"link": {"id": "lk1", "status": "active"}}
        ) as ai:
            resp = self.client.post("/api/care/accept", json={"code": "ABCDEF"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["link"]["status"], "active")
        ai.assert_called_once_with(token="ptok", code="ABCDEF")

    def test_empty_code_is_422(self) -> None:
        resp = self.client.post("/api/care/accept", json={"code": ""})
        self.assertEqual(resp.status_code, 422)

    def test_requires_auth(self) -> None:
        app.dependency_overrides.clear()
        resp = self.client.post("/api/care/accept", json={"code": "ABCDEF"})
        self.assertEqual(resp.status_code, 401)


class CareClinicianListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: PATIENT
        self.addCleanup(app.dependency_overrides.clear)

    def test_lists_the_callers_clinicians(self) -> None:
        rows = [{"link_id": "lk1", "clinician_id": "c1", "display_name": "Dr. A"}]
        with mock.patch.object(clinic, "my_clinicians", return_value=rows) as mc:
            resp = self.client.get("/api/care/clinicians")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"clinicians": rows})
        mc.assert_called_once_with(token="ptok")


class CareRevokeLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: PATIENT
        self.addCleanup(app.dependency_overrides.clear)

    def test_not_found_becomes_404(self) -> None:
        with mock.patch.object(clinic, "revoke_link", return_value={"error": "not_found"}):
            resp = self.client.delete(f"/api/care/links/{LINK_ID}")
        self.assertEqual(resp.status_code, 404)

    def test_success(self) -> None:
        with mock.patch.object(
            clinic, "revoke_link", return_value={"link": {"id": LINK_ID, "status": "revoked"}}
        ) as rl:
            resp = self.client.delete(f"/api/care/links/{LINK_ID}")
        self.assertEqual(resp.status_code, 200)
        rl.assert_called_once_with(token="ptok", link_id=LINK_ID)

    def test_non_uuid_is_404(self) -> None:
        resp = self.client.delete("/api/care/links/not-a-uuid")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
