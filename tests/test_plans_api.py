"""The training-plan HTTP endpoints, over the real app.

These run the router END TO END against the in-memory PostgREST stand-in from
``test_plans_store`` rather than mocking the service out: the interesting behaviour of this router
is the translation layer (``completed`` -> ``completed_at``, movement canonicalization, which id
becomes a 404 and which a 400), and a mocked service would let all of it pass untested.

The auth dependency is overridden per suite, except in ``PlanAuthTests``, which is the one place
that must go through the real dependency to pin that these endpoints require a session at all.
"""

from __future__ import annotations

import unittest
import uuid
from typing import Any
from unittest import mock

from fastapi.testclient import TestClient

from backend.app.auth import CurrentUser, get_current_user
from backend.app.main import app
from backend.app.routers import plans as plans_router
from backend.app.services import plans as plans_store
from backend.app.services import store
from tests.test_plans_store import _FakeDb


class _PlanApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.db = _FakeDb()
        patcher = mock.patch.object(plans_store, "_user_client", return_value=self.db)
        self.addCleanup(patcher.stop)
        patcher.start()
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)

    def _create(self, **body: Any) -> dict[str, Any]:
        payload = {"name": "My week"}
        payload.update(body)
        resp = self.client.post("/api/plans", json=payload)
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()


class TemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        # No auth override on purpose: this catalog is public, and a leaked override from a sibling
        # suite would make that impossible to tell apart from a gated endpoint.
        self.assertEqual(app.dependency_overrides, {})

    def test_templates_are_public(self) -> None:
        resp = self.client.get("/api/plans/templates")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["templates"])

    def test_templates_route_is_not_swallowed_by_the_plan_id_route(self) -> None:
        """``/plans/templates`` must be declared before ``/plans/{plan_id}``.

        Swapping the two declarations makes "templates" parse as a plan id and this returns 404 (or
        401), which is exactly the failure this ordering exists to prevent.
        """
        body = self.client.get("/api/plans/templates").json()
        self.assertIn("templates", body)

    def test_every_template_movement_is_analysable(self) -> None:
        """Templates only ever seed movements a detector can actually run.

        Jumping Jacks and High Knee are addable by hand but are deliberately not put in anyone's
        default path, since a plan item pointing at them can only ever be ticked manually.
        """
        from src.pose.movements import registry

        registered = {d.name for d in registry.list_detectors()}
        for template in plans_router.TEMPLATES:
            for item in template.items:
                self.assertIn(item.movement, registered, f"{template.key}: {item.movement}")

    def test_every_template_item_is_within_the_bounds_the_database_enforces(self) -> None:
        """Template rows are the one item payload that never passes through request validation --
        they are copied straight into the insert -- so a day 8 or a 50-set exercise in TEMPLATES
        would pass every other test and fail at the migration's check constraint as a 500."""
        for template in plans_router.TEMPLATES:
            for item in template.items:
                with self.subTest(template=template.key, movement=item.movement):
                    self.assertIn(item.day_index, range(1, plans_router.MAX_DAY + 1))
                    self.assertIn(item.sets, range(1, 21))
                    self.assertIn(item.reps, range(1, 201))

    def test_a_template_item_outside_those_bounds_is_rejected_at_construction(self) -> None:
        """The bounds live on TemplateItem, so a bad literal fails when the module is loaded rather
        than when a user tries to use it."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            plans_router.TemplateItem(day_index=8, movement="Squat", sets=3, reps=10)
        with self.assertRaises(ValidationError):
            plans_router.TemplateItem(day_index=1, movement="Squat", sets=50, reps=10)

    def test_template_keys_are_unique(self) -> None:
        keys = [t.key for t in plans_router.TEMPLATES]
        self.assertEqual(len(set(keys)), len(keys))


class PlanAuthTests(unittest.TestCase):
    """The plan endpoints are one user's own data and must 401 without a session.

    Verified by mutation, not assumption: removing ``= Depends(get_current_user)`` from
    ``plans.list_plans`` makes ``test_list_requires_auth`` fail.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.assertEqual(
            app.dependency_overrides, {}, "a leaked dependency override would hide a 401 regression"
        )

    def test_list_requires_auth(self) -> None:
        self.assertEqual(self.client.get("/api/plans").status_code, 401)

    def test_create_requires_auth(self) -> None:
        # A VALID body, so a 422 from request validation cannot stand in for the 401.
        self.assertEqual(self.client.post("/api/plans", json={"name": "x"}).status_code, 401)

    def test_get_requires_auth(self) -> None:
        self.assertEqual(self.client.get("/api/plans/some-id").status_code, 401)

    def test_start_requires_auth(self) -> None:
        self.assertEqual(self.client.post("/api/plans/some-id/start").status_code, 401)


class CreatePlanTests(_PlanApiTestCase):
    def test_creates_an_empty_plan(self) -> None:
        plan = self._create()
        self.assertEqual(plan["name"], "My week")
        self.assertEqual(plan["items"], [])

    def test_trims_the_name(self) -> None:
        self.assertEqual(self._create(name="  Leg day  ")["name"], "Leg day")

    def test_creates_from_explicit_items(self) -> None:
        plan = self._create(
            items=[
                {"day_index": 1, "movement": "Squat", "sets": 4, "reps": 12},
                {"day_index": 2, "movement": "Row"},
            ]
        )
        self.assertEqual([i["movement"] for i in plan["items"]], ["Squat", "Row"])
        self.assertEqual(plan["items"][0]["sets"], 4)
        # The unspecified item takes the schema defaults rather than nulls.
        self.assertEqual((plan["items"][1]["sets"], plan["items"][1]["reps"]), (3, 10))

    def test_canonicalizes_item_movements(self) -> None:
        plan = self._create(items=[{"day_index": 1, "movement": "  push-UP "}])
        self.assertEqual(plan["items"][0]["movement"], "Push-up")

    def test_accepts_a_catalog_movement_with_no_detector(self) -> None:
        # All sixteen are plannable; only fourteen are analysable. This is the difference.
        plan = self._create(items=[{"day_index": 1, "movement": "Jumping Jacks"}])
        self.assertEqual(plan["items"][0]["movement"], "Jumping Jacks")

    def test_rejects_an_unknown_movement(self) -> None:
        resp = self.client.post(
            "/api/plans", json={"name": "x", "items": [{"day_index": 1, "movement": "Burpee"}]}
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_rejects_a_day_index_outside_the_week(self) -> None:
        resp = self.client.post(
            "/api/plans", json={"name": "x", "items": [{"day_index": 8, "movement": "Squat"}]}
        )
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_rejects_a_blank_name(self) -> None:
        self.assertEqual(self.client.post("/api/plans", json={"name": ""}).status_code, 422)

    def test_creates_from_a_template(self) -> None:
        template = plans_router.TEMPLATES[0]
        plan = self._create(template_key=template.key)
        self.assertEqual(plan["template_key"], template.key)
        self.assertEqual(len(plan["items"]), len(template.items))

    def test_template_items_are_copied_not_referenced(self) -> None:
        """Editing TEMPLATES later must never mutate a plan a user already owns."""
        template = plans_router.TEMPLATES[0]
        plan = self._create(template_key=template.key)
        item_id = plan["items"][0]["id"]
        self.client.patch(f"/api/plans/{plan['id']}/items/{item_id}", json={"sets": 99})
        self.assertEqual(plans_router.TEMPLATES[0].items[0].sets, template.items[0].sets)

    def test_a_template_create_ignores_explicit_items(self) -> None:
        # One or the other — merging the two would make the result hard to predict.
        template = plans_router.TEMPLATES[0]
        plan = self._create(
            template_key=template.key, items=[{"day_index": 7, "movement": "Sit-up"}]
        )
        self.assertEqual(len(plan["items"]), len(template.items))
        self.assertNotIn(7, [i["day_index"] for i in plan["items"]])

    def test_rejects_an_unknown_template(self) -> None:
        resp = self.client.post("/api/plans", json={"name": "x", "template_key": "nope"})
        self.assertEqual(resp.status_code, 400, resp.text)


class ReadAndUpdatePlanTests(_PlanApiTestCase):
    def test_lists_the_callers_plans_with_a_summary(self) -> None:
        self._create(name="A", items=[{"day_index": 1, "movement": "Squat"}])
        body = self.client.get("/api/plans").json()
        self.assertEqual(len(body["plans"]), 1)
        self.assertEqual(body["plans"][0]["item_count"], 1)

    def test_gets_one_plan_with_its_items(self) -> None:
        plan = self._create(items=[{"day_index": 1, "movement": "Squat"}])
        resp = self.client.get(f"/api/plans/{plan['id']}")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(len(resp.json()["items"]), 1)

    def test_missing_plan_is_404(self) -> None:
        resp = self.client.get(f"/api/plans/{uuid.uuid4()}")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_a_non_uuid_plan_id_is_404_not_500(self) -> None:
        # Forwarded raw it would reach PostgREST as 22P02 and surface as a 500.
        self.assertEqual(self.client.get("/api/plans/not-a-uuid").status_code, 404)

    def test_renames_a_plan(self) -> None:
        plan = self._create()
        resp = self.client.patch(f"/api/plans/{plan['id']}", json={"name": "  Renamed "})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["name"], "Renamed")

    def test_an_empty_patch_returns_the_plan_unchanged(self) -> None:
        plan = self._create()
        resp = self.client.patch(f"/api/plans/{plan['id']}", json={})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["name"], "My week")

    def test_patching_a_missing_plan_is_404(self) -> None:
        # A well-formed id that matches nothing, so the 404 comes from the store returning None
        # rather than from the uuid guard -- the two are different lines and both must answer 404.
        resp = self.client.patch(f"/api/plans/{uuid.uuid4()}", json={"name": "x"})
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_patching_a_non_uuid_plan_id_is_404(self) -> None:
        self.assertEqual(self.client.patch("/api/plans/not-a-uuid", json={}).status_code, 404)

    def test_deletes_a_plan(self) -> None:
        plan = self._create()
        self.assertEqual(self.client.delete(f"/api/plans/{plan['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/plans/{plan['id']}").status_code, 404)

    def test_deleting_a_missing_plan_is_404(self) -> None:
        self.assertEqual(self.client.delete(f"/api/plans/{uuid.uuid4()}").status_code, 404)

    def test_deleting_a_non_uuid_plan_id_is_404(self) -> None:
        self.assertEqual(self.client.delete("/api/plans/not-a-uuid").status_code, 404)


class StartPlanTests(_PlanApiTestCase):
    def test_start_stamps_and_clears_progress(self) -> None:
        plan = self._create(items=[{"day_index": 1, "movement": "Squat"}])
        item_id = plan["items"][0]["id"]
        self.client.patch(f"/api/plans/{plan['id']}/items/{item_id}", json={"completed": True})

        resp = self.client.post(f"/api/plans/{plan['id']}/start")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIsNotNone(body["started_at"])
        self.assertIsNone(body["items"][0]["completed_at"])

    def test_starting_a_missing_plan_is_404(self) -> None:
        self.assertEqual(self.client.post(f"/api/plans/{uuid.uuid4()}/start").status_code, 404)

    def test_starting_a_non_uuid_plan_id_is_404(self) -> None:
        self.assertEqual(self.client.post("/api/plans/not-a-uuid/start").status_code, 404)


class ItemEndpointTests(_PlanApiTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.plan = self._create()
        self.plan_id = self.plan["id"]

    def _add(self, **body: Any) -> dict[str, Any]:
        payload = {"day_index": 1, "movement": "Squat"}
        payload.update(body)
        resp = self.client.post(f"/api/plans/{self.plan_id}/items", json=payload)
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_adds_an_item(self) -> None:
        item = self._add(sets=5, reps=6, notes="tempo")
        self.assertEqual((item["sets"], item["reps"], item["notes"]), (5, 6, "tempo"))

    def test_adding_to_a_missing_plan_is_404(self) -> None:
        resp = self.client.post(
            f"/api/plans/{uuid.uuid4()}/items", json={"day_index": 1, "movement": "Squat"}
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_adding_an_unknown_movement_is_400(self) -> None:
        resp = self.client.post(
            f"/api/plans/{self.plan_id}/items", json={"day_index": 1, "movement": "Burpee"}
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_moves_an_item_to_another_day(self) -> None:
        item = self._add()
        resp = self.client.patch(
            f"/api/plans/{self.plan_id}/items/{item['id']}", json={"day_index": 4, "position": 2}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual((resp.json()["day_index"], resp.json()["position"]), (4, 2))

    def test_patch_canonicalizes_a_changed_movement(self) -> None:
        item = self._add()
        resp = self.client.patch(
            f"/api/plans/{self.plan_id}/items/{item['id']}", json={"movement": "row"}
        )
        self.assertEqual(resp.json()["movement"], "Row")

    def test_an_explicit_null_for_a_not_null_column_is_ignored(self) -> None:
        # Every PlanItemPatch field is optional and therefore typed `X | None`, so a client can
        # send `{"movement": null}`. Forwarded, it would hit a NOT NULL column and surface as a
        # 500; dropped, the patch is simply a no-op for that field.
        item = self._add(movement="Squat", sets=4)
        resp = self.client.patch(
            f"/api/plans/{self.plan_id}/items/{item['id']}",
            json={"movement": None, "sets": None, "day_index": None, "position": None, "reps": None},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual((resp.json()["movement"], resp.json()["sets"]), ("Squat", 4))

    def test_an_explicit_null_note_clears_it(self) -> None:
        # `notes` is NULLABLE, and null is how a note is cleared -- so it is deliberately not in
        # the dropped set above.
        item = self._add(notes="tempo")
        resp = self.client.patch(
            f"/api/plans/{self.plan_id}/items/{item['id']}", json={"notes": None}
        )
        self.assertIsNone(resp.json()["notes"])

    def test_patch_rejects_an_unknown_movement(self) -> None:
        item = self._add()
        resp = self.client.patch(
            f"/api/plans/{self.plan_id}/items/{item['id']}", json={"movement": "Burpee"}
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_completing_stamps_completed_at(self) -> None:
        item = self._add()
        resp = self.client.patch(
            f"/api/plans/{self.plan_id}/items/{item['id']}", json={"completed": True}
        )
        self.assertIsNotNone(resp.json()["completed_at"])

    def test_unticking_clears_both_the_stamp_and_the_analysis_link(self) -> None:
        item = self._add()
        with mock.patch.object(store, "get_analysis", return_value={"id": "an-1"}):
            self.client.patch(
                f"/api/plans/{self.plan_id}/items/{item['id']}",
                json={"completed": True, "analysis_id": "11111111-1111-1111-1111-111111111111"},
            )
        resp = self.client.patch(
            f"/api/plans/{self.plan_id}/items/{item['id']}", json={"completed": False}
        )
        body = resp.json()
        self.assertIsNone(body["completed_at"])
        # Leaving a stale link would light 看報告 on an item the user just said they had not done.
        self.assertIsNone(body["analysis_id"])

    def test_links_an_analysis_the_caller_owns(self) -> None:
        item = self._add()
        with mock.patch.object(store, "get_analysis", return_value={"id": "an-1"}) as get:
            resp = self.client.patch(
                f"/api/plans/{self.plan_id}/items/{item['id']}",
                json={"completed": True, "analysis_id": "11111111-1111-1111-1111-111111111111"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["analysis_id"], "11111111-1111-1111-1111-111111111111")
        get.assert_called_once()

    def test_rejects_an_analysis_the_caller_does_not_own(self) -> None:
        # FK checks do not run under RLS, so without this check a client could pin someone else's
        # analysis id onto its own item and leave itself a 看報告 link that 404s.
        item = self._add()
        with mock.patch.object(store, "get_analysis", return_value=None):
            resp = self.client.patch(
                f"/api/plans/{self.plan_id}/items/{item['id']}",
                json={"completed": True, "analysis_id": "11111111-1111-1111-1111-111111111111"},
            )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_rejects_a_non_uuid_analysis_id(self) -> None:
        item = self._add()
        with mock.patch.object(store, "get_analysis") as get:
            resp = self.client.patch(
                f"/api/plans/{self.plan_id}/items/{item['id']}",
                json={"completed": True, "analysis_id": "not-a-uuid"},
            )
        self.assertEqual(resp.status_code, 400, resp.text)
        # Refused before it costs a lookup.
        get.assert_not_called()

    def test_patching_a_missing_item_is_404(self) -> None:
        resp = self.client.patch(
            f"/api/plans/{self.plan_id}/items/{uuid.uuid4()}", json={"sets": 4}
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_a_non_uuid_item_id_is_404(self) -> None:
        resp = self.client.patch(
            f"/api/plans/{self.plan_id}/items/not-a-uuid", json={"sets": 4}
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_deletes_an_item(self) -> None:
        item = self._add()
        self.assertEqual(
            self.client.delete(f"/api/plans/{self.plan_id}/items/{item['id']}").status_code, 200
        )
        self.assertEqual(self.client.get(f"/api/plans/{self.plan_id}").json()["items"], [])

    def test_deleting_a_missing_item_is_404(self) -> None:
        resp = self.client.delete(f"/api/plans/{self.plan_id}/items/{uuid.uuid4()}")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_deleting_a_non_uuid_item_id_is_404(self) -> None:
        resp = self.client.delete(f"/api/plans/{self.plan_id}/items/not-a-uuid")
        self.assertEqual(resp.status_code, 404, resp.text)
