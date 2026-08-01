from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.app.main import app


class TestMovementsEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_lists_the_registered_movements(self) -> None:
        # Lunge and Row appear here as soon as they are registered in
        # src/pose/movements/registry.py -- this router's own docstring states that as the
        # DESIGN ("registering a fourth detector surfaces it in the UI with no backend or
        # frontend edit"), not an oversight this test should special-case around.
        resp = self.client.get("/api/movements")
        self.assertEqual(resp.status_code, 200)
        names = [m["name"] for m in resp.json()["movements"]]
        self.assertEqual(names, ["Squat", "Overhead Press", "Push-up", "Lunge", "Row"])

    def test_reports_validation_status(self) -> None:
        # Lunge and Row surface with validated=False -- the Beta tag -- because their thresholds
        # are spec-derived and unchecked against labeled data at registration time (Phase 2 is
        # what changes that). See src/pose/movements/lunge.py and row.py's registration comments.
        resp = self.client.get("/api/movements")
        flags = {m["name"]: m["validated"] for m in resp.json()["movements"]}
        self.assertEqual(
            flags,
            {
                "Squat": True,
                "Overhead Press": False,
                "Push-up": False,
                "Lunge": False,
                "Row": False,
            },
        )

    def test_is_public(self) -> None:
        """/app is the anonymous public demo and needs this list to render its selector and
        validate ?movement= before enabling the dropzone, so no auth header is required."""
        self.assertEqual(self.client.get("/api/movements").status_code, 200)

    def test_derives_from_the_registry_not_a_literal(self) -> None:
        """If someone replaces the body with a hardcoded list, registering a fourth detector
        stops surfacing it -- the exact drift this endpoint exists to prevent."""
        from src.pose.movements import registry

        resp = self.client.get("/api/movements")
        self.assertEqual(
            [m["name"] for m in resp.json()["movements"]],
            [d.name for d in registry.list_detectors()],
        )
