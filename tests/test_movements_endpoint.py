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
        self.assertEqual(
            names,
            [
                "Squat", "Overhead Press", "Push-up", "Lunge", "Deadlift", "Row",
                "Band Pull Apart", "Bicep Curl", "Arm Abduction", "Arm VW", "Sit-up",
                "Shoulder Bridge", "Leg Abduction",
            ],
        )

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
                # Deadlift ships Beta: there is no labeled deadlift data anywhere in this
                # repository, so its thresholds have never been checked against ground truth.
                "Deadlift": False,
                "Row": False,
                # Band pull apart ships Beta: no REHAB24-6 or labeled-correctness data exists
                # for it anywhere in this repository (src/pose/movements/band_pull_apart.py's
                # registration comment).
                "Band Pull Apart": False,
                "Bicep Curl": False,
                # Arm Abduction ships Beta for a NEW reason: labeled data finally exists
                # (REHAB24-6 Ex1, 178 reps, 90 correct / 88 incorrect) and nothing has run the
                # check. See src/pose/movements/arm_abduction.py's registration comment.
                "Arm Abduction": False,
                # Arm VW ships Beta for the same new reason, on the largest labeled non-squat
                # set yet (REHAB24-6 Ex2, 208 reps, 94 correct / 114 incorrect) and the first
                # that is BILATERAL, i.e. the variant the app actually models. Nothing has run
                # the check. See src/pose/movements/arm_vw.py's registration comment.
                "Arm VW": False,
                # Sit-up ships Beta for a THIRD reason: labeled data exists (EgoExo-Fitness, 82
                # human-judged sit-up actions) but describes a DIFFERENT VARIANT -- its canonical
                # guidance is a full sit-up while the parent spec specifies a curl-up, so a
                # validation run against it would be measuring another exercise. See
                # src/pose/movements/situp.py's registration comment.
                "Sit-up": False,
                # Shoulder Bridge ships Beta for a FOURTH reason, and the only one a DOWNLOAD
                # fixes: the labels exist AND match the variant (EgoExo-Fitness, 77 human-judged
                # Shoulder Bridge actions, one of whose twelve criteria is verbatim this
                # detector's rule, faulted on 16/77) but the PIXELS are missing -- `frames_open`
                # part `.ac` is absent from the download, leaving 2 of the 77 recoverable. See
                # src/pose/movements/shoulder_bridge.py's registration comment.
                "Shoulder Bridge": False,
                # Leg Abduction ships Beta for a FIFTH reason, and it is the first one that is
                # not a gap: the check WAS run. REHAB24-6 Ex4 is standing leg abduction, 210
                # human-labeled repetitions of the variant the app models, and the run decided
                # this detector's roster -- it silenced one rule and confirmed the other. What
                # it cannot do is confirm a FAULT-level claim: REHAB24-6 labels repetitions
                # correct/incorrect and never names the fault. See
                # src/pose/movements/leg_abduction.py's registration comment.
                "Leg Abduction": False,
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
