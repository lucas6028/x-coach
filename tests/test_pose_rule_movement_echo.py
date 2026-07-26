from __future__ import annotations

import unittest


def _empty_payload() -> dict:
    """A structurally valid payload with no frames -- enough to exercise routing without
    depending on any fixture video."""
    return {"metadata": {"fps": 30.0}, "frames": []}


class TestMovementEcho(unittest.TestCase):
    def _detect(self, movement: str | None) -> dict:
        from src.pose.pose_rule_detector import detect_pose_rules_from_payload

        return detect_pose_rules_from_payload(_empty_payload(), movement=movement)

    def test_echoes_each_registered_movement(self) -> None:
        for movement in ("Squat", "Push-up", "Overhead Press"):
            with self.subTest(movement=movement):
                self.assertEqual(self._detect(movement)["movement"], movement)

    def test_normalises_caller_casing_to_the_canonical_name(self) -> None:
        """get_detector lowercases its lookup key but detector.name keeps its case. Echoing
        detector.name (not the caller's string) is what keeps the KG scope and the frontend
        i18n key correct for '--movement push-up'."""
        self.assertEqual(self._detect("push-up")["movement"], "Push-up")
        self.assertEqual(self._detect("OVERHEAD PRESS")["movement"], "Overhead Press")

    def test_defaults_to_squat_when_movement_is_none(self) -> None:
        self.assertEqual(self._detect(None)["movement"], "Squat")

    def test_unknown_movement_still_raises(self) -> None:
        with self.assertRaises(KeyError):
            self._detect("Cartwheel")
