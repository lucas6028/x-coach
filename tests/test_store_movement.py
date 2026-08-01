from __future__ import annotations

import unittest


class TestSummarizeMovement(unittest.TestCase):
    def test_promotes_the_movement_out_of_the_result(self) -> None:
        from backend.app.services.store import _summarize

        result = {
            "view": {"view_type": "side"},
            "detections": [{"fault_id": "a"}, {"fault_id": "b"}],
            "pipeline_version": "v1",
            "movement": "Push-up",
        }
        view_type, fault_count, pipeline_version, movement = _summarize(result)
        self.assertEqual(view_type, "side")
        self.assertEqual(fault_count, 2)
        self.assertEqual(pipeline_version, "v1")
        self.assertEqual(movement, "Push-up")

    def test_movement_is_none_when_absent(self) -> None:
        """Analyses produced before the echo landed have no movement; the column is nullable
        and the frontend falls back rather than inventing 'Squat' at the storage layer."""
        from backend.app.services.store import _summarize

        self.assertIsNone(_summarize({"view": {}, "detections": []})[3])

    def test_history_select_includes_movement(self) -> None:
        """The history badge reads the promoted column; if the select drops it, every row
        renders the fallback and the badge silently lies."""
        import inspect

        from backend.app.services import store

        source = inspect.getsource(store.list_analyses)
        self.assertIn("movement", source)
