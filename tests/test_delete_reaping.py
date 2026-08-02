"""Deleting an analysis must also reap its stored objects.

Before object storage, `delete_analysis`'s own docstring admitted the uploaded file was
"deliberately left on disk". With a `delete_prefix` to call, that orphan is closed.
"""

from __future__ import annotations

import unittest
from unittest import mock

from backend.app.services import storage, store


class _Result:
    def __init__(self, data=None, count=None):
        self.data = data if data is not None else []
        self.count = count


class _Table:
    """Records the operation order so a test can assert reads happen before deletes."""

    def __init__(self, log, name, responses):
        self._log, self._name, self._responses = log, name, responses
        self._op = None

    def select(self, *args, **kwargs):
        self._op = "select"
        self._log.append((self._name, "select"))
        return self

    def delete(self):
        self._op = "delete"
        self._log.append((self._name, "delete"))
        return self

    def eq(self, *args, **kwargs):
        return self

    def in_(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        return self._responses.get((self._name, self._op), _Result())


class _Client:
    def __init__(self, log, responses):
        self._log, self._responses = log, responses

    def table(self, name):
        return _Table(self._log, name, self._responses)


class _RecordingStore:
    def __init__(self, fail=False):
        self.deleted: list[str] = []
        self._fail = fail

    def delete_prefix(self, prefix):
        if self._fail:
            raise storage.StorageError("down")
        self.deleted.append(prefix)


class DeleteAnalysisReapsObjectsTests(unittest.TestCase):
    def _run(self, obj_store, *, siblings=0):
        log: list[tuple[str, str]] = []
        responses = {
            ("analyses", "select"): _Result(data=[{"video_id": "upload_a"}], count=siblings),
            ("analyses", "delete"): _Result(data=[{"id": "a1"}]),
            ("videos", "select"): _Result(data=[{"storage_key": "uploads/u1/upload_a"}]),
        }
        client = _Client(log, responses)
        with mock.patch.object(store, "_user_client", return_value=client):
            with mock.patch.object(storage, "get_object_store", return_value=obj_store):
                ok = store.delete_analysis(token="t", analysis_id="a1", user_id="u1")
        return ok, log

    def test_reaps_the_prefix_when_it_was_the_last_analysis(self) -> None:
        obj = _RecordingStore()
        ok, _ = self._run(obj, siblings=0)
        self.assertTrue(ok)
        self.assertEqual(obj.deleted, ["uploads/u1/upload_a"])

    def test_keeps_the_objects_when_a_sibling_analysis_remains(self) -> None:
        """Re-analysing one clip inserts a second `analyses` row against the same video."""
        obj = _RecordingStore()
        self._run(obj, siblings=1)
        self.assertEqual(obj.deleted, [])

    def test_a_storage_failure_does_not_undo_the_db_delete(self) -> None:
        obj = _RecordingStore(fail=True)
        ok, _ = self._run(obj, siblings=0)
        self.assertTrue(ok, "the row is gone; a failed reap leaves an orphan, not a stuck record")


class DeleteAllAnalysesReapsObjectsTests(unittest.TestCase):
    def test_reads_every_storage_key_before_deleting_anything(self) -> None:
        """Selecting after the bulk delete would find no rows and silently reap nothing."""
        log: list[tuple[str, str]] = []
        responses = {
            ("analyses", "delete"): _Result(data=[{"id": "a1"}, {"id": "a2"}]),
            ("videos", "select"): _Result(
                data=[
                    {"storage_key": "uploads/u1/upload_a"},
                    {"storage_key": "uploads/u1/upload_b"},
                ]
            ),
        }
        obj = _RecordingStore()
        client = _Client(log, responses)
        with mock.patch.object(store, "_user_client", return_value=client):
            with mock.patch.object(storage, "get_object_store", return_value=obj):
                deleted = store.delete_all_analyses(token="t", user_id="u1")
        self.assertEqual(deleted, 2)
        self.assertEqual(sorted(obj.deleted), ["uploads/u1/upload_a", "uploads/u1/upload_b"])
        first_delete = next(i for i, (_, op) in enumerate(log) if op == "delete")
        videos_select = log.index(("videos", "select"))
        self.assertLess(videos_select, first_delete)

    def test_a_storage_failure_does_not_stop_the_db_delete(self) -> None:
        log: list[tuple[str, str]] = []
        responses = {
            ("analyses", "delete"): _Result(data=[{"id": "a1"}]),
            ("videos", "select"): _Result(data=[{"storage_key": "uploads/u1/upload_a"}]),
        }
        client = _Client(log, responses)
        with mock.patch.object(store, "_user_client", return_value=client):
            with mock.patch.object(storage, "get_object_store", return_value=_RecordingStore(fail=True)):
                deleted = store.delete_all_analyses(token="t", user_id="u1")
        self.assertEqual(deleted, 1)

    def test_a_failing_object_store_does_not_escape(self) -> None:
        """_reap_objects runs after the rows are gone, so a raise here would 500 a delete that
        already succeeded — including a failure from get_object_store() itself."""
        log: list[tuple[str, str]] = []
        responses = {
            ("analyses", "delete"): _Result(data=[{"id": "a1"}]),
            ("videos", "select"): _Result(data=[{"storage_key": "uploads/u1/upload_a"}]),
        }
        client = _Client(log, responses)
        with mock.patch.object(store, "_user_client", return_value=client):
            with mock.patch.object(storage, "get_object_store", side_effect=RuntimeError("boom")):
                deleted = store.delete_all_analyses(token="t", user_id="u1")
        self.assertEqual(deleted, 1)

    def test_one_failing_prefix_does_not_stop_the_others(self) -> None:
        """The try sits INSIDE the loop, so a bad prefix must not abandon the rest."""

        class _PartiallyFailing:
            def __init__(self):
                self.deleted: list[str] = []

            def delete_prefix(self, prefix):
                if prefix.endswith("upload_a"):
                    raise storage.StorageError("down")
                self.deleted.append(prefix)

        log: list[tuple[str, str]] = []
        responses = {
            ("analyses", "delete"): _Result(data=[{"id": "a1"}, {"id": "a2"}]),
            ("videos", "select"): _Result(
                data=[
                    {"storage_key": "uploads/u1/upload_a"},
                    {"storage_key": "uploads/u1/upload_b"},
                ]
            ),
        }
        obj = _PartiallyFailing()
        client = _Client(log, responses)
        with mock.patch.object(store, "_user_client", return_value=client):
            with mock.patch.object(storage, "get_object_store", return_value=obj):
                store.delete_all_analyses(token="t", user_id="u1")
        self.assertEqual(obj.deleted, ["uploads/u1/upload_b"])


if __name__ == "__main__":
    unittest.main()
