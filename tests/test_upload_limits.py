"""Per-file upload cap and per-user storage quota.

Offline throughout — the Supabase client is a fake and the object store is never reached.
"""

from __future__ import annotations

import unittest
from unittest import mock

from backend.app.services import store


class _Result:
    def __init__(self, data=None):
        self.data = data if data is not None else []


class _Table:
    def __init__(self, name, responses, log):
        self._name, self._responses, self._log = name, responses, log
        self._op = None

    def select(self, *args, **kwargs):
        self._op = "select"
        self._log.append((self._name, "select", args))
        return self

    def upsert(self, payload, **kwargs):
        self._op = "upsert"
        self._log.append((self._name, "upsert", payload))
        return self

    def insert(self, payload, **kwargs):
        self._op = "insert"
        self._log.append((self._name, "insert", payload))
        return self

    def eq(self, *args, **kwargs):
        return self

    def execute(self):
        return self._responses.get((self._name, self._op), _Result())


class _Client:
    def __init__(self, responses, log):
        self._responses, self._log = responses, log

    def table(self, name):
        return _Table(name, self._responses, self._log)


class GetStorageUsedTests(unittest.TestCase):
    def _used(self, rows):
        client = _Client({("videos", "select"): _Result(data=rows)}, [])
        with mock.patch.object(store, "_user_client", return_value=client):
            return store.get_storage_used(token="t", user_id="u1")

    def test_sums_the_callers_rows(self) -> None:
        self.assertEqual(self._used([{"size_bytes": 10}, {"size_bytes": 32}]), 42)

    def test_no_rows_is_zero_not_an_error(self) -> None:
        self.assertEqual(self._used([]), 0)

    def test_a_null_size_counts_as_zero(self) -> None:
        """Rows predating the column read back as NULL — they must not poison the sum."""
        self.assertEqual(self._used([{"size_bytes": None}, {"size_bytes": 5}]), 5)

    def test_a_row_missing_the_key_counts_as_zero(self) -> None:
        self.assertEqual(self._used([{}, {"size_bytes": 7}]), 7)


class PersistAnalysisRecordsSizeTests(unittest.TestCase):
    def test_writes_size_bytes_onto_the_videos_row(self) -> None:
        log: list[tuple] = []
        client = _Client({("analyses", "insert"): _Result(data=[{"id": "a1"}])}, log)
        with mock.patch.object(store, "_user_client", return_value=client):
            store.persist_analysis(
                token="t",
                user_id="u1",
                video_id="upload_a",
                source="upload",
                storage_key="uploads/u1/upload_a",
                size_bytes=1234,
                result={},
            )
        upserts = [payload for name, op, payload in log if name == "videos" and op == "upsert"]
        self.assertEqual(len(upserts), 1)
        self.assertEqual(upserts[0]["size_bytes"], 1234)
