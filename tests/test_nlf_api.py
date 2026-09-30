"""Tests for the NLF 3D view feature: services.store's new seams, routers/nlf.py, and the
admin `enable_nlf` toggle on `PUT /api/admin/users/{id}/role`.

Supabase is faked locally (rather than reusing tests/test_backend.py's `_FakeQuery`/`_fake_client`,
per the task's own guidance) since these store functions only ever chain
select/eq/limit/upsert/delete/rpc -- a narrower fake is simpler than widening the shared one.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from fastapi.testclient import TestClient

from backend.app.auth import CurrentUser, get_current_user
from backend.app.main import app
from backend.app.routers import nlf as nlf_router
from backend.app.services import storage, store

_AID = "3f2a5c1e-0000-4000-8000-000000000001"


# ------------------------------------------------------------------------------------- fakes


class _Resp:
    def __init__(self, data=None) -> None:
        self.data = data


class _FakeQuery:
    """Records chained PostgREST calls and returns a preset response on execute()."""

    def __init__(self, resp: _Resp) -> None:
        self._resp = resp
        self.eq_calls: list[tuple] = []
        self.upserted: dict | None = None
        self.deleted = False

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        self.eq_calls.append(a)
        return self

    def limit(self, *a, **k):
        return self

    def upsert(self, row, **k):
        self.upserted = row
        return self

    def delete(self, *a, **k):
        self.deleted = True
        return self

    def execute(self):
        return self._resp


def _fake_client(resp: _Resp) -> tuple[mock.Mock, _FakeQuery]:
    """A client whose `.table(...)` and `.rpc(...)` both return one shared, chainable query."""
    query = _FakeQuery(resp)
    client = mock.Mock()
    client.table.return_value = query
    client.rpc.return_value = query
    return client, query


class _ApiError(Exception):
    """Stand-in for a postgrest `APIError` carrying a Postgres/PostgREST error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


# ---------------------------------------------------------------------------- services.store


class StoreNlfTests(unittest.TestCase):
    def test_get_nlf_enabled_true(self) -> None:
        client, _ = _fake_client(_Resp(data=[{"user_id": "u1"}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertTrue(store.get_nlf_enabled(token="t", user_id="u1"))
        client.table.assert_called_with("user_roles")

    def test_get_nlf_enabled_false(self) -> None:
        client, _ = _fake_client(_Resp(data=[]))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertFalse(store.get_nlf_enabled(token="t", user_id="u1"))

    def test_request_nlf_job_returns_row_from_list(self) -> None:
        client, _ = _fake_client(_Resp(data=[{"id": "job1", "status": "queued"}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            job = store.request_nlf_job(token="t", video_id="v1")
        self.assertEqual(job["id"], "job1")
        client.rpc.assert_called_once_with("request_nlf_job", {"p_video_id": "v1"})

    def test_request_nlf_job_returns_row_from_dict(self) -> None:
        client, _ = _fake_client(_Resp(data={"id": "job1", "status": "queued"}))
        with mock.patch.object(store, "_user_client", return_value=client):
            job = store.request_nlf_job(token="t", video_id="v1")
        self.assertEqual(job["id"], "job1")

    def test_request_nlf_job_empty_list_is_empty_dict(self) -> None:
        client, _ = _fake_client(_Resp(data=[]))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertEqual(store.request_nlf_job(token="t", video_id="v1"), {})

    def test_request_nlf_job_propagates_rpc_error(self) -> None:
        client = mock.Mock()
        client.rpc.return_value.execute.side_effect = _ApiError("42501")
        with mock.patch.object(store, "_user_client", return_value=client):
            with self.assertRaises(_ApiError) as ctx:
                store.request_nlf_job(token="t", video_id="v1")
        self.assertEqual(ctx.exception.code, "42501")

    def test_get_nlf_job_found(self) -> None:
        client, _ = _fake_client(_Resp(data=[{"id": "job1", "video_id": "v1"}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            job = store.get_nlf_job(token="t", user_id="u1", video_id="v1")
        self.assertEqual(job["id"], "job1")
        client.table.assert_called_with("nlf_jobs")

    def test_get_nlf_job_not_found(self) -> None:
        client, _ = _fake_client(_Resp(data=[]))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertIsNone(store.get_nlf_job(token="t", user_id="u1", video_id="v1"))

    def test_nlf_worker_last_seen_present(self) -> None:
        client, _ = _fake_client(_Resp(data="2026-09-28T12:00:00+00:00"))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertEqual(
                store.nlf_worker_last_seen(token="t"), "2026-09-28T12:00:00+00:00"
            )
        client.rpc.assert_called_once_with("nlf_worker_last_seen")

    def test_nlf_worker_last_seen_absent(self) -> None:
        client, _ = _fake_client(_Resp(data=None))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertIsNone(store.nlf_worker_last_seen(token="t"))

    def test_set_nlf_access_grant(self) -> None:
        client, query = _fake_client(_Resp(data=[{"user_id": "u2"}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            store.set_nlf_access(token="t", user_id="u2", enabled=True)
        self.assertEqual(query.upserted, {"user_id": "u2", "role": "nlf_user"})
        self.assertFalse(query.deleted)
        client.table.assert_called_with("user_roles")

    def test_set_nlf_access_revoke(self) -> None:
        client, query = _fake_client(_Resp(data=[]))
        with mock.patch.object(store, "_user_client", return_value=client):
            store.set_nlf_access(token="t", user_id="u2", enabled=False)
        self.assertTrue(query.deleted)
        self.assertIsNone(query.upserted)


# --------------------------------------------------------------------- routers.nlf: /nlf/status


class NlfStatusRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)

    def test_status_enabled(self) -> None:
        with mock.patch.object(store, "get_nlf_enabled", return_value=True) as ge:
            resp = self.client.get("/api/nlf/status")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"enabled": True})
        ge.assert_called_once_with(token="tok", user_id="u1")

    def test_status_disabled(self) -> None:
        with mock.patch.object(store, "get_nlf_enabled", return_value=False):
            resp = self.client.get("/api/nlf/status")
        self.assertEqual(resp.json(), {"enabled": False})

    def test_status_requires_auth(self) -> None:
        app.dependency_overrides.clear()
        resp = self.client.get("/api/nlf/status")
        self.assertEqual(resp.status_code, 401)


# --------------------------------------------------- routers.nlf: /analyses/{id}/nlf (POST/GET)


class NlfAnalysisRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)

    @staticmethod
    def _analysis_row(video_id: str = "vid1", user_id: str = "u1") -> dict:
        return {"id": _AID, "user_id": user_id, "video_id": video_id, "result": {}}

    # -- POST ---------------------------------------------------------------------------------

    def test_post_happy_path_202(self) -> None:
        job = {
            "status": "queued",
            "error": None,
            "created_at": "c",
            "updated_at": "u",
            "meta": None,
        }
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ) as ga, mock.patch.object(
            store, "request_nlf_job", return_value=job
        ) as rj, mock.patch.object(
            store, "nlf_worker_last_seen", return_value=None
        ):
            resp = self.client.post(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 202)
        body = resp.json()
        self.assertEqual(body["status"], "queued")
        self.assertEqual(body["key_frames"], [])
        self.assertFalse(body["worker_online"])
        self.assertEqual(body["expires_in"], storage.DEFAULT_URL_TTL)
        ga.assert_called_once_with(token="tok", analysis_id=_AID)
        rj.assert_called_once_with(token="tok", video_id="vid1")

    def test_post_forbidden_when_rpc_raises_42501(self) -> None:
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "request_nlf_job", side_effect=_ApiError("42501")):
            resp = self.client.post(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 403)

    def test_post_missing_analysis_is_404(self) -> None:
        with mock.patch.object(store, "get_analysis", return_value=None), mock.patch.object(
            store, "request_nlf_job"
        ) as rj:
            resp = self.client.post(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 404)
        rj.assert_not_called()

    def test_post_missing_video_is_404_via_p0002(self) -> None:
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "request_nlf_job", side_effect=_ApiError("P0002")):
            resp = self.client.post(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 404)

    def test_post_someone_elses_analysis_is_404(self) -> None:
        # RLS can hand back a row that exists but isn't the caller's (e.g. a linked clinician
        # reading a patient's analysis, per routers/checkins.py's header note) -- the explicit
        # user_id check must still 404 it, and never reach the RPC with someone else's video.
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row(user_id="u2")
        ), mock.patch.object(store, "request_nlf_job") as rj:
            resp = self.client.post(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 404)
        rj.assert_not_called()

    def test_post_bad_uuid_is_404(self) -> None:
        with mock.patch.object(store, "get_analysis") as ga:
            resp = self.client.post("/api/analyses/not-a-uuid/nlf")
        self.assertEqual(resp.status_code, 404)
        ga.assert_not_called()

    def test_post_requires_auth(self) -> None:
        app.dependency_overrides.clear()
        resp = self.client.post(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 401)

    def test_post_unrecognised_rpc_error_code_propagates(self) -> None:
        # Only 42501/P0002 are translated; anything else (a genuine bug, a transport error) is
        # left to propagate (FastAPI's own 500 handler) -- swallowing it silently would hide a
        # real failure behind a wrong 4xx. Called directly (bypassing TestClient, which re-raises
        # rather than converting to a response) to assert on the propagated exception itself.
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "request_nlf_job", side_effect=_ApiError("XX000")):
            with self.assertRaises(_ApiError):
                nlf_router.request_nlf(_AID, user=CurrentUser(id="u1", token="tok"))

    def test_post_storage_error_is_503(self) -> None:
        job = {
            "status": "done",
            "error": None,
            "created_at": "c",
            "updated_at": "u",
            "meta": {
                "angles": 24,
                "tile_px": 384,
                "key_frames": [
                    {
                        "t_s": 1.0,
                        "kind": "fault_peak",
                        "fault_ids": [],
                        "rep_index": None,
                        "strip_key": "uploads/u1/vid1/nlf.v1.a0.webp",
                    }
                ],
            },
        }

        class _Failing:
            def presigned_url(self, key, *, expires_in=storage.DEFAULT_URL_TTL):
                raise storage.StorageError("down")

        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "request_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value=None
        ), mock.patch.object(
            store, "get_storage_key", return_value="uploads/u1/vid1"
        ), mock.patch.object(storage, "get_object_store", return_value=_Failing()):
            resp = self.client.post(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 503)

    # -- GET ----------------------------------------------------------------------------------

    def test_get_no_job_is_404(self) -> None:
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=None):
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 404)

    def test_get_missing_analysis_is_404(self) -> None:
        with mock.patch.object(store, "get_analysis", return_value=None), mock.patch.object(
            store, "get_nlf_job"
        ) as gj:
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 404)
        gj.assert_not_called()

    def test_get_someone_elses_analysis_is_404(self) -> None:
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row(user_id="u2")
        ), mock.patch.object(store, "get_nlf_job") as gj:
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 404)
        gj.assert_not_called()

    def test_get_bad_uuid_is_404(self) -> None:
        with mock.patch.object(store, "get_analysis") as ga:
            resp = self.client.get("/api/analyses/not-a-uuid/nlf")
        self.assertEqual(resp.status_code, 404)
        ga.assert_not_called()

    def test_get_queued_has_empty_key_frames(self) -> None:
        job = {
            "status": "queued",
            "error": None,
            "created_at": "c",
            "updated_at": "u",
            "meta": None,
        }
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value=None
        ):
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "queued")
        self.assertEqual(body["key_frames"], [])
        self.assertIsNone(body["angles"])
        self.assertIsNone(body["tile_px"])

    def test_get_done_returns_presigned_key_frames(self) -> None:
        meta = {
            "schema_version": 1,
            "angles": 24,
            "tile_px": 384,
            "prefix": "uploads/u1/vid1/nlf.v1.a24",
            "key_frames": [
                {
                    "t_s": 1.5,
                    "kind": "rep_bottom",
                    "fault_ids": ["knee_valgus"],
                    "rep_index": 0,
                    "strip_key": "uploads/u1/vid1/nlf.v1.a0.webp",
                }
            ],
        }
        job = {"status": "done", "error": None, "created_at": "c", "updated_at": "u", "meta": meta}

        class _FakeObjStore:
            def presigned_url(self, key, *, expires_in=storage.DEFAULT_URL_TTL):
                return f"https://signed.example/{key}"

        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value=None
        ), mock.patch.object(
            store, "get_storage_key", return_value="uploads/u1/vid1"
        ), mock.patch.object(
            storage, "get_object_store", return_value=_FakeObjStore()
        ):
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["angles"], 24)
        self.assertEqual(body["tile_px"], 384)
        self.assertEqual(len(body["key_frames"]), 1)
        kf = body["key_frames"][0]
        self.assertEqual(
            kf["strip_url"], "https://signed.example/uploads/u1/vid1/nlf.v1.a0.webp"
        )
        self.assertEqual(kf["fault_ids"], ["knee_valgus"])
        self.assertEqual(kf["rep_index"], 0)
        self.assertEqual(kf["kind"], "rep_bottom")
        self.assertEqual(body["expires_in"], storage.DEFAULT_URL_TTL)
        # No pose3d_url is ever returned -- only the pre-rendered turntable strips.
        self.assertNotIn("pose3d_url", body)

    def test_get_worker_online_true_when_recent(self) -> None:
        job = {
            "status": "queued",
            "error": None,
            "created_at": "c",
            "updated_at": "u",
            "meta": None,
        }
        recent = datetime.now(timezone.utc).isoformat()
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value=recent
        ):
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertTrue(resp.json()["worker_online"])

    def test_get_worker_online_false_when_stale(self) -> None:
        job = {
            "status": "queued",
            "error": None,
            "created_at": "c",
            "updated_at": "u",
            "meta": None,
        }
        stale = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value=stale
        ):
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertFalse(resp.json()["worker_online"])

    def test_get_worker_online_false_when_unparsable(self) -> None:
        job = {
            "status": "queued",
            "error": None,
            "created_at": "c",
            "updated_at": "u",
            "meta": None,
        }
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value="not-a-timestamp"
        ):
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertFalse(resp.json()["worker_online"])

    def test_get_worker_online_true_for_naive_recent_timestamp(self) -> None:
        # A timezone-naive ISO string (no offset) is treated as UTC, not rejected.
        job = {
            "status": "queued",
            "error": None,
            "created_at": "c",
            "updated_at": "u",
            "meta": None,
        }
        naive_recent = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value=naive_recent
        ):
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertTrue(resp.json()["worker_online"])

    def test_get_worker_online_false_when_null(self) -> None:
        job = {
            "status": "queued",
            "error": None,
            "created_at": "c",
            "updated_at": "u",
            "meta": None,
        }
        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value=None
        ):
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertFalse(resp.json()["worker_online"])

    def test_get_storage_error_is_503(self) -> None:
        meta = {
            "angles": 24,
            "tile_px": 384,
            "key_frames": [
                {
                    "t_s": 1.0,
                    "kind": "fault_peak",
                    "fault_ids": [],
                    "rep_index": None,
                    "strip_key": "uploads/u1/vid1/nlf.v1.a0.webp",
                }
            ],
        }
        job = {"status": "done", "error": None, "created_at": "c", "updated_at": "u", "meta": meta}

        class _Failing:
            def presigned_url(self, key, *, expires_in=storage.DEFAULT_URL_TTL):
                raise storage.StorageError("down")

        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value=None
        ), mock.patch.object(
            store, "get_storage_key", return_value="uploads/u1/vid1"
        ), mock.patch.object(storage, "get_object_store", return_value=_Failing()):
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 503)

    def test_get_done_drops_key_frame_outside_the_video_prefix(self) -> None:
        # The worker writes meta.key_frames itself; a strip_key outside THIS video's own storage
        # prefix must never be presigned and handed back -- that would turn the endpoint into an
        # oracle for unrelated storage. The in-prefix sibling is still returned.
        meta = {
            "angles": 24,
            "tile_px": 384,
            "key_frames": [
                {
                    "t_s": 1.0,
                    "kind": "fault_peak",
                    "fault_ids": [],
                    "rep_index": None,
                    "strip_key": "uploads/OTHER-USER/other-video/nlf.v1.a0.webp",
                },
                {
                    "t_s": 2.0,
                    "kind": "rep_bottom",
                    "fault_ids": [],
                    "rep_index": 1,
                    "strip_key": "uploads/u1/vid1/nlf.v1.a1.webp",
                },
            ],
        }
        job = {"status": "done", "error": None, "created_at": "c", "updated_at": "u", "meta": meta}

        class _FakeObjStore:
            def presigned_url(self, key, *, expires_in=storage.DEFAULT_URL_TTL):
                return f"https://signed.example/{key}"

        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value=None
        ), mock.patch.object(
            store, "get_storage_key", return_value="uploads/u1/vid1"
        ), mock.patch.object(
            storage, "get_object_store", return_value=_FakeObjStore()
        ):
            with self.assertLogs(nlf_router.logger, level="WARNING") as logs:
                resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["key_frames"]), 1)
        self.assertEqual(
            body["key_frames"][0]["strip_url"],
            "https://signed.example/uploads/u1/vid1/nlf.v1.a1.webp",
        )
        # The offending strip_key itself is never logged, only the expected prefix.
        self.assertTrue(any("uploads/u1/vid1/nlf.v1." in line for line in logs.output))
        self.assertFalse(any("OTHER-USER" in line for line in logs.output))

    def test_get_done_returns_no_key_frames_when_video_storage_key_missing(self) -> None:
        # If the video's own storage_key can't be resolved, there is nothing safe to check a
        # strip_key against, so every key frame is dropped rather than trusting any of them.
        meta = {
            "angles": 24,
            "tile_px": 384,
            "key_frames": [
                {
                    "t_s": 1.0,
                    "kind": "fault_peak",
                    "fault_ids": [],
                    "rep_index": None,
                    "strip_key": "uploads/u1/vid1/nlf.v1.a0.webp",
                }
            ],
        }
        job = {"status": "done", "error": None, "created_at": "c", "updated_at": "u", "meta": meta}

        with mock.patch.object(
            store, "get_analysis", return_value=self._analysis_row()
        ), mock.patch.object(store, "get_nlf_job", return_value=job), mock.patch.object(
            store, "nlf_worker_last_seen", return_value=None
        ), mock.patch.object(store, "get_storage_key", return_value=None):
            resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["key_frames"], [])

    def test_get_requires_auth(self) -> None:
        app.dependency_overrides.clear()
        resp = self.client.get(f"/api/analyses/{_AID}/nlf")
        self.assertEqual(resp.status_code, 401)


# --------------------------------------------------------- routers.admin: enable_nlf + has_nlf


class NlfAdminRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)

    def test_enable_nlf_grants(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), mock.patch.object(
            store, "set_nlf_access"
        ) as sna:
            resp = self.client.put("/api/admin/users/u2/role", json={"enable_nlf": True})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})
        sna.assert_called_once_with(token="tok", user_id="u2", enabled=True)

    def test_enable_nlf_revokes(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), mock.patch.object(
            store, "set_nlf_access"
        ) as sna:
            resp = self.client.put("/api/admin/users/u2/role", json={"enable_nlf": False})
        self.assertEqual(resp.status_code, 200)
        sna.assert_called_once_with(token="tok", user_id="u2", enabled=False)

    def test_enable_nlf_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False), mock.patch.object(
            store, "set_nlf_access"
        ) as sna:
            resp = self.client.put("/api/admin/users/u2/role", json={"enable_nlf": True})
        self.assertEqual(resp.status_code, 403)
        sna.assert_not_called()

    def test_empty_body_is_still_rejected(self) -> None:
        # F3 (unchanged): an empty body with no role field at all is a client bug, not a no-op.
        with mock.patch.object(store, "is_admin", return_value=True):
            resp = self.client.put("/api/admin/users/u2/role", json={})
        self.assertEqual(resp.status_code, 422)

    def test_has_nlf_flows_through_user_list(self) -> None:
        rows = [{"id": "u2", "email": "a@x.com", "has_nlf": True}]
        with mock.patch.object(store, "is_admin", return_value=True), mock.patch.object(
            store, "admin_list_users", return_value=rows
        ):
            resp = self.client.get("/api/admin/users")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["users"][0]["has_nlf"])


if __name__ == "__main__":
    unittest.main()
