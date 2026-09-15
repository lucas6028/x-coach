"""Unit tests for POST /api/jobs/daily -- the nightly care-loop job (WP5).

The service-role client (services/line_bot._service_client) and the push helper
(services/line_bot.push) are both patched, so no real network call to Supabase or LINE is ever
made. The fake client's ``.table`` raises, proving the router never falls back to it.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest import mock

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routers import jobs as jobs_router
from backend.app.services import line_bot
from backend.app.settings import Settings

_TOKEN = "shared-job-token"

_PATIENT_A = "U" + "a" * 32
_PATIENT_B = "U" + "b" * 32
_CLINICIAN_A = "U" + "c" * 32


def _settings(**overrides) -> Settings:
    values = {
        "supabase_url": "https://proj.supabase.co",
        "supabase_anon_key": "anon-key",
        "supabase_service_role_key": "service-key",
        "line_messaging_access_token": "chan-token",
        "job_token": _TOKEN,
    }
    values.update(overrides)
    return Settings(**values)


class _FakeResponse:
    def __init__(self, data) -> None:
        self.data = data


class _FakeRpcBuilder:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def execute(self) -> _FakeResponse:
        return self._response


class _NoTableClient:
    """A fake service-role client: answers .rpc() from a table, raises on .table()."""

    def __init__(self, rpc_data: dict, *, raise_on: set[str] | None = None) -> None:
        self._rpc_data = rpc_data
        self._raise_on = raise_on or set()
        self.calls: list[tuple[str, dict]] = []

    def table(self, *args, **kwargs):  # pragma: no cover - only hit on a regression
        raise AssertionError("routers/jobs.py must never call .table(...)")

    def rpc(self, name: str, params: dict) -> _FakeRpcBuilder:
        self.calls.append((name, params))
        if name in self._raise_on:
            raise RuntimeError(f"RPC {name} failed")
        return _FakeRpcBuilder(_FakeResponse(self._rpc_data.get(name)))


def _default_rpc_data(*, claimed: bool = True, reminders=None, summaries=None) -> dict:
    return {
        "claim_job_run": claimed,
        "daily_patient_reminders": reminders if reminders is not None else [],
        "daily_clinician_summaries": summaries if summaries is not None else [],
    }


class JobsDailyAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_no_token_is_401(self) -> None:
        with mock.patch.object(jobs_router, "get_settings", return_value=_settings()):
            resp = self.client.post("/api/jobs/daily")
        self.assertEqual(resp.status_code, 401)

    def test_wrong_token_is_401(self) -> None:
        with mock.patch.object(jobs_router, "get_settings", return_value=_settings()):
            resp = self.client.post("/api/jobs/daily", headers={"X-Job-Token": "nope"})
        self.assertEqual(resp.status_code, 401)

    def test_empty_token_header_is_401(self) -> None:
        with mock.patch.object(jobs_router, "get_settings", return_value=_settings()):
            resp = self.client.post("/api/jobs/daily", headers={"X-Job-Token": ""})
        self.assertEqual(resp.status_code, 401)

    def test_unconfigured_is_503(self) -> None:
        unconfigured = _settings(job_token="")
        with mock.patch.object(jobs_router, "get_settings", return_value=unconfigured):
            resp = self.client.post("/api/jobs/daily", headers={"X-Job-Token": _TOKEN})
        self.assertEqual(resp.status_code, 503)

    def test_unconfigured_missing_service_role_key_is_503(self) -> None:
        unconfigured = _settings(supabase_service_role_key="")
        with mock.patch.object(jobs_router, "get_settings", return_value=unconfigured):
            resp = self.client.post("/api/jobs/daily", headers={"X-Job-Token": _TOKEN})
        self.assertEqual(resp.status_code, 503)


class JobsDailyRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def _post(self, *, rpc_data, force: int | None = None, pushes=None):
        client = _NoTableClient(rpc_data)
        params = {} if force is None else {"force": force}
        push_mock = mock.MagicMock(side_effect=pushes) if pushes is not None else mock.MagicMock(return_value=True)
        with mock.patch.object(jobs_router, "get_settings", return_value=_settings()), \
             mock.patch.object(line_bot, "_service_client", return_value=client), \
             mock.patch.object(line_bot, "push", push_mock):
            resp = self.client.post(
                "/api/jobs/daily", headers={"X-Job-Token": _TOKEN}, params=params
            )
        return resp, client, push_mock

    def test_already_ran_when_not_claimed(self) -> None:
        resp, client, push_mock = self._post(rpc_data=_default_rpc_data(claimed=False))
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["ran"], False)
        self.assertEqual(body["reason"], "already_ran")
        self.assertIn("date", body)
        push_mock.assert_not_called()
        # Only the claim RPC ran -- no reminders/summaries fetched once already claimed today.
        self.assertEqual([c[0] for c in client.calls], ["claim_job_run"])

    def test_force_skips_the_claim(self) -> None:
        resp, client, push_mock = self._post(
            rpc_data=_default_rpc_data(claimed=False), force=1
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ran"])
        self.assertNotIn("claim_job_run", [c[0] for c in client.calls])

    def test_no_rows_means_no_pushes(self) -> None:
        resp, client, push_mock = self._post(rpc_data=_default_rpc_data())
        body = resp.json()
        self.assertEqual(body["ran"], True)
        self.assertEqual(body["reminders"], 0)
        self.assertEqual(body["reminders_sent"], 0)
        self.assertEqual(body["summaries"], 0)
        self.assertEqual(body["summaries_sent"], 0)
        push_mock.assert_not_called()

    def test_one_push_per_reminder_row_with_exact_text(self) -> None:
        reminders = [
            {
                "patient_id": "p1",
                "line_user_id": _PATIENT_A,
                "plan_id": "plan1",
                "plan_name": "膝關節術後六週菜單",
            },
            {
                "patient_id": "p2",
                "line_user_id": _PATIENT_B,
                "plan_id": "plan2",
                # 41 'x' chars -- must be truncated to 40 in the pushed text.
                "plan_name": "x" * 41,
            },
        ]
        resp, client, push_mock = self._post(
            rpc_data=_default_rpc_data(reminders=reminders)
        )
        body = resp.json()
        self.assertEqual(body["reminders"], 2)
        self.assertEqual(body["reminders_sent"], 2)
        self.assertEqual(push_mock.call_count, 2 + 0)  # 2 reminders, 0 summaries
        calls = {c.args[0]: c.args[1] for c in push_mock.call_args_list}
        self.assertEqual(
            calls[_PATIENT_A],
            "今天記得完成治療師安排的「膝關節術後六週菜單」，做完後在 App 回報一下狀況喔。",
        )
        self.assertEqual(
            calls[_PATIENT_B],
            "今天記得完成治療師安排的「" + "x" * 40 + "」，做完後在 App 回報一下狀況喔。",
        )

    def test_one_push_per_summary_row_with_exact_text(self) -> None:
        summaries = [
            {
                "clinician_id": "c1",
                "line_user_id": _CLINICIAN_A,
                "patients": 5,
                "checked_in_today": 3,
                "open_flags": 2,
                "inactive_7d": 1,
            }
        ]
        resp, client, push_mock = self._post(
            rpc_data=_default_rpc_data(summaries=summaries)
        )
        body = resp.json()
        self.assertEqual(body["summaries"], 1)
        self.assertEqual(body["summaries_sent"], 1)
        push_mock.assert_called_once_with(
            _CLINICIAN_A, "今日個案回報：3/5 位已回報；未處理紅旗 2 則；1 位超過 7 天未回報。"
        )

    def test_counts_reflect_failed_pushes(self) -> None:
        reminders = [
            {"line_user_id": _PATIENT_A, "plan_name": "A"},
            {"line_user_id": _PATIENT_B, "plan_name": "B"},
        ]
        resp, client, push_mock = self._post(
            rpc_data=_default_rpc_data(reminders=reminders),
            pushes=[True, False],
        )
        body = resp.json()
        self.assertEqual(body["reminders"], 2)
        self.assertEqual(body["reminders_sent"], 1)

    def test_rpc_error_is_502(self) -> None:
        client = _NoTableClient(_default_rpc_data(), raise_on={"daily_patient_reminders"})
        with mock.patch.object(jobs_router, "get_settings", return_value=_settings()), \
             mock.patch.object(line_bot, "_service_client", return_value=client), \
             mock.patch.object(line_bot, "push", return_value=True):
            resp = self.client.post("/api/jobs/daily", headers={"X-Job-Token": _TOKEN})
        self.assertEqual(resp.status_code, 502)

    def test_claim_rpc_error_is_502(self) -> None:
        client = _NoTableClient(_default_rpc_data(), raise_on={"claim_job_run"})
        with mock.patch.object(jobs_router, "get_settings", return_value=_settings()), \
             mock.patch.object(line_bot, "_service_client", return_value=client), \
             mock.patch.object(line_bot, "push", return_value=True):
            resp = self.client.post("/api/jobs/daily", headers={"X-Job-Token": _TOKEN})
        self.assertEqual(resp.status_code, 502)

    def test_never_calls_table(self) -> None:
        # Regression guard for the "service_role only through narrow RPCs" posture: if the
        # router ever reaches for .table(...), the fake raises and the request 500s/502s instead
        # of silently reading raw rows.
        client = _NoTableClient(_default_rpc_data())
        with mock.patch.object(jobs_router, "get_settings", return_value=_settings()), \
             mock.patch.object(line_bot, "_service_client", return_value=client), \
             mock.patch.object(line_bot, "push", return_value=True):
            self.client.post("/api/jobs/daily", headers={"X-Job-Token": _TOKEN})
        for name, _params in client.calls:
            self.assertIn(
                name,
                {"claim_job_run", "daily_patient_reminders", "daily_clinician_summaries"},
            )


class JobsDailyTaipeiBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def _run_at(self, iso_utc: str):
        fixed_now = datetime.fromisoformat(iso_utc).replace(tzinfo=timezone.utc)
        client = _NoTableClient(_default_rpc_data())
        with mock.patch.object(jobs_router, "get_settings", return_value=_settings()), \
             mock.patch.object(line_bot, "_service_client", return_value=client), \
             mock.patch.object(line_bot, "push", return_value=True), \
             mock.patch.object(jobs_router, "_utc_now", return_value=fixed_now):
            resp = self.client.post("/api/jobs/daily", headers={"X-Job-Token": _TOKEN})
        return resp, client

    def test_just_before_taipei_midnight(self) -> None:
        resp, client = self._run_at("2026-09-16T15:59:59")
        self.assertEqual(resp.json()["date"], "2026-09-16")
        params_by_name = dict(client.calls)
        self.assertEqual(
            params_by_name["daily_patient_reminders"]["p_since"],
            "2026-09-16T00:00:00+08:00",
        )
        # 2026-09-16T00:00:00+08:00 == 2026-09-15T16:00:00Z
        since = datetime.fromisoformat(params_by_name["daily_patient_reminders"]["p_since"])
        self.assertEqual(since.astimezone(timezone.utc).isoformat(), "2026-09-15T16:00:00+00:00")

    def test_at_taipei_midnight(self) -> None:
        resp, client = self._run_at("2026-09-16T16:00:00")
        self.assertEqual(resp.json()["date"], "2026-09-17")


if __name__ == "__main__":
    unittest.main()
