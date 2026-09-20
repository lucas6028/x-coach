"""POST /api/jobs/daily -- the nightly care-loop job (WP5).

A GitHub Actions cron (``.github/workflows/daily-jobs.yml``) POSTs here at 20:00 Asia/Taipei
with no user JWT, so it authenticates via a shared ``X-Job-Token`` header (``settings.job_token``,
compared with ``hmac.compare_digest``) instead of ``get_current_user``. Two things happen:
patients on a therapist-assigned plan who have not checked in today get a LINE reminder, and
each clinician with linked patients gets a one-line summary push.

Like ``services/line_bot``'s webhook, this endpoint touches data with the service_role client
ONLY through narrow SECURITY DEFINER RPCs -- ``claim_job_run``, ``daily_patient_reminders``,
``daily_clinician_summaries`` -- and NEVER through ``.table(...)``. That is the smallest possible
widening of the "backend never touches data with service_role" posture; see the migration
``db/migrations/20260927000000_daily_jobs.sql`` for exactly what each RPC returns and why. The
service-role client is built by ``services/line_bot._service_client`` -- reused rather than
duplicated, same as every other service_role caller in this backend.

Idempotency is per Asia/Taipei calendar date, enforced by ``claim_job_run`` (a unique-key insert
on ``(job, run_date)``): a cron retry or a manual ``workflow_dispatch`` on the same day is a
no-op unless ``force=1`` is passed, which skips the claim entirely.
"""

from __future__ import annotations

import hmac
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query

from backend.app.services import line_bot
from backend.app.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Display is fixed to Taipei regardless of the server's local timezone (mirrors line_bot's
# _DISPLAY_TZ): the job's "today" is always Asia/Taipei, matching the cron's 20:00 Taipei firing.
_TAIPEI = timezone(timedelta(hours=8))

_JOB_NAME = "daily"

_REMINDER_TEXT = "今天記得完成治療師安排的「{plan_name}」，做完後在 App 回報一下狀況喔。"
_SUMMARY_TEXT = (
    "今日個案回報：{checked_in_today}/{patients} 位已回報；"
    "未處理紅旗 {open_flags} 則；{inactive_7d} 位超過 7 天未回報。"
)
# Keeps a long plan name from blowing out the fixed reminder template.
_PLAN_NAME_MAX = 40


def _utc_now() -> datetime:
    """The current instant, UTC. A seam so tests can pin the clock without monkeypatching
    ``datetime`` itself."""
    return datetime.now(timezone.utc)


def _taipei_date_and_since(now: datetime) -> tuple[date, datetime]:
    """The Asia/Taipei calendar date for ``now``, and the start of that day (00:00 UTC+8).

    ``now`` must be UTC-aware. The Taipei date is computed by shifting the UTC instant forward
    8 hours and taking its date part (no ``zoneinfo``/``tzdata`` dependency, same trick as
    ``line_bot._DISPLAY_TZ`` for the same Windows-CI reason).
    """
    taipei_date = (now + timedelta(hours=8)).date()
    since = datetime(taipei_date.year, taipei_date.month, taipei_date.day, tzinfo=_TAIPEI)
    return taipei_date, since


def _check_job_token(x_job_token: str | None) -> None:
    """503 when the job isn't configured; 401 for a missing/wrong/empty token."""
    settings = get_settings()
    if not getattr(settings, "jobs_configured", False):
        raise HTTPException(status_code=503, detail="Daily job is not configured.")
    # ``not x_job_token`` rejects both a missing header and an empty one, before it ever reaches
    # compare_digest -- an empty token must never be treated as "no token supplied, compare
    # anyway" (compare_digest("", "") is trivially True, and job_token is non-empty here anyway
    # per jobs_configured, but the empty-header case is guarded explicitly regardless).
    if not x_job_token or not hmac.compare_digest(x_job_token, settings.job_token):
        raise HTTPException(status_code=401, detail="Invalid job token.")


def _rpc_rows(client: Any, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Call a table-returning RPC and return its rows, or ``[]`` for an unexpected shape."""
    response = client.rpc(name, params).execute()
    data = getattr(response, "data", None)
    return data if isinstance(data, list) else []


@router.post("/daily")
def run_daily_job(
    force: int = Query(0, ge=0, le=1),
    x_job_token: str | None = Header(default=None, alias="X-Job-Token"),
) -> dict[str, Any]:
    _check_job_token(x_job_token)

    now = _utc_now()
    taipei_date, since = _taipei_date_and_since(now)
    date_str = taipei_date.isoformat()

    client = line_bot._service_client()

    # Gather FIRST, claim second. `claim_job_run` burns the day permanently, so claiming before the
    # reads means one transient Supabase failure costs that day's reminders entirely: the 502 tells
    # the workflow to retry, and the retry gets `already_ran`. Reading first narrows the
    # unrecoverable window to the pushes themselves, which is as far as an at-most-once design can
    # go without per-message state. The claim is still the one atomic gate, so two concurrent runs
    # can both read but only one sends.
    try:
        reminder_rows = _rpc_rows(
            client, "daily_patient_reminders", {"p_since": since.isoformat()}
        )
        summary_rows = _rpc_rows(
            client,
            "daily_clinician_summaries",
            {"p_since": since.isoformat(), "p_now": now.isoformat()},
        )

        if not force:
            claim_response = client.rpc(
                "claim_job_run", {"p_job": _JOB_NAME, "p_run_date": date_str}
            ).execute()
            if getattr(claim_response, "data", None) is not True:
                return {"ran": False, "reason": "already_ran", "date": date_str}
    except Exception as exc:  # noqa: BLE001 — a scheduler call must degrade to a clear 502, never a 500.
        logger.exception("daily job: RPC call failed")
        raise HTTPException(status_code=502, detail="Daily job failed.") from exc

    reminders_sent = 0
    for row in reminder_rows:
        plan_name = str(row.get("plan_name") or "")[:_PLAN_NAME_MAX]
        text = _REMINDER_TEXT.format(plan_name=plan_name)
        if line_bot.push(row.get("line_user_id"), text):
            reminders_sent += 1

    summaries_sent = 0
    for row in summary_rows:
        text = _SUMMARY_TEXT.format(
            checked_in_today=row.get("checked_in_today"),
            patients=row.get("patients"),
            open_flags=row.get("open_flags"),
            inactive_7d=row.get("inactive_7d"),
        )
        if line_bot.push(row.get("line_user_id"), text):
            summaries_sent += 1

    return {
        "ran": True,
        "date": date_str,
        "reminders": len(reminder_rows),
        "reminders_sent": reminders_sent,
        "summaries": len(summary_rows),
        "summaries_sent": summaries_sent,
    }
