import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowClockwise, Cube } from "@phosphor-icons/react";
import { api, type Analysis, type NlfJob } from "../../api";
import { useAuth } from "../../lib/auth";
import { useI18n } from "../../lib/i18n";
import StudioCard from "../studio/StudioCard";
import TurntableViewer from "./TurntableViewer";

// How often to re-poll a job that is actively moving toward "done" (worker crunching, or queued
// with a live worker that will pick it up).
const POLL_MS = 15_000;
// After a FAILED poll, back off to a minute rather than hammering every 15s — the job is still
// pollable (worker/network hiccups are usually transient), just not worth retrying as eagerly.
const POLL_BACKOFF_MS = 60_000;
// At most this many image-error re-fetches per mount: a strip URL can expire mid-session, and one
// re-fetch gets a fresh presigned URL, but a strip that keeps failing is a real problem, not an
// expiry — retrying forever would spam the backend.
const MAX_IMAGE_ERROR_REFETCHES = 2;

function isPollable(job: NlfJob): boolean {
  return job.status === "claimed" || (job.status === "queued" && job.worker_online);
}

// Whether a visibilitychange -> visible transition should re-fetch: only while there is
// something still moving (no job yet, or queued/claimed) — a done/failed job cannot change
// underneath the worker, so there is nothing to catch up on.
function isNonTerminal(job: NlfJob | null | undefined): boolean {
  return job === undefined || job === null || job.status === "queued" || job.status === "claimed";
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

interface Props {
  analysis: Analysis;
  /** Seeks the page's own <video> element. Omitted where the caller has no seek handle in scope. */
  onSeek?: (t_s: number) => void;
}

/**
 * The "3D view" card: request a turntable render of this analysis, watch it move through the home
 * worker's queue, and view the result once done. Gated on `canUse3d` (an admin-enabled, per-user
 * flag — see lib/auth.tsx) AND the analysis actually having a persisted `analysis_id` (an
 * anonymous upload or a failed-persist upload has neither a server-side job to attach to nor
 * anywhere for the worker to write results back to).
 *
 * POLLING DESIGN: rescheduling is driven from each fetch's own completion (success OR failure),
 * never from a `useEffect` keyed on the `job` state object. Keying on `job` identity is a trap: a
 * poll that resolves to a value React considers unchanged (e.g. an identical object, or simply a
 * setState the runtime coalesces) never re-runs a `job`-dependent effect, silently killing the
 * poll loop after one such tick. Here, `fetchJob` schedules its OWN next call directly out of its
 * `.then`/`.catch`, so a poll is always followed by another one while conditions hold — a failed
 * poll included, just at a slower `POLL_BACKOFF_MS` cadence instead of `POLL_MS`.
 */
export default function Nlf3dPanel({ analysis, onSeek }: Props) {
  const { canUse3d } = useAuth();
  const { t } = useI18n();
  const id = analysis.analysis_id ?? null;
  const active = canUse3d && !!id;

  // `undefined` = not yet fetched (first load spinner); `null` = fetched, no job requested yet.
  const [job, setJob] = useState<NlfJob | null | undefined>(undefined);
  const [apiError, setApiError] = useState<string | null>(null);
  const [visible, setVisible] = useState(
    () => typeof document === "undefined" || document.visibilityState === "visible"
  );
  const [requesting, setRequesting] = useState(false);
  const imageErrorCountRef = useRef(0);

  // Latest-value refs for state read from inside async callbacks (fetch completions, the
  // visibilitychange listener, the poll timer) — those callbacks are scheduled ahead of time and
  // must see what's true when they actually RUN, not what was true when they were created. Synced
  // every render, which is safe: nothing here is used to compute this render's own output.
  const idRef = useRef(id);
  idRef.current = id;
  const activeRef = useRef(active);
  activeRef.current = active;
  const visibleRef = useRef(visible);
  visibleRef.current = visible;
  const jobRef = useRef(job);
  jobRef.current = job;

  // Monotonic guard: only the newest in-flight fetch is allowed to land, same pattern as
  // lib/auth.tsx's probeIdRef — needed because fetchJob is called from several independent
  // triggers (mount, the poll, visibilitychange, a manual Refresh, an image-error retry).
  const fetchSeqRef = useRef(0);
  const mountedRef = useRef(true);

  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const clearPollTimer = useCallback(() => {
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  // A stable handle on the latest `fetchJob` (defined below), so the two scheduling helpers can
  // call it from inside a `setTimeout` without depending on `fetchJob` itself — that would be a
  // circular dependency (fetchJob schedules via these; these call fetchJob).
  const fetchJobRef = useRef<() => void>(() => {});

  const schedulePoll = useCallback(
    (freshJob: NlfJob | null) => {
      clearPollTimer();
      if (activeRef.current && visibleRef.current && freshJob && isPollable(freshJob)) {
        pollTimerRef.current = setTimeout(() => fetchJobRef.current(), POLL_MS);
      }
    },
    [clearPollTimer]
  );

  // After a FAILED fetch there is no fresh job to check — fall back to whatever the panel already
  // held before this attempt, since that is still the best guess at whether polling should
  // continue (a transient error doesn't retroactively make a claimed/queued+online job terminal).
  const scheduleBackoff = useCallback(() => {
    clearPollTimer();
    const prevJob = jobRef.current;
    if (activeRef.current && visibleRef.current && prevJob && isPollable(prevJob)) {
      pollTimerRef.current = setTimeout(() => fetchJobRef.current(), POLL_BACKOFF_MS);
    }
  }, [clearPollTimer]);

  const fetchJob = useCallback(() => {
    const currentId = idRef.current;
    if (!currentId) return;
    clearPollTimer(); // this fetch supersedes any pending scheduled one.
    const seq = ++fetchSeqRef.current;
    api
      .getNlf(currentId)
      .then((j) => {
        if (!mountedRef.current || seq !== fetchSeqRef.current) return;
        setJob(j);
        setApiError(null);
        schedulePoll(j);
      })
      .catch((e) => {
        if (!mountedRef.current || seq !== fetchSeqRef.current) return;
        setApiError(errorMessage(e));
        scheduleBackoff();
      });
  }, [clearPollTimer, schedulePoll, scheduleBackoff]);

  useEffect(() => {
    fetchJobRef.current = fetchJob;
  }, [fetchJob]);

  // Initial fetch (and re-fetch if the analysis itself changes under the same mounted panel), or
  // a clean stop if the panel becomes inactive (e.g. the flag or the analysis id disappears).
  useEffect(() => {
    if (!active) {
      clearPollTimer();
      return;
    }
    setJob(undefined);
    setApiError(null);
    imageErrorCountRef.current = 0;
    fetchJob();
  }, [active, id, fetchJob, clearPollTimer]);

  // Mount/unmount: `mountedRef` must go back to `true` on every mount, not just start there —
  // React 18 StrictMode (dev only; see main.tsx) mounts, cleans up, and remounts every component
  // once to shake out non-idempotent effects. Without resetting it here, that first synthetic
  // unmount would leave `mountedRef.current` false forever, and every `if (!mountedRef.current)
  // return;` guard above would then silently no-op on the REAL mount too — the panel would stay
  // on "Loading…" forever. Clearing the poll timer on cleanup still stands: a real unmount (or
  // StrictMode's rehearsal one) must not leave a timer running against a torn-down component.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      clearPollTimer();
    };
  }, [clearPollTimer]);

  // Track visibility. Going hidden stops polling immediately (clearing any pending timer, rather
  // than letting it fire once more into a hidden tab). Coming back visible re-fetches once, but
  // ONLY while there is something that could plausibly have changed — a done/failed job is final,
  // so re-fetching one just to get the same answer back would be pointless network chatter.
  useEffect(() => {
    if (!active) return;
    const onVisibilityChange = () => {
      const nowVisible = document.visibilityState === "visible";
      setVisible(nowVisible);
      if (!nowVisible) {
        clearPollTimer();
        return;
      }
      if (isNonTerminal(jobRef.current)) fetchJob();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [active, fetchJob, clearPollTimer]);

  const handleImageError = useCallback(() => {
    if (imageErrorCountRef.current >= MAX_IMAGE_ERROR_REFETCHES) return;
    imageErrorCountRef.current += 1;
    fetchJob();
  }, [fetchJob]);

  const requestJob = useCallback(() => {
    if (!idRef.current || requesting) return;
    setRequesting(true);
    setApiError(null);
    // Invalidate any in-flight fetchJob (the initial fetch, a poll tick, a manual Refresh, an
    // image-error retry) — its eventual response, and whatever poll/backoff it would have
    // scheduled, must not land on top of what THIS request is about to produce. Also drop any
    // already-scheduled timer for the same reason: it was scheduled for the job as of before this
    // request, which this request is about to replace.
    fetchSeqRef.current += 1;
    clearPollTimer();
    api
      .requestNlf(idRef.current)
      .then((j) => {
        if (!mountedRef.current) return;
        setJob(j);
        schedulePoll(j);
      })
      .catch((e) => {
        if (!mountedRef.current) return;
        setApiError(errorMessage(e));
      })
      .finally(() => {
        if (mountedRef.current) setRequesting(false);
      });
  }, [requesting, schedulePoll, clearPollTimer]);

  if (!active) return null;

  // Show the manual Refresh affordance whenever the job is still moving (queued/claimed) but
  // isn't ALSO being kept fresh automatically right now: queued with the worker offline (isPollable
  // is false — the main case, a visible user just waiting for the PC), the tab is hidden (so the
  // poll loop is deliberately paused), or the last attempt errored (apiError set — the panel is
  // sitting on a backoff timer, and the person may not want to wait a full minute for the retry).
  const showRefreshButton =
    !!job &&
    (job.status === "queued" || job.status === "claimed") &&
    (!isPollable(job) || !visible || !!apiError);

  return (
    <StudioCard icon={<Cube size={14} weight="bold" />} title={t("nlf.title")}>
      <div className="space-y-3">
        {/* Skipped for `job === undefined`: that state renders its OWN error + Retry block below
            instead of this generic banner, since there is nothing else to show alongside it. */}
        {apiError && job !== undefined && (
          <p className="rounded-lg bg-[#fff5f5] px-3 py-2 text-[11px] leading-relaxed text-[#c14343]">
            {t("nlf.apiError", { detail: apiError })}
          </p>
        )}

        {job === undefined ? (
          apiError ? (
            <div className="space-y-2">
              <p className="rounded-lg bg-[#fff5f5] px-3 py-2 text-[11px] leading-relaxed text-[#c14343]">
                {t("nlf.apiError", { detail: apiError })}
              </p>
              <RefreshButton t={t} onClick={fetchJob} />
            </div>
          ) : (
            <p className="text-[11px] text-[#63709f]">{t("nlf.loading")}</p>
          )
        ) : job === null ? (
          <div className="space-y-2">
            <button
              type="button"
              onClick={requestJob}
              disabled={requesting}
              className="rounded-full bg-primary px-4 py-1.5 text-[12px] font-semibold text-white disabled:opacity-60"
            >
              {t("nlf.get")}
            </button>
            <p className="text-[10.5px] leading-relaxed text-[#8a90ac]">{t("nlf.privacy")}</p>
          </div>
        ) : job.status === "queued" ? (
          <div className="space-y-1.5">
            <p className="text-[11px] font-semibold text-[#1e2142]">{t("nlf.queued")}</p>
            {!job.worker_online && (
              <p className="text-[10.5px] leading-relaxed text-[#8a90ac]">{t("nlf.workerOffline")}</p>
            )}
            {showRefreshButton && <RefreshButton t={t} onClick={fetchJob} />}
          </div>
        ) : job.status === "claimed" ? (
          <div className="space-y-1.5">
            <p className="text-[11px] font-semibold text-[#1e2142]">{t("nlf.claimed")}</p>
            {showRefreshButton && <RefreshButton t={t} onClick={fetchJob} />}
          </div>
        ) : job.status === "failed" ? (
          <div className="space-y-2">
            <p className="text-[11px] leading-relaxed text-[#c14343]">
              {job.error ?? t("nlf.failedGeneric")}
            </p>
            <button
              type="button"
              onClick={requestJob}
              disabled={requesting}
              className="rounded-full border border-[#ddd8f5] px-4 py-1.5 text-[12px] font-semibold text-[#59648f] disabled:opacity-60"
            >
              {t("nlf.tryAgain")}
            </button>
          </div>
        ) : (
          <TurntableViewer
            keyFrames={job.key_frames}
            angles={job.angles}
            tilePx={job.tile_px}
            onSeek={onSeek}
            onImageError={handleImageError}
          />
        )}
      </div>
    </StudioCard>
  );
}

function RefreshButton({
  t,
  onClick,
}: {
  t: (key: string) => string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-1 rounded-full border border-[#ddd8f5] px-3 py-1 text-[11px] font-semibold text-[#59648f] hover:bg-[#f3f0ff]"
    >
      <ArrowClockwise size={12} weight="bold" />
      {t("nlf.refresh")}
    </button>
  );
}
