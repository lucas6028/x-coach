import { useCallback, useEffect, useRef, useState } from "react";
import { ClipboardText } from "@phosphor-icons/react";
import { Link, useSearchParams } from "react-router-dom";
import { api, UploadLimitError, type Analysis } from "./api";
import AppLayout from "./components/AppLayout";
import VideoPanel from "./components/VideoPanel";
import CoachTray from "./components/CoachTray";
import DemoIntro from "./components/DemoIntro";
import StudioTitleBar from "./components/StudioTitleBar";
import StudioMobile from "./components/mobile/StudioMobile";
import KeyMetricsCard from "./components/studio/KeyMetricsCard";
import PreviousSessionsCard from "./components/studio/PreviousSessionsCard";
import TipsCard from "./components/studio/TipsCard";
import { captureThumbnail } from "./lib/thumbnail";
import { loadAnalysisTier, saveAnalysisTier, type PoseTier } from "./lib/poseTier";
import { movementLabel, useI18n } from "./lib/i18n";
import { useIsMobile } from "./lib/useIsMobile";
import { useLiffContext } from "./lib/liffContext";
import type { AnalyzableMovement } from "./lib/movements";

export default function App() {
  const { t } = useI18n();
  // The phone tree covers both phone surfaces: a narrow viewport, and the LINE in-app browser
  // (which can be wide on a tablet but still gets the phone shell).
  const { isInClient } = useLiffContext();
  const phone = useIsMobile() || isInClient;
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [currentTime, setCurrentTime] = useState(0);
  const [activeFaultId, setActiveFaultId] = useState<string | null>(null);

  // The extraction tier now lives in the page header (StudioTitleBar) rather than inside the
  // capture panel, so the studio owns it and forwards it down. Persisted, same as before.
  const [tier, setTier] = useState<PoseTier>(() => loadAnalysisTier());
  const changeTier = useCallback((next: PoseTier) => {
    setTier(next);
    saveAnalysisTier(next);
  }, []);

  const [searchParams, setSearchParams] = useSearchParams();

  // Entered from a training plan: /app?movement=X&plan=<planId>&plan_item=<itemId>. Both ids are
  // needed — the item PATCH is scoped by plan, so an item id alone cannot be written back.
  const planId = searchParams.get("plan");
  const planItemId = searchParams.get("plan_item");
  // The plan's name and this item's day, for the banner. Fetched rather than passed in the URL:
  // a name in a query string is a second copy that goes stale the moment the plan is renamed.
  const [planCtx, setPlanCtx] = useState<{ name: string; day: number } | null>(null);
  // Whether this analysis has been ticked off in the plan, so the banner can say so.
  const [planLinked, setPlanLinked] = useState(false);
  useEffect(() => {
    if (!planId || !planItemId) {
      setPlanCtx(null);
      return;
    }
    let cancelled = false;
    api
      .getPlan(planId)
      .then((p) => {
        if (cancelled) return;
        const item = p.items.find((it) => it.id === planItemId);
        setPlanCtx({ name: p.name, day: item?.day_index ?? 1 });
        // Arriving on an item that is already ticked (a re-record, or a back-button return) must
        // show the linked state rather than claiming it is still outstanding.
        setPlanLinked(!!item?.completed_at);
      })
      // Silent: the plan banner is context, and a signed-out or failed fetch must not stop the
      // studio from analysing the clip the user came here to analyse.
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [planId, planItemId]);

  const videoRef = useRef<HTMLVideoElement>(null);
  // The analysis id we just reflected into the URL after an upload — so the replay effect below can
  // skip re-fetching an analysis we already hold in state.
  const skipReloadId = useRef<string | null>(null);

  const seek = useCallback((t: number) => {
    const v = videoRef.current;
    if (v) {
      v.currentTime = t;
      v.play().catch(() => undefined);
    }
  }, []);

  const [movements, setMovements] = useState<AnalyzableMovement[]>([
    { name: "Squat", validated: true },
  ]);
  // Tracked separately from the list itself: the seed value is a FALLBACK, not an answer, and
  // treating it as one would flash "Push-up cannot be analysed yet" on every slow load before
  // the real list arrives.
  const [movementsLoaded, setMovementsLoaded] = useState(false);
  useEffect(() => {
    let cancelled = false;
    api
      .getMovements()
      .then((ms) => {
        if (!cancelled && ms.length) setMovements(ms);
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setMovementsLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // The movement is user-asserted input, taken from the URL when the studio is entered from the
  // /movements menu. Validate it against the fetched list BEFORE enabling the dropzone: the
  // backend would 400 anyway, but only after the user picked and uploaded a file.
  // `?capture=record` is a one-shot instruction from the movement detail page's "Start recording"
  // card: open the capture panel on the camera rather than the dropzone. Read once and then
  // stripped from the URL — left there it would re-apply every time the capture panel remounts
  // (DemoIntro swaps it for the loader while an analysis runs), snapping the panel back to the
  // camera and overriding whichever tab the user has picked since. `?movement=` is deliberately
  // NOT stripped with it: that one describes the session, not a single action.
  const [captureRecord] = useState(() => searchParams.get("capture") === "record");
  useEffect(() => {
    if (!searchParams.has("capture")) return;
    const next = new URLSearchParams(searchParams);
    next.delete("capture");
    setSearchParams(next, { replace: true });
  }, [searchParams, setSearchParams]);

  const requestedMovement = searchParams.get("movement");
  const [movement, setMovement] = useState<string>(requestedMovement || "Squat");
  useEffect(() => {
    if (requestedMovement) setMovement(requestedMovement);
  }, [requestedMovement]);

  // Resolve to the catalog's OWN spelling, matched the way the BACKEND matches it: the movement
  // registry lowercases its lookup key (registry.get_detector) and `_validated_movement` returns
  // the registry's canonical name, so `?movement=push-up` is a request the API accepts and
  // canonicalizes to "Push-up". Comparing exactly here locked the user out of a movement the
  // server would have analyzed fine.
  //
  // Canonicalizing rather than just loosening the comparison, because the canonical spelling is
  // load-bearing in three more places: it is the <select> option value, the `movement.<Name>`
  // i18n key, and the key the Beta badge looks up. A case-insensitive `known` alone would clear
  // the error but still show a phantom duplicate option and drop the Beta tag.
  const resolved = movements.find(
    (m) => m.name.toLowerCase() === movement.trim().toLowerCase()
  );
  const known = resolved !== undefined;
  const canonicalMovement = resolved?.name ?? movement;
  // Only an ANSWERED "not analyzable" is an error. While the list is in flight we know nothing,
  // and "we don't know yet" must not render as "no". The message quotes what the USER asked for,
  // not a canonical name we could not resolve.
  const movementError =
    !movementsLoaded || known ? "" : t("studio.movementUnavailable", { movement });

  // The server's 413 detail is English and structured; the message the user reads is neither.
  const errorMessage = useCallback(
    (e: unknown): string => {
      if (e instanceof UploadLimitError) {
        return e.code === "upload_too_large"
          ? t("upload.tooLarge", { limit: e.limitMb })
          : t("upload.quotaFull", { used: e.usedMb ?? 0, limit: e.limitMb });
      }
      return e instanceof Error ? e.message : String(e);
    },
    [t]
  );

  // Client-side capture path: extraction happens in-browser (extractPoseFromBlob), then the pose
  // JSON + original video POST to /api/analyze/pose. Mirrors the old runUpload's state handling.
  const runPoseAnalysis = useCallback(async (blob: Blob, chosenTier: PoseTier) => {
    setLoading(true);
    setError("");
    setAnalysis(null);
    setStatusMsg(t("app.analysing"));
    try {
      // MediaPipe is a cold path: defer its WASM graph until the user explicitly supplies video.
      const { extractPoseFromBlob } = await import("./lib/poseExtract");
      const pose = await extractPoseFromBlob(blob, chosenTier);
      // Captured from the same blob the browser just decoded for MediaPipe, so it costs one
      // extra seek. Resolves to null on any failure — a missing thumbnail never blocks analysis.
      const thumbnail = await captureThumbnail(blob);
      // The user's selected movement, not a hardcoded "Squat". `analyzePose` has taken a movement
      // since the client-capture path landed; this is the caller that finally supplies a real one.
      const data = await api.analyzePose(canonicalMovement, pose, blob, thumbnail);
      setAnalysis(data);
      // Reflect a persisted upload in the URL so it's shareable and survives a refresh (which then
      // restores the chat thread via the replay path). Only signed-in uploads get an analysis_id;
      // an anonymous upload has nothing durable to link to, so the URL stays put. Guard the replay
      // effect from re-fetching the analysis we already hold.
      if (data.analysis_id) {
        skipReloadId.current = data.analysis_id;
        // The plan ids ride along, so the banner and its "back to plan" link survive the rewrite.
        setSearchParams(
          planId && planItemId
            ? { analysis: data.analysis_id, plan: planId, plan_item: planItemId }
            : { analysis: data.analysis_id },
          { replace: true }
        );
        // Tick the plan item off and link this analysis to it. Guarded on `analysis_id`, which
        // ONLY a signed-in upload has — an anonymous visitor who lands here with a ?plan_item= in
        // the URL analyses their clip normally and this is simply skipped, rather than erroring on
        // a write they could not have been allowed to make.
        if (planId && planItemId) {
          try {
            await api.updatePlanItem(planId, planItemId, {
              completed: true,
              analysis_id: data.analysis_id,
            });
            setPlanLinked(true);
          } catch {
            // The analysis is saved either way. A failed tick is worth neither an error banner
            // over a successful analysis nor losing the result the user just waited for.
          }
        }
      }
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
      setStatusMsg("");
    }
  }, [t, setSearchParams, canonicalMovement, errorMessage, planId, planItemId]);

  // Replay a saved analysis when arriving from history via /app?analysis=<id>.
  const loadStored = useCallback(async (id: string) => {
    setLoading(true);
    setError("");
    setStatusMsg(t("app.loading", { id }));
    setAnalysis(null);
    try {
      const row = await api.getStoredAnalysis(id);
      setAnalysis(row.result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      setStatusMsg("");
    }
  }, [t]);

  // Reset the studio to a fresh upload state — clears the loaded analysis, any transient
  // status/error, and the shareable ?analysis= param so the URL matches the empty view.
  const newAnalysis = useCallback(() => {
    setAnalysis(null);
    setError("");
    setStatusMsg("");
    skipReloadId.current = null;
    if (searchParams.get("analysis")) setSearchParams({}, { replace: true });
  }, [searchParams, setSearchParams]);

  const storedId = searchParams.get("analysis");
  useEffect(() => {
    if (!storedId) return;
    // A just-uploaded analysis is already in state — skip the redundant re-fetch (consume the guard
    // once, so a later manual navigation back to the same id still reloads).
    if (storedId === skipReloadId.current) {
      skipReloadId.current = null;
      return;
    }
    void loadStored(storedId);
    // Re-run only when the requested id changes (not on unrelated re-renders).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storedId]);

  // Reset playback state whenever a new analysis arrives.
  useEffect(() => {
    setCurrentTime(0);
    setActiveFaultId(null);
  }, [analysis?.video_id]);

  // Adopt the loaded analysis's own movement. Without this the page header keeps naming whatever
  // the selector happened to hold — replaying a saved Overhead Press from history rendered
  // "Squat Motion Analysis" over a clip the coach panel was correctly calling an overhead press.
  // Syncing the selector rather than just the title keeps all three surfaces (title, dropdown,
  // coach banner) saying one thing, and leaves the next upload defaulted to the same movement.
  //
  // The `?? "Squat"` is the SAME fallback CoachTray applies to its clean-rep banner, and it has to
  // be: analyses predating per-movement selection carry no `movement` at all, so leaving those
  // alone let the title keep the previous clip's name — an overhead-press header over a squat
  // replayed from history. Two surfaces guessing differently is the failure this whole sync
  // exists to prevent, so they guess together.
  useEffect(() => {
    if (analysis) setMovement(analysis.movement ?? "Squat");
  }, [analysis?.video_id, analysis?.movement]);

  const hasResult = !!analysis;

  return (
    <AppLayout
      onNewAnalysis={newAnalysis}
      title={t("studio.title", { movement: movementLabel(t, canonicalMovement) })}
      // The studio's header goes in the shell's top row, so its movement / precision / start
      // controls sit on the title's own line, beside the account cluster rather than under it.
      header={
        <StudioTitleBar
          movement={canonicalMovement}
          movements={movements}
          onMovementChange={setMovement}
          tier={tier}
          onTierChange={changeTier}
          // Only once a result is up: in the empty state the dropzone right below is already the
          // call to action, so this would be a second, weaker copy of it.
          onNewSession={hasResult ? newAnalysis : undefined}
        />
      }
    >
      {/* Where this session came from, when the studio was entered from a plan item. It stays up
          through the whole session — before the upload it says which exercise is being recorded,
          after it confirms the item was ticked off. */}
      {planCtx && (
        <div className="mt-3 flex flex-wrap items-center gap-2 rounded-xl border border-primary/25 bg-primary/[0.04] px-3 py-2 text-xs">
          <ClipboardText size={14} weight="duotone" className="shrink-0 text-primary" />
          <span className="font-medium text-content">
            {t("plans.studioBanner", { plan: planCtx.name, day: planCtx.day })}
          </span>
          {planLinked && <span className="text-secondary">{t("plans.studioLinked")}</span>}
          <Link
            to={`/plans/${planId}`}
            className="ml-auto font-semibold text-primary underline-offset-2 hover:underline"
          >
            {t("plans.studioBackToPlan")}
          </Link>
        </div>
      )}

      {hasResult && phone ? (
        // The phone layout (motion_analysis_mobile.png). Chosen here rather than by CSS: both
        // trees mount a <video> and a skeleton canvas, so rendering the two and hiding one would
        // decode the clip twice and run two rAF loops.
        <StudioMobile
          analysis={analysis!}
          videoRef={videoRef}
          currentTime={currentTime}
          onTimeUpdate={setCurrentTime}
          onActiveFault={setActiveFaultId}
          activeFaultId={activeFaultId}
          onSeek={seek}
          onNewSession={newAnalysis}
        />
      ) : !hasResult ? (
        <DemoIntro
          onBlob={runPoseAnalysis}
          onError={setError}
          loading={loading}
          statusMsg={statusMsg}
          error={error}
          movement={canonicalMovement}
          movementError={movementError}
          movementsLoaded={movementsLoaded}
          tier={tier}
          record={captureRecord}
        />
      ) : (
        // The reference's 12-column split: the clip and its dashboard cards on the left, the
        // coach column on the right. Mobile stacks and scrolls as one page; on desktop each
        // column scrolls independently inside the card.
        <div className="mt-4 grid min-h-0 flex-1 grid-cols-12 gap-4 overflow-y-auto scrollbar-thin lg:gap-5 lg:overflow-hidden">
          <div className="col-span-12 flex min-w-0 flex-col gap-4 lg:col-span-8 lg:min-h-0 lg:overflow-y-auto lg:pr-1 scrollbar-thin">
            <VideoPanel
              analysis={analysis!}
              videoRef={videoRef}
              onTimeUpdate={setCurrentTime}
              onActiveFault={setActiveFaultId}
              onSeek={seek}
              activeFaultId={activeFaultId}
            />

            <div className="grid shrink-0 grid-cols-1 gap-4 md:grid-cols-3">
              <PreviousSessionsCard currentVideoId={analysis!.video_id} />
              <KeyMetricsCard analysis={analysis!} />
              <TipsCard analysis={analysis!} />
            </div>
          </div>

          {/* One unified "coach chat" column — the grounded fault-card analysis, the knowledge
              graph below it, and the follow-up conversation, all in one thread. */}
          <aside className="col-span-12 flex min-h-0 lg:col-span-4 lg:h-full">
            <CoachTray
              analysis={analysis!}
              currentTime={currentTime}
              onSeek={seek}
              activeFaultId={activeFaultId}
            />
          </aside>
        </div>
      )}
    </AppLayout>
  );
}
