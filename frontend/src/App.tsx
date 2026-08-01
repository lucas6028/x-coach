import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, type Analysis } from "./api";
import AppLayout from "./components/AppLayout";
import VideoPanel from "./components/VideoPanel";
import CoachTray from "./components/CoachTray";
import LibraryPicker from "./components/LibraryPicker";
import DemoIntro from "./components/DemoIntro";
import ResizeHandle from "./components/ResizeHandle";
import { extractPoseFromBlob } from "./lib/poseExtract";
import type { PoseTier } from "./lib/poseTier";
import { useI18n } from "./lib/i18n";
import type { AnalyzableMovement } from "./lib/movements";

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

const FEEDBACK_MIN = 280;
const FEEDBACK_MAX = 640;

export default function App() {
  const { t } = useI18n();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeFaultId, setActiveFaultId] = useState<string | null>(null);
  const [feedbackWidth, setFeedbackWidth] = useState(384);

  const [searchParams, setSearchParams] = useSearchParams();
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

  const loadLibrary = useCallback(async (videoId: string) => {
    setLoading(true);
    setError("");
    setStatusMsg(t("app.loading", { id: videoId }));
    setAnalysis(null);
    try {
      const data = await api.getAnalysis(videoId);
      setAnalysis(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      setStatusMsg("");
    }
  }, [t]);

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

  // Client-side capture path: extraction happens in-browser (extractPoseFromBlob), then the pose
  // JSON + original video POST to /api/analyze/pose. Mirrors the old runUpload's state handling.
  const runPoseAnalysis = useCallback(async (blob: Blob, tier: PoseTier) => {
    setLoading(true);
    setError("");
    setAnalysis(null);
    setStatusMsg(t("app.analysing"));
    try {
      const pose = await extractPoseFromBlob(blob, tier);
      // The user's selected movement, not a hardcoded "Squat". `analyzePose` has taken a movement
      // since the client-capture path landed; this is the caller that finally supplies a real one.
      const data = await api.analyzePose(canonicalMovement, pose, blob);
      setAnalysis(data);
      // Reflect a persisted upload in the URL so it's shareable and survives a refresh (which then
      // restores the chat thread via the replay path). Only signed-in uploads get an analysis_id;
      // an anonymous upload has nothing durable to link to, so the URL stays put. Guard the replay
      // effect from re-fetching the analysis we already hold.
      if (data.analysis_id) {
        skipReloadId.current = data.analysis_id;
        setSearchParams({ analysis: data.analysis_id }, { replace: true });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      setStatusMsg("");
    }
  }, [t, setSearchParams, canonicalMovement]);

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
  // status/error, the picker, and the shareable ?analysis= param so the URL matches the empty view.
  const newAnalysis = useCallback(() => {
    setAnalysis(null);
    setError("");
    setStatusMsg("");
    setPickerOpen(false);
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

  const hasResult = !!analysis;

  return (
    <AppLayout
      analysis={analysis}
      loading={loading}
      movement={canonicalMovement}
      onOpenLibrary={() => setPickerOpen(true)}
      onNewAnalysis={newAnalysis}
    >
      {!hasResult ? (
        <DemoIntro
          onBlob={runPoseAnalysis}
          onError={setError}
          onOpenLibrary={() => setPickerOpen(true)}
          loading={loading}
          statusMsg={statusMsg}
          error={error}
          movements={movements}
          movement={canonicalMovement}
          onMovementChange={setMovement}
          movementError={movementError}
          movementsLoaded={movementsLoaded}
        />
      ) : (
        <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden scrollbar-thin">
          {/* Left: video (with overlaid metrics HUD) + timeline. Mobile scrolls
              with the page; desktop is a bounded, independently-scrolling column. */}
          <div className="min-w-0 flex flex-col gap-4 p-4 bg-content/[0.03] lg:flex-1 lg:min-h-0 lg:overflow-hidden">
            <VideoPanel
              analysis={analysis!}
              videoRef={videoRef}
              onTimeUpdate={setCurrentTime}
              onActiveFault={setActiveFaultId}
              onSeek={seek}
            />
          </div>

          {/* Drag to resize video vs. feedback (desktop only — panes stack on mobile). */}
          <ResizeHandle
            className="hidden lg:block"
            onResize={(d) => setFeedbackWidth((w) => clamp(w - d, FEEDBACK_MIN, FEEDBACK_MAX))}
          />

          {/* Right: one unified "coach chat" tray — the grounded fault-card analysis, the
              knowledge graph below it, and the follow-up conversation, all in one thread. */}
          <aside
            style={{ ["--fbw" as string]: `${feedbackWidth}px` }}
            className="w-full lg:w-[var(--fbw)] flex flex-col border-t lg:border-t-0 lg:border-l border-border-dark bg-surface-dark min-h-0 shrink-0"
          >
            <CoachTray
              analysis={analysis!}
              currentTime={currentTime}
              onSeek={seek}
              activeFaultId={activeFaultId}
            />
          </aside>
        </div>
      )}

      {pickerOpen && (
        <LibraryPicker
          onClose={() => setPickerOpen(false)}
          onPick={(id) => {
            setPickerOpen(false);
            loadLibrary(id);
          }}
        />
      )}
    </AppLayout>
  );
}
