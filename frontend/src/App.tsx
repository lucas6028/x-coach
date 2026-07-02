import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, type Analysis } from "./api";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import VideoPanel from "./components/VideoPanel";
import ReasoningLog from "./components/ReasoningLog";
import KnowledgeGraphWidget from "./components/KnowledgeGraphWidget";
import LibraryPicker from "./components/LibraryPicker";
import DemoIntro from "./components/DemoIntro";
import ChatInput from "./components/ChatInput";
import ResizeHandle from "./components/ResizeHandle";
import { useI18n } from "./lib/i18n";

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

const SIDEBAR_MIN = 160;
const SIDEBAR_MAX = 480;
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
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(240);
  const [feedbackWidth, setFeedbackWidth] = useState(384);
  const [resizing, setResizing] = useState(false);

  const [searchParams] = useSearchParams();
  const videoRef = useRef<HTMLVideoElement>(null);

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

  const runUpload = useCallback(async (file: File) => {
    setLoading(true);
    setError("");
    setAnalysis(null);
    setStatusMsg(t("app.analysing"));
    try {
      const data = await api.analyzeUpload(file);
      setAnalysis(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      setStatusMsg("");
    }
  }, [t]);

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

  const storedId = searchParams.get("analysis");
  useEffect(() => {
    if (storedId) void loadStored(storedId);
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
    <div className="h-[100dvh] w-full flex bg-background-dark text-content overflow-hidden">
      {/* Desktop: inline, resizable sidebar */}
      <div className="hidden lg:flex shrink-0">
        <Sidebar
          open={sidebarOpen}
          width={sidebarOpen ? sidebarWidth : 64}
          animate={!resizing}
          onToggle={() => setSidebarOpen((v) => !v)}
          onOpenLibrary={() => setPickerOpen(true)}
        />
        {sidebarOpen && (
          <ResizeHandle
            onResize={(d) => setSidebarWidth((w) => clamp(w + d, SIDEBAR_MIN, SIDEBAR_MAX))}
            onResizeStart={() => setResizing(true)}
            onResizeEnd={() => setResizing(false)}
          />
        )}
      </div>

      {/* Mobile: off-canvas drawer + backdrop */}
      {mobileNav && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => setMobileNav(false)}
        />
      )}
      <div
        className={`fixed inset-y-0 left-0 z-40 w-[270px] max-w-[80vw] transition-transform duration-200 ease-in-out lg:hidden ${
          mobileNav ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Sidebar
          open
          width={270}
          animate={false}
          onToggle={() => setMobileNav(false)}
          onOpenLibrary={() => {
            setMobileNav(false);
            setPickerOpen(true);
          }}
        />
      </div>

      <main className="flex-1 flex flex-col min-w-0 min-h-0">
        <Header analysis={analysis} loading={loading} onMenu={() => setMobileNav(true)} />

        {!hasResult ? (
          <DemoIntro
            onFile={runUpload}
            onOpenLibrary={() => setPickerOpen(true)}
            loading={loading}
            statusMsg={statusMsg}
            error={error}
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
              onResizeStart={() => setResizing(true)}
              onResizeEnd={() => setResizing(false)}
            />

            {/* Right: coaching feedback, then a compact knowledge-graph card and
                the follow-up input pinned to the foot of the column. */}
            <aside
              style={{ ["--fbw" as string]: `${feedbackWidth}px` }}
              className="w-full lg:w-[var(--fbw)] flex flex-col border-t lg:border-t-0 lg:border-l border-border-dark bg-surface-dark min-h-0 shrink-0"
            >
              <ReasoningLog
                analysis={analysis!}
                currentTime={currentTime}
                onSeek={seek}
              />
              <KnowledgeGraphWidget analysis={analysis!} activeFaultId={activeFaultId} />
              <ChatInput analysis={analysis!} />
            </aside>
          </div>
        )}
      </main>

      {pickerOpen && (
        <LibraryPicker
          onClose={() => setPickerOpen(false)}
          onPick={(id) => {
            setPickerOpen(false);
            loadLibrary(id);
          }}
        />
      )}
    </div>
  );
}
