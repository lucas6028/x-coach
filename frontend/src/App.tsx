import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Analysis } from "./api";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import VideoPanel from "./components/VideoPanel";
import MetricsCards from "./components/MetricsCards";
import ReasoningLog from "./components/ReasoningLog";
import KnowledgeGraphWidget from "./components/KnowledgeGraphWidget";
import LibraryPicker from "./components/LibraryPicker";
import UploadDropzone from "./components/UploadDropzone";
import ChatInput from "./components/ChatInput";
import ResizeHandle from "./components/ResizeHandle";
import { useI18n } from "./lib/i18n";

type MobileTab = "feedback" | "graph";

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
  const [mobileTab, setMobileTab] = useState<MobileTab>("feedback");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(240);
  const [feedbackWidth, setFeedbackWidth] = useState(384);
  const [resizing, setResizing] = useState(false);

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

  // Reset playback state whenever a new analysis arrives.
  useEffect(() => {
    setCurrentTime(0);
    setActiveFaultId(null);
  }, [analysis?.video_id]);

  const hasResult = !!analysis;

  return (
    <div className="h-screen w-screen flex bg-background-dark text-content overflow-hidden">
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

      <main className="flex-1 flex flex-col min-w-0">
        <Header analysis={analysis} loading={loading} />

        {!hasResult ? (
          <div className="flex-1 overflow-y-auto p-6 flex flex-col items-center justify-center gap-6">
            <UploadDropzone onFile={runUpload} loading={loading} statusMsg={statusMsg} />
            <button
              onClick={() => setPickerOpen(true)}
              className="text-sm text-primary hover:underline"
            >
              {t("app.pickSample")}
            </button>
            {error && <p className="text-danger text-sm max-w-md text-center">{error}</p>}
          </div>
        ) : (
          <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
            {/* Left: video + timeline + metrics */}
            <div className="flex-1 lg:flex-1 min-w-0 flex flex-col gap-4 p-4 overflow-y-auto scrollbar-thin bg-content/[0.03]">
              <VideoPanel
                analysis={analysis!}
                videoRef={videoRef}
                onTimeUpdate={setCurrentTime}
                onActiveFault={setActiveFaultId}
                onSeek={seek}
              />
              <MetricsCards analysis={analysis!} />
            </div>

            {/* Drag to resize video vs. feedback (desktop only — panes stack on mobile). */}
            <ResizeHandle
              className="hidden lg:block"
              onResize={(d) => setFeedbackWidth((w) => clamp(w - d, FEEDBACK_MIN, FEEDBACK_MAX))}
              onResizeStart={() => setResizing(true)}
              onResizeEnd={() => setResizing(false)}
            />

            {/* Right: knowledge graph + reasoning (desktop). Tabbed on mobile. */}
            <aside
              style={{ ["--fbw" as string]: `${feedbackWidth}px` }}
              className="w-full lg:w-[var(--fbw)] flex flex-col border-t lg:border-t-0 lg:border-l border-border-dark bg-surface-dark min-h-0 shrink-0"
            >
              <div className="flex lg:hidden border-b border-border-dark">
                {(["feedback", "graph"] as MobileTab[]).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setMobileTab(tab)}
                    className={`flex-1 py-2 text-xs uppercase tracking-wide ${
                      mobileTab === tab ? "text-primary border-b-2 border-primary" : "text-muted"
                    }`}
                  >
                    {t(tab === "feedback" ? "tab.coaching" : "tab.graph")}
                  </button>
                ))}
              </div>

              <div className={`${mobileTab === "graph" ? "block" : "hidden"} lg:block`}>
                <KnowledgeGraphWidget analysis={analysis!} activeFaultId={activeFaultId} />
              </div>
              <div
                className={`${
                  mobileTab === "feedback" ? "flex" : "hidden"
                } lg:flex flex-col flex-1 min-h-0`}
              >
                <ReasoningLog
                  analysis={analysis!}
                  currentTime={currentTime}
                  onSeek={seek}
                />
                <ChatInput />
              </div>
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
