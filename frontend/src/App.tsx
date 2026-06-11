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

type MobileTab = "feedback" | "graph";

export default function App() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeFaultId, setActiveFaultId] = useState<string | null>(null);
  const [mobileTab, setMobileTab] = useState<MobileTab>("feedback");
  const [sidebarOpen, setSidebarOpen] = useState(true);

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
    setStatusMsg(`Loading ${videoId}…`);
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
  }, []);

  const runUpload = useCallback(async (file: File) => {
    setLoading(true);
    setError("");
    setAnalysis(null);
    setStatusMsg("Extracting pose & analysing… (this can take ~20s)");
    try {
      const data = await api.analyzeUpload(file);
      setAnalysis(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      setStatusMsg("");
    }
  }, []);

  // Reset playback state whenever a new analysis arrives.
  useEffect(() => {
    setCurrentTime(0);
    setActiveFaultId(null);
  }, [analysis?.video_id]);

  const hasResult = !!analysis;

  return (
    <div className="h-screen w-screen flex bg-background-dark text-gray-100 overflow-hidden">
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((v) => !v)}
        onOpenLibrary={() => setPickerOpen(true)}
      />

      <main className="flex-1 flex flex-col min-w-0">
        <Header analysis={analysis} loading={loading} />

        {!hasResult ? (
          <div className="flex-1 overflow-y-auto p-6 flex flex-col items-center justify-center gap-6">
            <UploadDropzone onFile={runUpload} loading={loading} statusMsg={statusMsg} />
            <button
              onClick={() => setPickerOpen(true)}
              className="text-sm text-primary hover:underline"
            >
              …or pick a clip from the sample library
            </button>
            {error && <p className="text-danger text-sm max-w-md text-center">{error}</p>}
          </div>
        ) : (
          <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
            {/* Left: video + timeline + metrics */}
            <div className="flex-[2] min-w-0 flex flex-col gap-4 p-4 overflow-y-auto scrollbar-thin bg-black/20">
              <VideoPanel
                analysis={analysis!}
                videoRef={videoRef}
                onTimeUpdate={setCurrentTime}
                onActiveFault={setActiveFaultId}
                onSeek={seek}
              />
              <MetricsCards analysis={analysis!} />
            </div>

            {/* Right: knowledge graph + reasoning (desktop). Tabbed on mobile. */}
            <aside className="w-full lg:max-w-md lg:w-[24rem] flex flex-col border-t lg:border-t-0 lg:border-l border-border-dark bg-surface-dark min-h-0">
              <div className="flex lg:hidden border-b border-border-dark">
                {(["feedback", "graph"] as MobileTab[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => setMobileTab(t)}
                    className={`flex-1 py-2 text-xs uppercase tracking-wide ${
                      mobileTab === t ? "text-primary border-b-2 border-primary" : "text-gray-500"
                    }`}
                  >
                    {t === "feedback" ? "Coaching" : "Knowledge Graph"}
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
