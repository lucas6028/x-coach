import { useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import type { Analysis } from "../api";
import Sidebar from "./Sidebar";
import Header from "./Header";
import ResizeHandle from "./ResizeHandle";

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

const SIDEBAR_MIN = 160;
const SIDEBAR_MAX = 480;

interface Props {
  children: ReactNode;
  // Header content: analysis-mode (studio) shows the session + status pill; a plain `title`
  // (History/Settings) shows just the page name. Analysis-mode wins when both are given.
  analysis?: Analysis | null;
  loading?: boolean;
  title?: string;
  // The studio supplies a picker opener; other pages fall back to navigating into the studio.
  onOpenLibrary?: () => void;
  // The studio resets its own state for a fresh session; other pages just route into the studio.
  onNewAnalysis?: () => void;
}

// The shared app shell: collapsible/resizable desktop sidebar, off-canvas mobile drawer, and the
// top navbar (Header). Every signed-in page (studio, history, settings) renders its content as
// `children` so the sidebar + navbar stay identical across the app. Only the home/landing and the
// pre-auth login gateway opt out.
export default function AppLayout({
  children,
  analysis = null,
  loading = false,
  title,
  onOpenLibrary,
  onNewAnalysis,
}: Props) {
  const navigate = useNavigate();
  // Desktop sidebar starts open but on the narrow side; the user can widen it via the drag handle.
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileNav, setMobileNav] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(200);
  const [resizing, setResizing] = useState(false);

  // Off the studio there is no picker, so "Library" just routes into the studio.
  const openLibrary = onOpenLibrary ?? (() => navigate("/app"));
  // Off the studio, "New analysis" routes into the studio; the studio resets in place.
  const newAnalysis = onNewAnalysis ?? (() => navigate("/app"));

  return (
    <div className="h-[100dvh] w-full flex bg-background-dark text-content overflow-hidden">
      {/* Desktop: inline, resizable sidebar */}
      <div className="hidden lg:flex shrink-0">
        <Sidebar
          open={sidebarOpen}
          width={sidebarOpen ? sidebarWidth : 64}
          animate={!resizing}
          onToggle={() => setSidebarOpen((v) => !v)}
          onOpenLibrary={openLibrary}
          onNewAnalysis={newAnalysis}
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
            openLibrary();
          }}
          onNewAnalysis={() => {
            setMobileNav(false);
            newAnalysis();
          }}
        />
      </div>

      <main className="flex-1 flex flex-col min-w-0 min-h-0">
        <Header analysis={analysis} loading={loading} title={title} onMenu={() => setMobileNav(true)} />
        {children}
      </main>
    </div>
  );
}
