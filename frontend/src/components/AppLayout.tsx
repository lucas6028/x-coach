import { useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "./Sidebar";
import Header from "./Header";
import LiffAppShell from "./LiffAppShell";
import { useLiffContext } from "../lib/liffContext";

// Fixed expanded width — the sidebar is no longer drag-resizable; it only toggles
// between this and the 64px icon rail.
const SIDEBAR_WIDTH = 200;

interface Props {
  children: ReactNode;
  // The studio supplies a picker opener; other pages fall back to navigating into the studio.
  onOpenLibrary?: () => void;
  // The studio resets its own state for a fresh session; other pages just route into the studio.
  onNewAnalysis?: () => void;
  // Whether the desktop sidebar starts expanded. Defaults to open; the games opt to start it
  // collapsed so the camera/play area gets near-full width.
  initialSidebarOpen?: boolean;
}

// The shared app shell: collapsible/resizable desktop sidebar, off-canvas mobile drawer, and the
// top navbar (Header). Every signed-in page (studio, history, settings) renders its content as
// `children` so the sidebar + navbar stay identical across the app. Only the home/landing and the
// pre-auth login gateway opt out. Inside the LINE in-app browser this delegates to LiffAppShell
// instead (see lib/liffContext).
export default function AppLayout({
  children,
  onOpenLibrary,
  onNewAnalysis,
  initialSidebarOpen = true,
}: Props) {
  const navigate = useNavigate();
  const { isInClient } = useLiffContext();
  // Desktop sidebar is a fixed-width rail; it only toggles between expanded and the icon rail.
  const [sidebarOpen, setSidebarOpen] = useState(initialSidebarOpen);
  const [mobileNav, setMobileNav] = useState(false);

  // Off the studio there is no picker, so "Library" just routes into the studio.
  const openLibrary = onOpenLibrary ?? (() => navigate("/app"));
  // Off the studio, "New analysis" routes into the studio; the studio resets in place.
  const newAnalysis = onNewAnalysis ?? (() => navigate("/app"));

  // Inside the LINE app the whole web chrome is replaced by the app shell: bottom tabs instead
  // of a sidebar, no marketing navbar. Every page renders through here, so this one branch
  // converts the entire app without any page knowing about LIFF. openLibrary/newAnalysis are the
  // same resolved actions the (now-hidden) Sidebar would have gotten — without threading them
  // through, the shell's four tabs would be the whole app and there'd be no way to start a
  // second analysis without leaving LINE.
  if (isInClient) {
    return (
      <LiffAppShell onOpenLibrary={openLibrary} onNewAnalysis={newAnalysis}>
        {children}
      </LiffAppShell>
    );
  }

  return (
    <div className="h-[100dvh] w-full flex flex-col bg-background-dark text-content overflow-hidden">
      {/* Full-width top navbar, spanning both the sidebar and main columns (the reference layout).
          It carries the brand and the desktop sidebar-collapse toggle. */}
      <Header
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
        onMenu={() => setMobileNav(true)}
      />

      {/* Below the navbar: the sidebar and main sit side by side, split by a vertical divider. */}
      <div className="flex flex-1 min-h-0 min-w-0">
        {/* Desktop: inline, fixed-width sidebar with a divider between it and the main content. */}
        <div className="hidden lg:flex shrink-0 border-r border-border-dark">
          <Sidebar
            open={sidebarOpen}
            width={sidebarOpen ? SIDEBAR_WIDTH : 64}
            animate
            onOpenLibrary={openLibrary}
            onNewAnalysis={newAnalysis}
          />
        </div>

        {/* Mobile: off-canvas drawer + backdrop (overlays the navbar too) */}
        {mobileNav && (
          <div
            className="fixed inset-0 z-40 bg-black/50 lg:hidden"
            onClick={() => setMobileNav(false)}
          />
        )}
        <div
          className={`fixed inset-y-0 left-0 z-50 w-[270px] max-w-[80vw] bg-background-dark transition-transform duration-200 ease-in-out lg:hidden ${
            mobileNav ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <Sidebar
            open
            width={270}
            animate={false}
            onClose={() => setMobileNav(false)}
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

        <main className="flex-1 flex flex-col min-w-0 min-h-0">{children}</main>
      </div>
    </div>
  );
}
