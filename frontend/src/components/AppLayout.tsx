import { useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "./Sidebar";
import Header from "./Header";
import LiffAppShell from "./LiffAppShell";
import { useLiffContext } from "../lib/liffContext";

// The reference design's rail: 84px labelled, 64px icon-only when collapsed.
const SIDEBAR_WIDTH = 84;

interface Props {
  children: ReactNode;
  /** The page's own header row (breadcrumb / title / its controls). Rendered inside the shell's
   *  top row, beside the account cluster — see the branch in the body for why that matters. */
  header?: ReactNode;
  // The studio supplies a picker opener; other pages fall back to navigating into the studio.
  onOpenLibrary?: () => void;
  // The studio resets its own state for a fresh session; other pages just route into the studio.
  onNewAnalysis?: () => void;
  // Whether the desktop sidebar starts expanded. Defaults to open; the games opt to start it
  // collapsed so the camera/play area gets near-full width.
  initialSidebarOpen?: boolean;
}

// The shared app shell, in the motion_analysis_muse-spark idiom: a lavender canvas carrying a
// floating white nav rail beside one big rounded content card, with the pill action row and the
// account controls along the top of that card. Every signed-in page (studio, history, settings)
// renders its content as `children` so the shell stays identical across the app. Only the
// home/landing and the pre-auth login gateway opt out. Inside the LINE in-app browser this
// delegates to LiffAppShell instead (see lib/liffContext).
//
// The palette is the reference's own, fixed light hexes — it is a light-only design, so the shell
// does not follow the theme toggle (which still governs the token-styled page bodies).
export default function AppLayout({
  children,
  header,
  onOpenLibrary,
  onNewAnalysis,
  initialSidebarOpen = true,
}: Props) {
  const navigate = useNavigate();
  const { isInClient } = useLiffContext();
  // The desktop rail no longer has a collapse control (the top row carries no toggle), so this is
  // fixed for the life of the layout: labelled 84px rail, or the 64px icon strip when a page asks
  // for it (the games want the extra width for their camera area).
  const sidebarOpen = initialSidebarOpen;
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
    // `ms-shell` re-declares the light design tokens for everything inside the frame — see
    // index.css. The reference design is light-only, and a dark token set under these white
    // cards puts dark text and dark scrollbars inside them.
    <div className="ms-shell relative h-[100dvh] w-full overflow-hidden bg-[#eef0fb] p-2 font-body text-[#211f39] sm:p-3 lg:p-[14px]">
      {/* Background wash: the reference's two soft colour blooms behind the cards. */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
        <div className="absolute -right-20 -top-20 h-[600px] w-[600px] rounded-full bg-[#e9e3ff] opacity-40 blur-[120px]" />
        <div className="absolute -left-40 top-40 h-[500px] w-[500px] rounded-full bg-[#e0e7ff] opacity-30 blur-[100px]" />
      </div>

      <div className="relative mx-auto flex h-full max-w-[1500px] gap-3 lg:gap-4">
        {/* Desktop: the floating nav rail. */}
        <div className="hidden lg:flex">
          <Sidebar
            open={sidebarOpen}
            width={sidebarOpen ? SIDEBAR_WIDTH : 64}
            animate={false}
            onOpenLibrary={openLibrary}
            onNewAnalysis={newAnalysis}
          />
        </div>

        {/* The one content card. `glass-shell` is the theme's outermost frosted pane: the tinted
            gradient that the translucent panels inside sample from, plus the page's single
            content-level blur. Everything nested in it stays unblurred by design (index.css). */}
        <main className="glass-shell relative flex min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-hidden rounded-[28px] border border-white/80 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_22px_58px_rgba(105,112,175,0.16)] sm:p-4 lg:gap-0 lg:rounded-[32px] lg:p-5">
          {/* A page that supplies its own header puts it INSIDE this row, so its controls and the
              account cluster are siblings in one flex line and cannot overlap. A page that does
              not (history, settings, games) would otherwise be left with a row holding nothing but
              the account cluster, so there the cluster floats in the card's top-right corner from
              `lg` and the page's content starts at the top of the card instead. */}
          {header ? (
            <Header onMenu={() => setMobileNav(true)}>{header}</Header>
          ) : (
            <div className="lg:absolute lg:right-5 lg:top-5 lg:z-30">
              <Header onMenu={() => setMobileNav(true)} />
            </div>
          )}
          <div className="flex min-h-0 min-w-0 flex-1 flex-col">{children}</div>
        </main>

        {/* Mobile: off-canvas drawer + backdrop. Rendered LAST although it sits on the left —
            it is `fixed`, so DOM order costs nothing visually, and keeping it after the content
            card puts the navbar's collapse toggle ahead of the drawer's close button in the
            accessibility tree (both are labelled "Hide navigation"). */}
        {mobileNav && (
          <div
            className="fixed inset-0 z-40 bg-black/40 lg:hidden"
            onClick={() => setMobileNav(false)}
          />
        )}
        <div
          className={`fixed inset-y-2 left-2 z-50 w-[240px] max-w-[80vw] transition-transform duration-200 ease-in-out lg:hidden ${
            mobileNav ? "translate-x-0" : "-translate-x-[110%]"
          }`}
        >
          <Sidebar
            open
            width={240}
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
      </div>
    </div>
  );
}
