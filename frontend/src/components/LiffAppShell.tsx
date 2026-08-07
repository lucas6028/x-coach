import { type ReactNode } from "react";
import MobileTabBar from "./mobile/MobileTabBar";
import MobileTopBar from "./mobile/MobileTopBar";

// The in-LINE shell: what AppLayout renders instead of the navbar + sidebar when the SPA is
// running inside the LINE app. A bottom tab bar is the one thing that makes a web app read as
// a native one on a phone, so the marketing header, the brand lockup, the sign-in button and
// the sidebar all go away — inside LINE the user is already signed in silently (see
// lib/auth's LINE auto-login) and the four destinations below are the whole app.
//
// Movements, Games and Admin deliberately have no tab: their routes still resolve, but only via a
// rich menu entry or a direct link opened outside the app — there is no in-app link to any of them
// while in-client (Sidebar's links only render in the web shell, see AppLayout). That's fine: the
// bar is the thumb-reachable primary nav, not a full site map.
//
// The bar and the header are MobileTabBar / MobileTopBar, shared with the phone web shell — both
// surfaces are phones, and the design (motion_analysis_mobile.png) is one design.

interface Props {
  children: ReactNode;
  // Same "start a fresh session" / "open a saved clip" actions the web sidebar carries on every
  // page (AppLayout already falls back to navigating into the studio off the studio itself — see
  // AppLayout's openLibrary/newAnalysis). Surfaced here in the header because there is no sidebar
  // to hold them, and without them the studio is a one-shot: once a result is on screen, tapping
  // the Analyse tab is a same-route Link that doesn't remount or reset it, so there'd be no way
  // to start a second analysis without leaving LINE entirely.
  onOpenLibrary: () => void;
  onNewAnalysis: () => void;
  /** Short page name for the header's centred title. */
  title?: string;
}

export default function LiffAppShell({ children, onOpenLibrary, onNewAnalysis, title }: Props) {
  return (
    // 100dvh (not 100vh) so the LINE in-app browser's collapsing toolbar can't clip the tab
    // bar; the top inset covers a notch when LINE renders the LIFF view full-bleed.
    // `ms-shell` pins the light design tokens (index.css): the pages rendered inside this shell
    // are the same reference-design pages the web shell carries, and they are light-only.
    <div className="ms-shell flex h-[100dvh] w-full flex-col overflow-hidden bg-[#eef0fb] pt-[env(safe-area-inset-top)] font-body text-[#1e2142]">
      <MobileTopBar title={title ?? "X-Coach"} onNewAnalysis={onNewAnalysis} />

      {/* Same contract as AppLayout's main: bounded height, pages own their own scrolling. */}
      <main className="flex min-h-0 min-w-0 flex-1 flex-col">{children}</main>

      {/* The same five-slot bar the phone web shell uses, so the two phone surfaces are one
          design. In normal flow, not fixed — the flex column already pins it to the bottom. */}
      <MobileTabBar onNewAnalysis={onNewAnalysis} onOpenLibrary={onOpenLibrary} />
    </div>
  );
}
