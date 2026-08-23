import { type ReactNode } from "react";
import MobileTabBar from "./mobile/MobileTabBar";
import MobileTopBar from "./mobile/MobileTopBar";

// The in-LINE shell: what AppLayout renders instead of the navbar + sidebar when the SPA is
// running inside the LINE app. A bottom tab bar is the one thing that makes a web app read as
// a native one on a phone, so the marketing header, the brand lockup, the sign-in button and
// the sidebar all go away — inside LINE the user is already signed in silently (see
// lib/auth's LINE auto-login) and the four destinations below are the whole app.
//
// The studio, Settings and Admin deliberately have no tab: their routes still resolve, but the bar
// doesn't name them. The studio is the raised centre action instead; Settings lives behind the
// header's account menu (inside LINE the user is always signed in, so that avatar is always
// there); Admin is only ever a direct link opened outside the app. That's fine: the bar is the
// thumb-reachable primary nav, not a full site map.
//
// The bar and the header are MobileTabBar / MobileTopBar, shared with the phone web shell — both
// surfaces are phones, and the design (motion_analysis_mobile.png) is one design.

interface Props {
  children: ReactNode;
  // The same "start a fresh session" action the web sidebar carries on every page (AppLayout
  // already falls back to navigating into the studio off the studio itself — see AppLayout's
  // newAnalysis). Surfaced here in the header because there is no sidebar to hold it, and without
  // it the studio is a one-shot: once a result is on screen no tab returns to it, so there'd be
  // no way to start a second analysis without leaving LINE entirely.
  onNewAnalysis: () => void;
  /** Short page name for the header's centred title. */
  title?: string;
}

export default function LiffAppShell({ children, onNewAnalysis, title }: Props) {
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
      <MobileTabBar onNewAnalysis={onNewAnalysis} />
    </div>
  );
}
