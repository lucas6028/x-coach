import { ClockCounterClockwise, Folders, GameController, GearSix, Plus, VideoCamera } from "@phosphor-icons/react";
import { Link, useLocation } from "react-router-dom";
import { type ReactNode } from "react";
import { movementLabel, useI18n } from "../lib/i18n";

// The in-LINE shell: what AppLayout renders instead of the navbar + sidebar when the SPA is
// running inside the LINE app. A bottom tab bar is the one thing that makes a web app read as
// a native one on a phone, so the marketing header, the brand lockup, the sign-in button and
// the sidebar all go away — inside LINE the user is already signed in silently (see
// lib/auth's LINE auto-login) and the four destinations below are the whole app.
//
// Movements and Admin deliberately have no tab: their routes still resolve, but only via a rich
// menu entry or a direct link opened outside the app — there is no in-app link to either while
// in-client (Sidebar's Movements/Admin links only render in the web shell, see AppLayout).
// That's fine: the four tabs are the thumb-reachable primary nav, not a full site map.

interface Props {
  children: ReactNode;
  // The page name, same prop AppLayout takes. The studio passes none and gets the studio title
  // — mirroring Header's own fallback, so the copy matches the web build.
  title?: string;
  // The studio's currently-selected movement, mirroring Header's `movement` prop — so the LINE
  // shell's fallback title is "{movement} Analysis" too, not a hardcoded "Squat Analysis".
  movement?: string;
  // Same "start a fresh session" / "open a saved clip" actions the web sidebar carries on every
  // page (AppLayout already falls back to navigating into the studio off the studio itself — see
  // AppLayout's openLibrary/newAnalysis). Surfaced here in the header because there is no sidebar
  // to hold them, and without them the studio is a one-shot: once a result is on screen, tapping
  // the Analyse tab is a same-route Link that doesn't remount or reset it, so there'd be no way
  // to start a second analysis without leaving LINE entirely.
  onOpenLibrary: () => void;
  onNewAnalysis: () => void;
}

export default function LiffAppShell({ children, title, movement, onOpenLibrary, onNewAnalysis }: Props) {
  const { t } = useI18n();
  const { pathname } = useLocation();

  const tabs = [
    { to: "/app", label: t("nav.analyse"), Icon: VideoCamera, active: pathname === "/app" },
    {
      to: "/history",
      label: t("nav.history"),
      Icon: ClockCounterClockwise,
      active: pathname === "/history",
    },
    {
      to: "/games",
      label: t("nav.games"),
      Icon: GameController,
      // One tab for the hub and both individual games (mirrors Sidebar's onGames check).
      active: pathname === "/games" || pathname === "/67" || pathname === "/ninja",
    },
    { to: "/settings", label: t("settings.title"), Icon: GearSix, active: pathname === "/settings" },
  ];

  return (
    // 100dvh (not 100vh) so the LINE in-app browser's collapsing toolbar can't clip the tab
    // bar; the top inset covers a notch when LINE renders the LIFF view full-bleed.
    <div className="h-[100dvh] w-full flex flex-col overflow-hidden bg-background-dark text-content pt-[env(safe-area-inset-top)]">
      <header className="h-12 shrink-0 border-b border-border-dark bg-surface flex items-center gap-1 px-4">
        <h1 className="flex-1 min-w-0 text-sm font-semibold tracking-tight truncate">
          {title ?? t("header.title", { movement: movementLabel(t, movement ?? "Squat") })}
        </h1>
        {/* Icon-only — the header has no room for labels, so the accessible name carries the copy. */}
        <button
          onClick={onNewAnalysis}
          aria-label={t("nav.newAnalysis")}
          title={t("nav.newAnalysis")}
          className="shrink-0 w-9 h-9 flex items-center justify-center rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors"
        >
          <Plus size={20} weight="bold" />
        </button>
        <button
          onClick={onOpenLibrary}
          aria-label={t("nav.library")}
          title={t("nav.library")}
          className="shrink-0 w-9 h-9 flex items-center justify-center rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors"
        >
          <Folders size={20} weight="duotone" />
        </button>
      </header>

      {/* Same contract as AppLayout's main: bounded height, pages own their own scrolling. */}
      <main className="flex-1 flex flex-col min-w-0 min-h-0">{children}</main>

      {/* In normal flow, not fixed — the flex column already keeps it pinned to the bottom, and
          nothing can slide under it, so no compensating padding is needed. */}
      <nav
        aria-label={t("nav.tabBar")}
        className="shrink-0 grid grid-cols-4 border-t border-border-dark bg-surface pb-[env(safe-area-inset-bottom)]"
      >
        {tabs.map(({ to, label, Icon, active }) => (
          <Link
            key={to}
            to={to}
            aria-current={active ? "page" : undefined}
            className={`flex flex-col items-center gap-0.5 py-2 text-[11px] font-medium transition-colors ${
              active ? "text-primary" : "text-muted"
            }`}
          >
            <Icon size={22} weight={active ? "fill" : "duotone"} />
            <span className="truncate max-w-full px-1">{label}</span>
          </Link>
        ))}
      </nav>
    </div>
  );
}
