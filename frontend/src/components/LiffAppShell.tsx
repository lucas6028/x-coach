import { ClockCounterClockwise, GameController, GearSix, VideoCamera } from "@phosphor-icons/react";
import { Link, useLocation } from "react-router-dom";
import { type ReactNode } from "react";
import { useI18n } from "../lib/i18n";

// The in-LINE shell: what AppLayout renders instead of the navbar + sidebar when the SPA is
// running inside the LINE app. A bottom tab bar is the one thing that makes a web app read as
// a native one on a phone, so the marketing header, the brand lockup, the sign-in button and
// the sidebar all go away — inside LINE the user is already signed in silently (see
// lib/auth's LINE auto-login) and the four destinations below are the whole app.
//
// Movements and Admin deliberately have no tab: their routes still resolve (rich menu, direct
// link, or a link from the studio), they just don't earn one of four thumb-reachable slots.

interface Props {
  children: ReactNode;
  // The page name, same prop AppLayout takes. The studio passes none and gets the studio title
  // — mirroring Header's own fallback, so the copy matches the web build.
  title?: string;
}

export default function LiffAppShell({ children, title }: Props) {
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
      <header className="h-12 shrink-0 border-b border-border-dark bg-surface flex items-center px-4">
        <h1 className="text-sm font-semibold tracking-tight truncate">{title ?? t("header.title")}</h1>
      </header>

      {/* Same contract as AppLayout's main: bounded height, pages own their own scrolling. */}
      <main className="flex-1 flex flex-col min-w-0 min-h-0">{children}</main>

      {/* In normal flow, not fixed — the flex column already keeps it pinned to the bottom, and
          nothing can slide under it, so no compensating padding is needed. */}
      <nav className="shrink-0 grid grid-cols-4 border-t border-border-dark bg-surface pb-[env(safe-area-inset-bottom)]">
        {tabs.map(({ to, label, Icon, active }) => (
          <Link
            key={to}
            to={to}
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
