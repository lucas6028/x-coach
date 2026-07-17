import { ClockCounterClockwise, Folders, GameController, Graph, List, Plus, ShieldCheck, VideoCamera } from "@phosphor-icons/react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";

interface Props {
  open: boolean;
  width: number;
  // Animate width changes (toggle) but not while the user is dragging the resize handle.
  animate: boolean;
  onToggle: () => void;
  onOpenLibrary: () => void;
  // Start a fresh studio session (clears the current analysis / routes into the studio).
  onNewAnalysis: () => void;
}

// Labelled bar when `open`, slim icon rail when collapsed — mirrors demo/index.html.
// The menu toggle lives in the top row so it stays reachable in either state.
// Width is driven from the parent so it can be dragged to resize.
// Account, language and theme controls live in the top-right Header, not here.
export default function Sidebar({ open, width, animate, onToggle, onOpenLibrary, onNewAnalysis }: Props) {
  const { t } = useI18n();
  const { isAdmin } = useAuth();
  const { pathname } = useLocation();
  // Shared shell: highlight whichever destination the current route matches.
  const onStudio = pathname === "/app";
  const onHistory = pathname === "/history";
  const onExplore = pathname === "/explore";
  const onAdmin = pathname === "/admin";

  // The Admin link is admin-only UX gating (the /admin page + backend re-check are the real
  // defence). `isAdmin` is resolved once per session by AuthProvider, so switching pages no longer
  // re-probes the endpoint; a non-admin (or an errored probe) leaves it false and shows no link.

  // The games hub, plus the individual game routes it links into, all light up the one Games entry.
  const onGames = pathname === "/games" || pathname === "/67" || pathname === "/ninja";
  const navBase =
    "flex items-center gap-3 px-3 py-3 rounded-lg transition-colors";
  const navActive = "bg-primary/10 text-primary border border-primary/20";
  const navIdle = "text-muted hover:bg-content/5 hover:text-content";
  return (
    <aside
      style={{ width }}
      className={`h-full shrink-0 border-r border-border-dark bg-surface-dark flex flex-col justify-between overflow-hidden ${
        animate ? "transition-[width] duration-200 ease-in-out" : ""
      }`}
    >
      <div>
        <div className="h-16 flex items-center gap-2 px-3 border-b border-border-dark">
          <button
            onClick={onToggle}
            aria-label={open ? t("nav.hide") : t("nav.show")}
            title={open ? t("nav.hide") : t("nav.show")}
            className="shrink-0 w-10 h-10 flex items-center justify-center rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors"
          >
            <List size={20} />
          </button>
          {open && (
            <div className="flex items-center min-w-0">
              <img src="/icon.svg" alt="" className="w-8 h-8 rounded shrink-0" />
              <span className="ml-2 font-bold tracking-wide truncate">X-Coach</span>
            </div>
          )}
        </div>
        <nav className="flex flex-col gap-1 p-2">
          {/* Primary CTA: start a fresh analysis from anywhere in the app. */}
          <button
            onClick={onNewAnalysis}
            title={t("nav.newAnalysis")}
            className={`${navBase} mb-1 bg-primary text-primary-content font-medium hover:bg-primary/90 active:scale-[0.99] ${
              open ? "" : "justify-center"
            }`}
          >
            <Plus size={22} weight="bold" />
            {open && <span className="text-sm">{t("nav.newAnalysis")}</span>}
          </button>
          <Link
            to="/app"
            title={t("nav.analyse")}
            className={`${navBase} ${onStudio ? navActive : navIdle} ${open ? "" : "justify-center"}`}
          >
            <VideoCamera size={22} weight="duotone" />
            {open && <span className="text-sm font-medium">{t("nav.analyse")}</span>}
          </Link>
          <button
            onClick={onOpenLibrary}
            className={`${navBase} ${navIdle} ${open ? "" : "justify-center"}`}
          >
            <Folders size={22} weight="duotone" />
            {open && <span className="text-sm font-medium">{t("nav.library")}</span>}
          </button>
          <Link
            to="/games"
            title={t("nav.games")}
            className={`${navBase} ${onGames ? navActive : navIdle} ${open ? "" : "justify-center"}`}
          >
            <GameController size={22} weight="duotone" />
            {open && <span className="text-sm font-medium">{t("nav.games")}</span>}
          </Link>
          <Link
            to="/explore"
            title={t("nav.explore")}
            className={`${navBase} ${onExplore ? navActive : navIdle} ${open ? "" : "justify-center"}`}
          >
            <Graph size={22} weight="duotone" />
            {open && <span className="text-sm font-medium">{t("nav.explore")}</span>}
          </Link>
          <Link
            to="/history"
            title={t("nav.history")}
            className={`${navBase} ${onHistory ? navActive : navIdle} ${open ? "" : "justify-center"}`}
          >
            <ClockCounterClockwise size={22} weight="duotone" />
            {open && <span className="text-sm font-medium">{t("nav.history")}</span>}
          </Link>
          {isAdmin && (
            <Link
              to="/admin"
              title={t("admin.nav")}
              className={`${navBase} ${onAdmin ? navActive : navIdle} ${open ? "" : "justify-center"}`}
            >
              <ShieldCheck size={22} weight="duotone" />
              {open && <span className="text-sm font-medium">{t("admin.nav")}</span>}
            </Link>
          )}
        </nav>
      </div>
      {open && (
        <div className="p-2 border-t border-border-dark px-3 flex flex-col gap-0.5">
          <p className="text-[10px] text-faint uppercase tracking-wider">{t("sidebar.version")}</p>
          <p className="text-[10px] text-faint">{t("sidebar.tagline")}</p>
        </div>
      )}
    </aside>
  );
}
