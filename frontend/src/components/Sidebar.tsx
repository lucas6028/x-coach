import { Link } from "react-router-dom";
import ThemeToggle from "./ThemeToggle";
import LanguageToggle from "./LanguageToggle";
import { useI18n } from "../lib/i18n";
import { useAuth } from "../lib/auth";

interface Props {
  open: boolean;
  width: number;
  // Animate width changes (toggle) but not while the user is dragging the resize handle.
  animate: boolean;
  onToggle: () => void;
  onOpenLibrary: () => void;
}

// Labelled bar when `open`, slim icon rail when collapsed — mirrors demo/index.html.
// The menu toggle lives in the top row so it stays reachable in either state.
// Width is driven from the parent so it can be dragged to resize.
export default function Sidebar({ open, width, animate, onToggle, onOpenLibrary }: Props) {
  const { t } = useI18n();
  const { user, signOut } = useAuth();
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
            <span className="material-symbols-outlined">{open ? "menu_open" : "menu"}</span>
          </button>
          {open && (
            <div className="flex items-center min-w-0">
              <img src="/icon.svg" alt="" className="w-8 h-8 rounded shrink-0" />
              <span className="ml-2 font-bold tracking-wide truncate">X-Coach</span>
            </div>
          )}
        </div>
        <nav className="flex flex-col gap-1 p-2">
          <a
            className={`flex items-center gap-3 px-3 py-3 rounded-lg bg-primary/10 text-primary border border-primary/20 ${
              open ? "" : "justify-center"
            }`}
            href="#"
          >
            <span className="material-symbols-outlined">video_camera_front</span>
            {open && <span className="text-sm font-medium">{t("nav.analyse")}</span>}
          </a>
          <button
            onClick={onOpenLibrary}
            className={`flex items-center gap-3 px-3 py-3 rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors ${
              open ? "" : "justify-center"
            }`}
          >
            <span className="material-symbols-outlined">folder_data</span>
            {open && <span className="text-sm font-medium">{t("nav.library")}</span>}
          </button>
          <Link
            to="/history"
            title={t("nav.history")}
            className={`flex items-center gap-3 px-3 py-3 rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors ${
              open ? "" : "justify-center"
            }`}
          >
            <span className="material-symbols-outlined">history</span>
            {open && <span className="text-sm font-medium">{t("nav.history")}</span>}
          </Link>
        </nav>
      </div>
      <div className="p-2 border-t border-border-dark flex flex-col gap-2">
        {/* Account: signed-in identity + sign out, or a sign-in entry point. */}
        {user ? (
          open ? (
            <div className="flex items-center gap-2 px-1">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                <span className="material-symbols-outlined text-lg">person</span>
              </span>
              <span className="min-w-0 flex-1 truncate text-xs text-muted" title={user.email ?? ""}>
                {user.email}
              </span>
              <button
                onClick={() => void signOut()}
                aria-label={t("account.signout")}
                title={t("account.signout")}
                className="shrink-0 flex h-8 w-8 items-center justify-center rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors"
              >
                <span className="material-symbols-outlined text-lg">logout</span>
              </button>
            </div>
          ) : (
            <button
              onClick={() => void signOut()}
              aria-label={t("account.signout")}
              title={t("account.signout")}
              className="w-10 h-10 mx-auto flex items-center justify-center rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors"
            >
              <span className="material-symbols-outlined">logout</span>
            </button>
          )
        ) : (
          <Link
            to="/login"
            aria-label={t("account.signin")}
            title={t("account.signin")}
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors ${
              open ? "" : "justify-center px-0"
            }`}
          >
            <span className="material-symbols-outlined">login</span>
            {open && <span className="text-sm font-medium">{t("account.signin")}</span>}
          </Link>
        )}
        <LanguageToggle expanded={open} />
        <ThemeToggle expanded={open} />
        {open && (
          <div className="px-1 flex flex-col gap-0.5">
            <p className="text-[10px] text-faint uppercase tracking-wider">{t("sidebar.version")}</p>
            <p className="text-[10px] text-faint">{t("sidebar.tagline")}</p>
          </div>
        )}
      </div>
    </aside>
  );
}
