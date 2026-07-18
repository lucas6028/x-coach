import { CircleNotch, List, SignIn } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import type { Analysis } from "../api";
import { useI18n, viewLabel } from "../lib/i18n";
import { useAuth } from "../lib/auth";
import ThemeToggle from "./ThemeToggle";
import LanguageToggle from "./LanguageToggle";
import AccountMenu from "./AccountMenu";

interface Props {
  analysis: Analysis | null;
  loading: boolean;
  // A plain page title (History/Settings). When set, the navbar shows just this name and drops the
  // analysis status line; the studio leaves it unset to show the session + status pill.
  title?: string;
  onMenu?: () => void;
  // Desktop sidebar collapse toggle. Rendered only when supplied (i.e. inside AppLayout); a
  // standalone Header omits it so its accessible name can't collide with the mobile menu button.
  onToggleSidebar?: () => void;
  sidebarOpen?: boolean;
}

export default function Header({ analysis, loading, title, onMenu, onToggleSidebar, sidebarOpen = true }: Props) {
  const { t } = useI18n();
  const { user, lineAuthenticating } = useAuth();
  return (
    <header className="relative z-30 h-16 shrink-0 border-b border-border-dark bg-surface flex items-center gap-2 sm:gap-3 px-3 lg:px-4">
      {/* Mobile: open the off-canvas drawer */}
      <button
        onClick={onMenu}
        aria-label={t("nav.show")}
        className="lg:hidden shrink-0 w-10 h-10 flex items-center justify-center rounded-xl text-muted hover:bg-content/5 hover:text-content transition-colors"
      >
        <List size={22} />
      </button>
      {/* Desktop: collapse the sidebar rail (this is the navbar's role in the reference layout) */}
      {onToggleSidebar && (
        <button
          onClick={onToggleSidebar}
          aria-label={sidebarOpen ? t("nav.hide") : t("nav.show")}
          title={sidebarOpen ? t("nav.hide") : t("nav.show")}
          className="hidden lg:flex shrink-0 w-10 h-10 items-center justify-center rounded-xl text-muted hover:bg-content/5 hover:text-content transition-colors"
        >
          <List size={20} />
        </button>
      )}
      {/* Brand lockup — lives in the full-width navbar, above where the sidebar begins */}
      <Link to="/app" aria-label="X-Coach" className="flex items-center gap-2.5 shrink-0">
        <img src="/icon.svg" alt="" className="w-9 h-9 rounded-xl shadow-accent ring-1 ring-black/5" />
        <span className="hidden sm:block font-display font-bold tracking-tight">X-Coach</span>
      </Link>
      <div className="hidden sm:block h-6 w-px bg-border-dark mx-1 shrink-0" />
      <div className="flex flex-1 flex-col min-w-0">
        <h1 className="text-content text-sm lg:text-base font-semibold tracking-tight truncate">
          {title ?? (analysis ? t("header.session", { id: analysis.video_id }) : t("header.title"))}
        </h1>
        {!title && (
          <div className="flex items-center gap-2 text-[11px] text-muted font-mono">
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                loading ? "bg-yellow-400 animate-pulse" : analysis ? "bg-green-500" : "bg-faint"
              }`}
            />
            {loading
              ? t("header.processing")
              : analysis
                ? t("header.complete")
                : t("header.awaiting")}
            {analysis && (
              <>
                <span className="text-faint">|</span>
                <span>{t("header.view", { type: viewLabel(t, analysis.view.view_type) })}</span>
                <span className="text-faint hidden sm:inline">|</span>
                <span className="hidden sm:inline uppercase">{analysis.source}</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Top-right controls: language, theme, account. */}
      <div className="flex items-center gap-1 shrink-0">
        <LanguageToggle />
        <ThemeToggle />
        <div className="mx-1 h-6 w-px bg-border-dark" />
        {user ? (
          <AccountMenu />
        ) : lineAuthenticating ? (
          // Silent LINE auto-login is in flight (typically the web redirect-return): show a
          // "signing in" affordance instead of the log-in link, which would read as failed.
          <span
            aria-live="polite"
            className="flex items-center gap-1.5 h-10 px-2.5 text-sm font-medium text-muted"
          >
            <CircleNotch size={18} weight="bold" className="animate-spin" />
            <span className="hidden sm:inline">{t("account.lineSigningIn")}</span>
          </span>
        ) : (
          <Link
            to="/login"
            aria-label={t("account.signin")}
            title={t("account.signin")}
            className="flex items-center gap-1.5 h-10 px-2.5 rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors"
          >
            <SignIn size={20} />
            <span className="hidden sm:inline text-sm font-medium">{t("account.signin")}</span>
          </Link>
        )}
      </div>
    </header>
  );
}
