import { CircleNotch, List, SignIn } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import { useI18n } from "../lib/i18n";
import { useAuth } from "../lib/auth";
import ThemeToggle from "./ThemeToggle";
import LanguageToggle from "./LanguageToggle";
import AccountMenu from "./AccountMenu";

interface Props {
  onMenu?: () => void;
  // Desktop sidebar collapse toggle. Rendered only when supplied (i.e. inside AppLayout); a
  // standalone Header omits it so its accessible name can't collide with the mobile menu button.
  onToggleSidebar?: () => void;
  sidebarOpen?: boolean;
}

export default function Header({ onMenu, onToggleSidebar, sidebarOpen = true }: Props) {
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
      {/* No page title or status line: the sidebar's active pill (bottom tabs under LIFF) is what
          says which page you're on. This spacer holds the controls at the right edge. */}
      <div className="flex-1" />

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
