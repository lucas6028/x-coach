import { List, SignIn } from "@phosphor-icons/react";
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
}

export default function Header({ analysis, loading, title, onMenu }: Props) {
  const { t } = useI18n();
  const { user } = useAuth();
  return (
    <header className="relative z-30 h-16 shrink-0 border-b border-border-dark bg-background-dark/95 backdrop-blur flex items-center gap-2 justify-between px-4 lg:px-6">
      <button
        onClick={onMenu}
        aria-label={t("nav.show")}
        className="lg:hidden shrink-0 -ml-1 w-10 h-10 flex items-center justify-center rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors"
      >
        <List size={22} />
      </button>
      <div className="flex flex-1 flex-col min-w-0">
        <h1 className="text-content text-base lg:text-lg font-bold tracking-tight truncate">
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
