import { CircleNotch, List, SignIn } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import { useI18n } from "../lib/i18n";
import { useAuth } from "../lib/auth";
import ThemeToggle from "./ThemeToggle";
import LanguageToggle from "./LanguageToggle";
import AccountMenu from "./AccountMenu";

interface Props {
  onMenu?: () => void;
}

// The top row of the content card: account controls, and nothing else. The brand lives on the
// rail (Sidebar's mark) and every destination is a rail entry, so the row carries no lockup, no
// action pills and no rail-collapse toggle — the rail is only 84px wide, so collapsing it bought
// 20px in exchange for a permanent control. The mobile menu button stays: below `lg` the rail is
// an off-canvas drawer, and it is the only way to open it.
// Colours are the reference's own hexes; the design is light-only, so this row does not follow
// the theme toggle (which still governs token-styled pages).
export default function Header({ onMenu }: Props) {
  const { t } = useI18n();
  const { user, lineAuthenticating } = useAuth();

  const railBtn =
    "shrink-0 w-10 h-10 flex items-center justify-center rounded-2xl text-[#59648f] hover:bg-[#f5f6fb] hover:text-[#1e2142] transition-colors";

  return (
    <header className="relative z-30 flex shrink-0 items-center gap-2 sm:gap-3">
      {/* Mobile only: open the off-canvas drawer. */}
      <button onClick={onMenu} aria-label={t("nav.show")} className={`lg:hidden ${railBtn}`}>
        <List size={22} />
      </button>

      <div className="flex-1" />

      {/* Top-right controls: language, theme, account — inside the reference's rounded chip. */}
      <div className="flex items-center gap-1 shrink-0">
        <LanguageToggle />
        <ThemeToggle />
        <div className="mx-1 h-6 w-px bg-[#ebeaf6]" />
        {user ? (
          <AccountMenu />
        ) : lineAuthenticating ? (
          // Silent LINE auto-login is in flight (typically the web redirect-return): show a
          // "signing in" affordance instead of the log-in link, which would read as failed.
          <span
            aria-live="polite"
            className="flex items-center gap-1.5 h-10 px-2.5 text-sm font-medium text-[#59648f]"
          >
            <CircleNotch size={18} weight="bold" className="animate-spin" />
            <span className="hidden sm:inline">{t("account.lineSigningIn")}</span>
          </span>
        ) : (
          <Link
            to="/login"
            aria-label={t("account.signin")}
            title={t("account.signin")}
            className="glass-control flex items-center gap-1.5 h-10 px-3 rounded-full text-[#59648f] hover:text-[#1e2142] transition-colors"
          >
            <SignIn size={18} />
            <span className="hidden sm:inline text-sm font-medium">{t("account.signin")}</span>
          </Link>
        )}
      </div>
    </header>
  );
}
