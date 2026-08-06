import { CircleNotch, List, SignIn } from "@phosphor-icons/react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { useI18n } from "../lib/i18n";
import { useAuth } from "../lib/auth";
import ThemeToggle from "./ThemeToggle";
import LanguageToggle from "./LanguageToggle";
import AccountMenu from "./AccountMenu";

interface Props {
  onMenu?: () => void;
  /** The page's own header (breadcrumb, title, its controls), rendered between the drawer button
   *  and the account cluster. Supplying it puts the page's controls and the account controls in
   *  ONE flex row, which is the only arrangement in which they cannot overlap — the alternative,
   *  floating the account cluster over the page's header band, ran the studio's movement and
   *  precision cards under the avatar. */
  children?: ReactNode;
}

// The top row of the content card: account controls, and nothing else. The brand lives on the
// rail (Sidebar's mark) and every destination is a rail entry, so the row carries no lockup, no
// action pills and no rail-collapse toggle — the rail is only 84px wide, so collapsing it bought
// 20px in exchange for a permanent control. The mobile menu button stays: below `lg` the rail is
// an off-canvas drawer, and it is the only way to open it.
// Colours are the reference's own hexes; the design is light-only, so this row does not follow
// the theme toggle (which still governs token-styled pages).
export default function Header({ onMenu, children }: Props) {
  const { t } = useI18n();
  const { user, lineAuthenticating } = useAuth();

  const railBtn =
    "shrink-0 w-10 h-10 flex items-center justify-center rounded-2xl text-[#59648f] hover:bg-[#f5f6fb] hover:text-[#1e2142] transition-colors";

  return (
    // `items-start`: the page header can be three lines tall (breadcrumb, title, subtitle) while
    // the account cluster is one control high, and the cluster belongs at the top of that band.
    <header className="relative z-30 flex shrink-0 items-start gap-2 sm:gap-3">
      {/* Mobile only: open the off-canvas drawer. */}
      <button onClick={onMenu} aria-label={t("nav.show")} className={`lg:hidden ${railBtn}`}>
        <List size={22} />
      </button>

      {/* The page's own header, or just a spacer when a page supplies none. */}
      <div className="min-w-0 flex-1">{children}</div>

      {/* Top-right controls: language, theme, account — inside the reference's rounded chip. */}
      <div className="flex shrink-0 items-center gap-1">
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
