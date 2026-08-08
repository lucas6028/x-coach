import {
  Barbell,
  CaretDoubleLeft,
  CaretDoubleRight,
  ClockCounterClockwise,
  Folders,
  GameController,
  GearSix,
  Plus,
  ShieldCheck,
  VideoCamera,
  X,
  type Icon,
} from "@phosphor-icons/react";
import { CircleNotch, SignIn } from "@phosphor-icons/react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";
import AccountMenu from "./AccountMenu";

// The app's own brand mark. The reference design's chevron placeholder is gone: with the top row
// carrying no lockup any more, the rail shows the real X-Coach icon.
function Mark({ className = "" }: { className?: string }) {
  return (
    <img
      src="/icon.svg"
      alt=""
      className={`h-10 w-10 rounded-xl shadow-accent ring-1 ring-black/5 ${className}`}
    />
  );
}

interface Props {
  open: boolean;
  width: number;
  // Animate width changes (toggle) but not while the user is dragging the resize handle.
  animate: boolean;
  onOpenLibrary: () => void;
  // Start a fresh studio session (clears the current analysis / routes into the studio).
  onNewAnalysis: () => void;
  // Mobile drawer only: when provided, the sidebar renders a brand + close row at the top and
  // wires the ✕ button to it. The desktop rail omits this — its brand is the mark below.
  onClose?: () => void;
  // Desktop rail only: flips `open`. Omitted by the drawer, which is always labelled and closes
  // outright rather than shrinking to a strip.
  onToggle?: () => void;
}

// The navigation rail: a floating white card, one horizontal icon + label row per destination, and
// a soft violet pill under the active one. `open` is the labelled 236px rail; collapsed drops to a
// 76px icon-only strip with the labels living in the rows' tooltips. The mobile drawer reuses the
// same component with a brand + close row on top and no collapse control.
export default function Sidebar({
  open,
  width,
  animate,
  onOpenLibrary,
  onNewAnalysis,
  onClose,
  onToggle,
}: Props) {
  const { t } = useI18n();
  const { isAdmin, user, lineAuthenticating } = useAuth();
  const { pathname } = useLocation();
  // Shared shell: highlight whichever destination the current route matches.
  const onStudio = pathname === "/app";
  const onHistory = pathname === "/history";
  const onMovements = pathname === "/movements";
  const onSettings = pathname === "/settings";
  const onAdmin = pathname === "/admin";

  // The Admin link is admin-only UX gating (the /admin page + backend re-check are the real
  // defence). `isAdmin` is resolved once per session by AuthProvider, so switching pages no longer
  // re-probes the endpoint; a non-admin (or an errored probe) leaves it false and shows no link.

  // The games hub, plus the individual game routes it links into, all light up the one Games entry.
  const onGames = pathname === "/games" || pathname === "/67" || pathname === "/ninja";

  // One rail row: icon beside its label, the whole row a rounded target. Collapsed, the label is
  // gone and the icon centres itself in the 76px strip — the row keeps its height either way, so
  // toggling only moves things horizontally.
  const cell = `w-full flex items-center gap-3 min-h-[46px] px-3 rounded-[14px] transition-colors ${
    open ? "justify-start" : "justify-center"
  }`;
  // `primary` is the reference's violet, so primary/10 over white lands on its #f3f0ff pill —
  // using the token keeps the active state one definition instead of two. The lift under the
  // selected row is the reference's own, and it is what separates "selected" from a plain hover.
  const cellActive = "bg-primary/10 text-primary shadow-[0_14px_34px_rgba(112,70,255,0.14)]";
  const cellIdle = "text-[#59648f] hover:bg-[#f8f8fb] hover:text-[#1e2142]";
  const label = "text-sm leading-none tracking-tight truncate";

  const Cell = ({ icon: Ico, text, active }: { icon: Icon; text: string; active: boolean }) => (
    <>
      <Ico size={21} weight="duotone" className="shrink-0" />
      {open && (
        <span className={`${label} ${active ? "font-semibold" : "font-medium"}`}>{text}</span>
      )}
    </>
  );

  return (
    <aside
      style={{ width }}
      // `glass-rail` — the second of the page's three blurred surfaces (shell, rail, popovers).
      // The scroll lives on the inner block, NOT here: an overflow container clips absolutely
      // positioned descendants, and the account menu at the foot opens upward out of the rail.
      className={`glass-rail h-full shrink-0 flex flex-col rounded-[28px] ${
        animate ? "transition-[width] duration-200 ease-in-out" : ""
      }`}
    >
      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-none rounded-t-[28px]">
        {/* Mobile drawer only: brand + close. */}
        {onClose && (
          <div className="h-16 flex items-center gap-2 px-3 border-b border-[#f0f1f8]">
            <div className="flex items-center min-w-0 flex-1">
              <Mark className="!h-9 !w-9 shrink-0" />
              <span className="ml-2.5 font-display font-bold tracking-tight truncate text-[#1e2142]">
                X-Coach
              </span>
            </div>
            <button
              onClick={onClose}
              aria-label={t("nav.hide")}
              title={t("nav.hide")}
              className="shrink-0 w-10 h-10 flex items-center justify-center rounded-xl text-[#59648f] hover:bg-[#f5f6fb] hover:text-[#1e2142] transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        )}

        {/* The rail mark is now the app's brand — the top row no longer carries a lockup, so this
            is the one labelled "X-Coach" in the shell and it links home. */}
        {!onClose && (
          <div className="flex justify-center pt-5 pb-2">
            <Link to="/app" aria-label="X-Coach" title="X-Coach" className="rounded-xl p-1">
              <Mark />
            </Link>
          </div>
        )}

        <nav className="flex flex-col gap-1 px-2 py-3">
          {/* Primary CTA: start a fresh analysis from anywhere in the app. */}
          <button
            onClick={onNewAnalysis}
            title={t("nav.newAnalysis")}
            className={`${cell} bg-gradient-to-br from-[#a48bff] to-[#7b5cff] text-white shadow-[0_8px_20px_rgba(123,92,255,0.3)] hover:from-[#9a80ff] hover:to-[#6e4bff] active:scale-[0.98] mb-1`}
          >
            <Plus size={21} weight="bold" className="shrink-0" />
            {open && <span className={`${label} font-semibold`}>{t("nav.newAnalysis")}</span>}
          </button>
          <Link
            to="/app"
            title={t("nav.analyse")}
            className={`${cell} ${onStudio ? cellActive : cellIdle}`}
          >
            <Cell icon={VideoCamera} text={t("nav.analyse")} active={onStudio} />
          </Link>
          <button onClick={onOpenLibrary} title={t("nav.library")} className={`${cell} ${cellIdle}`}>
            <Cell icon={Folders} text={t("nav.library")} active={false} />
          </button>
          <Link
            to="/movements"
            title={t("nav.movements")}
            className={`${cell} ${onMovements ? cellActive : cellIdle}`}
          >
            <Cell icon={Barbell} text={t("nav.movements")} active={onMovements} />
          </Link>
          <Link
            to="/history"
            title={t("nav.history")}
            className={`${cell} ${onHistory ? cellActive : cellIdle}`}
          >
            <Cell icon={ClockCounterClockwise} text={t("nav.history")} active={onHistory} />
          </Link>
          <Link
            to="/games"
            title={t("nav.games")}
            className={`${cell} ${onGames ? cellActive : cellIdle}`}
          >
            <Cell icon={GameController} text={t("nav.games")} active={onGames} />
          </Link>
          <Link
            to="/settings"
            title={t("nav.settings")}
            className={`${cell} ${onSettings ? cellActive : cellIdle}`}
          >
            <Cell icon={GearSix} text={t("nav.settings")} active={onSettings} />
          </Link>
          {isAdmin && (
            <Link
              to="/admin"
              title={t("admin.nav")}
              className={`${cell} ${onAdmin ? cellActive : cellIdle}`}
            >
              <Cell icon={ShieldCheck} text={t("admin.nav")} active={onAdmin} />
            </Link>
          )}
        </nav>
      </div>

      {/* The width toggle stays INSIDE the rail rather than pinned to the shell's edge: the shell
          is a rounded card, and anything absolutely positioned on the seam ends up outside the
          radius. Drawer variant has no toggle — it closes outright instead of shrinking. */}
      {onToggle && (
        <div className={`flex px-2 py-1 ${open ? "justify-end" : "justify-center"}`}>
          <button
            onClick={onToggle}
            aria-label={open ? t("nav.collapse") : t("nav.expand")}
            title={open ? t("nav.collapse") : t("nav.expand")}
            className="flex h-9 w-9 items-center justify-center rounded-xl text-[#59648f] transition-colors hover:bg-[#f5f6fb] hover:text-[#1e2142]"
          >
            {open ? <CaretDoubleLeft size={16} weight="bold" /> : <CaretDoubleRight size={16} weight="bold" />}
          </button>
        </div>
      )}

      {/* The account cluster now lives at the foot of the rail rather than the content card's top
          row. The same slot carries all three states, so the corner is never just empty: the
          avatar when signed in, the in-flight LINE auto-login, otherwise the way in. */}
      <div className="border-t border-[#f0f1f8] p-2">
        {user ? (
          <AccountMenu rail={open ? "open" : "closed"} />
        ) : lineAuthenticating ? (
          // Silent LINE auto-login is in flight (typically the web redirect-return): show a
          // "signing in" affordance instead of the log-in link, which would read as failed.
          <span aria-live="polite" className={`${cell} ${cellIdle} pointer-events-none`}>
            <CircleNotch size={21} weight="bold" className="shrink-0 animate-spin" />
            {open && <span className={`${label} font-medium`}>{t("account.lineSigningIn")}</span>}
          </span>
        ) : (
          <Link to="/login" title={t("account.signin")} className={`${cell} ${cellIdle}`}>
            <SignIn size={21} className="shrink-0" />
            {open && <span className={`${label} font-medium`}>{t("account.signin")}</span>}
          </Link>
        )}
      </div>
    </aside>
  );
}
