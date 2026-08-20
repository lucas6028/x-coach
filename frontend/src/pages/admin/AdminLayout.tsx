import { useState } from "react";
import {
  ArrowLeft,
  CaretDoubleLeft,
  CaretDoubleRight,
  ChatCircleText,
  Gauge,
  List,
  ShieldWarning,
  SlidersHorizontal,
  Brain,
  Graph,
  Users,
  WarningCircle,
  X,
  type Icon,
} from "@phosphor-icons/react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../../lib/auth";
import { useI18n, type TFunc } from "../../lib/i18n";
import AccountMenu from "../../components/AccountMenu";
import AdminLoading from "./AdminLoading";
import {
  RAIL_CELL_ACTIVE,
  RAIL_CELL_IDLE,
  RAIL_CTA,
  RAIL_FRAME,
  RAIL_LABEL,
  RailMark,
  railCell,
} from "../../components/railStyles";

interface NavItem {
  to: string;
  end?: boolean;
  labelKey: string;
  Icon: Icon;
}

const NAV: NavItem[] = [
  { to: "/admin", end: true, labelKey: "admin.nav.overview", Icon: Gauge },
  { to: "/admin/users", labelKey: "admin.nav.users", Icon: Users },
  { to: "/admin/line", labelKey: "admin.nav.line", Icon: ChatCircleText },
  { to: "/admin/settings/llm", labelKey: "admin.nav.settingsLlm", Icon: Brain },
  { to: "/admin/settings/rag", labelKey: "admin.nav.settingsRag", Icon: Graph },
  { to: "/admin/settings/analyze", labelKey: "admin.nav.settingsAnalyze", Icon: SlidersHorizontal },
];

// The rail: 236px with labels beside the icons, 76px icon-only when collapsed. Same two widths as
// the app shell (components/AppLayout.tsx) so the two consoles collapse to the same strip.
const WIDTH_OPEN = 236;
const WIDTH_CLOSED = 76;

// Standalone admin console shell. It now wears the SAME chrome as the signed-in app (AppLayout):
// the lavender canvas with its two colour blooms, a floating frosted nav rail, and one rounded
// glass content card. Only the rail's destinations differ — they are the admin pages, plus a way
// back to the app. The auth+admin gate runs ONCE here (RequireAuth in the router already handles
// the logged-out redirect); only when the server confirms the admin role does the child page
// (<Outlet/>) mount.
export default function AdminLayout() {
  const { t } = useI18n();
  // The admin role is probed once per session by AuthProvider; this shell just reflects that state.
  const { user, isAdmin, adminState, refreshAdmin } = useAuth();
  const [mobileNav, setMobileNav] = useState(false);
  // Defaults to open, like the app rail. The admin pages are dense forms and tables, so nothing
  // here wants the extra width badly enough to start collapsed.
  const [railOpen, setRailOpen] = useState(true);

  return (
    // `ms-shell` is the layout-root styling hook the app shell also carries (index.css).
    <div className="ms-shell relative flex h-[100dvh] w-full flex-col overflow-hidden bg-[#eef0fb] p-2 font-body text-[#211f39] sm:p-3 lg:p-[14px]">
      {/* Background wash: the reference's two soft colour blooms behind the cards. */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
        <div className="absolute -right-20 -top-20 h-[600px] w-[600px] rounded-full bg-[#e9e3ff] opacity-40 blur-[120px]" />
        <div className="absolute -left-40 top-40 h-[500px] w-[500px] rounded-full bg-[#e0e7ff] opacity-30 blur-[100px]" />
      </div>

      {/* The frame is capped in BOTH axes and centred in whatever is left, exactly as in AppLayout —
          without the height cap the content card stretches to fill a tall monitor. */}
      <div className="relative mx-auto my-auto flex h-full max-h-[940px] w-full max-w-[1500px] gap-3 lg:gap-4">
        {/* Desktop: the floating nav rail. */}
        <div className="hidden lg:flex">
          <AdminNav
            t={t}
            open={railOpen}
            width={railOpen ? WIDTH_OPEN : WIDTH_CLOSED}
            animate
            signedIn={Boolean(user)}
            onToggle={() => setRailOpen((v) => !v)}
          />
        </div>

        <main className="glass-shell relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-[28px] border border-white/80 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_22px_58px_rgba(105,112,175,0.16)] sm:p-4 lg:rounded-[32px] lg:p-5">
          {/* Top row: the drawer button, plus the console's name for the phone/tablet widths where
              the rail (which carries the same lockup) is off-canvas. Collapses to nothing at `lg`. */}
          <div className="flex shrink-0 items-center gap-2 lg:hidden">
            <button
              onClick={() => setMobileNav(true)}
              aria-label={t("nav.show")}
              className="-ml-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-[#59648f] transition-colors hover:bg-[#f5f6fb] hover:text-[#1e2142]"
            >
              <List size={22} />
            </button>
            <h1 className="min-w-0 truncate font-display text-base font-bold tracking-tight text-[#1e2142]">
              {t("admin.console.title")}
            </h1>
          </div>

          {/* The scroll lives here, INSIDE the content card — the card itself is `overflow-hidden`
              so nothing escapes its radius. The gate states render in this same container so a
              denied/errored console still sits on the card rather than outside it. */}
          <div className="min-h-0 flex-1 overflow-y-auto">
            {/* Wide enough for the LINE page's four status cards to share one row (its grid goes
                4-across at `xl`). Pages whose content is a long form set their own narrower
                measure rather than stretching label+input rows across the full column. */}
            <div className="mx-auto max-w-6xl px-1 py-6 lg:px-2 lg:py-8">
              {adminState === "loading" && <AdminLoading />}

              {adminState === "error" && (
                <div className="flex flex-col gap-3 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
                  <div className="flex items-start gap-2.5">
                    <WarningCircle size={18} className="shrink-0" />
                    <p className="font-medium">{t("admin.error")}</p>
                  </div>
                  <button
                    onClick={() => refreshAdmin()}
                    className="inline-flex w-fit items-center gap-1.5 rounded-xl border border-danger/40 px-3 py-1.5 text-sm font-semibold text-danger transition-colors hover:bg-danger/10 active:scale-[0.99]"
                  >
                    {t("admin.retry")}
                  </button>
                </div>
              )}

              {adminState === "ready" && !isAdmin && (
                <div className="flex flex-col items-center gap-4 rounded-2xl border border-dashed border-border-dark bg-content/[0.02] px-6 py-16 text-center">
                  <span className="flex h-14 w-14 items-center justify-center rounded-full bg-danger/10 text-danger">
                    <ShieldWarning size={30} weight="duotone" />
                  </span>
                  <p className="font-medium text-content">{t("admin.denied")}</p>
                  <Link
                    to="/admin/login"
                    className="inline-flex items-center gap-2 rounded-xl bg-primary px-4 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99]"
                  >
                    {t("adminLogin.switchAccount")}
                  </Link>
                </div>
              )}

              {adminState === "ready" && isAdmin && <Outlet context={{ currentUserId: user?.id }} />}
            </div>
          </div>
        </main>

        {/* Mobile: off-canvas drawer + backdrop. Rendered LAST although it sits on the left — it is
            `fixed`, so DOM order costs nothing visually. Insets match AppLayout's so the drawer
            floats on the canvas instead of running flush to the viewport edge. */}
        {mobileNav && (
          <div className="fixed inset-0 z-40 bg-black/40 lg:hidden" onClick={() => setMobileNav(false)} />
        )}
        <div
          className={`fixed inset-y-2 left-2 z-50 w-[240px] max-w-[80vw] transition-transform duration-200 ease-in-out lg:hidden ${
            mobileNav ? "translate-x-0" : "-translate-x-[110%]"
          }`}
        >
          <AdminNav
            t={t}
            open
            width={240}
            animate={false}
            signedIn={Boolean(user)}
            onNavigate={() => setMobileNav(false)}
          />
        </div>
      </div>
    </div>
  );
}

interface NavProps {
  t: TFunc;
  open: boolean;
  width: number;
  // Animate width changes (the toggle) but not the drawer, which slides rather than resizes.
  animate: boolean;
  // Drives the account cluster at the foot: the menu needs a signed-in user to render.
  signedIn: boolean;
  // Mobile drawer only: renders a brand + close row on top and closes on navigation.
  onNavigate?: () => void;
  // Desktop rail only: flips `open`. The drawer closes outright rather than shrinking to a strip.
  onToggle?: () => void;
}

// The admin console's navigation rail. Deliberately NOT a variant of components/Sidebar: that
// component owns the app's own destinations, its New-analysis CTA and the LINE auto-login branch,
// none of which belong in the console. It shares the LOOK through components/railStyles.
function AdminNav({ t, open, width, animate, signedIn, onNavigate, onToggle }: NavProps) {
  const cell = railCell(open);

  return (
    <aside
      style={{ width }}
      // The scroll lives on the inner block, NOT here: an overflow container clips absolutely
      // positioned descendants, and the account menu at the foot opens upward out of the rail.
      className={`${RAIL_FRAME} ${animate ? "transition-[width] duration-200 ease-in-out" : ""}`}
    >
      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-none rounded-t-[28px]">
        {/* Mobile drawer only: brand + close. */}
        {onNavigate && (
          <div className="flex h-16 items-center gap-2 border-b border-[#f0f1f8] px-3">
            <div className="flex min-w-0 flex-1 items-center">
              <RailMark className="!h-9 !w-9 shrink-0" />
              <span className="ml-2.5 truncate font-display font-bold tracking-tight text-[#1e2142]">
                {t("admin.console.title")}
              </span>
            </div>
            <button
              onClick={onNavigate}
              aria-label={t("nav.hide")}
              title={t("nav.hide")}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-[#59648f] transition-colors hover:bg-[#f5f6fb] hover:text-[#1e2142]"
            >
              <X size={20} />
            </button>
          </div>
        )}

        {/* The rail's own brand, linking to the console home. The wordmark goes when the rail
            collapses to its 76px strip — there is no room for it, and the link keeps its
            aria-label so the bare mark is still named. */}
        {!onNavigate && (
          <div className={`flex pt-5 pb-2 ${open ? "px-4" : "justify-center"}`}>
            <Link
              to="/admin"
              aria-label={t("admin.console.title")}
              title={t("admin.console.title")}
              className="flex min-w-0 items-center gap-2.5 rounded-xl p-1"
            >
              <RailMark />
              {open && (
                <span className="truncate font-display text-lg font-bold tracking-tight text-[#1e2142]">
                  {t("admin.console.title")}
                </span>
              )}
            </Link>
          </div>
        )}

        <nav className="flex flex-col gap-1 px-2 py-3">
          {/* The rail's primary action, in the slot (and the treatment) the app rail gives "New
              analysis": the console is somewhere you visit, so the way back out is the one thing
              here that isn't a destination. */}
          <Link
            to="/app"
            onClick={onNavigate}
            title={t("admin.nav.backToApp")}
            className={`${cell} ${RAIL_CTA}`}
          >
            <ArrowLeft size={21} weight="bold" className="shrink-0" />
            {open && <span className={`${RAIL_LABEL} font-semibold`}>{t("admin.nav.backToApp")}</span>}
          </Link>
          {NAV.map(({ to, end, labelKey, Icon: Ico }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={onNavigate}
              title={t(labelKey)}
              className={({ isActive }) => `${cell} ${isActive ? RAIL_CELL_ACTIVE : RAIL_CELL_IDLE}`}
            >
              {({ isActive }) => (
                <>
                  <Ico size={21} weight="duotone" className="shrink-0" />
                  {open && (
                    <span className={`${RAIL_LABEL} ${isActive ? "font-semibold" : "font-medium"}`}>
                      {t(labelKey)}
                    </span>
                  )}
                </>
              )}
            </NavLink>
          ))}
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

      {/* Account cluster at the foot, the same slot the app rail uses. The console is behind the
          auth gate, so there is no signed-out affordance to carry here — only the menu itself. */}
      {signedIn && (
        <div className="border-t border-[#f0f1f8] p-2">
          <AccountMenu rail={open ? "open" : "closed"} />
        </div>
      )}
    </aside>
  );
}
