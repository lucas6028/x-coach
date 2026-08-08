import { useState } from "react";
import {
  ArrowLeft,
  ChatCircleText,
  CircleNotch,
  Gauge,
  List,
  ShieldCheck,
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

// Standalone admin console shell. Distinct from the user-facing AppLayout: its own sidebar lists the
// admin pages, plus a "back to app" link and the shared language / theme / account controls. The
// auth+admin gate runs ONCE here (RequireAuth in the router already handles the logged-out redirect);
// only when the server confirms the admin role does the child page (<Outlet/>) mount.
export default function AdminLayout() {
  const { t } = useI18n();
  // The admin role is probed once per session by AuthProvider; this shell just reflects that state.
  const { user, isAdmin, adminState, refreshAdmin } = useAuth();
  const [mobileNav, setMobileNav] = useState(false);

  return (
    <div className="h-[100dvh] w-full flex bg-background-dark text-content overflow-hidden">
      {/* Desktop sidebar */}
      <div className="hidden lg:flex shrink-0">
        <AdminNav t={t} />
      </div>

      {/* Mobile drawer + backdrop */}
      {mobileNav && (
        <div className="fixed inset-0 z-30 bg-black/50 lg:hidden" onClick={() => setMobileNav(false)} />
      )}
      <div
        className={`fixed inset-y-0 left-0 z-40 w-[260px] max-w-[80vw] transition-transform duration-200 ease-in-out lg:hidden ${
          mobileNav ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <AdminNav t={t} onNavigate={() => setMobileNav(false)} />
      </div>

      <main className="flex-1 flex flex-col min-w-0 min-h-0">
        {/* Top bar: mobile menu + language / theme / account. */}
        <header className="relative z-20 h-16 shrink-0 border-b border-border-dark bg-background-dark/95 backdrop-blur flex items-center gap-2 justify-between px-4 lg:px-6">
          <button
            onClick={() => setMobileNav(true)}
            aria-label={t("nav.show")}
            className="lg:hidden shrink-0 -ml-1 w-10 h-10 flex items-center justify-center rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors"
          >
            <List size={22} />
          </button>
          <h1 className="flex flex-1 items-center gap-2 min-w-0 text-content text-base lg:text-lg font-bold tracking-tight truncate">
            <ShieldCheck size={20} weight="duotone" className="shrink-0 text-primary" />
            <span className="truncate">{t("admin.console.title")}</span>
          </h1>
          {/* Language moved behind this avatar (the settings dialog it opens) along with the rest
              of the app's chrome; the theme picker is gone entirely. */}
          <div className="flex items-center gap-1 shrink-0">
            <AccountMenu />
          </div>
        </header>

        <div className="flex-1 min-h-0 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 py-8 lg:px-6 lg:py-12">
            {adminState === "loading" && (
              <div className="grid place-items-center py-24 text-muted">
                <CircleNotch size={28} className="animate-spin" />
                <span className="sr-only">{t("admin.loading")}</span>
              </div>
            )}

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
    </div>
  );
}

// The admin console's own navigation rail (used for both the desktop sidebar and the mobile drawer).
function AdminNav({ t, onNavigate }: { t: TFunc; onNavigate?: () => void }) {
  const base = "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors";
  const active = "bg-primary/10 text-primary border border-primary/20";
  const idle = "text-muted hover:bg-content/5 hover:text-content";
  return (
    <aside className="h-full w-[240px] shrink-0 border-r border-border-dark bg-surface-dark flex flex-col justify-between overflow-hidden">
      <div>
        <div className="h-16 flex items-center gap-2 px-4 border-b border-border-dark">
          <img src="/icon.svg" alt="" className="w-8 h-8 rounded shrink-0" />
          <span className="font-bold tracking-wide truncate">{t("admin.console.title")}</span>
          {onNavigate && (
            <button
              onClick={onNavigate}
              aria-label={t("nav.hide")}
              className="ml-auto w-9 h-9 flex items-center justify-center rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors lg:hidden"
            >
              <X size={18} />
            </button>
          )}
        </div>
        <nav className="flex flex-col gap-1 p-2">
          {NAV.map(({ to, end, labelKey, Icon: Ico }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={onNavigate}
              className={({ isActive }) => `${base} ${isActive ? active : idle}`}
            >
              <Ico size={20} weight="duotone" />
              <span>{t(labelKey)}</span>
            </NavLink>
          ))}
        </nav>
      </div>
      <div className="p-2 border-t border-border-dark">
        <Link
          to="/app"
          onClick={onNavigate}
          className={`${base} ${idle}`}
        >
          <ArrowLeft size={20} />
          <span>{t("admin.nav.backToApp")}</span>
        </Link>
      </div>
    </aside>
  );
}
