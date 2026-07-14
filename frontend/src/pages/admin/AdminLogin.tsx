import { useState, type FormEvent } from "react";
import {
  CircleNotch,
  Info,
  ShieldCheck,
  WarningCircle,
} from "@phosphor-icons/react";
import { Navigate, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../../lib/auth";
import { useI18n } from "../../lib/i18n";
import LanguageToggle from "../../components/LanguageToggle";
import ThemeToggle from "../../components/ThemeToggle";

// Dedicated admin console sign-in. Standalone (NOT inside AppLayout or the AdminLayout console
// chrome): a compact centered card with email + password ONLY — no Google/OAuth. Authorization is
// still the server-side is_admin gate (AdminLayout enforces it after sign-in); this page only
// establishes a Supabase session using a dedicated admin email+password account.
export default function AdminLogin() {
  const { t } = useI18n();
  const navigate = useNavigate();
  const location = useLocation();
  const [params] = useSearchParams();
  const { user, configured, signInWithPassword } = useAuth();

  // Where to go after a successful sign-in: explicit ?redirect=, else the path RequireAuth stashed
  // in router state, else the console root. AdminLayout gates the destination regardless.
  const stateFrom = (location.state as { from?: string } | null)?.from;
  const dest = params.get("redirect") || stateFrom || "/admin";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // Already authenticated: don't force a re-auth. We can't cheaply know is_admin here, so hand off
  // to the destination and let AdminLayout's server check gate it.
  if (user) return <Navigate to={dest} replace />;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await signInWithPassword(email, password);
      navigate(dest, { replace: true });
    } catch {
      setError(t("adminLogin.error"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="relative grid min-h-[100dvh] place-items-center bg-background-dark px-5 py-10 text-content">
      {/* Corner controls */}
      <div className="absolute right-4 top-4 flex items-center gap-1">
        <LanguageToggle />
        <ThemeToggle />
      </div>

      <div className="w-full max-w-sm">
        <div className="flex flex-col items-center text-center">
          <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <ShieldCheck size={30} weight="duotone" />
          </span>
          <h1 className="mt-5 font-display text-2xl font-bold tracking-tight">
            {t("adminLogin.title")}
          </h1>
          <p className="mt-1.5 text-sm text-muted">{t("adminLogin.subtitle")}</p>
        </div>

        {!configured && (
          <div className="mt-6 flex items-start gap-2.5 rounded-xl border border-border-dark bg-content/[0.03] p-3.5 text-sm text-muted">
            <Info size={18} className="shrink-0 text-faint" />
            <span>{t("adminLogin.notConfigured")}</span>
          </div>
        )}

        <form onSubmit={submit} className="mt-7 flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="admin-email" className="text-sm font-medium text-content">
              {t("auth.email")}
            </label>
            <input
              id="admin-email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="rounded-xl border border-border-dark bg-content/[0.02] px-3.5 py-2.5 text-sm text-content outline-none transition-colors placeholder:text-faint focus:border-primary focus:bg-content/[0.04]"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="admin-password" className="text-sm font-medium text-content">
              {t("auth.password")}
            </label>
            <input
              id="admin-password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="rounded-xl border border-border-dark bg-content/[0.02] px-3.5 py-2.5 text-sm text-content outline-none transition-colors placeholder:text-faint focus:border-primary focus:bg-content/[0.04]"
            />
          </div>

          {error && (
            <div className="flex items-start gap-2.5 rounded-xl border border-danger/30 bg-danger/[0.06] p-3.5 text-sm text-danger">
              <WarningCircle size={18} className="shrink-0" />
              <span className="break-words">{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={busy || !configured}
            className="mt-1 flex items-center justify-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99] disabled:opacity-60"
          >
            {busy && <CircleNotch size={18} className="animate-spin" />}
            {t(busy ? "adminLogin.submitting" : "adminLogin.submit")}
          </button>
        </form>
      </div>
    </div>
  );
}
