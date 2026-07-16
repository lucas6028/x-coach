import { useState, type FormEvent } from "react";
import {
  ArrowCounterClockwise,
  ArrowLeft,
  CircleNotch,
  ClockCounterClockwise,
  EnvelopeSimple,
  Info,
  Lock,
  WarningCircle,
  type Icon,
} from "@phosphor-icons/react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { motion, useReducedMotion } from "motion/react";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";

type Mode = "signin" | "signup";

// Brand-side value points (why make an account): persistence is the whole reason to sign in.
const POINTS: { Icon: Icon; key: string }[] = [
  { Icon: ClockCounterClockwise, key: "auth.point1" },
  { Icon: ArrowCounterClockwise, key: "auth.point2" },
  { Icon: Lock, key: "auth.point3" },
];

// The LINE mark, simplified: the brand-green rounded square holding a white speech bubble.
function LineBubble() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
      <rect width="24" height="24" rx="6" fill="#06C755" />
      <path
        fill="#fff"
        d="M12 5.3c-4.2 0-7.6 2.77-7.6 6.18 0 3.05 2.7 5.61 6.35 6.1.25.05.42.17.38.44l-.16 1.1c-.05.33.27.58.56.42 3.15-1.73 8.07-4.13 8.07-8.06 0-3.41-3.4-6.18-7.6-6.18z"
      />
    </svg>
  );
}

// The Google "G" is a brand mark (the OAuth convention), not a hand-rolled generic icon.
function GoogleG() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62z" />
      <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.8.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 0 0 9 18z" />
      <path fill="#FBBC05" d="M3.97 10.72a5.4 5.4 0 0 1 0-3.44V4.95H.96a9 9 0 0 0 0 8.1l3.01-2.33z" />
      <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58A9 9 0 0 0 .96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z" />
    </svg>
  );
}

// Auth gateway into the studio. Lives in the app's theme-aware token system (dual-mode), with
// an asymmetric split: brand + reason-to-sign-up on the left, the form on the right.
export default function Login() {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  const navigate = useNavigate();
  const location = useLocation();
  const { user, configured, signInWithPassword, signUpWithPassword, signInWithGoogle, signInWithLine } =
    useAuth();

  const from = (location.state as { from?: string } | null)?.from ?? "/app";

  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  // Already authenticated: skip the form.
  if (user) return <Navigate to={from} replace />;

  const isSignup = mode === "signup";

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setNotice("");
    setBusy(true);
    try {
      if (isSignup) {
        const { needsConfirmation } = await signUpWithPassword(email, password);
        if (needsConfirmation) setNotice(t("auth.confirmEmail"));
        else navigate(from, { replace: true });
      } else {
        await signInWithPassword(email, password);
        navigate(from, { replace: true });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function google() {
    setError("");
    setBusy(true);
    try {
      await signInWithGoogle(); // navigates away on success
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  async function line() {
    setError("");
    setBusy(true);
    try {
      // Web: navigates away (OAuth redirect). Inside LIFF: resolves in place with a live
      // session — `user` flips and the <Navigate> above leaves this page.
      await signInWithLine();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-[100dvh] bg-background-dark text-content lg:grid-cols-2">
      {/* Brand / value panel — desktop only */}
      <aside className="relative hidden flex-col justify-between overflow-hidden border-r border-border-dark bg-surface-dark p-12 lg:flex">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -right-24 -top-24 h-80 w-80 rounded-full bg-primary/10 blur-3xl"
        />
        <Link to="/" className="flex items-center gap-2.5">
          <img src="/icon.svg" alt="" className="h-9 w-9 rounded" />
          <span className="font-display text-lg font-bold tracking-tight">X-Coach</span>
        </Link>
        <div className="relative">
          <h2 className="max-w-sm font-display text-3xl font-bold leading-tight tracking-tight">
            {t("auth.brandHeadline")}
          </h2>
          <p className="mt-3 max-w-sm leading-relaxed text-muted">{t("auth.brandSub")}</p>
          <ul className="mt-8 flex flex-col gap-4">
            {POINTS.map((p) => (
              <li key={p.key} className="flex items-center gap-3">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <p.Icon size={20} weight="duotone" />
                </span>
                <span className="text-sm text-content">{t(p.key)}</span>
              </li>
            ))}
          </ul>
        </div>
        <p className="relative font-mono text-[11px] uppercase tracking-wider text-faint">
          {t("sidebar.tagline")}
        </p>
      </aside>

      {/* Form panel */}
      <main className="flex items-center justify-center px-5 py-10 sm:px-8">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="w-full max-w-sm"
        >
          <Link
            to="/"
            className="mb-8 inline-flex items-center gap-1.5 text-sm text-muted transition-colors hover:text-content"
          >
            <ArrowLeft size={18} />
            {t("auth.back")}
          </Link>

          <h1 className="font-display text-2xl font-bold tracking-tight">
            {t(isSignup ? "auth.signUpTitle" : "auth.signInTitle")}
          </h1>
          <p className="mt-1.5 text-sm text-muted">
            {t(isSignup ? "auth.signUpSub" : "auth.signInSub")}
          </p>

          {!configured && (
            <div className="mt-6 flex items-start gap-2.5 rounded-xl border border-border-dark bg-content/[0.03] p-3.5 text-sm text-muted">
              <Info size={18} className="shrink-0 text-faint" />
              <span>{t("auth.notConfigured")}</span>
            </div>
          )}

          <form onSubmit={submit} className="mt-6 flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <label htmlFor="email" className="text-sm font-medium text-content">
                {t("auth.email")}
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="rounded-xl border border-border-dark bg-content/[0.02] px-3.5 py-2.5 text-sm text-content outline-none transition-colors placeholder:text-faint focus:border-primary focus:bg-content/[0.04]"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label htmlFor="password" className="text-sm font-medium text-content">
                {t("auth.password")}
              </label>
              <input
                id="password"
                type="password"
                autoComplete={isSignup ? "new-password" : "current-password"}
                required
                minLength={6}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="rounded-xl border border-border-dark bg-content/[0.02] px-3.5 py-2.5 text-sm text-content outline-none transition-colors placeholder:text-faint focus:border-primary focus:bg-content/[0.04]"
              />
            </div>

            {error && (
              <div className="flex items-start gap-2.5 rounded-xl border border-danger/30 bg-danger/[0.06] p-3.5 text-sm text-danger">
                <WarningCircle size={18} className="shrink-0" />
                <div className="min-w-0">
                  <p className="font-medium">{t("auth.errorTitle")}</p>
                  <p className="mt-0.5 break-words text-danger/80">{error}</p>
                </div>
              </div>
            )}

            {notice && (
              <div className="flex items-start gap-2.5 rounded-xl border border-primary/30 bg-primary/[0.06] p-3.5 text-sm text-primary">
                <EnvelopeSimple size={18} className="shrink-0" />
                <span>{notice}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={busy || !configured}
              className="mt-1 flex items-center justify-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99] disabled:opacity-60"
            >
              {busy && <CircleNotch size={18} className="animate-spin" />}
              {t(isSignup ? "auth.signUpBtn" : "auth.signInBtn")}
            </button>
          </form>

          <div className="my-5 flex items-center gap-3 text-[11px] uppercase tracking-wider text-faint">
            <span className="h-px flex-1 bg-border-dark" />
            {t("auth.or")}
            <span className="h-px flex-1 bg-border-dark" />
          </div>

          <button
            type="button"
            onClick={google}
            disabled={busy || !configured}
            className="flex w-full items-center justify-center gap-2.5 rounded-xl border border-border-dark bg-surface-dark px-5 py-2.5 text-sm font-medium text-content transition-colors hover:bg-content/[0.05] active:scale-[0.99] disabled:opacity-60"
          >
            <GoogleG />
            {t("auth.google")}
          </button>

          <button
            type="button"
            onClick={line}
            disabled={busy || !configured}
            className="mt-3 flex w-full items-center justify-center gap-2.5 rounded-xl border border-border-dark bg-surface-dark px-5 py-2.5 text-sm font-medium text-content transition-colors hover:bg-content/[0.05] active:scale-[0.99] disabled:opacity-60"
          >
            <LineBubble />
            {t("auth.lineBtn")}
          </button>

          <p className="mt-6 text-center text-sm text-muted">
            {t(isSignup ? "auth.haveAccount" : "auth.noAccount")}{" "}
            <button
              type="button"
              onClick={() => {
                setMode(isSignup ? "signin" : "signup");
                setError("");
                setNotice("");
              }}
              className="font-medium text-primary hover:underline"
            >
              {t(isSignup ? "auth.toSignin" : "auth.toSignup")}
            </button>
          </p>

          <p className="mt-4 text-center">
            <Link to="/app" className="text-sm text-faint transition-colors hover:text-muted">
              {t("auth.demoLink")}
            </Link>
          </p>
        </motion.div>
      </main>
    </div>
  );
}
