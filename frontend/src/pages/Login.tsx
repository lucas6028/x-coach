import { useState, type FormEvent, type ReactNode } from "react";
import {
  Barbell,
  CircleNotch,
  EnvelopeSimple,
  Eye,
  EyeSlash,
  Info,
  Lock,
  SignIn,
  Sparkle,
  TrendUp,
  WarningCircle,
  type Icon,
} from "@phosphor-icons/react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";
import { useLiffContext } from "../lib/liffContext";
import LineLogo from "../components/LineLogo";

type Mode = "signin" | "signup";

// The Google "G" is a brand mark (the OAuth convention), not a hand-rolled generic icon.
function GoogleG() {
  return (
    <svg width="20" height="20" viewBox="0 0 18 18" aria-hidden="true">
      <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62z" />
      <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.8.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 0 0 9 18z" />
      <path fill="#FBBC05" d="M3.97 10.72a5.4 5.4 0 0 1 0-3.44V4.95H.96a9 9 0 0 0 0 8.1l3.01-2.33z" />
      <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58A9 9 0 0 0 .96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z" />
    </svg>
  );
}

// One of the three floating capability cards on the brand stage. `extra` is the card's little
// data ornament (sparkline / progress bar / bar chart) — the thing that makes it read as a
// screenshot of the product rather than as a feature bullet.
function StageCard({
  className,
  Glyph,
  title,
  body,
  extra,
}: {
  className: string;
  Glyph: Icon;
  title: string;
  body: string;
  extra?: ReactNode;
}) {
  return (
    <div
      className={`lgn-card absolute z-[7] w-[214px] rounded-2xl border border-white/95 bg-white/[0.88] p-4 pb-3.5 shadow-card backdrop-blur-[10px] ${className}`}
    >
      <div className="flex gap-2.5">
        <span className="grid h-[34px] w-[34px] shrink-0 place-items-center rounded-[10px] bg-[#eee9ff] text-primary">
          <Glyph size={21} weight="duotone" />
        </span>
        <div className="min-w-0">
          <strong className="block text-xs font-bold leading-snug text-content">{title}</strong>
          <span className="mt-1 block text-[10.5px] leading-[1.5] text-muted">{body}</span>
        </div>
      </div>
      {extra}
    </div>
  );
}

// Auth gateway into the studio. Ported from the login-page design study: a single rounded shell
// holding an illustrated brand stage on the left and the form on the right. The stage is desktop
// only — its callouts are absolutely positioned in the stage's own coordinate space, which has no
// sensible phone equivalent, and the LIFF path lands here on small screens.
export default function Login() {
  const { t } = useI18n();
  const navigate = useNavigate();
  const location = useLocation();
  const { user, configured, signInWithPassword, signUpWithPassword, signInWithGoogle, signInWithLine } =
    useAuth();
  const { ready, isInClient } = useLiffContext();

  const from = (location.state as { from?: string } | null)?.from ?? "/app";

  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  // Already authenticated: skip the form.
  if (user) return <Navigate to={from} replace />;
  // Same reasoning as Landing: the in-client guess is a synchronous heuristic, not a confirmed
  // LIFF context (see lib/liffContext) — a shared link opened in LINE's in-app browser matches
  // the guess but isn't really LIFF. Hold on a neutral loading state rather than firing the
  // irreversible redirect below on an unconfirmed guess. Same idiom as RequireAuth's
  // session-check placeholder (not LumenLoader's "scan" analysis narration — nothing in the
  // analysis pipeline is running here), so the app's waits read as one system.
  if (!ready && isInClient) {
    return (
      <div className="grid min-h-[100dvh] place-items-center bg-background text-muted" role="status">
        <CircleNotch size={24} className="animate-spin" />
        <span className="sr-only">{t("loader.neutral")}</span>
      </div>
    );
  }
  // Inside LINE the silent LIFF token exchange (see lib/auth's auto-login effect) is already
  // running — showing a sign-in form would suggest it failed. Preserve the query string for the
  // same reason as Landing: LINE can append liff.state here too on a direct /login deep link.
  if (isInClient) return <Navigate to={`/app${location.search}`} replace />;

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
      // Web (external browser): navigates away to LINE's login redirect, then finishes on
      // return. Inside LIFF: resolves in place with a live session — `user` flips and the
      // <Navigate> above leaves this page.
      await signInWithLine();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const fieldWrap =
    "flex h-[52px] items-center rounded-[14px] border border-border bg-white/70 transition-colors focus-within:border-primary focus-within:ring-[3px] focus-within:ring-primary/10";
  const fieldInput =
    "h-full w-full min-w-0 bg-transparent px-3.5 text-[15px] text-content outline-none placeholder:text-faint";
  const socialButton =
    "flex h-[52px] w-full items-center justify-center gap-3 rounded-[13px] border border-border bg-white text-[15px] font-semibold text-content shadow-[0_5px_11px_rgba(35,46,97,0.035)] transition-colors hover:border-primary/40 hover:bg-primary/[0.03] active:scale-[0.995] disabled:opacity-60";

  return (
    <div className="min-h-[100dvh] bg-background xl:p-4">
      {/* Below `lg` the brand stage is gone, so the shell keeps the lavender canvas and the form
          reads as a card on it; from `lg` up the shell itself is the white surface the stage
          paints on. */}
      <div className="relative grid min-h-[100dvh] grid-rows-[1fr_auto] overflow-hidden bg-background lg:bg-white xl:min-h-[calc(100dvh-2rem)] xl:rounded-[2rem] xl:border xl:border-white xl:shadow-card">
        <Link
          to="/"
          aria-label={t("auth.brandHome")}
          className="absolute left-6 top-6 z-20 inline-flex items-center gap-3.5 lg:left-11 lg:top-9"
        >
          <img src="/icon.svg" alt="" width={52} height={52} className="h-[52px] w-[52px] rounded-[12px] shadow-accent" />
          <span className="font-display text-2xl font-extrabold tracking-tight text-content">X-Coach</span>
        </Link>

        {/* 61/39 split, matching the design study the percentage offsets below were measured in. */}
        <div className="grid pt-28 lg:grid-cols-[minmax(0,1.564fr)_minmax(430px,1fr)] lg:pt-[4.8rem]">
          {/* ── Brand stage ─────────────────────────────────────────────── */}
          <section
            aria-labelledby="lgn-hero-title"
            className="lgn-stage relative isolate hidden min-h-[716px] overflow-hidden lg:block"
          >
            {/* -z-[1], not lower: the stage's floor gradient sits at -3/-4, and the mock shows the
                dot fields ON the violet corner rather than buried under it. */}
            <div className="lgn-dots absolute -z-[1] h-[126px] w-[126px] opacity-50 right-[10%] top-11" aria-hidden="true" />
            <div className="lgn-dots absolute -z-[1] h-[126px] w-[126px] opacity-50 bottom-[18px] left-[4%]" aria-hidden="true" />
            <span className="lgn-orb absolute right-[19%] top-[48%] h-2 w-2 rounded-full border-2 border-[#ad92ff] opacity-70" aria-hidden="true" />
            <span className="lgn-orb lgn-orb-b absolute right-[4.5%] top-[56%] h-[9px] w-[9px] rounded-full border-2 border-[#ad92ff] opacity-70" aria-hidden="true" />

            <div className="lgn-copy absolute left-[13%] top-[72px] z-[6]">
              {/* h2, not h1: the stage is desktop-only, so making it the document's h1 would leave
                  phones with no h1 at all. The page's subject is the form, and its title keeps the
                  h1 it has always had. */}
              <h2
                id="lgn-hero-title"
                className="font-display text-[clamp(38px,3vw,49px)] font-extrabold leading-[1.13] tracking-[-0.03em] text-content"
              >
                {t("auth.heroLine1")}
                <br />
                {t("auth.heroLine2Lead")} <span className="text-primary">{t("auth.heroLine2Accent")}</span>
              </h2>
              {/* Capped so the copy wraps before it reaches the Form-score card, which floats into
                  this line's track. The cap widens with the stage. */}
              <p className="mt-4 max-w-[24rem] text-base leading-relaxed text-muted 2xl:max-w-[29rem]">
                {t("auth.heroSub")}
              </p>
            </div>

            <img
              src="/assets/squat-hero.webp"
              width={920}
              height={814}
              alt=""
              className="lgn-art absolute bottom-[2.7%] left-[29.5%] z-[2] h-[61%] w-[56%] object-contain mix-blend-multiply"
            />
            <div
              aria-hidden="true"
              className="lgn-rings absolute bottom-[2.8%] left-[26%] z-[1] h-[13%] w-[54%] rounded-[50%] border border-white/90 shadow-[0_0_40px_rgba(121,80,255,0.2)]"
            >
              <i />
              <i />
              <i />
            </div>

            <StageCard
              className="left-[10%] top-[287px]"
              Glyph={Sparkle}
              title={t("auth.card1Title")}
              body={t("auth.card1Body")}
              extra={
                <svg className="mt-1 w-[122px] translate-x-[44px]" viewBox="0 0 122 32" fill="none" aria-hidden="true">
                  <path d="M1 27.5 17 20l13 3.5 15-7 12 3 14-6 14 4 13-8 11 6L121 4" stroke="#9b80ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M1 31h120" stroke="#e5e1ff" strokeWidth="1" />
                  <circle cx="121" cy="4" r="2.2" fill="#7a52ff" />
                </svg>
              }
            />

            {/* Form score — the one card that shows a verdict rather than a capability. */}
            <div className="lgn-card lgn-card-b absolute right-[7%] top-[190px] z-[7] w-[184px] rounded-2xl border border-white/95 bg-white/[0.88] p-4 shadow-card backdrop-blur-[10px]">
              <strong className="block text-xs font-bold leading-snug text-content">{t("auth.scoreTitle")}</strong>
              <div className="mt-3 flex items-center gap-3.5">
                <div className="lgn-ring relative grid h-[64px] w-[64px] shrink-0 place-items-center rounded-full">
                  <span className="z-[1] text-[17px] font-extrabold text-content">100%</span>
                </div>
                <div>
                  <b className="block text-sm font-bold text-[#0fbe50]">{t("auth.scoreVerdict")}</b>
                  <small className="mt-1 block text-[10px] leading-tight text-muted">{t("auth.scoreNote")}</small>
                </div>
              </div>
            </div>

            <StageCard
              className="lgn-card-c bottom-[148px] left-[14%]"
              Glyph={Barbell}
              title={t("auth.card2Title")}
              body={t("auth.card2Body")}
              extra={
                <div className="mt-3 h-[5px] w-full overflow-hidden rounded-full bg-[#eeeefe]" aria-hidden="true">
                  <span className="block h-full w-[34%] rounded-full bg-gradient-to-r from-[#6f42f7] to-[#9d7bff]" />
                </div>
              }
            />

            <StageCard
              className="lgn-card-d bottom-[150px] right-[6%]"
              Glyph={TrendUp}
              title={t("auth.card3Title")}
              body={t("auth.card3Body")}
              extra={
                <div className="mt-2 flex h-[33px] items-end justify-end gap-[18px] pr-1" aria-hidden="true">
                  <i className="h-[7px] w-[7px] rounded-t-sm bg-gradient-to-b from-[#bda9ff] to-[#8967fa]" />
                  <i className="h-[20px] w-[7px] rounded-t-sm bg-gradient-to-b from-[#bda9ff] to-[#8967fa]" />
                  <i className="h-[31px] w-[7px] rounded-t-sm bg-gradient-to-b from-[#bda9ff] to-[#8967fa]" />
                  <i className="h-[15px] w-[7px] rounded-t-sm bg-gradient-to-b from-[#bda9ff] to-[#8967fa]" />
                  <i className="h-[36px] w-[7px] rounded-t-sm bg-gradient-to-b from-[#bda9ff] to-[#8967fa]" />
                </div>
              }
            />
          </section>

          {/* ── Form panel ──────────────────────────────────────────────── */}
          <section className="flex items-center justify-center px-5 pb-10 sm:px-8 lg:px-6">
            <div className="lgn-panel w-full max-w-[554px] rounded-[28px] border border-border bg-white px-6 py-9 shadow-card sm:px-10">
              <h1 className="font-display text-[30px] font-extrabold tracking-[-0.03em] text-content">
                {t(isSignup ? "auth.signUpTitle" : "auth.signInTitle")}
              </h1>
              <p className="mt-2 text-base text-muted">
                {t(isSignup ? "auth.signUpSub" : "auth.signInSub")}
              </p>

              {!configured && (
                <div className="mt-6 flex items-start gap-2.5 rounded-xl border border-border bg-background p-3.5 text-sm text-muted">
                  <Info size={18} className="shrink-0 text-faint" />
                  <span>{t("auth.notConfigured")}</span>
                </div>
              )}

              <form onSubmit={submit} className="mt-7">
                <label htmlFor="email" className="mb-2 block text-sm font-bold text-content">
                  {t("auth.email")}
                </label>
                <div className={fieldWrap}>
                  <EnvelopeSimple size={21} className="ml-4 shrink-0 text-faint" />
                  <input
                    id="email"
                    type="email"
                    autoComplete="email"
                    required
                    placeholder="you@example.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className={fieldInput}
                  />
                </div>

                <label htmlFor="password" className="mb-2 mt-5 block text-sm font-bold text-content">
                  {t("auth.password")}
                </label>
                <div className={fieldWrap}>
                  <Lock size={21} className="ml-4 shrink-0 text-faint" />
                  <input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    autoComplete={isSignup ? "new-password" : "current-password"}
                    required
                    minLength={6}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className={fieldInput}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((v) => !v)}
                    aria-label={t(showPassword ? "auth.hidePassword" : "auth.showPassword")}
                    className="grid h-full w-12 shrink-0 place-items-center rounded-r-[14px] text-faint transition-colors hover:text-muted"
                  >
                    {showPassword ? <EyeSlash size={21} /> : <Eye size={21} />}
                  </button>
                </div>

                {error && (
                  <div className="mt-5 flex items-start gap-2.5 rounded-xl border border-danger/30 bg-danger/[0.06] p-3.5 text-sm text-danger">
                    <WarningCircle size={18} className="shrink-0" />
                    <div className="min-w-0">
                      <p className="font-medium">{t("auth.errorTitle")}</p>
                      <p className="mt-0.5 break-words text-danger/80">{error}</p>
                    </div>
                  </div>
                )}

                {notice && (
                  <div className="mt-5 flex items-start gap-2.5 rounded-xl border border-primary/30 bg-primary/[0.06] p-3.5 text-sm text-primary">
                    <EnvelopeSimple size={18} className="shrink-0" />
                    <span>{notice}</span>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={busy || !configured}
                  className="mt-7 flex h-[54px] w-full items-center justify-center gap-3 rounded-[13px] bg-gradient-to-r from-[#7447ff] to-[#6d3cf5] text-base font-bold text-primary-content shadow-accent transition-transform hover:-translate-y-0.5 active:translate-y-0 disabled:translate-y-0 disabled:opacity-60"
                >
                  {busy ? <CircleNotch size={20} className="animate-spin" /> : <SignIn size={20} />}
                  {t(isSignup ? "auth.signUpBtn" : "auth.signInBtn")}
                </button>
              </form>

              <div className="my-6 flex items-center gap-4 text-[13px] text-muted">
                <span className="h-px flex-1 bg-border" />
                {t("auth.orContinue")}
                <span className="h-px flex-1 bg-border" />
              </div>

              <button type="button" onClick={google} disabled={busy || !configured} className={socialButton}>
                <GoogleG />
                {t("auth.google")}
              </button>

              <button type="button" onClick={line} disabled={busy || !configured} className={`mt-3 ${socialButton}`}>
                <LineLogo />
                {t("auth.lineBtn")}
              </button>

              <p className="mt-7 text-center text-[15px] text-muted">
                {t(isSignup ? "auth.haveAccount" : "auth.noAccount")}{" "}
                <button
                  type="button"
                  onClick={() => {
                    setMode(isSignup ? "signin" : "signup");
                    setError("");
                    setNotice("");
                  }}
                  className="font-semibold text-primary hover:underline"
                >
                  {t(isSignup ? "auth.toSignin" : "auth.toSignup")}
                </button>
              </p>

              <p className="mt-3 text-center">
                <Link to="/app" className="text-sm text-faint transition-colors hover:text-muted">
                  {t("auth.demoLink")}
                </Link>
              </p>
            </div>
          </section>
        </div>

        <footer className="z-[15] flex items-center justify-center px-6 py-5 text-xs text-muted">
          {t("auth.footer", { year: String(new Date().getFullYear()) })}
        </footer>
      </div>
    </div>
  );
}
