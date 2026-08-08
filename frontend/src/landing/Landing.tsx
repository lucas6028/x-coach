import { type ReactNode } from "react";
import { Link, Navigate, useLocation } from "react-router-dom";
import { motion, useReducedMotion } from "motion/react";
import {
  ArrowRight,
  CircleNotch,
  Eye,
  TreeStructure,
  Brain,
  ChatCircleText,
  FilmStrip,
  Ruler,
  PersonSimpleRun,
} from "@phosphor-icons/react";
import Reveal from "./Reveal";
import PosePreview from "./PosePreview";
import MovementShowcase from "./MovementShowcase";
import { LANGS, useI18n, type Lang } from "../lib/i18n";
import { useLiffContext } from "../lib/liffContext";

// The landing page runs on the SAME palette as the app it fronts (the muse-spark tokens in
// index.css / tailwind.config.js): the lavender canvas #eef0fb under white cards, violet #7b61ff
// as the one brand accent, coral #ff6b6b for a fault and green #22c55e for a clean rep. It used to
// be a dark near-black page with a neon green→teal gradient, which meant the first click into /app
// dropped the visitor into what looked like a different product.
//
// The two places that stay dark are the two that are dark IN the app: the video stages. The studio
// renders a black clip inside a pale rounded card with white glass cards floating over it
// (components/VideoPanel), and PosePreview / SkeletonStage below are built to that same shape.
const SECTION = "mx-auto w-full max-w-6xl px-5 sm:px-8";

// Violet gradient for accent words in headings. Deep enough at both stops to clear body-copy
// contrast on the lavender canvas — the brand's lighter tints are for artwork, not for text.
const ACCENT_TEXT = "bg-gradient-to-r from-[#8b6bff] to-[#5a3fe0] bg-clip-text text-transparent";

function PrimaryCTA({ className = "", label }: { className?: string; label?: string }) {
  const { t } = useI18n();
  return (
    <Link
      to="/app"
      className={`group inline-flex items-center gap-2 rounded-full bg-primary px-5 py-3 text-sm font-semibold text-primary-content shadow-accent transition-colors hover:bg-primary/90 active:scale-[0.98] ${className}`}
    >
      {label ?? t("landing.cta.open")}
      <ArrowRight weight="bold" className="transition-transform group-hover:translate-x-0.5" size={16} />
    </Link>
  );
}

// Slim segmented language switcher, on the app's control vocabulary (`glass-control`).
function LangSwitch() {
  const { lang, setLang } = useI18n();
  return (
    <div className="glass-control flex items-center gap-0.5 rounded-full p-0.5">
      {LANGS.map((l) => {
        const active = lang === l.value;
        return (
          <button
            key={l.value}
            onClick={() => setLang(l.value as Lang)}
            aria-pressed={active}
            className={`rounded-full px-2.5 py-1 text-xs font-semibold transition-colors ${
              active ? "bg-primary text-primary-content" : "text-muted hover:text-content"
            }`}
          >
            {l.short}
          </button>
        );
      })}
    </div>
  );
}

// Brand lockup: the skeleton mark (public/icon.svg — the same artwork as the favicon and the app
// rail) beside the name set in the display face. The mark is rendered WITH its violet plate, not
// as the bare figure this used to hand-draw inline: that figure was tuned for the old near-black
// nav, and on the lavender canvas a near-white skeleton has nothing to sit against. The name stays
// in flat ink — the mark is already the violet in this lockup, and the app's own lockups (the
// rail, the login panel) set the name the same way.
function Brand({ markSize = 26, textClass = "text-lg" }: { markSize?: number; textClass?: string }) {
  return (
    <span className="flex shrink-0 items-center gap-2.5">
      <img
        src="/icon.svg"
        width={markSize}
        height={markSize}
        alt=""
        aria-hidden="true"
        className="rounded-md"
        style={{ width: markSize, height: markSize }}
      />
      <span className={`whitespace-nowrap font-display font-bold tracking-tight text-content ${textClass}`}>
        x-coach
      </span>
    </span>
  );
}

function Nav() {
  const { t } = useI18n();
  return (
    <header className="sticky top-0 z-50 border-b border-border-dark bg-background/80 backdrop-blur-md">
      <nav className={`${SECTION} flex h-16 items-center justify-between`}>
        <a href="#top" className="flex shrink-0 items-center">
          <Brand />
        </a>
        <div className="hidden items-center gap-8 text-sm text-muted md:flex">
          <a href="#how" className="transition-colors hover:text-content">{t("landing.nav.how")}</a>
          <a href="#pipeline" className="transition-colors hover:text-content">{t("landing.nav.pipeline")}</a>
          <a href="#eval" className="transition-colors hover:text-content">{t("landing.nav.eval")}</a>
        </div>
        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
          <LangSwitch />
          {/* Redundant with the hero CTA on small screens — hide it there so the
              nav doesn't overflow / wrap on narrow phones. */}
          <div className="hidden sm:block">
            <PrimaryCTA className="px-4 py-2" />
          </div>
        </div>
      </nav>
    </header>
  );
}

function Hero() {
  const { t } = useI18n();
  return (
    <section id="top" className="relative overflow-hidden">
      {/* Two soft violet blooms — the same tint the shell's canvas carries under its glass. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(620px 420px at 78% 8%, rgba(123,97,255,0.20), transparent 70%), radial-gradient(520px 360px at 8% 30%, rgba(155,123,255,0.14), transparent 70%)",
        }}
      />
      <div className={`${SECTION} relative grid grid-cols-1 items-center gap-12 pb-20 pt-12 sm:gap-14 sm:pb-28 sm:pt-16 lg:grid-cols-12 lg:gap-10 lg:pt-24`}>
        <div className="lg:col-span-6">
          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="font-display text-4xl font-bold leading-[1.05] tracking-tight text-content md:text-5xl lg:text-6xl"
          >
            {t("landing.hero.titlePre")}
            <span className={ACCENT_TEXT}>{t("landing.hero.titleAccent")}</span>
            {t("landing.hero.titlePost")}
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
            className="mt-6 max-w-[34rem] text-lg leading-relaxed text-muted"
          >
            {t("landing.hero.sub")}
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="mt-9 flex flex-wrap items-center gap-3"
          >
            <PrimaryCTA />
            <a
              href="#pipeline"
              className="glass-control inline-flex items-center gap-2 rounded-full px-5 py-3 text-sm font-medium text-content transition-colors active:scale-[0.98]"
            >
              {t("landing.hero.readMethod")}
            </a>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="lg:col-span-6 lg:pl-6"
        >
          <PosePreview />
        </motion.div>
      </div>
    </section>
  );
}

function Problem() {
  const { t } = useI18n();
  return (
    <section className={`${SECTION} border-t border-border-dark py-16 sm:py-24`}>
      <Reveal>
        <h2 className="max-w-2xl font-display text-3xl font-bold tracking-tight text-content md:text-4xl">
          {t("landing.problem.title")}
        </h2>
        <p className="mt-4 max-w-xl text-muted">
          {t("landing.problem.sub")}
        </p>
      </Reveal>

      <div className="mt-12 grid gap-6 md:grid-cols-5">
        <Reveal className="md:col-span-3" delay={0.05}>
          <div className="grid h-full gap-0 divide-y divide-border-dark rounded-2xl border border-border-dark bg-surface shadow-card">
            <div className="p-6">
              <p className="font-mono text-xs uppercase tracking-wider text-faint">{t("landing.problem.aqs.label")}</p>
              <p className="mt-2 text-muted">
                {t("landing.problem.aqs.body")}
              </p>
            </div>
            <div className="p-6">
              <p className="font-mono text-xs uppercase tracking-wider text-faint">{t("landing.problem.llm.label")}</p>
              <p className="mt-2 text-muted">
                {t("landing.problem.llm.body")}
              </p>
            </div>
          </div>
        </Reveal>

        <Reveal className="md:col-span-2" delay={0.12}>
          <div className="flex h-full flex-col rounded-2xl border border-primary/25 bg-gradient-to-br from-[#f3f0ff] to-surface p-6 shadow-card">
            <p className="font-mono text-xs uppercase tracking-wider text-primary">x-coach</p>
            <p className="mt-2 font-display text-xl font-semibold text-content">
              {t("landing.problem.xcoach.title")}
            </p>
            <ul className="mt-5 space-y-3 text-sm text-muted">
              {[
                t("landing.problem.point1"),
                t("landing.problem.point2"),
                t("landing.problem.point3"),
              ].map((point) => (
                <li key={point} className="flex gap-2.5">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                  {point}
                </li>
              ))}
            </ul>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

const STAGES = [
  { icon: Eye, n: "01", key: "perceive" },
  { icon: TreeStructure, n: "02", key: "retrieve" },
  { icon: Brain, n: "03", key: "reason" },
  { icon: ChatCircleText, n: "04", key: "coach" },
];

function Pipeline() {
  const reduce = useReducedMotion();
  const { t } = useI18n();
  return (
    <section id="pipeline" className="border-t border-border-dark bg-white/55 py-16 sm:py-24">
      <div className={SECTION}>
        <Reveal>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-primary">{t("landing.pipeline.kicker")}</p>
          <h2 className="mt-3 max-w-2xl font-display text-3xl font-bold tracking-tight text-content md:text-4xl">
            {t("landing.pipeline.title")}
          </h2>
        </Reveal>

        <div className="relative mt-16">
          {/* connecting spine */}
          <motion.div
            aria-hidden
            initial={reduce ? false : { scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            viewport={{ once: true, amount: 0.4 }}
            transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
            className="absolute left-0 right-0 top-7 hidden h-px origin-left bg-gradient-to-r from-primary via-primary/60 to-primary/0 lg:block"
          />
          <div className="grid gap-10 lg:grid-cols-4 lg:gap-6">
            {STAGES.map((s, i) => {
              const Icon = s.icon;
              return (
                <Reveal key={s.key} delay={i * 0.1}>
                  <div className="relative">
                    <div className="relative z-10 flex h-14 w-14 items-center justify-center rounded-2xl border border-[#ece8ff] bg-[#f3f0ff] text-primary shadow-card">
                      <Icon size={26} weight="duotone" />
                    </div>
                    <span className="pointer-events-none absolute -top-3 right-2 font-display text-5xl font-bold text-content/[0.07]">
                      {s.n}
                    </span>
                    <h3 className="mt-5 font-display text-lg font-semibold text-content">{t(`landing.stage.${s.key}.title`)}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-muted">{t(`landing.stage.${s.key}.body`)}</p>
                  </div>
                </Reveal>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

const STEPS = ["observation", "attribution", "prescription"];

function Diagnosis() {
  const { t } = useI18n();
  return (
    <section id="how" className={`${SECTION} border-t border-border-dark py-16 sm:py-24`}>
      <Reveal>
        <h2 className="max-w-2xl font-display text-3xl font-bold tracking-tight text-content md:text-4xl">
          {t("landing.diagnosis.title")}
        </h2>
        <p className="mt-4 max-w-xl text-muted">
          {t("landing.diagnosis.sub")}
        </p>
      </Reveal>

      <div className="relative mt-14 grid gap-0">
        {STEPS.map((key, i) => (
          <Reveal key={key} delay={i * 0.1}>
            <div className="relative grid grid-cols-[auto_1fr] gap-6 pb-10 last:pb-0">
              {/* rail */}
              <div className="flex flex-col items-center">
                <div className="flex h-11 w-11 items-center justify-center rounded-full border border-primary/30 bg-[#f3f0ff] font-display text-sm font-bold text-primary">
                  {i + 1}
                </div>
                {i < STEPS.length - 1 && <div className="mt-1 w-px flex-1 bg-gradient-to-b from-primary/40 to-border-dark" />}
              </div>
              <div className="pb-2">
                <span className="font-mono text-xs uppercase tracking-wider text-faint">{t(`landing.step.${key}.tag`)}</span>
                <h3 className="mt-1 font-display text-xl font-semibold text-content">{t(`landing.step.${key}.title`)}</h3>
                <p className="mt-2 max-w-2xl leading-relaxed text-muted">{t(`landing.step.${key}.body`)}</p>
              </div>
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}

function FrameStrip() {
  const { t } = useI18n();
  // Real frames sampled from the analyzed demo clips (public/demo), not stock placeholders.
  const frames = ["squat", "pushups", "highknee"];
  return (
    <div className="flex gap-2">
      {frames.map((name, i) => (
        <div key={name} className="relative flex-1 overflow-hidden rounded-lg border border-border-dark">
          <img
            src={`/demo/${name}.jpg`}
            alt={t("landing.frame.alt")}
            loading="lazy"
            className="h-20 w-full object-cover sm:h-24"
          />
          {/* Violet wash instead of the old screen-blend teal: `mix-blend-screen` lifts a frame
              toward white, which on a white card erases it. `multiply` tints it downward. */}
          <div className="absolute inset-0 bg-gradient-to-t from-primary/35 to-transparent mix-blend-multiply" />
          <span className="absolute bottom-1 left-1 rounded bg-content/70 px-1.5 py-0.5 font-mono text-[9px] text-white">
            0:0{i + 3}
          </span>
        </div>
      ))}
    </div>
  );
}

function Bento() {
  const { t } = useI18n();
  return (
    <section className="border-t border-border-dark bg-white/55 py-16 sm:py-24">
      <div className={SECTION}>
        <Reveal>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-primary">{t("landing.bento.kicker")}</p>
          <h2 className="mt-3 max-w-2xl font-display text-3xl font-bold tracking-tight text-content md:text-4xl">
            {t("landing.bento.title")}
          </h2>
        </Reveal>

        <div className="mt-12 grid gap-4 md:grid-cols-6 md:grid-rows-2">
          {/* Pose (wide) */}
          <Reveal className="md:col-span-4 md:row-span-1" delay={0.05}>
            <div className="flex h-full flex-col justify-between overflow-hidden rounded-2xl border border-primary/25 bg-gradient-to-br from-[#f3f0ff] to-surface p-6 shadow-card">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <PersonSimpleRun size={26} weight="duotone" className="text-primary" />
                  <h3 className="mt-3 font-display text-xl font-semibold text-content">{t("landing.bento.pose.title")}</h3>
                  <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted">
                    {t("landing.bento.pose.body")}
                  </p>
                </div>
                <svg viewBox="0 0 90 120" className="hidden w-16 shrink-0 sm:block" aria-hidden>
                  <g stroke="url(#m)" strokeWidth="6" strokeLinecap="round" fill="none">
                    <line x1="34" y1="30" x2="30" y2="58" />
                    <line x1="30" y1="58" x2="62" y2="62" />
                    <line x1="62" y1="62" x2="54" y2="92" />
                  </g>
                  <circle cx="36" cy="20" r="9" fill="url(#m)" />
                  <circle cx="62" cy="62" r="5" fill="#ff6b6b" />
                </svg>
              </div>
            </div>
          </Reveal>

          {/* Knowledge graph (tall right) */}
          <Reveal className="md:col-span-2 md:row-span-2" delay={0.12}>
            <div className="flex h-full flex-col overflow-hidden rounded-2xl border border-border-dark bg-surface p-6 shadow-card">
              <TreeStructure size={26} weight="duotone" className="text-primary" />
              <h3 className="mt-3 font-display text-xl font-semibold text-content">{t("landing.bento.kg.title")}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">
                {t("landing.bento.kg.body")}
              </p>
              <svg viewBox="0 0 220 200" className="mt-auto w-full" aria-hidden>
                <g stroke="#7b61ff" strokeOpacity="0.45" strokeWidth="2">
                  <line x1="48" y1="40" x2="120" y2="96" />
                  <line x1="120" y1="96" x2="60" y2="156" />
                  <line x1="120" y1="96" x2="186" y2="150" />
                </g>
                <g fontFamily="monospace" fontSize="10" fill="#63709f">
                  <g>
                    <circle cx="48" cy="40" r="7" fill="#ff6b6b" />
                    <text x="48" y="26" textAnchor="middle">valgus</text>
                  </g>
                  <g>
                    <circle cx="120" cy="96" r="7" fill="#7b61ff" />
                    <text x="120" y="84" textAnchor="middle">abductor</text>
                  </g>
                  <g>
                    <circle cx="60" cy="156" r="6" fill="#b8bcd3" />
                    <text x="60" y="176" textAnchor="middle">glute med.</text>
                  </g>
                  <g>
                    <circle cx="186" cy="150" r="6" fill="#b8bcd3" />
                    <text x="186" y="170" textAnchor="middle">band squat</text>
                  </g>
                </g>
              </svg>
            </div>
          </Reveal>

          {/* Rules */}
          <Reveal className="md:col-span-2" delay={0.18}>
            <div className="flex h-full flex-col overflow-hidden rounded-2xl border border-border-dark bg-surface p-6 shadow-card">
              <Ruler size={24} weight="duotone" className="text-primary" />
              <h3 className="mt-3 font-display text-lg font-semibold text-content">{t("landing.bento.rules.title")}</h3>
              <div className="mt-3 space-y-1.5 font-mono text-[11px] text-muted">
                <p><span className="font-semibold text-[#e05252]">knee_valgus</span> &gt; thr</p>
                <p><span className="text-content">depth</span> below parallel</p>
                <p><span className="text-content">torso_lean</span> within band</p>
              </div>
            </div>
          </Reveal>

          {/* VideoMAE (real frames) */}
          <Reveal className="md:col-span-2" delay={0.24}>
            <div className="flex h-full flex-col overflow-hidden rounded-2xl border border-border-dark bg-surface p-6 shadow-card">
              <FilmStrip size={24} weight="duotone" className="text-primary" />
              <h3 className="mt-3 font-display text-lg font-semibold text-content">{t("landing.bento.videomae.title")}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">
                {t("landing.bento.videomae.body")}
              </p>
              <div className="mt-4">
                <FrameStrip />
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

const METHODS = ["m1", "m2", "m3"];

function Evaluation() {
  const { t } = useI18n();
  return (
    <section id="eval" className={`${SECTION} border-t border-border-dark py-16 sm:py-24`}>
      <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
        <Reveal>
          <h2 className="font-display text-3xl font-bold tracking-tight text-content md:text-4xl">
            {t("landing.eval.title")}
          </h2>
          <p className="mt-4 max-w-md text-muted">
            {t("landing.eval.sub")}
          </p>
        </Reveal>
        <div className="grid gap-8">
          {METHODS.map((m, i) => (
            <Reveal key={m} delay={i * 0.08}>
              <div className="border-l-2 border-primary/35 pl-5">
                <p className="font-mono text-xs uppercase tracking-wider text-primary">{t(`landing.eval.${m}.label`)}</p>
                <p className="mt-2 text-muted">{t(`landing.eval.${m}.body`)}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTA() {
  const { t } = useI18n();
  return (
    <section className="relative overflow-hidden border-t border-border-dark py-16 sm:py-24">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(560px 300px at 50% 0%, rgba(123,97,255,0.20), transparent 70%)" }}
      />
      <div className={`${SECTION} relative text-center`}>
        <Reveal>
          <h2 className="mx-auto max-w-2xl font-display text-3xl font-bold tracking-tight text-content md:text-5xl">
            {t("landing.cta.title")}
          </h2>
          <p className="mx-auto mt-4 max-w-md text-muted">
            {t("landing.cta.sub")}
          </p>
          <div className="mt-9 flex justify-center">
            <PrimaryCTA />
          </div>
        </Reveal>
      </div>
    </section>
  );
}

function Footer() {
  const { t } = useI18n();
  return (
    <footer className="border-t border-border-dark py-12">
      <div className={`${SECTION} flex flex-col items-center justify-between gap-6 sm:flex-row`}>
        <Brand markSize={24} textClass="text-base" />
        <div className="flex items-center gap-7 text-sm text-muted">
          <a href="#how" className="transition-colors hover:text-content">{t("landing.nav.how")}</a>
          <a href="#pipeline" className="transition-colors hover:text-content">{t("landing.footer.pipeline")}</a>
          <a href="#eval" className="transition-colors hover:text-content">{t("landing.nav.eval")}</a>
        </div>
        <p className="text-sm text-faint">{t("landing.footer.tagline")}</p>
      </div>
    </footer>
  );
}

// Shared SVG gradient def so the inline bento marks can reference url(#m).
function Defs() {
  return (
    <svg width="0" height="0" className="absolute" aria-hidden>
      <defs>
        <linearGradient id="m" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#9b7bff" />
          <stop offset="1" stopColor="#6b4dff" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function Layout({ children }: { children: ReactNode }) {
  return <div className="min-h-screen bg-background font-body text-content antialiased">{children}</div>;
}

export default function Landing() {
  const { t } = useI18n();
  const { ready, isInClient } = useLiffContext();
  const location = useLocation();

  // The in-client guess is a synchronous heuristic (LINE user agent / liff.state query params —
  // see lib/liffContext), not a confirmed LIFF context. Sharing the site URL in a LINE chat opens
  // it in LINE's in-app browser too, which matches the same user-agent signal, but
  // liff.isInClient() answers false there because it isn't actually a LIFF app context. A shell
  // swap on a wrong guess is just a visible flash; redirecting off the marketing page on one is
  // an irreversible navigation with no way back (Header's brand lockup points at /app, not "/").
  // So: while the guess says in-client but the SDK hasn't confirmed yet, hold on a neutral
  // loading state instead of committing to either the marketing page or the redirect. Once ready,
  // behave exactly as before. Rendering the marketing page itself during this window isn't an
  // option either — the LIFF endpoint URL is the site root, so every real LINE launch would hit
  // this page and flash it for the ~1s the SDK takes to confirm.
  //
  // A neutral spinner, not LumenLoader's "scan" narration (Reading pose / Checking mechanics /
  // Lighting the why): this fires before any video exists, so the analysis-pipeline copy would
  // describe work that isn't happening. Same idiom as RequireAuth's session-check placeholder,
  // so the app's waits read as one system.
  if (!ready && isInClient) {
    return (
      <div className="grid min-h-[100dvh] place-items-center bg-background text-muted" role="status">
        <CircleNotch size={24} className="animate-spin" />
        <span className="sr-only">{t("loader.neutral")}</span>
      </div>
    );
  }
  // Inside LINE the marketing page is dead weight — the user arrived from a rich menu to use
  // the app, and the LIFF endpoint URL is the site root (see the comment above), so every
  // real LINE launch lands here and needs this redirect, not just a stray "/" hit.
  // Preserve the query string: it can carry liff.state, which LINE appends on a LIFF redirect.
  if (isInClient) return <Navigate to={`/app${location.search}`} replace />;
  return (
    <Layout>
      <Defs />
      <Nav />
      <main>
        <Hero />
        <Problem />
        <Pipeline />
        <Diagnosis />
        <MovementShowcase />
        <Bento />
        <Evaluation />
        <CTA />
      </main>
      <Footer />
    </Layout>
  );
}
