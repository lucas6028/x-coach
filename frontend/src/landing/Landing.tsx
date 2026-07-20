import { type ReactNode } from "react";
import { Link, Navigate } from "react-router-dom";
import { motion, useReducedMotion } from "motion/react";
import {
  ArrowRight,
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

const SECTION = "mx-auto w-full max-w-6xl px-5 sm:px-8";

function PrimaryCTA({ className = "", label }: { className?: string; label?: string }) {
  const { t } = useI18n();
  return (
    <Link
      to="/app"
      className={`group inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-[#5ffb6f] to-[#19c2b0] px-5 py-3 text-sm font-semibold text-[#06140c] transition-shadow active:scale-[0.98] hover:shadow-[0_0_34px_-6px_rgba(60,224,122,0.55)] ${className}`}
    >
      {label ?? t("landing.cta.open")}
      <ArrowRight weight="bold" className="transition-transform group-hover:translate-x-0.5" size={16} />
    </Link>
  );
}

// Slim segmented language switcher tuned for the landing's dark palette.
function LangSwitch() {
  const { lang, setLang } = useI18n();
  return (
    <div className="flex items-center gap-0.5 rounded-full border border-white/10 bg-white/5 p-0.5">
      {LANGS.map((l) => {
        const active = lang === l.value;
        return (
          <button
            key={l.value}
            onClick={() => setLang(l.value as Lang)}
            aria-pressed={active}
            className={`rounded-full px-2.5 py-1 text-xs font-semibold transition-colors ${
              active ? "bg-white/15 text-zinc-50" : "text-zinc-400 hover:text-zinc-100"
            }`}
          >
            {l.short}
          </button>
        );
      })}
    </div>
  );
}

// Brand mark — kept in sync with the site favicon/logo (public/icon.svg,
// public/logo.svg): the full squatting-skeleton figure, minus the rounded
// background so it sits cleanly on the dark nav.
function Mark() {
  return (
    <svg width="26" height="26" viewBox="0 0 128 128" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="markBone" x1="28" y1="20" x2="104" y2="116" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#5ffb6f" />
          <stop offset="1" stopColor="#16b8a8" />
        </linearGradient>
      </defs>
      {/* ground line */}
      <line x1="22" y1="110" x2="106" y2="110" stroke="#3a4a4f" strokeWidth="4" strokeLinecap="round" />
      {/* knee-angle arc: the "explainable" measurement accent */}
      <path d="M84 84 A 22 22 0 0 1 70 110" fill="none" stroke="#ffd23f" strokeWidth="4" strokeLinecap="round" />
      {/* skeleton bones: torso, thigh, shin, foot, arm */}
      <g stroke="url(#markBone)" strokeWidth="9" strokeLinecap="round" strokeLinejoin="round" fill="none">
        <line x1="47" y1="40" x2="40" y2="78" />
        <line x1="40" y1="78" x2="86" y2="84" />
        <line x1="86" y1="84" x2="74" y2="110" />
        <line x1="60" y1="110" x2="94" y2="110" />
        <line x1="47" y1="44" x2="92" y2="50" />
      </g>
      {/* pose keypoints */}
      <g fill="#eafff0" stroke="#0d1113" strokeWidth="2.5">
        <circle cx="47" cy="42" r="6" />
        <circle cx="40" cy="78" r="6" />
        <circle cx="86" cy="84" r="7.5" />
        <circle cx="74" cy="110" r="6" />
        <circle cx="92" cy="50" r="5.5" />
      </g>
      {/* head */}
      <circle cx="49" cy="26" r="12" fill="url(#markBone)" stroke="#eafff0" strokeWidth="3" />
    </svg>
  );
}

function Nav() {
  const { t } = useI18n();
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#0d0f10]/80 backdrop-blur-md">
      <nav className={`${SECTION} flex h-16 items-center justify-between`}>
        <a href="#top" className="flex shrink-0 items-center gap-2.5">
          <Mark />
          <span className="whitespace-nowrap font-display text-lg font-bold tracking-tight text-zinc-50">
            x-<span className="bg-gradient-to-r from-[#5ffb6f] to-[#16b8a8] bg-clip-text text-transparent">coach</span>
          </span>
        </a>
        <div className="hidden items-center gap-8 text-sm text-zinc-400 md:flex">
          <a href="#how" className="transition-colors hover:text-zinc-100">{t("landing.nav.how")}</a>
          <a href="#pipeline" className="transition-colors hover:text-zinc-100">{t("landing.nav.pipeline")}</a>
          <a href="#eval" className="transition-colors hover:text-zinc-100">{t("landing.nav.eval")}</a>
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
      {/* restrained brand glow, not AI-purple mesh */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(620px 420px at 78% 8%, rgba(22,184,168,0.16), transparent 70%), radial-gradient(520px 360px at 8% 30%, rgba(95,251,111,0.08), transparent 70%)",
        }}
      />
      <div className={`${SECTION} relative grid grid-cols-1 items-center gap-12 pb-20 pt-12 sm:gap-14 sm:pb-28 sm:pt-16 lg:grid-cols-12 lg:gap-10 lg:pt-24`}>
        <div className="lg:col-span-6">
          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="font-display text-4xl font-bold leading-[1.05] tracking-tight text-zinc-50 md:text-5xl lg:text-6xl"
          >
            {t("landing.hero.titlePre")}
            <span className="bg-gradient-to-r from-[#5ffb6f] to-[#16b8a8] bg-clip-text text-transparent">
              {t("landing.hero.titleAccent")}
            </span>
            {t("landing.hero.titlePost")}
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
            className="mt-6 max-w-[34rem] text-lg leading-relaxed text-zinc-400"
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
              className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-5 py-3 text-sm font-medium text-zinc-200 transition-colors hover:bg-white/10 active:scale-[0.98]"
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
    <section className={`${SECTION} border-t border-white/10 py-16 sm:py-24`}>
      <Reveal>
        <h2 className="max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
          {t("landing.problem.title")}
        </h2>
        <p className="mt-4 max-w-xl text-zinc-400">
          {t("landing.problem.sub")}
        </p>
      </Reveal>

      <div className="mt-12 grid gap-6 md:grid-cols-5">
        <Reveal className="md:col-span-3" delay={0.05}>
          <div className="grid h-full gap-0 divide-y divide-white/10 rounded-2xl border border-white/10 bg-white/[0.02]">
            <div className="p-6">
              <p className="font-mono text-xs uppercase tracking-wider text-zinc-500">{t("landing.problem.aqs.label")}</p>
              <p className="mt-2 text-zinc-300">
                {t("landing.problem.aqs.body")}
              </p>
            </div>
            <div className="p-6">
              <p className="font-mono text-xs uppercase tracking-wider text-zinc-500">{t("landing.problem.llm.label")}</p>
              <p className="mt-2 text-zinc-300">
                {t("landing.problem.llm.body")}
              </p>
            </div>
          </div>
        </Reveal>

        <Reveal className="md:col-span-2" delay={0.12}>
          <div className="flex h-full flex-col rounded-2xl border border-[#16b8a8]/30 bg-gradient-to-br from-[#16b8a8]/[0.10] to-transparent p-6">
            <p className="font-mono text-xs uppercase tracking-wider text-[#3ee07a]">x-coach</p>
            <p className="mt-2 font-display text-xl font-semibold text-zinc-50">
              {t("landing.problem.xcoach.title")}
            </p>
            <ul className="mt-5 space-y-3 text-sm text-zinc-300">
              {[
                t("landing.problem.point1"),
                t("landing.problem.point2"),
                t("landing.problem.point3"),
              ].map((point) => (
                <li key={point} className="flex gap-2.5">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[#3ee07a]" />
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
    <section id="pipeline" className="border-t border-white/10 bg-white/[0.015] py-16 sm:py-24">
      <div className={SECTION}>
        <Reveal>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-[#3ee07a]">{t("landing.pipeline.kicker")}</p>
          <h2 className="mt-3 max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
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
            className="absolute left-0 right-0 top-7 hidden h-px origin-left bg-gradient-to-r from-[#5ffb6f] via-[#16b8a8] to-[#16b8a8]/0 lg:block"
          />
          <div className="grid gap-10 lg:grid-cols-4 lg:gap-6">
            {STAGES.map((s, i) => {
              const Icon = s.icon;
              return (
                <Reveal key={s.key} delay={i * 0.1}>
                  <div className="relative">
                    <div className="relative z-10 flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-[#15191b] text-[#3ee07a]">
                      <Icon size={26} weight="duotone" />
                    </div>
                    <span className="pointer-events-none absolute -top-3 right-2 font-display text-5xl font-bold text-white/[0.05]">
                      {s.n}
                    </span>
                    <h3 className="mt-5 font-display text-lg font-semibold text-zinc-50">{t(`landing.stage.${s.key}.title`)}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-zinc-400">{t(`landing.stage.${s.key}.body`)}</p>
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
    <section id="how" className={`${SECTION} border-t border-white/10 py-16 sm:py-24`}>
      <Reveal>
        <h2 className="max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
          {t("landing.diagnosis.title")}
        </h2>
        <p className="mt-4 max-w-xl text-zinc-400">
          {t("landing.diagnosis.sub")}
        </p>
      </Reveal>

      <div className="relative mt-14 grid gap-0">
        {STEPS.map((key, i) => (
          <Reveal key={key} delay={i * 0.1}>
            <div className="relative grid grid-cols-[auto_1fr] gap-6 pb-10 last:pb-0">
              {/* rail */}
              <div className="flex flex-col items-center">
                <div className="flex h-11 w-11 items-center justify-center rounded-full border border-[#16b8a8]/40 bg-[#16b8a8]/10 font-display text-sm font-bold text-[#3ee07a]">
                  {i + 1}
                </div>
                {i < STEPS.length - 1 && <div className="mt-1 w-px flex-1 bg-gradient-to-b from-[#16b8a8]/40 to-white/5" />}
              </div>
              <div className="pb-2">
                <span className="font-mono text-xs uppercase tracking-wider text-zinc-500">{t(`landing.step.${key}.tag`)}</span>
                <h3 className="mt-1 font-display text-xl font-semibold text-zinc-50">{t(`landing.step.${key}.title`)}</h3>
                <p className="mt-2 max-w-2xl leading-relaxed text-zinc-300">{t(`landing.step.${key}.body`)}</p>
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
        <div key={name} className="relative flex-1 overflow-hidden rounded-lg border border-white/10">
          <img
            src={`/demo/${name}.jpg`}
            alt={t("landing.frame.alt")}
            loading="lazy"
            className="h-20 w-full object-cover opacity-80 grayscale contrast-[1.05] sm:h-24"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#16b8a8]/25 to-transparent mix-blend-screen" />
          <span className="absolute bottom-1 left-1 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[9px] text-zinc-200">
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
    <section className="border-t border-white/10 bg-white/[0.015] py-16 sm:py-24">
      <div className={SECTION}>
        <Reveal>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-[#3ee07a]">{t("landing.bento.kicker")}</p>
          <h2 className="mt-3 max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
            {t("landing.bento.title")}
          </h2>
        </Reveal>

        <div className="mt-12 grid gap-4 md:grid-cols-6 md:grid-rows-2">
          {/* Pose (wide) */}
          <Reveal className="md:col-span-4 md:row-span-1" delay={0.05}>
            <div className="flex h-full flex-col justify-between overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-[#16b8a8]/[0.08] to-transparent p-6">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <PersonSimpleRun size={26} weight="duotone" className="text-[#3ee07a]" />
                  <h3 className="mt-3 font-display text-xl font-semibold text-zinc-50">{t("landing.bento.pose.title")}</h3>
                  <p className="mt-2 max-w-sm text-sm leading-relaxed text-zinc-400">
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
                  <circle cx="62" cy="62" r="5" fill="#f5b945" />
                </svg>
              </div>
            </div>
          </Reveal>

          {/* Knowledge graph (tall right) */}
          <Reveal className="md:col-span-2 md:row-span-2" delay={0.12}>
            <div className="flex h-full flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#121517] p-6">
              <TreeStructure size={26} weight="duotone" className="text-[#3ee07a]" />
              <h3 className="mt-3 font-display text-xl font-semibold text-zinc-50">{t("landing.bento.kg.title")}</h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">
                {t("landing.bento.kg.body")}
              </p>
              <svg viewBox="0 0 220 200" className="mt-auto w-full" aria-hidden>
                <g stroke="#16b8a8" strokeOpacity="0.5" strokeWidth="2">
                  <line x1="48" y1="40" x2="120" y2="96" />
                  <line x1="120" y1="96" x2="60" y2="156" />
                  <line x1="120" y1="96" x2="186" y2="150" />
                </g>
                <g fontFamily="monospace" fontSize="10" fill="#a1a1aa">
                  <g>
                    <circle cx="48" cy="40" r="7" fill="#f5b945" />
                    <text x="48" y="26" textAnchor="middle">valgus</text>
                  </g>
                  <g>
                    <circle cx="120" cy="96" r="7" fill="#3ee07a" />
                    <text x="120" y="84" textAnchor="middle">abductor</text>
                  </g>
                  <g>
                    <circle cx="60" cy="156" r="6" fill="#52525b" />
                    <text x="60" y="176" textAnchor="middle">glute med.</text>
                  </g>
                  <g>
                    <circle cx="186" cy="150" r="6" fill="#52525b" />
                    <text x="186" y="170" textAnchor="middle">band squat</text>
                  </g>
                </g>
              </svg>
            </div>
          </Reveal>

          {/* Rules */}
          <Reveal className="md:col-span-2" delay={0.18}>
            <div className="flex h-full flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#121517] p-6">
              <Ruler size={24} weight="duotone" className="text-[#3ee07a]" />
              <h3 className="mt-3 font-display text-lg font-semibold text-zinc-50">{t("landing.bento.rules.title")}</h3>
              <div className="mt-3 space-y-1.5 font-mono text-[11px] text-zinc-400">
                <p><span className="text-[#f5b945]">knee_valgus</span> &gt; thr</p>
                <p><span className="text-zinc-200">depth</span> below parallel</p>
                <p><span className="text-zinc-200">torso_lean</span> within band</p>
              </div>
            </div>
          </Reveal>

          {/* VideoMAE (real frames) */}
          <Reveal className="md:col-span-2" delay={0.24}>
            <div className="flex h-full flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#121517] p-6">
              <FilmStrip size={24} weight="duotone" className="text-[#3ee07a]" />
              <h3 className="mt-3 font-display text-lg font-semibold text-zinc-50">{t("landing.bento.videomae.title")}</h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">
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
    <section id="eval" className={`${SECTION} border-t border-white/10 py-16 sm:py-24`}>
      <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
        <Reveal>
          <h2 className="font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
            {t("landing.eval.title")}
          </h2>
          <p className="mt-4 max-w-md text-zinc-400">
            {t("landing.eval.sub")}
          </p>
        </Reveal>
        <div className="grid gap-8">
          {METHODS.map((m, i) => (
            <Reveal key={m} delay={i * 0.08}>
              <div className="border-l-2 border-[#16b8a8]/40 pl-5">
                <p className="font-mono text-xs uppercase tracking-wider text-[#3ee07a]">{t(`landing.eval.${m}.label`)}</p>
                <p className="mt-2 text-zinc-300">{t(`landing.eval.${m}.body`)}</p>
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
    <section className="relative overflow-hidden border-t border-white/10 py-16 sm:py-24">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(560px 300px at 50% 0%, rgba(22,184,168,0.16), transparent 70%)" }}
      />
      <div className={`${SECTION} relative text-center`}>
        <Reveal>
          <h2 className="mx-auto max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-5xl">
            {t("landing.cta.title")}
          </h2>
          <p className="mx-auto mt-4 max-w-md text-zinc-400">
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
    <footer className="border-t border-white/10 py-12">
      <div className={`${SECTION} flex flex-col items-center justify-between gap-6 sm:flex-row`}>
        <div className="flex items-center gap-2.5">
          <Mark />
          <span className="font-display text-base font-bold tracking-tight text-zinc-50">
            x-<span className="bg-gradient-to-r from-[#5ffb6f] to-[#16b8a8] bg-clip-text text-transparent">coach</span>
          </span>
        </div>
        <div className="flex items-center gap-7 text-sm text-zinc-400">
          <a href="#how" className="transition-colors hover:text-zinc-100">{t("landing.nav.how")}</a>
          <a href="#pipeline" className="transition-colors hover:text-zinc-100">{t("landing.footer.pipeline")}</a>
          <a href="#eval" className="transition-colors hover:text-zinc-100">{t("landing.nav.eval")}</a>
        </div>
        <p className="text-sm text-zinc-500">{t("landing.footer.tagline")}</p>
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
          <stop offset="0" stopColor="#5ffb6f" />
          <stop offset="1" stopColor="#16b8a8" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function Layout({ children }: { children: ReactNode }) {
  return <div className="min-h-screen bg-[#0d0f10] font-display text-zinc-100 antialiased">{children}</div>;
}

export default function Landing() {
  const { isInClient } = useLiffContext();
  // Inside LINE the marketing page is dead weight — the user arrived from a rich menu to use
  // the app, and the LIFF endpoint URL points at /app anyway. This covers a stray "/" hit.
  if (isInClient) return <Navigate to="/app" replace />;
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
