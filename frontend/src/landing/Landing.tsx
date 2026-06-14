import { type ReactNode } from "react";
import { Link } from "react-router-dom";
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

const SECTION = "mx-auto w-full max-w-6xl px-5 sm:px-8";

function PrimaryCTA({ className = "", label = "Open the demo" }: { className?: string; label?: string }) {
  return (
    <Link
      to="/app"
      className={`group inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-[#5ffb6f] to-[#19c2b0] px-5 py-3 text-sm font-semibold text-[#06140c] transition-shadow active:scale-[0.98] hover:shadow-[0_0_34px_-6px_rgba(60,224,122,0.55)] ${className}`}
    >
      {label}
      <ArrowRight weight="bold" className="transition-transform group-hover:translate-x-0.5" size={16} />
    </Link>
  );
}

function Mark() {
  return (
    <svg width="26" height="26" viewBox="0 0 124 124" aria-hidden="true">
      <defs>
        <linearGradient id="m" x1="28" y1="20" x2="104" y2="116" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#5ffb6f" />
          <stop offset="1" stopColor="#16b8a8" />
        </linearGradient>
      </defs>
      <g stroke="url(#m)" strokeWidth="9" strokeLinecap="round" strokeLinejoin="round" fill="none">
        <line x1="49" y1="42" x2="42" y2="80" />
        <line x1="42" y1="80" x2="88" y2="86" />
        <line x1="88" y1="86" x2="76" y2="112" />
      </g>
      <circle cx="51" cy="28" r="12" fill="url(#m)" />
      <circle cx="88" cy="86" r="7" fill="#f5b945" />
    </svg>
  );
}

function Nav() {
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#0d0f10]/80 backdrop-blur-md">
      <nav className={`${SECTION} flex h-16 items-center justify-between`}>
        <a href="#top" className="flex items-center gap-2.5">
          <Mark />
          <span className="font-display text-lg font-bold tracking-tight text-zinc-50">
            x-<span className="bg-gradient-to-r from-[#5ffb6f] to-[#16b8a8] bg-clip-text text-transparent">coach</span>
          </span>
        </a>
        <div className="hidden items-center gap-8 text-sm text-zinc-400 md:flex">
          <a href="#how" className="transition-colors hover:text-zinc-100">How it works</a>
          <a href="#pipeline" className="transition-colors hover:text-zinc-100">The pipeline</a>
          <a href="#eval" className="transition-colors hover:text-zinc-100">Evaluation</a>
        </div>
        <PrimaryCTA className="px-4 py-2" />
      </nav>
    </header>
  );
}

function Hero() {
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
      <div className={`${SECTION} relative grid items-center gap-14 pb-28 pt-16 lg:grid-cols-12 lg:gap-10 lg:pt-24`}>
        <div className="lg:col-span-6">
          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="font-display text-4xl font-bold leading-[1.05] tracking-tight text-zinc-50 md:text-5xl lg:text-6xl"
          >
            Coaching cues you can{" "}
            <span className="bg-gradient-to-r from-[#5ffb6f] to-[#16b8a8] bg-clip-text text-transparent">
              trace to the joint
            </span>
            .
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
            className="mt-6 max-w-[34rem] text-lg leading-relaxed text-zinc-400"
          >
            x-coach reads a squat video, locates the fault, traces its cause in a biomechanics
            knowledge graph, and explains the fix.
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
              Read the method
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
  return (
    <section className={`${SECTION} border-t border-white/10 py-24`}>
      <Reveal>
        <h2 className="max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
          Scores don't coach. Generic models guess.
        </h2>
        <p className="mt-4 max-w-xl text-zinc-400">
          Action-quality models hand back a number with no instruction. Ask a general language
          model and it sounds confident while inventing the biomechanics.
        </p>
      </Reveal>

      <div className="mt-12 grid gap-6 md:grid-cols-5">
        <Reveal className="md:col-span-3" delay={0.05}>
          <div className="grid h-full gap-0 divide-y divide-white/10 rounded-2xl border border-white/10 bg-white/[0.02]">
            <div className="p-6">
              <p className="font-mono text-xs uppercase tracking-wider text-zinc-500">Action quality scoring</p>
              <p className="mt-2 text-zinc-300">
                Returns a 0 to 100 rating. The lifter learns they scored a 71, not what to change
                or why.
              </p>
            </div>
            <div className="p-6">
              <p className="font-mono text-xs uppercase tracking-wider text-zinc-500">General language models</p>
              <p className="mt-2 text-zinc-300">
                Produce fluent advice untethered from the video, and hallucinate causes that the
                footage never showed.
              </p>
            </div>
          </div>
        </Reveal>

        <Reveal className="md:col-span-2" delay={0.12}>
          <div className="flex h-full flex-col rounded-2xl border border-[#16b8a8]/30 bg-gradient-to-br from-[#16b8a8]/[0.10] to-transparent p-6">
            <p className="font-mono text-xs uppercase tracking-wider text-[#3ee07a]">x-coach</p>
            <p className="mt-2 font-display text-xl font-semibold text-zinc-50">
              Grounded by construction
            </p>
            <ul className="mt-5 space-y-3 text-sm text-zinc-300">
              {[
                "Sees the fault in the actual frames",
                "Retrieves the cause from a sourced knowledge graph",
                "Explains the fix it can point back to",
              ].map((t) => (
                <li key={t} className="flex gap-2.5">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[#3ee07a]" />
                  {t}
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
  { icon: Eye, n: "01", title: "Perceive", body: "Pose landmarks and VideoMAE motion features extract geometry, then localize the fault in time." },
  { icon: TreeStructure, n: "02", title: "Retrieve", body: "GraphRAG walks the fitness knowledge graph from the visible symptom to its deeper cause." },
  { icon: Brain, n: "03", title: "Reason", body: "A chain of thought moves from observation to attribution to prescription, grounded in the retrieved evidence." },
  { icon: ChatCircleText, n: "04", title: "Coach", body: "A diagnosis report and corrective cues come back, with the exact frames highlighted." },
];

function Pipeline() {
  const reduce = useReducedMotion();
  return (
    <section id="pipeline" className="border-t border-white/10 bg-white/[0.015] py-24">
      <div className={SECTION}>
        <Reveal>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-[#3ee07a]">The system</p>
          <h2 className="mt-3 max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
            Four modules, one closed loop from pixels to prescription.
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
                <Reveal key={s.title} delay={i * 0.1}>
                  <div className="relative">
                    <div className="relative z-10 flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-[#15191b] text-[#3ee07a]">
                      <Icon size={26} weight="duotone" />
                    </div>
                    <span className="pointer-events-none absolute -top-3 right-2 font-display text-5xl font-bold text-white/[0.05]">
                      {s.n}
                    </span>
                    <h3 className="mt-5 font-display text-lg font-semibold text-zinc-50">{s.title}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-zinc-400">{s.body}</p>
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

const STEPS = [
  {
    tag: "Perception",
    title: "Observation",
    body: "The left knee crosses inward of the foot through the bottom of the rep, flagged across frames 96 to 118.",
  },
  {
    tag: "Knowledge graph",
    title: "Attribution",
    body: "Multi-hop retrieval links medial knee travel to weak hip abductors, with glute medius as the primary node.",
  },
  {
    tag: "Reasoning",
    title: "Prescription",
    body: "Cue the lifter to drive the knees out over the toes, and program banded goblet squats as the accessory.",
  },
];

function Diagnosis() {
  return (
    <section id="how" className={`${SECTION} border-t border-white/10 py-24`}>
      <Reveal>
        <h2 className="max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
          Every cue carries its reasoning.
        </h2>
        <p className="mt-4 max-w-xl text-zinc-400">
          One detected fault, walked from what the camera saw to the exercise you should do about it.
        </p>
      </Reveal>

      <div className="relative mt-14 grid gap-0">
        {STEPS.map((s, i) => (
          <Reveal key={s.title} delay={i * 0.1}>
            <div className="relative grid grid-cols-[auto_1fr] gap-6 pb-10 last:pb-0">
              {/* rail */}
              <div className="flex flex-col items-center">
                <div className="flex h-11 w-11 items-center justify-center rounded-full border border-[#16b8a8]/40 bg-[#16b8a8]/10 font-display text-sm font-bold text-[#3ee07a]">
                  {i + 1}
                </div>
                {i < STEPS.length - 1 && <div className="mt-1 w-px flex-1 bg-gradient-to-b from-[#16b8a8]/40 to-white/5" />}
              </div>
              <div className="pb-2">
                <span className="font-mono text-xs uppercase tracking-wider text-zinc-500">{s.tag}</span>
                <h3 className="mt-1 font-display text-xl font-semibold text-zinc-50">{s.title}</h3>
                <p className="mt-2 max-w-2xl leading-relaxed text-zinc-300">{s.body}</p>
              </div>
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}

function FrameStrip() {
  const seeds = ["squat-descent", "barbell-lift", "gym-athlete"];
  return (
    <div className="flex gap-2">
      {seeds.map((seed, i) => (
        <div key={seed} className="relative flex-1 overflow-hidden rounded-lg border border-white/10">
          <img
            src={`https://picsum.photos/seed/${seed}/240/200`}
            alt="Sampled video frame under analysis"
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
  return (
    <section className="border-t border-white/10 bg-white/[0.015] py-24">
      <div className={SECTION}>
        <Reveal>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-[#3ee07a]">Under the hood</p>
          <h2 className="mt-3 max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
            Four signals, read locally and fused.
          </h2>
        </Reveal>

        <div className="mt-12 grid gap-4 md:grid-cols-6 md:grid-rows-2">
          {/* Pose (wide) */}
          <Reveal className="md:col-span-4 md:row-span-1" delay={0.05}>
            <div className="flex h-full flex-col justify-between overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-[#16b8a8]/[0.08] to-transparent p-6">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <PersonSimpleRun size={26} weight="duotone" className="text-[#3ee07a]" />
                  <h3 className="mt-3 font-display text-xl font-semibold text-zinc-50">Pose perception</h3>
                  <p className="mt-2 max-w-sm text-sm leading-relaxed text-zinc-400">
                    MediaPipe and MMPose landmarks on 33 keypoints. Joint geometry like knee
                    valgus maps straight into language the graph understands.
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
              <h3 className="mt-3 font-display text-xl font-semibold text-zinc-50">Knowledge graph</h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">
                A fitness knowledge graph links fault to cause to fix, with multi-hop retrieval over
                sourced biomechanics.
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
              <h3 className="mt-3 font-display text-lg font-semibold text-zinc-50">Interpretable rules</h3>
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
              <h3 className="mt-3 font-display text-lg font-semibold text-zinc-50">VideoMAE motion</h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">
                Spatio-temporal features tell a clean rep from a subtle error.
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

const METHODS = [
  { label: "Score agreement", body: "Spearman correlation against expert ranking, so the model's ordering tracks a human judge." },
  { label: "Grounding and hallucination", body: "RAGAS faithfulness checks that every claim stays anchored to the retrieved evidence." },
  { label: "Usefulness", body: "A user study with lifters across experience levels rates the skeleton overlay and the written advice." },
];

function Evaluation() {
  return (
    <section id="eval" className={`${SECTION} border-t border-white/10 py-24`}>
      <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
        <Reveal>
          <h2 className="font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
            Held to a measurable bar.
          </h2>
          <p className="mt-4 max-w-md text-zinc-400">
            Explainability only counts if it is checkable. x-coach is validated on agreement,
            grounding, and whether lifters actually act on it.
          </p>
        </Reveal>
        <div className="grid gap-8">
          {METHODS.map((m, i) => (
            <Reveal key={m.label} delay={i * 0.08}>
              <div className="border-l-2 border-[#16b8a8]/40 pl-5">
                <p className="font-mono text-xs uppercase tracking-wider text-[#3ee07a]">{m.label}</p>
                <p className="mt-2 text-zinc-300">{m.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTA() {
  return (
    <section className="relative overflow-hidden border-t border-white/10 py-24">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(560px 300px at 50% 0%, rgba(22,184,168,0.16), transparent 70%)" }}
      />
      <div className={`${SECTION} relative text-center`}>
        <Reveal>
          <h2 className="mx-auto max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-5xl">
            See it analyze a real squat.
          </h2>
          <p className="mx-auto mt-4 max-w-md text-zinc-400">
            Upload a clip or open a labeled sample, and watch the skeleton, the faults, and the
            grounded feedback come back together.
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
          <a href="#how" className="transition-colors hover:text-zinc-100">How it works</a>
          <a href="#pipeline" className="transition-colors hover:text-zinc-100">Pipeline</a>
          <a href="#eval" className="transition-colors hover:text-zinc-100">Evaluation</a>
        </div>
        <p className="text-sm text-zinc-500">Explainable squat coaching, research prototype.</p>
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
  return (
    <Layout>
      <Defs />
      <Nav />
      <main>
        <Hero />
        <Problem />
        <Pipeline />
        <Diagnosis />
        <Bento />
        <Evaluation />
        <CTA />
      </main>
      <Footer />
    </Layout>
  );
}
