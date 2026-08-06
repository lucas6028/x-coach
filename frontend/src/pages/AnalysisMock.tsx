import { useState } from "react";
import {
  ArrowsClockwise,
  ArrowsOut,
  Barbell,
  BookOpen,
  Camera,
  CaretDown,
  CaretRight,
  ChartBar,
  ClockCounterClockwise,
  Lightbulb,
  Paperclip,
  PaperPlaneRight,
  Play,
  SpeakerHigh,
  Sparkle,
  WarningCircle,
} from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import AppLayout from "../components/AppLayout";

/**
 * Design comp for the squat-analysis dashboard (route: /mockup/analysis).
 *
 * This is a PRESENTATIONAL PAGE ONLY — every number, message and session below is fixed demo
 * content declared at the top of this file. It calls no API, reads no session and mutates
 * nothing, so it can be opened signed-out to review the layout. The working studio is /app.
 *
 * It reproduces the reference mock-up's LAYOUT (stage + overlay readouts on the left, a coach
 * column on the right, three summary cards underneath) but renders it in the app's OWN design
 * system — primary teal, Space Grotesk, the surface/border/content/muted tokens and `shadow-card`
 * — rather than the reference's violet/glass palette, so it drops into the product without
 * forking the theme. Nothing outside this file and the route table changed.
 *
 * Copy is inline English rather than i18n keys: this is a comp, and seeding ~40 throwaway keys
 * into both dictionaries would outlive the design review. Anything promoted into the real studio
 * gets translated at that point.
 */

const DETECTED_ERRORS = [
  { name: "Knee angle too small", detail: "Knees collapse forward" },
  { name: "Back not neutral", detail: "Slight forward lean" },
] as const;

const PREVIOUS_SESSIONS = [
  { movement: "Squat", date: "Jan 20, 2025", score: 68 },
  { movement: "Squat", date: "Jan 18, 2025", score: 72 },
  { movement: "Squat", date: "Jan 15, 2025", score: 65 },
] as const;

// `tone` is the measured verdict, not decoration: "off" means the value sits outside the optimal
// band and is what the matching detected error was raised from.
const KEY_METRICS = [
  { label: "Knee Angle", value: "72°", optimal: "Optimal 80–100°", fill: 32, tone: "off" },
  { label: "Back Angle", value: "8°", optimal: "Optimal 0–10°", fill: 80, tone: "ok" },
  { label: "Depth", value: "Good", optimal: "Hips below parallel", fill: 88, tone: "ok" },
] as const;

const TIPS = [
  {
    title: "Push your hips back",
    body: "Keep your hips back and down, as if sitting into a chair.",
  },
  {
    title: "Keep chest up",
    body: "Maintain a neutral spine with chest open and core engaged.",
  },
  {
    title: "Knees in line with toes",
    body: "Keep your knees aligned with your toes and avoid them collapsing inward.",
  },
] as const;

const COACH_FEEDBACK = [
  {
    title: "Knee angle too small (72°)",
    body: "Your knees are collapsing forward past your toes. Try to keep your knees aligned with your toes and push your hips back more.",
  },
  {
    title: "Back not neutral",
    body: "You have a slight forward lean in your back. Keep your chest up and core engaged to maintain a neutral spine.",
  },
] as const;

const RELATED_INSIGHTS = [
  'Journal: "The Effect of Knee Position on Squat Performance"',
  'Journal: "Core Stability and Spinal Alignment During Squats"',
  'Article: "How to Improve Your Squat Form"',
] as const;

const FORM_SCORE = 68;

/** Card chrome shared by every panel on the page. */
const CARD = "rounded-2xl border border-border-dark bg-surface shadow-card";
/** Frosted readout floating over the dark stage — white-on-dark in both themes, like MetricsCards. */
const GLASS = "rounded-2xl bg-black/55 ring-1 ring-white/15 backdrop-blur-md";

/**
 * Side-view squat skeleton, drawn rather than photographed so the comp carries no binary asset.
 * Joint coordinates follow the same hand-placed idiom as landing/PosePreview. `annotated` adds the
 * knee-angle callout that the reference puts over the athlete's knee.
 */
function PoseFigure({ annotated = false }: { annotated?: boolean }) {
  const j = {
    head: [196, 132],
    shoulder: [204, 192],
    elbow: [246, 224],
    wrist: [220, 184],
    hip: [166, 292],
    knee: [274, 324],
    ankle: [224, 422],
    toe: [280, 432],
    heel: [190, 432],
  } as const;

  const bones: [keyof typeof j, keyof typeof j][] = [
    ["head", "shoulder"],
    ["shoulder", "hip"],
    ["hip", "knee"],
    ["knee", "ankle"],
    ["ankle", "toe"],
    ["ankle", "heel"],
    ["shoulder", "elbow"],
    ["elbow", "wrist"],
  ];

  return (
    <svg
      // Each variant gets a viewBox cropped to the content it actually shows, so the figure fills
      // its box instead of letterboxing: a 16:10 window including the angle callout on the stage,
      // and a tight portrait crop (sliced to fill) in the thumbnails.
      viewBox={annotated ? "-27 80 640 400" : "146 100 168 356"}
      preserveAspectRatio={annotated ? "xMidYMid meet" : "xMidYMid slice"}
      className="h-full w-full"
      role="img"
      aria-label="Pose skeleton overlay on a side-view squat"
    >
      <defs>
        <linearGradient id="xc-mock-bone" x1="150" y1="128" x2="300" y2="430" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#42d159" />
          <stop offset="1" stopColor="#0f758a" />
        </linearGradient>
      </defs>

      {bones.map(([a, b]) => (
        <line
          key={`${a}-${b}`}
          x1={j[a][0]}
          y1={j[a][1]}
          x2={j[b][0]}
          y2={j[b][1]}
          stroke="url(#xc-mock-bone)"
          strokeWidth={annotated ? 6 : 10}
          strokeLinecap="round"
        />
      ))}
      {Object.values(j).map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r={annotated ? 6 : 10} fill="#ffffff" />
      ))}

      {annotated && (
        <g>
          {/* The flagged knee: the two segments that form the angle, plus its arc and value. */}
          <path
            d={`M ${j.hip[0]} ${j.hip[1]} L ${j.knee[0]} ${j.knee[1]} L ${j.ankle[0]} ${j.ankle[1]}`}
            fill="none"
            stroke="#ef4444"
            strokeWidth="3"
            strokeDasharray="7 6"
          />
          {/* Arc swept between the two segments, 34px out from the knee vertex along each. */}
          <path d="M 241 314 A 34 34 0 0 0 258 354" fill="none" stroke="#ef4444" strokeWidth="3" />
          <text x="300" y="330" fill="#ef4444" fontSize="17" fontWeight="700">
            Knee Angle
          </text>
          <text x="300" y="352" fill="#ef4444" fontSize="17" fontWeight="700">
            72°
          </text>
        </g>
      )}
    </svg>
  );
}

/** A dropdown-shaped control. Inert on purpose — the comp has nothing to switch between. */
function SelectPill({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    // Grouped and named so "Squat" reads as the Exercise value rather than as loose text — the
    // page says "Squat" in three places (selector, stage, session list).
    <div role="group" aria-label={label} className={`flex items-center gap-3 px-4 py-2.5 ${CARD}`}>
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
        {icon}
      </span>
      <span className="min-w-0">
        <span className="block text-[11px] leading-none text-faint">{label}</span>
        <span className="mt-1 block truncate text-sm font-semibold leading-none text-content">{value}</span>
      </span>
      <CaretDown size={14} weight="bold" className="shrink-0 text-faint" />
    </div>
  );
}

/** Panel heading: icon + title, with an optional trailing link slot. */
function CardHead({
  icon,
  title,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 px-4 pt-4">
      <span className="text-primary">{icon}</span>
      <h2 className="flex-1 font-display text-sm font-bold text-content">{title}</h2>
      {action}
    </div>
  );
}

/** The 68% form-score dial. `dasharray` on a circle of known circumference draws the arc. */
function ScoreRing({ score }: { score: number }) {
  const r = 26;
  const c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 64 64" className="h-16 w-16 shrink-0 -rotate-90" role="img" aria-label={`Form score ${score} percent`}>
      <circle cx="32" cy="32" r={r} fill="none" stroke="currentColor" strokeWidth="6" className="text-white/20" />
      <circle
        cx="32"
        cy="32"
        r={r}
        fill="none"
        stroke="#42d159"
        strokeWidth="6"
        strokeLinecap="round"
        strokeDasharray={`${(c * score) / 100} ${c}`}
      />
    </svg>
  );
}

export default function AnalysisMock() {
  // The composer accepts typing so the control reads as real, but this page has no conversation
  // behind it: submitting is a no-op rather than a silently dropped message.
  const [draft, setDraft] = useState("");

  return (
    <AppLayout>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-[1500px] px-4 py-6 lg:px-6 lg:py-8">
          {/* ── page header: breadcrumb, title, and the session controls ─────────────── */}
          <nav aria-label="Breadcrumb" className="text-xs text-faint">
            <ol className="flex items-center gap-1.5">
              <li>
                <Link to="/" className="hover:text-content">
                  Home
                </Link>
              </li>
              <li aria-hidden>/</li>
              <li>
                <Link to="/app" className="hover:text-content">
                  Workout
                </Link>
              </li>
              <li aria-hidden>/</li>
              <li aria-current="page" className="text-muted">
                Squat Analysis
              </li>
            </ol>
          </nav>

          <div className="mt-3 flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div className="min-w-0">
              <h1 className="font-display text-3xl font-bold tracking-tight text-content">
                Squat Motion Analysis
              </h1>
              <p className="mt-1 text-sm text-muted">Get real-time feedback and improve your form</p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <SelectPill icon={<Barbell size={18} weight="duotone" />} label="Exercise" value="Squat" />
              <SelectPill icon={<Camera size={18} weight="duotone" />} label="Device" value="Webcam" />
              <Link
                to="/app"
                className="flex items-center gap-2 rounded-2xl bg-primary px-6 py-3.5 font-semibold text-primary-content shadow-accent transition-colors hover:bg-primary/90 active:translate-y-px"
              >
                <Play size={18} weight="fill" />
                Start / Upload Video
              </Link>
            </div>
          </div>

          {/* ── body: stage + summary cards on the left, coach column on the right ───── */}
          <div className="mt-6 grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1fr)_390px]">
            <div className="flex min-w-0 flex-col gap-4">
              {/* video stage */}
              <section
                aria-label="Live analysis"
                className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[#15191b] to-[#0d1011] shadow-card"
              >
                <div className="relative aspect-[16/10] w-full">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <PoseFigure annotated />
                  </div>

                  <span className="absolute left-4 top-4 flex items-center gap-2 rounded-full bg-black/55 px-3 py-1.5 text-xs font-semibold text-white ring-1 ring-white/15 backdrop-blur-md">
                    <span className="h-2 w-2 rounded-full bg-secondary" />
                    Live Analysis
                  </span>

                  <span
                    aria-hidden
                    className="absolute right-4 top-4 flex h-10 w-10 items-center justify-center rounded-full bg-black/55 text-white ring-1 ring-white/15 backdrop-blur-md"
                  >
                    <ArrowsOut size={18} weight="bold" />
                  </span>

                  {/* overlay readouts */}
                  <div className="absolute right-4 top-20 flex w-[15rem] max-w-[55%] flex-col gap-3">
                    <div className={`${GLASS} p-3.5`}>
                      <div className="flex items-center gap-2 text-white">
                        <WarningCircle size={16} weight="fill" className="text-danger" />
                        <span className="font-display text-sm font-bold">Detected Errors</span>
                      </div>
                      <ul className="mt-2.5 flex flex-col gap-2">
                        {DETECTED_ERRORS.map((e) => (
                          <li key={e.name} className="flex gap-2">
                            <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full ring-2 ring-danger" />
                            <span className="min-w-0">
                              <span className="block text-[13px] font-semibold leading-tight text-white">
                                {e.name}
                              </span>
                              <span className="block text-[11px] leading-tight text-white/60">{e.detail}</span>
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className={`${GLASS} p-3.5`}>
                      <span className="font-display text-sm font-bold text-white">Form Score</span>
                      <div className="mt-2.5 flex items-center gap-3">
                        <span className="relative flex items-center justify-center">
                          <ScoreRing score={FORM_SCORE} />
                          <span className="absolute font-display text-base font-bold text-white">
                            {FORM_SCORE}%
                          </span>
                        </span>
                        <span>
                          <span className="block font-display text-sm font-bold text-secondary">Good</span>
                          <span className="block text-[11px] text-white/60">Keep going!</span>
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* transport bar */}
                  <div className="absolute inset-x-0 bottom-0 flex items-center gap-3 bg-black/55 px-4 py-3 text-white backdrop-blur-md">
                    <Play size={20} weight="fill" className="shrink-0" />
                    <span className="shrink-0 font-mono text-xs tabular-nums text-white/80">0:12 / 0:30</span>
                    <span className="relative h-1 flex-1 rounded-full bg-white/20">
                      <span className="absolute inset-y-0 left-0 w-2/5 rounded-full bg-primary" />
                      {/* Arbitrary value, not `left-2/5`: Tailwind's inset scale has no fifths, so
                          the fraction silently resolved to 0 and parked the knob at the far left. */}
                      <span className="absolute -top-1 left-[40%] h-3 w-3 -translate-x-1/2 rounded-full bg-white" />
                    </span>
                    <SpeakerHigh size={18} className="shrink-0" />
                    <ArrowsClockwise size={18} className="shrink-0" />
                  </div>
                </div>
              </section>

              {/* three summary cards */}
              <div className="grid gap-4 md:grid-cols-3">
                <section className={CARD}>
                  <CardHead
                    icon={<ClockCounterClockwise size={16} weight="duotone" />}
                    title="Your Previous Sessions"
                    action={
                      <Link to="/history" className="text-xs font-semibold text-primary hover:underline">
                        View All
                      </Link>
                    }
                  />
                  <ul className="flex flex-col gap-2 p-3">
                    {PREVIOUS_SESSIONS.map((s) => (
                      <li key={s.date} className="flex items-center gap-3 rounded-xl p-1.5">
                        <span className="h-11 w-14 shrink-0 overflow-hidden rounded-lg bg-gradient-to-br from-[#15191b] to-[#0d1011]">
                          <PoseFigure />
                        </span>
                        {/* Movement over date rather than one line: at a third of the row the
                            single line truncated the date to "Jan …", which is the only part of
                            the entry that distinguishes one session from another. */}
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-xs font-semibold text-content">
                            {s.movement}
                          </span>
                          <span className="block truncate text-[11px] text-faint">{s.date}</span>
                        </span>
                        <span className="shrink-0 rounded-full bg-primary/10 px-2 py-1 text-[10px] font-semibold text-primary">
                          {s.score}%
                        </span>
                      </li>
                    ))}
                  </ul>
                </section>

                <section className={CARD}>
                  <CardHead icon={<ChartBar size={16} weight="duotone" />} title="Key Metrics" />
                  <dl className="grid grid-cols-3 gap-2 p-3">
                    {KEY_METRICS.map((m) => (
                      // `mt-auto` on the bar keeps the three rails on one line even though the
                      // optimal-band captions wrap to different heights.
                      <div
                        key={m.label}
                        className="flex flex-col rounded-xl border border-border-dark p-2.5"
                      >
                        <dt className="text-[10px] font-medium leading-tight text-faint">{m.label}</dt>
                        <dd className="mt-1.5 font-display text-xl font-bold leading-none text-content">
                          {m.value}
                        </dd>
                        <span className="mt-1.5 text-[9px] leading-tight text-faint">{m.optimal}</span>
                        <span className="mt-auto block pt-2">
                          <span className="block h-1.5 rounded-full bg-track">
                            <span
                              style={{ width: `${m.fill}%` }}
                              className={`block h-full rounded-full ${
                                m.tone === "off" ? "bg-danger" : "bg-secondary"
                              }`}
                            />
                          </span>
                        </span>
                      </div>
                    ))}
                  </dl>
                </section>

                <section className={CARD}>
                  <CardHead icon={<Lightbulb size={16} weight="duotone" />} title="Tips for Improvement" />
                  <ol className="flex flex-col gap-3 p-4">
                    {TIPS.map((tip, i) => (
                      <li key={tip.title} className="flex gap-2.5">
                        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-[11px] font-bold text-primary">
                          {i + 1}
                        </span>
                        <span className="min-w-0">
                          <span className="block text-xs font-semibold leading-tight text-content">
                            {tip.title}
                          </span>
                          <span className="mt-0.5 block text-[11px] leading-snug text-muted">{tip.body}</span>
                        </span>
                      </li>
                    ))}
                  </ol>
                </section>
              </div>
            </div>

            {/* ── coach column ──────────────────────────────────────────────────────── */}
            <section aria-label="AI Fitness Coach" className={`flex min-w-0 flex-col ${CARD}`}>
              <header className="flex items-center gap-2.5 border-b border-border-dark px-4 py-3.5">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <Sparkle size={18} weight="fill" />
                </span>
                <h2 className="flex-1 font-display text-base font-bold text-content">AI Fitness Coach</h2>
                <span className="flex items-center gap-1.5 text-xs font-medium text-secondary">
                  <span className="h-2 w-2 rounded-full bg-secondary" />
                  Online
                </span>
              </header>

              <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-4 scrollbar-thin">
                <div className="flex gap-2.5">
                  <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <Sparkle size={14} weight="fill" />
                  </span>
                  <div className="rounded-2xl rounded-tl-sm bg-content/5 p-3 text-[13px] leading-relaxed text-content">
                    <p>Hi! I've analyzed your squat. You're doing a great job overall!</p>
                    <p className="mt-2">
                      I noticed a couple of areas to improve. Would you like to see detailed feedback on your
                      form?
                    </p>
                  </div>
                </div>

                <div className="self-end rounded-2xl rounded-br-sm bg-primary px-3.5 py-2 text-[13px] font-medium text-primary-content">
                  Yes, please!
                </div>

                <div className="rounded-2xl rounded-tl-sm bg-content/5 p-3">
                  <p className="text-[13px] text-content">Here's the feedback based on your squat:</p>
                  <ul className="mt-3 flex flex-col gap-3">
                    {COACH_FEEDBACK.map((f) => (
                      <li key={f.title} className="flex gap-2.5">
                        <WarningCircle size={16} weight="fill" className="mt-0.5 shrink-0 text-danger" />
                        <span className="min-w-0 flex-1">
                          <span className="block text-[13px] font-semibold leading-tight text-content">
                            {f.title}
                          </span>
                          <span className="mt-1 block text-[11px] leading-snug text-muted">{f.body}</span>
                        </span>
                        <span className="h-14 w-11 shrink-0 overflow-hidden rounded-lg bg-gradient-to-br from-[#15191b] to-[#0d1011]">
                          <PoseFigure />
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* The retrieval provenance the studio's real answers carry — see lib/grounding. */}
                <div className="rounded-2xl border border-primary/25 bg-primary/[0.07] p-3">
                  <div className="flex items-center gap-2 text-primary">
                    <BookOpen size={15} weight="duotone" />
                    <span className="font-display text-xs font-bold">Related Insights</span>
                  </div>
                  <ul className="mt-2 flex flex-col gap-1.5">
                    {RELATED_INSIGHTS.map((s) => (
                      <li key={s} className="flex gap-2 text-[11px] leading-snug text-muted">
                        <CaretRight size={10} weight="bold" className="mt-1 shrink-0 text-primary" />
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              <form
                onSubmit={(e) => e.preventDefault()}
                className="flex items-center gap-2 border-t border-border-dark p-3"
              >
                <Paperclip size={18} className="shrink-0 text-faint" />
                <input
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  aria-label="Ask me anything about your workout"
                  placeholder="Ask me anything about your workout..."
                  className="min-w-0 flex-1 bg-transparent text-sm text-content outline-none placeholder:text-faint"
                />
                <button
                  type="submit"
                  aria-label="Send message"
                  className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-content shadow-accent transition-colors hover:bg-primary/90"
                >
                  <PaperPlaneRight size={17} weight="fill" />
                </button>
              </form>
            </section>
          </div>
        </main>
      </div>
    </AppLayout>
  );
}
