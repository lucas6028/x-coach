/**
 * Authentic preview of the product's own output: the pose-skeleton overlay the
 * dashboard renders over an analyzed squat, plus the grounded diagnosis card.
 * This is a real component preview (a mini version of the app UI), not a mock
 * dashboard built from stray divs. Side-view deep squat, faults flagged on the
 * knee where x-coach detects medial collapse.
 */
export default function PosePreview() {
  // Side-view skeleton joint coordinates (viewBox 460x520).
  const j = {
    head: [172, 150],
    shoulder: [180, 198],
    elbow: [224, 224],
    wrist: [272, 214],
    hip: [168, 286],
    knee: [266, 318],
    ankle: [214, 416],
    toe: [270, 424],
    heel: [182, 424],
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

  const dots = (["head", "shoulder", "elbow", "wrist", "hip", "ankle", "heel", "toe"] as const).map(
    (k) => j[k]
  );

  return (
    <div className="relative">
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-[#15191b] to-[#0d1011] shadow-[0_30px_80px_-30px_rgba(0,0,0,0.8)]">
        {/* playhead readout */}
        <div className="absolute left-4 top-4 z-10 flex items-center gap-2 rounded-full border border-white/10 bg-black/30 px-3 py-1 font-mono text-[11px] text-zinc-300 backdrop-blur">
          <span className="h-1.5 w-1.5 rounded-full bg-[#3ee07a]" />
          0:04 <span className="text-zinc-500">/</span> descent
        </div>

        <svg viewBox="0 0 460 520" className="block w-full" role="img" aria-label="Pose skeleton overlay on a squat, knee valgus flagged">
          <defs>
            <linearGradient id="bone" x1="120" y1="150" x2="300" y2="430" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#5ffb6f" />
              <stop offset="1" stopColor="#16b8a8" />
            </linearGradient>
            <radialGradient id="preglow" cx="0.5" cy="0.4" r="0.6">
              <stop offset="0" stopColor="#16b8a8" stopOpacity="0.18" />
              <stop offset="1" stopColor="#16b8a8" stopOpacity="0" />
            </radialGradient>
          </defs>

          <rect width="460" height="520" fill="url(#preglow)" />

          {/* faint measurement grid */}
          <g stroke="#ffffff" strokeOpacity="0.04" strokeWidth="1">
            {Array.from({ length: 9 }).map((_, i) => (
              <line key={`v${i}`} x1={50 * i + 30} y1="0" x2={50 * i + 30} y2="520" />
            ))}
            {Array.from({ length: 11 }).map((_, i) => (
              <line key={`h${i}`} x1="0" y1={50 * i} x2="460" y2={50 * i} />
            ))}
          </g>

          {/* ground line */}
          <line x1="120" y1="424" x2="340" y2="424" stroke="#3a4a4f" strokeWidth="3" strokeLinecap="round" />

          {/* knee-over-toe vertical reference */}
          <line x1="266" y1="318" x2="266" y2="424" stroke="#f5b945" strokeOpacity="0.55" strokeWidth="2" strokeDasharray="4 6" />

          {/* knee-angle arc (the product's signature biomechanics cue) */}
          <path d="M 238 309 A 30 30 0 0 0 252 345" fill="none" stroke="#f5b945" strokeWidth="3" strokeLinecap="round" />

          {/* bones */}
          <g stroke="url(#bone)" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" fill="none">
            {bones.map(([a, b], i) => (
              <line key={i} x1={j[a][0]} y1={j[a][1]} x2={j[b][0]} y2={j[b][1]} />
            ))}
          </g>

          {/* head */}
          <circle cx={j.head[0]} cy={j.head[1]} r="22" fill="url(#bone)" stroke="#eafff0" strokeWidth="3" />

          {/* keypoints */}
          <g fill="#eafff0" stroke="#0d1113" strokeWidth="2.5">
            {dots.map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r="5.5" />
            ))}
          </g>

          {/* flagged knee keypoint with tracking pulse (CSS-driven) */}
          <circle
            className="xc-pulse-ring"
            cx={j.knee[0]}
            cy={j.knee[1]}
            r="10"
            fill="none"
            stroke="#f5b945"
            strokeWidth="2"
          />
          <circle cx={j.knee[0]} cy={j.knee[1]} r="7" fill="#f5b945" stroke="#0d1113" strokeWidth="2.5" />
        </svg>

        {/* timeline strip */}
        <div className="border-t border-white/10 bg-black/30 px-4 py-3">
          <div className="relative h-1.5 w-full rounded-full bg-white/10">
            <div className="absolute left-[34%] h-1.5 w-[22%] rounded-full bg-[#f5b945]/70" />
            <div className="absolute left-[44%] top-1/2 h-3 w-3 -translate-y-1/2 rounded-full border-2 border-[#0d1011] bg-[#3ee07a]" />
          </div>
        </div>
      </div>

      {/* grounded diagnosis card: the chain-of-thought output.
          On phones it sits in normal flow just below the preview (floating it
          would cover the whole skeleton); from sm up it overlaps the corner. */}
      <div className="relative mx-auto mt-4 w-full max-w-xs rounded-2xl border border-white/10 bg-[#15191b]/90 p-4 shadow-[0_24px_60px_-20px_rgba(0,0,0,0.85)] backdrop-blur sm:absolute sm:-bottom-6 sm:-right-6 sm:mt-0 sm:w-64 sm:max-w-none">
        <div className="flex items-center justify-between">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-[#f5b945]/15 px-2.5 py-1 text-[11px] font-medium text-[#f5b945]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#f5b945]" />
            Moderate
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-zinc-500">fault</span>
        </div>
        <p className="mt-3 font-display text-lg font-semibold text-zinc-50">Knee valgus</p>
        <p className="mt-1 text-[13px] leading-snug text-zinc-400">
          Left knee drifts inward at the bottom of the rep.
        </p>
        <div className="mt-3 border-t border-white/10 pt-3">
          <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-500">Traced to</p>
          <p className="mt-1 text-[13px] text-zinc-200">Hip abductor weakness (glute medius)</p>
        </div>
      </div>
    </div>
  );
}
