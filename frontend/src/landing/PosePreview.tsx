/**
 * Authentic preview of the product's own output: the pose-skeleton overlay the
 * dashboard renders over an analyzed squat, plus the grounded diagnosis card.
 * This is a real component preview (a mini version of the app UI), not a mock
 * dashboard built from stray divs. Side-view deep squat, faults flagged on the
 * knee where x-coach detects medial collapse.
 *
 * Built to the studio's shape (components/VideoPanel): a pale rounded card holding a black
 * stage, white glass pills floating over the clip, and a dark control bar along the bottom.
 * The clip stays dark on this otherwise-light page BECAUSE it is dark in the app — a light
 * "video" here would be the one thing on the page that doesn't match what a visitor gets.
 * Skeleton violet, flagged joint coral: the app's own overlay colours.
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
      <div className="relative overflow-hidden rounded-[24px] border border-[#e6e8f0] bg-[#d6dbe3] shadow-card">
        <div className="relative bg-black">
          {/* playhead readout — the studio's top-left status pill */}
          <div className="glass-over-video absolute left-4 top-4 z-10 flex items-center gap-2 rounded-full px-3 py-1.5 font-mono text-[11px] text-[#1e2142]">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            0:04 <span className="text-[#63709f]">/</span> descent
          </div>

          <svg viewBox="0 0 460 520" className="block w-full" role="img" aria-label="Pose skeleton overlay on a squat, knee valgus flagged">
            <defs>
              <linearGradient id="bone" x1="120" y1="150" x2="300" y2="430" gradientUnits="userSpaceOnUse">
                <stop offset="0" stopColor="#a78bfa" />
                <stop offset="1" stopColor="#7b61ff" />
              </linearGradient>
              <radialGradient id="preglow" cx="0.5" cy="0.4" r="0.6">
                <stop offset="0" stopColor="#7b61ff" stopOpacity="0.22" />
                <stop offset="1" stopColor="#7b61ff" stopOpacity="0" />
              </radialGradient>
            </defs>

            <rect width="460" height="520" fill="url(#preglow)" />

            {/* faint measurement grid */}
            <g stroke="#ffffff" strokeOpacity="0.05" strokeWidth="1">
              {Array.from({ length: 9 }).map((_, i) => (
                <line key={`v${i}`} x1={50 * i + 30} y1="0" x2={50 * i + 30} y2="520" />
              ))}
              {Array.from({ length: 11 }).map((_, i) => (
                <line key={`h${i}`} x1="0" y1={50 * i} x2="460" y2={50 * i} />
              ))}
            </g>

            {/* ground line */}
            <line x1="120" y1="424" x2="340" y2="424" stroke="#4a4f6b" strokeWidth="3" strokeLinecap="round" />

            {/* knee-over-toe vertical reference */}
            <line x1="266" y1="318" x2="266" y2="424" stroke="#ffffff" strokeOpacity="0.35" strokeWidth="2" strokeDasharray="4 6" />

            {/* knee-angle arc (the product's signature biomechanics cue), on the flagged joint */}
            <path d="M 238 309 A 30 30 0 0 0 252 345" fill="none" stroke="#ff6b6b" strokeWidth="3" strokeLinecap="round" />

            {/* bones */}
            <g stroke="url(#bone)" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" fill="none">
              {bones.map(([a, b], i) => (
                <line key={i} x1={j[a][0]} y1={j[a][1]} x2={j[b][0]} y2={j[b][1]} />
              ))}
            </g>

            {/* head */}
            <circle cx={j.head[0]} cy={j.head[1]} r="22" fill="url(#bone)" stroke="#f4f2ff" strokeWidth="3" />

            {/* keypoints */}
            <g fill="#f4f2ff" stroke="#12142a" strokeWidth="2.5">
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
              stroke="#ff6b6b"
              strokeWidth="2"
            />
            <circle cx={j.knee[0]} cy={j.knee[1]} r="7" fill="#ff6b6b" stroke="#12142a" strokeWidth="2.5" />
          </svg>

          {/* timeline strip — the studio's dark control pill: violet track, coral fault band */}
          <div className="absolute inset-x-3 bottom-3 rounded-full border border-white/15 bg-[#373a4a]/85 px-3 py-2.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.14),0_10px_28px_rgba(0,0,0,0.32)]">
            <div className="relative h-1.5 w-full rounded-full bg-white/15">
              <div className="absolute left-0 h-1.5 w-[44%] rounded-full bg-[#8b7bff]" />
              <div className="absolute left-[34%] h-1.5 w-[22%] rounded-full bg-[#ff6b6b]" />
              <div className="absolute left-[44%] top-1/2 h-3 w-3 -translate-y-1/2 rounded-full border-2 border-[#8b7bff] bg-white" />
            </div>
          </div>
        </div>
      </div>

      {/* grounded diagnosis card: the chain-of-thought output, on the studio's over-video card
          face. On phones it sits in normal flow just below the preview (floating it would cover
          the whole skeleton); from sm up it overlaps the corner. */}
      <div className="glass-over-video relative mx-auto mt-4 w-full max-w-xs rounded-2xl p-4 sm:absolute sm:-bottom-6 sm:-right-6 sm:mt-0 sm:w-64 sm:max-w-none">
        <div className="flex items-center justify-between">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-[#ffe0e0] bg-[#fff5f5] px-2.5 py-1 text-[11px] font-semibold text-[#e05252]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#ff6b6b]" />
            Moderate
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-[#9aa0b8]">fault</span>
        </div>
        <p className="mt-3 font-display text-lg font-semibold text-[#1e2142]">Knee valgus</p>
        <p className="mt-1 text-[13px] leading-snug text-[#59648f]">
          Left knee drifts inward at the bottom of the rep.
        </p>
        <div className="mt-3 border-t border-[#ebeaf6] pt-3">
          <p className="font-mono text-[10px] uppercase tracking-wider text-[#9aa0b8]">Traced to</p>
          <p className="mt-1 text-[13px] text-[#1e2142]">Hip abductor weakness (glute medius)</p>
        </div>
      </div>
    </div>
  );
}
