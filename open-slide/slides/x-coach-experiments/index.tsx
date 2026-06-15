import type { DesignSystem, Page, SlideMeta, SlideTransition } from '@open-slide/core';
import { useSlidePageNumber } from '@open-slide/core';

export const design: DesignSystem = {
  palette: { bg: '#090d14', text: '#eef2f7', accent: '#a8ff35' },
  fonts: {
    display:
      'system-ui, -apple-system, "PingFang TC", "Microsoft JhengHei", "Noto Sans TC", sans-serif',
    body:
      'system-ui, -apple-system, "PingFang TC", "Microsoft JhengHei", "Noto Sans TC", sans-serif',
  },
  typeScale: { hero: 184, body: 36 },
  radius: 18,
};

// --- Extra tokens outside the DesignSystem shape -------------------------
const cyan = '#34d8f0';
const muted = '#7e8a9c';
const faint = '#566175';
const panel = '#0f1622';
const panelHi = '#131c2b';
const line = 'rgba(255,255,255,0.07)';
const amber = '#f5b942';
const danger = '#f0675a';
const mono =
  'ui-monospace, "SF Mono", "JetBrains Mono", "Cascadia Code", Menlo, monospace';
const GRAD = `linear-gradient(90deg, ${'#a8ff35'} 0%, ${cyan} 100%)`;

// --- One-time global keyframes (idempotent) ------------------------------
const KEYFRAMES = `
@keyframes osd-pulse { 0%,100% { opacity:.55; } 50% { opacity:1; } }
@keyframes osd-fadeup { from { opacity:0; transform:translateY(10px);} to { opacity:1; transform:translateY(0);} }
@media (prefers-reduced-motion: reduce){
  *{ animation: none !important; }
}`;
if (typeof document !== 'undefined' && !document.getElementById('osd-xcoach-kf')) {
  const s = document.createElement('style');
  s.id = 'osd-xcoach-kf';
  s.textContent = KEYFRAMES;
  document.head.appendChild(s);
}

const fill = {
  width: '100%',
  height: '100%',
  fontFamily: 'var(--osd-font-body)',
  background: 'var(--osd-bg)',
  color: 'var(--osd-text)',
  position: 'relative',
  overflow: 'hidden',
} as const;

// =========================================================================
// Decorative motion-capture skeleton (squat pose)
// =========================================================================
const PTS: Record<string, [number, number]> = {
  nose: [120, 44],
  neck: [120, 86],
  lsh: [90, 92],
  rsh: [150, 92],
  lel: [72, 146],
  rel: [168, 146],
  lwr: [106, 170],
  rwr: [134, 170],
  pel: [120, 198],
  lhip: [98, 194],
  rhip: [142, 194],
  lkn: [78, 270],
  rkn: [162, 270],
  lank: [92, 340],
  rank: [148, 340],
  lft: [76, 350],
  rft: [164, 350],
};
const BONES: [string, string][] = [
  ['nose', 'neck'],
  ['neck', 'lsh'],
  ['neck', 'rsh'],
  ['lsh', 'lel'],
  ['lel', 'lwr'],
  ['rsh', 'rel'],
  ['rel', 'rwr'],
  ['lsh', 'lhip'],
  ['rsh', 'rhip'],
  ['neck', 'pel'],
  ['lhip', 'rhip'],
  ['pel', 'lhip'],
  ['pel', 'rhip'],
  ['lhip', 'lkn'],
  ['lkn', 'lank'],
  ['rhip', 'rkn'],
  ['rkn', 'rank'],
  ['lank', 'lft'],
  ['rank', 'rft'],
];

const Skeleton = ({
  size = 520,
  opacity = 1,
  glow = true,
}: {
  size?: number;
  opacity?: number;
  glow?: boolean;
}) => (
  <svg
    width={size}
    height={size * (390 / 240)}
    viewBox="0 0 240 390"
    style={{ opacity, display: 'block' }}
  >
    <defs>
      <linearGradient id="boneGrad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stopColor="#a8ff35" />
        <stop offset="100%" stopColor={cyan} />
      </linearGradient>
    </defs>
    {BONES.map(([a, b], i) => (
      <line
        key={i}
        x1={PTS[a][0]}
        y1={PTS[a][1]}
        x2={PTS[b][0]}
        y2={PTS[b][1]}
        stroke="url(#boneGrad)"
        strokeWidth={2.4}
        strokeLinecap="round"
        opacity={0.5}
      />
    ))}
    {Object.keys(PTS).map((k, i) => {
      const [x, y] = PTS[k];
      const r = ['lkn', 'rkn', 'lhip', 'rhip', 'neck'].includes(k) ? 7.5 : 5.5;
      return (
        <g
          key={k}
          style={
            glow
              ? { animation: `osd-pulse 3.4s ease-in-out ${(i % 6) * 0.25}s infinite` }
              : undefined
          }
        >
          <circle cx={x} cy={y} r={r + 5} fill="#a8ff35" opacity={0.12} />
          <circle cx={x} cy={y} r={r} fill={i % 2 ? cyan : '#a8ff35'} />
        </g>
      );
    })}
  </svg>
);

// faint full-bleed grid for the technical-editorial feel
const GridBg = () => (
  <div
    style={{
      position: 'absolute',
      inset: 0,
      backgroundImage: `linear-gradient(${line} 1px, transparent 1px), linear-gradient(90deg, ${line} 1px, transparent 1px)`,
      backgroundSize: '120px 120px',
      maskImage: 'radial-gradient(circle at 70% 40%, #000 0%, transparent 75%)',
      WebkitMaskImage: 'radial-gradient(circle at 70% 40%, #000 0%, transparent 75%)',
      pointerEvents: 'none',
    }}
  />
);

// =========================================================================
// Shared chrome
// =========================================================================
const Eyebrow = ({ children }: { children: React.ReactNode }) => (
  <div
    style={{
      fontFamily: mono,
      fontSize: 22,
      letterSpacing: '0.32em',
      color: 'var(--osd-accent)',
      textTransform: 'uppercase',
      display: 'flex',
      alignItems: 'center',
      gap: 16,
    }}
  >
    <span style={{ width: 40, height: 2, background: GRAD, display: 'inline-block' }} />
    {children}
  </div>
);

const Footer = ({ tag }: { tag: string }) => {
  const { current, total } = useSlidePageNumber();
  return (
    <div
      style={{
        position: 'absolute',
        left: 120,
        right: 120,
        bottom: 48,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontFamily: mono,
        fontSize: 19,
        color: faint,
        letterSpacing: '0.12em',
      }}
    >
      <span>x-coach · {tag}</span>
      <span>
        <span style={{ color: 'var(--osd-text)' }}>{String(current).padStart(2, '0')}</span>
        {' / '}
        {String(total).padStart(2, '0')}
      </span>
    </div>
  );
};

const Shell = ({
  eyebrow,
  heading,
  tag,
  children,
}: {
  eyebrow: string;
  heading: React.ReactNode;
  tag: string;
  children: React.ReactNode;
}) => (
  <div style={{ ...fill, padding: '92px 120px 120px' }}>
    <GridBg />
    <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Eyebrow>{eyebrow}</Eyebrow>
      <h2
        style={{
          fontFamily: 'var(--osd-font-display)',
          fontSize: 60,
          fontWeight: 850,
          lineHeight: 1.12,
          margin: '24px 0 0',
          letterSpacing: '-0.01em',
        }}
      >
        {heading}
      </h2>
      <div style={{ marginTop: 44, flex: 1 }}>{children}</div>
    </div>
    <Footer tag={tag} />
  </div>
);

// =========================================================================
// Reusable content atoms
// =========================================================================
const Stat = ({
  value,
  unit,
  label,
  note,
  color = 'var(--osd-accent)',
}: {
  value: string;
  unit?: string;
  label: string;
  note?: string;
  color?: string;
}) => (
  <div
    style={{
      flex: 1,
      background: panel,
      border: `1px solid ${line}`,
      borderRadius: 'var(--osd-radius)',
      padding: '38px 36px',
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
    }}
  >
    <div style={{ fontFamily: mono, fontSize: 76, fontWeight: 800, color, lineHeight: 1 }}>
      {value}
      {unit && <span style={{ fontSize: 30, color: muted, marginLeft: 6 }}>{unit}</span>}
    </div>
    <div style={{ fontSize: 28, fontWeight: 600 }}>{label}</div>
    {note && <div style={{ fontSize: 21, color: muted, lineHeight: 1.5 }}>{note}</div>}
  </div>
);

// horizontal balanced-accuracy bar with a random-baseline marker at 0.50
const Bar = ({
  label,
  sub,
  value,
  display,
  color = GRAD,
  best = false,
}: {
  label: string;
  sub?: string;
  value: number; // 0..1
  display: string;
  color?: string;
  best?: boolean;
}) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 28 }}>
    <div style={{ width: 360, flexShrink: 0 }}>
      <div style={{ fontSize: 27, fontWeight: 700, color: best ? 'var(--osd-text)' : '#cdd6e3' }}>
        {label}
      </div>
      {sub && <div style={{ fontSize: 18, color: muted, fontFamily: mono, marginTop: 4 }}>{sub}</div>}
    </div>
    <div
      style={{
        flex: 1,
        height: 30,
        background: '#0c1320',
        border: `1px solid ${line}`,
        borderRadius: 8,
        position: 'relative',
      }}
    >
      {/* random baseline marker at 0.50 */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: -7,
          bottom: -7,
          width: 2,
          background: faint,
          opacity: 0.7,
        }}
      />
      <div
        style={{
          width: `${Math.min(value, 1) * 100}%`,
          height: '100%',
          background: color,
          borderRadius: 7,
          boxShadow: best ? `0 0 22px rgba(168,255,53,0.35)` : 'none',
        }}
      />
    </div>
    <div
      style={{
        width: 120,
        flexShrink: 0,
        textAlign: 'right',
        fontFamily: mono,
        fontSize: 34,
        fontWeight: 800,
        color: best ? 'var(--osd-accent)' : 'var(--osd-text)',
      }}
    >
      {display}
    </div>
  </div>
);

// =========================================================================
// 01 — Cover
// =========================================================================
const Cover: Page = () => (
  <div style={{ ...fill, padding: '0 140px', display: 'flex', alignItems: 'center' }}>
    <GridBg />
    <div
      style={{
        position: 'absolute',
        right: 120,
        top: '50%',
        transform: 'translateY(-50%)',
        opacity: 0.92,
      }}
    >
      <Skeleton size={500} />
    </div>
    <div style={{ position: 'relative', maxWidth: 1150 }}>
      <Eyebrow>研究進度報告 · 2026.06</Eyebrow>
      <h1
        style={{
          fontFamily: 'var(--osd-font-display)',
          fontSize: 'var(--osd-size-hero)',
          fontWeight: 900,
          margin: '30px 0 0',
          lineHeight: 0.94,
          letterSpacing: '-0.02em',
        }}
      >
        x-coach
      </h1>
      <div
        style={{
          marginTop: 28,
          fontSize: 46,
          fontWeight: 700,
          lineHeight: 1.3,
          maxWidth: 980,
        }}
      >
        結合知識檢索與多模態推理的
        <span
          style={{
            background: GRAD,
            WebkitBackgroundClip: 'text',
            backgroundClip: 'text',
            color: 'transparent',
          }}
        >
          可解釋 AI 運動教練
        </span>
      </div>
      <p style={{ fontSize: 28, color: muted, marginTop: 30, lineHeight: 1.6, maxWidth: 880 }}>
        感知 · 生物力學規則 · 知識檢索 — 五組實驗的成果彙整與下一步
      </p>
      <div
        style={{
          marginTop: 48,
          display: 'flex',
          gap: 14,
          fontFamily: mono,
          fontSize: 20,
          color: faint,
          flexWrap: 'wrap',
        }}
      >
        <span style={{ border: `1px solid ${line}`, borderRadius: 999, padding: '8px 18px' }}>
          MediaPipe · MMPose
        </span>
        <span style={{ border: `1px solid ${line}`, borderRadius: 999, padding: '8px 18px' }}>
          VideoMAE
        </span>
        <span style={{ border: `1px solid ${line}`, borderRadius: 999, padding: '8px 18px' }}>
          REHAB24-6 LOSO
        </span>
        <span style={{ border: `1px solid ${line}`, borderRadius: 999, padding: '8px 18px' }}>
          GraphRAG
        </span>
      </div>
    </div>
  </div>
);

// =========================================================================
// 02 — Problem framing
// =========================================================================
const GapCard = ({
  tag,
  title,
  body,
  color,
}: {
  tag: string;
  title: string;
  body: string;
  color: string;
}) => (
  <div
    style={{
      flex: 1,
      background: panel,
      border: `1px solid ${line}`,
      borderTop: `3px solid ${color}`,
      borderRadius: 'var(--osd-radius)',
      padding: '40px 40px 44px',
    }}
  >
    <div style={{ fontFamily: mono, fontSize: 20, letterSpacing: '0.2em', color }}>{tag}</div>
    <div style={{ fontSize: 38, fontWeight: 800, margin: '18px 0 16px' }}>{title}</div>
    <div style={{ fontSize: 27, color: '#c2ccd9', lineHeight: 1.6 }}>{body}</div>
  </div>
);

const Problem: Page = () => (
  <Shell eyebrow="問題定義" heading="為什麼需要「有據可依」的 AI 教練？" tag="motivation">
    <div style={{ display: 'flex', gap: 40, marginTop: 8 }}>
      <GapCard
        tag="現況 A"
        title="傳統 AQA 只給分"
        body="動作品質評估輸出一個分數，卻說不出哪裡錯、為什麼錯、怎麼改 — 對使用者沒有可執行的回饋。"
        color={amber}
      />
      <GapCard
        tag="現況 B"
        title="通用 LLM 會幻覺"
        body="大型語言模型能給建議，但缺乏生物力學依據、容易杜撰，無法對應到影片中真正發生的錯誤。"
        color={danger}
      />
    </div>
    <div
      style={{
        marginTop: 44,
        background: panelHi,
        border: `1px solid ${line}`,
        borderLeft: `4px solid ${'#a8ff35'}`,
        borderRadius: 'var(--osd-radius)',
        padding: '34px 40px',
        display: 'flex',
        alignItems: 'center',
        gap: 28,
      }}
    >
      <div style={{ fontFamily: mono, fontSize: 40, color: 'var(--osd-accent)' }}>→</div>
      <div style={{ fontSize: 30, lineHeight: 1.55 }}>
        <b>x-coach 的主張：</b>把視覺訊號接上可檢索的生物力學知識，
        產生<span style={{ color: 'var(--osd-accent)' }}>可解釋、可執行、有來源</span>的教練回饋。
      </div>
    </div>
  </Shell>
);

// =========================================================================
// 03 — System architecture (four pipelines)
// =========================================================================
const PipeNode = ({
  n,
  title,
  sub,
  done,
}: {
  n: string;
  title: string;
  sub: string;
  done: boolean;
}) => (
  <div
    style={{
      flex: 1,
      background: done ? panel : 'transparent',
      border: `1px solid ${done ? line : 'rgba(255,255,255,0.12)'}`,
      borderStyle: done ? 'solid' : 'dashed',
      borderRadius: 'var(--osd-radius)',
      padding: '30px 26px 32px',
      display: 'flex',
      flexDirection: 'column',
      gap: 12,
      opacity: done ? 1 : 0.7,
    }}
  >
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <span style={{ fontFamily: mono, fontSize: 30, fontWeight: 800, color: 'var(--osd-accent)' }}>
        {n}
      </span>
      <span
        style={{
          fontFamily: mono,
          fontSize: 15,
          letterSpacing: '0.14em',
          color: done ? cyan : faint,
          border: `1px solid ${done ? 'rgba(52,216,240,0.4)' : line}`,
          borderRadius: 999,
          padding: '4px 12px',
        }}
      >
        {done ? '已實作' : '建構中'}
      </span>
    </div>
    <div style={{ fontSize: 30, fontWeight: 800, lineHeight: 1.2 }}>{title}</div>
    <div style={{ fontSize: 20, color: muted, lineHeight: 1.55 }}>{sub}</div>
  </div>
);

const Arrow = () => (
  <div style={{ fontFamily: mono, fontSize: 34, color: faint, alignSelf: 'center', flexShrink: 0 }}>
    →
  </div>
);

const Architecture: Page = () => (
  <Shell
    eyebrow="系統架構"
    heading={<>四條管線、一個閉環：從細粒度感知到有據推理</>}
    tag="architecture"
  >
    <div style={{ display: 'flex', alignItems: 'stretch', gap: 18, marginTop: 20 }}>
      <PipeNode
        n="01"
        title="多模態感知"
        sub="Pose：MediaPipe / MMPose 骨架；Video：VideoMAE 時空特徵；錯誤的時間定位。"
        done
      />
      <Arrow />
      <PipeNode
        n="02"
        title="生物力學規則"
        sub="可解釋的 squat-fault 規則（膝前移、膝內扣、腳跟抬起），門檻可調。"
        done
      />
      <Arrow />
      <PipeNode
        n="03"
        title="GraphRAG 檢索"
        sub="離線向量資料庫 + 深蹲知識圖譜，把錯誤現象追溯到肌肉成因。"
        done
      />
      <Arrow />
      <PipeNode
        n="04"
        title="推理與生成"
        sub="LLM + Chain-of-Thought，輸出診斷與修正處方，串接互動前端。"
        done={false}
      />
    </div>
    <div
      style={{
        marginTop: 40,
        fontSize: 25,
        color: '#c2ccd9',
        lineHeight: 1.6,
        background: panel,
        border: `1px solid ${line}`,
        borderRadius: 'var(--osd-radius)',
        padding: '28px 36px',
      }}
    >
      本報告聚焦<span style={{ color: 'var(--osd-accent)' }}>前三層的實證基礎</span> — 感知、規則、檢索；
      推理／生成與前端仍在建構中。下面五組實驗，回答「哪一種感知訊號，最值得作為這個閉環的地基」。
    </div>
  </Shell>
);

// =========================================================================
// 04 — Datasets
// =========================================================================
const ViewRow = ({ label, n, pct }: { label: string; n: string; pct: number }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: 22 }}>
    <span style={{ width: 130, color: '#c2ccd9', fontFamily: mono }}>{label}</span>
    <div style={{ flex: 1, height: 16, background: '#0c1320', borderRadius: 6, overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: GRAD }} />
    </div>
    <span style={{ width: 70, textAlign: 'right', fontFamily: mono, color: muted }}>{n}</span>
  </div>
);

const DatasetCard = ({
  tag,
  title,
  children,
}: {
  tag: string;
  title: string;
  children: React.ReactNode;
}) => (
  <div
    style={{
      flex: 1,
      background: panel,
      border: `1px solid ${line}`,
      borderRadius: 'var(--osd-radius)',
      padding: '36px 38px 40px',
      display: 'flex',
      flexDirection: 'column',
    }}
  >
    <div style={{ fontFamily: mono, fontSize: 20, letterSpacing: '0.18em', color: cyan }}>{tag}</div>
    <div style={{ fontSize: 36, fontWeight: 800, margin: '14px 0 24px' }}>{title}</div>
    {children}
  </div>
);

const Datasets: Page = () => (
  <Shell
    eyebrow="資料集"
    heading="聚焦深蹲，並用 REHAB24-6 量測單目逼近度"
    tag="datasets"
  >
    <div style={{ display: 'flex', gap: 40, height: '100%' }}>
      <DatasetCard tag="主資料集" title="Squat Labeled Dataset">
        <div style={{ fontSize: 25, color: '#c2ccd9', lineHeight: 1.6, marginBottom: 26 }}>
          1,623 支深蹲影片，三種標籤：膝前移 / 膝內扣 / 合併。多數為斜後方視角 — 直接影響不同錯誤的可觀察性。
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <ViewRow label="rear_oblique" n="1075" pct={100} />
          <ViewRow label="rear" n="410" pct={38} />
          <ViewRow label="side" n="138" pct={13} />
        </div>
      </DatasetCard>
      <DatasetCard tag="對照基準" title="REHAB24-6">
        <div style={{ fontSize: 25, color: '#c2ccd9', lineHeight: 1.6, marginBottom: 26 }}>
          多模態復健資料集，以 repetition 為單位的二元 correctness。同時有 RGB 與高保真動捕，可量化「單目 vs 動捕」。
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14 }}>
          <Stat value="1,072" label="repetition 標註" color={cyan} />
          <Stat value="6" label="復健動作" color={cyan} />
        </div>
        <div style={{ fontSize: 21, color: muted, marginTop: 20, fontFamily: mono, lineHeight: 1.6 }}>
          10 受試者 · 2 RGB + 16 mocap 相機 · subject-wise 切分
        </div>
      </DatasetCard>
    </div>
  </Shell>
);

// =========================================================================
// 05 — Experiment 1: VideoMAE-only baseline / the F1 trap
// =========================================================================
const Exp1: Page = () => (
  <Shell
    eyebrow="實驗一 · VideoMAE-only"
    heading="高 F1 的陷阱：評估紀律比模型更先決"
    tag="exp-01"
  >
    <div style={{ display: 'flex', gap: 28, marginTop: 6 }}>
      <Stat
        value="0.83"
        label="always-positive 的 F1"
        note="全部判「有錯」就能拿到的分數 — F1 高不代表模型有效。"
        color={danger}
      />
      <Stat
        value="0.04"
        label="某次 run 的 specificity"
        note="recall ≈ 0.99 卻幾乎認不出正常深蹲，不能當教練回饋。"
        color={amber}
      />
      <Stat
        value="0.555"
        label="combined balanced acc."
        note="改用平衡準確率後，VideoMAE-only 僅略高於隨機 0.50。"
        color="var(--osd-accent)"
      />
    </div>
    <div
      style={{
        marginTop: 44,
        background: panelHi,
        border: `1px solid ${line}`,
        borderLeft: `4px solid ${cyan}`,
        borderRadius: 'var(--osd-radius)',
        padding: '30px 38px',
        fontSize: 27,
        lineHeight: 1.6,
      }}
    >
      <b style={{ color: cyan }}>教訓：</b>以 balanced accuracy、specificity、macro-F1 取代 positive-class F1；
      並固定 seed、改用 validation 最佳 checkpoint、回報 always-positive／negative 基準。
      問題不在分類器超參數，而在<span style={{ color: 'var(--osd-accent)' }}>特徵與標籤設計</span>。
    </div>
  </Shell>
);

// =========================================================================
// 06 — Experiment 2: pose geometry > single RGB embedding
// =========================================================================
const Exp2: Page = () => (
  <Shell
    eyebrow="實驗二 · Pose-only"
    heading="姿態幾何，勝過單一 RGB 嵌入"
    tag="exp-02"
  >
    <div
      style={{
        fontFamily: mono,
        fontSize: 19,
        color: muted,
        letterSpacing: '0.1em',
        marginBottom: 26,
      }}
    >
      COMBINED · TEST · SELECTED-THRESHOLD BALANCED ACCURACY（虛線 = 隨機 0.50）
    </div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 30 }}>
      <Bar label="VideoMAE-only" sub="generic RGB embedding" value={0.555} display="0.555" color={faint} />
      <Bar label="Pose-only（未正規化）" sub="MediaPipe geometry" value={0.581} display="0.581" color={cyan} />
      <Bar
        label="Pose-only（train-set 正規化）"
        sub="best MediaPipe baseline"
        value={0.635}
        display="0.635"
        best
      />
    </div>
    <div
      style={{
        marginTop: 42,
        fontSize: 26,
        lineHeight: 1.6,
        color: '#c2ccd9',
        background: panel,
        border: `1px solid ${line}`,
        borderRadius: 'var(--osd-radius)',
        padding: '28px 36px',
      }}
    >
      姿態幾何提供更貼近錯誤定義的生物力學訊號。train-set 正規化再 <b style={{ color: 'var(--osd-accent)' }}>+0.054</b>，
      主要來自 specificity 提升（不再過度預測「有錯」）— 已設為 pose-only 預設。
    </div>
  </Shell>
);

// =========================================================================
// 07 — Experiment 3: MMPose vs MediaPipe backend
// =========================================================================
const Exp3: Page = () => (
  <Shell
    eyebrow="實驗三 · Backend 比較"
    heading="換 backbone：MMPose 不是萬用升級"
    tag="exp-03"
  >
    <div
      style={{
        fontFamily: mono,
        fontSize: 19,
        color: muted,
        letterSpacing: '0.1em',
        marginBottom: 24,
      }}
    >
      KNEES_INWARD 分類器 · TEST BALANCED ACCURACY（5-seed mean）
    </div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
      <Bar label="MediaPipe（正規化）" sub="recall 0.578" value={0.608} display="0.608" color={cyan} />
      <Bar label="MMPose（正規化）" sub="recall 0.789 · +0.094" value={0.702} display="0.702" best />
    </div>
    <div style={{ display: 'flex', gap: 28, marginTop: 40 }}>
      <div
        style={{
          flex: 1,
          background: panel,
          border: `1px solid ${line}`,
          borderRadius: 'var(--osd-radius)',
          padding: '26px 32px',
          fontSize: 24,
          lineHeight: 1.55,
        }}
      >
        <b style={{ color: 'var(--osd-accent)' }}>各取所長：</b>MMPose 是最佳膝內扣偵測器；
        combined 上 MediaPipe（0.635）仍略勝（specificity 更穩）。
      </div>
      <div
        style={{
          flex: 1,
          background: panel,
          border: `1px solid ${line}`,
          borderRadius: 'var(--osd-radius)',
          padding: '26px 32px',
          fontSize: 24,
          lineHeight: 1.55,
        }}
      >
        <b style={{ color: amber }}>規則仍未解：</b>knees_forward 幾乎偵測不到（recall 0.005）；
        inward 規則換 backend 後 recall 升、但誤報翻倍 — 門檻需各自重調。
      </div>
    </div>
  </Shell>
);

// =========================================================================
// 08 — Experiment 4: REHAB24-6 LOSO, monocular vs mocap (headline)
// =========================================================================
const Exp4: Page = () => (
  <Shell
    eyebrow="實驗四 · REHAB24-6 LOSO"
    heading="便宜的單目骨架，逼近昂貴的光學動捕"
    tag="exp-04"
  >
    <div
      style={{
        fontFamily: mono,
        fontSize: 19,
        color: muted,
        letterSpacing: '0.1em',
        marginBottom: 22,
      }}
    >
      CORRECTNESS · LOSO 9-FOLD MEAN BALANCED ACCURACY（虛線 = 隨機 0.50）
    </div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
      <Bar label="Vicon 動捕骨架" sub="±0.078 · 上限基準" value={0.702} display="0.702" best />
      <Bar
        label="MediaPipe 單目（pseudo-3D）"
        sub="±0.055 · 可部署基準"
        value={0.633}
        display="0.633"
        color={cyan}
      />
      <Bar label="MMPose 單目（2D-only）" sub="±0.051 · 丟了深度通道" value={0.57} display="0.570" color={faint} />
      <Bar label="VideoMAE" sub="±0.044 · 近隨機" value={0.536} display="0.536" color={faint} />
    </div>
    <div style={{ display: 'flex', gap: 28, marginTop: 34 }}>
      <div
        style={{
          flex: 1.4,
          background: panelHi,
          border: `1px solid ${line}`,
          borderLeft: `4px solid ${'#a8ff35'}`,
          borderRadius: 'var(--osd-radius)',
          padding: '24px 32px',
          fontSize: 24,
          lineHeight: 1.55,
        }}
      >
        <b style={{ color: 'var(--osd-accent)' }}>深蹲 Ex6：</b>MediaPipe <b>0.714</b> ≥ Vicon 0.650 —
        在目標動作上，單目並不輸動捕。對「可部署單鏡頭深蹲教練」是最直接的正面證據。
      </div>
      <div
        style={{
          flex: 1,
          background: panel,
          border: `1px solid ${line}`,
          borderRadius: 'var(--osd-radius)',
          padding: '24px 32px',
          fontSize: 24,
          lineHeight: 1.55,
        }}
      >
        <b style={{ color: danger }}>VideoMAE 過擬合：</b>train→test 落差 <b>0.34</b>，
        subject-wise 下記住的是身份而非動作。
      </div>
    </div>
  </Shell>
);

// =========================================================================
// 09 — Experiment 5: R1 temporal smoothing
// =========================================================================
const Exp5: Page = () => (
  <Shell
    eyebrow="實驗五 · R1 時間平滑"
    heading="時間平滑只在深蹲有效 — 動作別效應"
    tag="exp-05"
  >
    <div style={{ display: 'flex', gap: 28, marginTop: 4 }}>
      <Stat
        value="+0.024"
        label="深蹲 Ex6（window=5）"
        note="0.720 → 0.744，五個視窗方向一致 — 真訊號，非單點僥倖。"
        color="var(--osd-accent)"
      />
      <Stat
        value="−0.045"
        label="上肢 arm abduction"
        note="0.684 → 0.639，平滑吃掉轉折訊號；與深蹲收益相互抵銷。"
        color={danger}
      />
      <Stat
        value="≈ 0"
        label="全 6 動作聚合 delta"
        note="Wilcoxon p ≥ 0.30；單一改動的效益 < 折間雜訊。"
        color={muted}
      />
    </div>
    <div
      style={{
        marginTop: 42,
        background: panelHi,
        border: `1px solid ${line}`,
        borderLeft: `4px solid ${cyan}`,
        borderRadius: 'var(--osd-radius)',
        padding: '30px 38px',
        fontSize: 26,
        lineHeight: 1.6,
      }}
    >
      <b style={{ color: cyan }}>評估紀律：</b>真實效益常小於評估雜訊，必須以
      <span style={{ color: 'var(--osd-accent)' }}> LOSO mean±std + 分動作</span> 才看得見。
      採用 <span style={{ fontFamily: mono }}>window=5、polyorder=2</span> 作為後續疊加 baseline。
    </div>
  </Shell>
);

// =========================================================================
// 10 — Closing: takeaways & next steps
// =========================================================================
const TakeRow = ({ n, children }: { n: string; children: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start' }}>
    <span
      style={{
        fontFamily: mono,
        fontSize: 24,
        fontWeight: 800,
        color: 'var(--osd-accent)',
        flexShrink: 0,
        marginTop: 2,
      }}
    >
      {n}
    </span>
    <span style={{ fontSize: 25, lineHeight: 1.5, color: '#d3dbe6' }}>{children}</span>
  </div>
);

const NextRow = ({ children }: { children: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
    <span style={{ color: cyan, fontSize: 25, flexShrink: 0, marginTop: 1 }}>▸</span>
    <span style={{ fontSize: 25, lineHeight: 1.5, color: '#d3dbe6' }}>{children}</span>
  </div>
);

const Closing: Page = () => (
  <Shell eyebrow="結論與下一步" heading="幾何路線可部署，推理層是下一座橋" tag="takeaways">
    <div style={{ display: 'flex', gap: 48, height: '100%' }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontFamily: mono, fontSize: 20, letterSpacing: '0.18em', color: 'var(--osd-accent)', marginBottom: 24 }}>
          三個站得住的結論
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <TakeRow n="01">
            幾何 / pose 路線可部署，<b>單目逼近動捕</b>，深蹲上甚至追平。
          </TakeRow>
          <TakeRow n="02">
            VideoMAE 單獨在 subject-wise 下<b>不可信</b> — 嚴重過擬合身份。
          </TakeRow>
          <TakeRow n="03">
            評估紀律：<b>balanced accuracy + LOSO + 分動作</b>，勝過 positive-class F1。
          </TakeRow>
        </div>
      </div>
      <div style={{ width: 1, background: line }} />
      <div style={{ flex: 1 }}>
        <div style={{ fontFamily: mono, fontSize: 20, letterSpacing: '0.18em', color: cyan, marginBottom: 24 }}>
          下一步
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
          <NextRow>
            Late fusion / gating：MediaPipe 顧 specificity、MMPose 顧 inward recall。
          </NextRow>
          <NextRow>2D→3D lifting 補回深度，改善上肢動作。</NextRow>
          <NextRow>高信心錯誤案例分析 + backend-specific 規則門檻。</NextRow>
          <NextRow>
            接上推理／生成層（GraphRAG + CoT）→ 可解釋回饋。
          </NextRow>
        </div>
      </div>
    </div>
  </Shell>
);

// =========================================================================
// Transitions — one DNA across the deck (RISE), cover gets SETTLE
// =========================================================================
const EASE_OUT = 'cubic-bezier(0, 0, 0.2, 1)';
const EASE_IN = 'cubic-bezier(0.4, 0, 1, 1)';

export const transition: SlideTransition = {
  duration: 200,
  exit: {
    duration: 140,
    easing: EASE_IN,
    keyframes: [
      { opacity: 1, transform: 'translateY(0)' },
      { opacity: 0, transform: 'translateY(-4px)' },
    ],
  },
  enter: {
    duration: 220,
    delay: 80,
    easing: EASE_OUT,
    keyframes: [
      { opacity: 0, transform: 'translateY(8px)' },
      { opacity: 1, transform: 'translateY(0)' },
    ],
  },
};

Cover.transition = {
  duration: 280,
  exit: {
    duration: 160,
    easing: EASE_IN,
    keyframes: [
      { opacity: 1, transform: 'translateY(0)' },
      { opacity: 0, transform: 'translateY(-6px)' },
    ],
  },
  enter: {
    duration: 280,
    delay: 100,
    easing: EASE_OUT,
    keyframes: [
      { opacity: 0, transform: 'translateY(12px)', filter: 'blur(4px)' },
      { opacity: 1, transform: 'translateY(0)', filter: 'blur(0)' },
    ],
  },
};

export const meta: SlideMeta = {
  title: 'x-coach 實驗成果報告',
  createdAt: '2026-06-15T14:08:42.440Z',
};

export default [
  Cover,
  Problem,
  Architecture,
  Datasets,
  Exp1,
  Exp2,
  Exp3,
  Exp4,
  Exp5,
  Closing,
] satisfies Page[];
