import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUpRight, Graph, X } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import type { Analysis, Retrieval } from "../api";
import { useI18n, faultLabel } from "../lib/i18n";

interface Props {
  analysis: Analysis;
  activeFaultId: string | null;
}

type Kind = "cause" | "risk" | "correction" | "evidence";

interface Neighbor {
  label: string;
  kind: Kind;
}

// Brand-semantic relation colors. These read on both light and dark canvases;
// neutral surfaces/labels use theme tokens (fill-content etc.) so the whole
// widget follows the app's light/dark toggle.
const KIND_COLOR: Record<Kind, string> = {
  cause: "#0f758a", // primary teal
  risk: "#ef4444", // danger red
  correction: "#42d159", // secondary green
  evidence: "#94a3b8", // slate (neutral)
};
const KIND_ORDER: Kind[] = ["cause", "risk", "correction", "evidence"];
const CENTER = "#f5b945"; // amber: the fault itself, distinct from red "risk"

function collect(retrieval: Retrieval | undefined): { center: string; neighbors: Neighbor[] } {
  if (!retrieval) return { center: "", neighbors: [] };
  const results = (retrieval.context.results as Array<Record<string, unknown>>) || [];
  const neighbors: Neighbor[] = [];
  const push = (key: string, kind: Kind) => {
    for (const seed of results) {
      const summary = (seed.summary as Record<string, { node_id: string }[]>) || {};
      for (const n of summary[key] || []) neighbors.push({ label: n.node_id, kind });
    }
  };
  // Pushed in KIND_ORDER so same-kind nodes stay adjacent around the ring.
  push("causes", "cause");
  push("risks", "risk");
  push("corrections", "correction");
  push("evidence", "evidence");
  const seen = new Set<string>();
  const unique = neighbors.filter((n) => (seen.has(n.label) ? false : seen.add(n.label)));
  return { center: retrieval.fault_name, neighbors: unique.slice(0, 16) };
}

// The radial node-link scene. `large` scales geometry/typography for the
// fullscreen view; the same markup powers the inline panel.
interface Pt {
  x: number;
  y: number;
}
type DragTarget = { kind: "node"; i: number } | { kind: "center" } | null;

function GraphScene({
  centerLabel,
  neighbors,
  large,
  animateKey,
}: {
  centerLabel: string;
  neighbors: Neighbor[];
  large: boolean;
  animateKey: string;
}) {
  const reduce = useReducedMotion();
  const W = large ? 1060 : 380;
  const H = large ? 600 : 222;
  const CX = W / 2;
  const CY = large ? H * 0.46 : 100;
  const R = large ? 235 : 78;
  const dotR = large ? 6 : 3.5;
  const haloR = large ? 13 : 6.5;
  const font = large ? 16 : 9;
  const centerFont = large ? 20 : 10.5;
  const maxLabel = large ? 44 : 20;
  const sw = large ? 4.5 : 3;
  const uid = large ? "lg" : "sm";

  const svgRef = useRef<SVGSVGElement>(null);

  // Initial radial layout. Nodes are then freely draggable, so positions live
  // in state; they reset whenever the subgraph (neighbors) or size changes.
  const layout = useMemo(() => {
    const center: Pt = { x: CX, y: CY };
    const nodes: Pt[] = neighbors.map((_, i) => {
      const a = (i / neighbors.length) * Math.PI * 2 - Math.PI / 2;
      return { x: CX + Math.cos(a) * R, y: CY + Math.sin(a) * R };
    });
    return { center, nodes };
  }, [neighbors, CX, CY, R]);

  const [pos, setPos] = useState<Pt[]>(layout.nodes);
  const [cpos, setCpos] = useState<Pt>(layout.center);
  const [drag, setDrag] = useState<DragTarget>(null);
  useEffect(() => {
    setPos(layout.nodes);
    setCpos(layout.center);
  }, [layout]);

  const toSvg = (clientX: number, clientY: number): Pt => {
    const svg = svgRef.current;
    if (!svg) return { x: CX, y: CY };
    const m = svg.getScreenCTM();
    if (!m) return { x: CX, y: CY };
    const p = new DOMPoint(clientX, clientY).matrixTransform(m.inverse());
    return { x: p.x, y: p.y };
  };

  const onDown = (target: DragTarget) => (e: React.PointerEvent) => {
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    setDrag(target);
  };
  const onMove = (e: React.PointerEvent) => {
    if (!drag) return;
    const p = toSvg(e.clientX, e.clientY);
    if (drag.kind === "center") setCpos(p);
    else setPos((prev) => prev.map((q, i) => (i === drag.i ? p : q)));
  };
  const onUp = (e: React.PointerEvent) => {
    if (drag) (e.target as Element).releasePointerCapture?.(e.pointerId);
    setDrag(null);
  };

  return (
    <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 h-full w-full select-none">
      <defs>
        <pattern id={`kg-dots-${uid}`} width="18" height="18" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="1" r="1" className="fill-content" opacity={0.07} />
        </pattern>
        <radialGradient id={`kg-glow-${uid}`} cx="50%" cy={`${(CY / H) * 100}%`} r="55%">
          <stop offset="0%" stopColor={CENTER} stopOpacity={0.1} />
          <stop offset="100%" stopColor={CENTER} stopOpacity={0} />
        </radialGradient>
      </defs>
      <rect x="0" y="0" width={W} height={H} fill={`url(#kg-dots-${uid})`} />
      <rect x="0" y="0" width={W} height={H} fill={`url(#kg-glow-${uid})`} />

      <motion.g
        key={animateKey}
        initial={reduce ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      >
        {/* Edges (follow live node positions) */}
        {neighbors.map((n, i) => {
          const x = pos[i]?.x ?? CX;
          const y = pos[i]?.y ?? CY;
          const dx = x - cpos.x;
          const dy = y - cpos.y;
          const len = Math.hypot(dx, dy) || 1;
          const off = large ? 26 : 10;
          const mx = (cpos.x + x) / 2 - (dy / len) * off;
          const my = (cpos.y + y) / 2 + (dx / len) * off;
          return (
            <path
              key={`e${i}`}
              d={`M ${cpos.x} ${cpos.y} Q ${mx} ${my} ${x} ${y}`}
              fill="none"
              stroke={KIND_COLOR[n.kind]}
              strokeOpacity={0.45}
              strokeWidth={large ? 2 : 1.25}
            />
          );
        })}

        {/* Neighbor nodes + labels (draggable) */}
        {neighbors.map((n, i) => {
          const x = pos[i]?.x ?? CX;
          const y = pos[i]?.y ?? CY;
          const ang = Math.atan2(y - cpos.y, x - cpos.x);
          const right = Math.cos(ang) > 0.15;
          const left = Math.cos(ang) < -0.15;
          const anchor = right ? "start" : left ? "end" : "middle";
          const lx = x + (right ? haloR + 4 : left ? -(haloR + 4) : 0);
          const ly = y + (Math.abs(Math.cos(ang)) < 0.15 ? (Math.sin(ang) > 0 ? font + 4 : -(haloR + 1)) : font / 3);
          const label = n.label.length > maxLabel ? n.label.slice(0, maxLabel - 1) + "…" : n.label;
          const grabbing = drag?.kind === "node" && drag.i === i;
          return (
            <g
              key={`n${i}`}
              onPointerDown={onDown({ kind: "node", i })}
              onPointerMove={onMove}
              onPointerUp={onUp}
              onClick={(e) => e.stopPropagation()}
              style={{ cursor: grabbing ? "grabbing" : "grab", touchAction: "none" }}
            >
              {/* generous invisible hit area */}
              <circle cx={x} cy={y} r={haloR + 8} fill="transparent" />
              <circle cx={x} cy={y} r={haloR} fill={KIND_COLOR[n.kind]} fillOpacity={grabbing ? 0.28 : 0.16} />
              <circle cx={x} cy={y} r={dotR} fill={KIND_COLOR[n.kind]} />
              <text
                x={lx}
                y={ly}
                textAnchor={anchor}
                fontSize={font}
                className="fill-content font-mono"
                style={{ stroke: "rgb(var(--c-bg))", strokeWidth: sw, paintOrder: "stroke" }}
              >
                {label}
              </text>
            </g>
          );
        })}

        {/* Center fault node (draggable; amber, distinct from neutral children) */}
        <g
          onPointerDown={onDown({ kind: "center" })}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onClick={(e) => e.stopPropagation()}
          style={{ cursor: drag?.kind === "center" ? "grabbing" : "grab", touchAction: "none" }}
        >
          <circle cx={cpos.x} cy={cpos.y} r={large ? 36 : 16} fill={CENTER} fillOpacity={0.12} />
          <circle
            cx={cpos.x}
            cy={cpos.y}
            r={large ? 23 : 10}
            fill="none"
            stroke={CENTER}
            strokeOpacity={0.6}
            strokeWidth={large ? 3 : 1.5}
          />
          <circle cx={cpos.x} cy={cpos.y} r={large ? 13 : 6} fill={CENTER} />
          <text
            x={cpos.x}
            y={cpos.y + (large ? 60 : 30)}
            textAnchor="middle"
            fontSize={centerFont}
            fontWeight={700}
            className="font-display"
            style={{
              fill: "rgb(var(--c-fault))",
              stroke: "rgb(var(--c-bg))",
              strokeWidth: sw + 0.5,
              paintOrder: "stroke",
            }}
          >
            {centerLabel}
          </text>
        </g>
      </motion.g>
    </svg>
  );
}

function Legend({ kinds, t }: { kinds: Kind[]; t: ReturnType<typeof useI18n>["t"] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {kinds.map((k) => (
        <span key={k} className="flex items-center gap-1.5 text-[10px] text-muted sm:text-xs">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: KIND_COLOR[k] }} />
          {t(`kg.${k}` as Parameters<typeof t>[0])}
        </span>
      ))}
    </div>
  );
}

export default function KnowledgeGraphWidget({ analysis, activeFaultId }: Props) {
  const { t } = useI18n();
  const [fullscreen, setFullscreen] = useState(false);

  const retrieval = useMemo(() => {
    const list = analysis.retrievals;
    return list.find((r) => r.fault_id === activeFaultId) || list[0];
  }, [analysis, activeFaultId]);

  const { center, neighbors } = useMemo(() => collect(retrieval), [retrieval]);
  const presentKinds = KIND_ORDER.filter((k) => neighbors.some((n) => n.kind === k));
  const hasGraph = neighbors.length > 0;
  const nodeCount = neighbors.length + (center ? 1 : 0);
  const centerLabel = center ? faultLabel(t, center) : "";
  const animKey = (retrieval?.fault_id ?? "none") + (fullscreen ? "-fs" : "");

  // Escape closes the fullscreen view.
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setFullscreen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen]);

  return (
    <>
      {/* Compact summary card — sits at the foot of the feedback column and
          opens the full node-link graph in an overlay. Shares the feedback
          panel's background so it reads as coaching content, not part of the
          follow-up input below it. */}
      <div className="bg-background px-3 py-3">
        <button
          onClick={() => hasGraph && setFullscreen(true)}
          disabled={!hasGraph}
          aria-label={hasGraph ? t("kg.expand") : t("kg.empty")}
          title={hasGraph ? t("kg.expand") : t("kg.empty")}
          className="group flex w-full items-center gap-3 rounded-xl border border-border-dark bg-surface px-3 py-2.5 text-left transition-colors hover:border-content/20 hover:bg-content/[0.03] disabled:cursor-default disabled:opacity-70 disabled:hover:border-border-dark disabled:hover:bg-surface"
        >
          <span
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg"
            style={{
              color: "rgb(var(--c-fault))",
              backgroundColor: "rgb(var(--c-fault) / 0.12)",
            }}
          >
            <Graph size={20} weight="duotone" />
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold text-content">{t("kg.title")}</h2>
            <p className="truncate text-xs text-muted">
              {hasGraph
                ? `${t("kg.chain")} · ${t("kg.nodes", { count: nodeCount })}`
                : t("kg.empty")}
            </p>
          </div>
          {hasGraph && (
            <ArrowUpRight
              size={18}
              weight="bold"
              className="shrink-0 text-muted transition-colors group-hover:text-content"
            />
          )}
        </button>
      </div>

      {/* Fullscreen overlay — AnimatePresence keeps it mounted long enough to
          play the fade/scale-out on close. */}
      <AnimatePresence>
        {fullscreen && (
          <motion.div
            key="kg-fullscreen"
            role="dialog"
            aria-modal="true"
            aria-label={t("kg.title")}
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            // A symmetric ease — the previous easeOut-expo front-loaded the
            // progress, so the fade-out finished within a few ms and read as an
            // instant vanish. easeInOut keeps the close perceptible.
            transition={{ duration: 0.22, ease: "easeInOut" }}
            className="fixed inset-0 z-50 flex flex-col bg-background"
          >
          <div className="flex items-center justify-between gap-2 border-b border-border-dark px-4 py-3 sm:px-6">
            <div className="flex items-center gap-2 min-w-0">
              <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-content">
                <Graph size={20} weight="duotone" className="text-primary" />
                {t("kg.title")}
              </h2>
              <span className="rounded-full bg-primary/10 px-2 py-0.5 font-mono text-[10px] font-medium text-primary">
                GraphRAG
              </span>
              {centerLabel && (
                <span className="truncate text-sm text-muted">
                  {t("kg.focus")}: <span className="font-medium text-content">{centerLabel}</span>
                </span>
              )}
            </div>
            <button
              onClick={() => setFullscreen(false)}
              aria-label={t("kg.collapse")}
              title={t("kg.collapse")}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-muted transition-colors hover:bg-content/5 hover:text-content active:scale-95"
            >
              <X size={20} />
            </button>
          </div>

          {/* Clicking empty canvas (anything but a draggable node) closes the
              overlay; nodes stopPropagation so dragging them stays put. */}
          <div
            className="relative min-h-0 flex-1"
            onClick={() => setFullscreen(false)}
          >
            <GraphScene centerLabel={centerLabel} neighbors={neighbors.slice(0, 14)} large animateKey={animKey} />
          </div>

          {presentKinds.length > 0 && (
            <div className="border-t border-border-dark px-4 py-3 sm:px-6">
              <Legend kinds={presentKinds} t={t} />
            </div>
          )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
