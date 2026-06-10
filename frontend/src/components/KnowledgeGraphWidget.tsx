import { useMemo } from "react";
import type { Analysis, Retrieval } from "../api";

interface Props {
  analysis: Analysis;
  activeFaultId: string | null;
}

interface Neighbor {
  label: string;
  kind: "cause" | "risk" | "correction" | "evidence";
}

const KIND_COLOR: Record<Neighbor["kind"], string> = {
  cause: "#0f758a",
  risk: "#ef4444",
  correction: "#42d159",
  evidence: "#9ca3af",
};

function collect(retrieval: Retrieval | undefined): { center: string; neighbors: Neighbor[] } {
  if (!retrieval) return { center: "—", neighbors: [] };
  const results = (retrieval.context.results as Array<Record<string, unknown>>) || [];
  const center = retrieval.fault_name;
  const neighbors: Neighbor[] = [];
  const push = (key: string, kind: Neighbor["kind"]) => {
    for (const seed of results) {
      const summary = (seed.summary as Record<string, { node_id: string }[]>) || {};
      for (const n of summary[key] || []) neighbors.push({ label: n.node_id, kind });
    }
  };
  push("causes", "cause");
  push("risks", "risk");
  push("corrections", "correction");
  push("evidence", "evidence");
  // De-dupe and cap for legibility.
  const seen = new Set<string>();
  const unique = neighbors.filter((n) => (seen.has(n.label) ? false : seen.add(n.label)));
  return { center, neighbors: unique.slice(0, 8) };
}

export default function KnowledgeGraphWidget({ analysis, activeFaultId }: Props) {
  const retrieval = useMemo(() => {
    const list = analysis.retrievals;
    return list.find((r) => r.fault_id === activeFaultId) || list[0];
  }, [analysis, activeFaultId]);

  const { center, neighbors } = useMemo(() => collect(retrieval), [retrieval]);

  const W = 360;
  const H = 240;
  const cx = W / 2;
  const cy = H / 2;
  const radius = 88;

  return (
    <div className="h-60 border-b border-border-dark relative overflow-hidden bg-[#16181b]">
      <div className="absolute top-3 left-3 z-10">
        <h2 className="text-[11px] font-bold text-primary uppercase tracking-widest flex items-center gap-2">
          <span className="material-symbols-outlined text-sm">hub</span> Knowledge Graph
        </h2>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
        {neighbors.map((n, i) => {
          const angle = (i / Math.max(1, neighbors.length)) * Math.PI * 2 - Math.PI / 2;
          const x = cx + Math.cos(angle) * radius;
          const y = cy + Math.sin(angle) * radius;
          return (
            <g key={i}>
              <line x1={cx} y1={cy} x2={x} y2={y} stroke={KIND_COLOR[n.kind]} strokeOpacity={0.4} strokeWidth={1} />
              <circle cx={x} cy={y} r={4} fill={KIND_COLOR[n.kind]} />
              <text
                x={x}
                y={y}
                dy={x < cx ? "1.1em" : "-0.7em"}
                textAnchor={x < cx ? "end" : "start"}
                className="font-mono"
                fontSize={8}
                fill="#cbd5e1"
              >
                {n.label.length > 22 ? n.label.slice(0, 21) + "…" : n.label}
              </text>
            </g>
          );
        })}
        {/* center fault node */}
        <circle cx={cx} cy={cy} r={7} fill="#ef4444" />
        <circle cx={cx} cy={cy} r={12} fill="none" stroke="#ef4444" strokeOpacity={0.4} />
        <text x={cx} y={cy} dy="-1.3em" textAnchor="middle" fontSize={9} fill="#ef4444" className="font-mono font-bold">
          {center}
        </text>
      </svg>
      {neighbors.length === 0 && (
        <p className="absolute inset-0 flex items-center justify-center text-xs text-gray-600">
          No graph context for this clip.
        </p>
      )}
    </div>
  );
}
