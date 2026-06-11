import { useMemo } from "react";
import type { Analysis, Detection, Retrieval, RagResult } from "../api";
import { fmtTime, severityLabel } from "../lib/format";

interface Props {
  analysis: Analysis;
  currentTime: number;
  onSeek: (t: number) => void;
}

interface SummaryNode {
  node_id: string;
  label: string;
}

function summaryCategory(retrieval: Retrieval | undefined, key: string): string[] {
  if (!retrieval) return [];
  const results = (retrieval.context.results as Array<Record<string, unknown>>) || [];
  const out: string[] = [];
  for (const seed of results) {
    const summary = (seed.summary as Record<string, SummaryNode[]>) || {};
    for (const n of summary[key] || []) out.push(n.node_id);
  }
  return Array.from(new Set(out)).slice(0, 4);
}

function ragSnippet(retrieval: Retrieval | undefined): string | null {
  if (!retrieval || retrieval.retrieval_mode !== "rag") return null;
  const results = (retrieval.context.results as RagResult[]) || [];
  return results.length ? results[0].text.slice(0, 220) + "…" : null;
}

function keyEvidence(d: Detection): string {
  const ev = d.evidence || {};
  const entries = Object.entries(ev).filter(([, v]) => typeof v === "number");
  if (!entries.length) return "";
  const [k, v] = entries[0];
  return `${k.replace(/_/g, " ")}: ${typeof v === "number" ? v.toFixed(2) : v}`;
}

function FaultCard({
  d,
  retrieval,
  active,
  onSeek,
}: {
  d: Detection;
  retrieval: Retrieval | undefined;
  active: boolean;
  onSeek: (t: number) => void;
}) {
  const causes = summaryCategory(retrieval, "causes");
  const risks = summaryCategory(retrieval, "risks");
  const corrections = summaryCategory(retrieval, "corrections");
  const snippet = ragSnippet(retrieval);

  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center gap-1 mt-1">
        <div className={`w-2 h-2 rounded-full ${active ? "bg-danger animate-pulse" : "bg-danger/60"}`} />
        <div className="w-px flex-1 bg-border-dark" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline justify-between mb-1">
          <span className="text-[11px] font-bold text-danger font-mono">
            [{fmtTime(d.start_time)}] {d.fault_name.toUpperCase()}
          </span>
          <span className="text-[10px] text-muted">{severityLabel(d.severity)}</span>
        </div>
        <button
          onClick={() => onSeek(d.start_time)}
          className={`w-full text-left bg-surface rounded border-l-2 p-3 transition-colors hover:bg-content/5 ${
            active ? "border-danger ring-1 ring-danger/40" : "border-danger/50"
          }`}
        >
          <p className="text-sm text-content mb-2 font-medium">
            {d.fault_name} during <span className="text-muted">{d.phase}</span> phase.
          </p>
          {keyEvidence(d) && (
            <p className="text-[11px] text-muted font-mono mb-2">{keyEvidence(d)}</p>
          )}

          {(causes.length > 0 || risks.length > 0) && (
            <div className="bg-black/40 rounded p-2 mb-2 border border-white/5">
              <div className="flex items-center gap-1.5 mb-1">
                <span className="material-symbols-outlined text-[12px] text-primary">lightbulb</span>
                <span className="text-[10px] text-primary font-bold uppercase">GraphRAG Context</span>
              </div>
              {causes.length > 0 && (
                <p className="text-[11px] text-muted leading-relaxed">
                  Likely cause:{" "}
                  <span className="text-primary">{causes.join(", ")}</span>
                </p>
              )}
              {risks.length > 0 && (
                <p className="text-[11px] text-muted leading-relaxed">
                  Injury risk: <span className="text-danger/90">{risks.join(", ")}</span>
                </p>
              )}
            </div>
          )}

          {corrections.length > 0 && (
            <div className="flex items-start gap-2">
              <span className="material-symbols-outlined text-sm text-secondary mt-0.5">
                check_circle
              </span>
              <div className="flex flex-col">
                <span className="text-[10px] text-muted uppercase font-bold">Cue</span>
                <p className="text-xs text-white">{corrections.join(" · ")}</p>
              </div>
            </div>
          )}

          {snippet && <p className="text-[11px] text-muted mt-2 italic">“{snippet}”</p>}
        </button>
      </div>
    </div>
  );
}

export default function ReasoningLog({ analysis, currentTime, onSeek }: Props) {
  const retrievalByFault = useMemo(() => {
    const m = new Map<string, Retrieval>();
    for (const r of analysis.retrievals) if (!m.has(r.fault_id)) m.set(r.fault_id, r);
    return m;
  }, [analysis]);

  const detections = analysis.detections;

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-background">
      <div className="p-3 border-b border-border-dark bg-surface-dark flex justify-between items-center">
        <h2 className="text-xs font-bold text-content uppercase tracking-widest flex items-center gap-2">
          <span className="material-symbols-outlined text-sm text-primary">psychology</span>
          Coaching Feedback
        </h2>
        <span className="text-[10px] bg-primary/20 text-primary px-1.5 py-0.5 rounded border border-primary/20 font-mono">
          rule + GraphRAG
        </span>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin">
        {detections.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center gap-2 text-muted">
            <span className="material-symbols-outlined text-3xl text-secondary">check_circle</span>
            <p className="text-sm">No biomechanical faults detected — clean rep.</p>
          </div>
        ) : (
          detections.map((d, i) => (
            <FaultCard
              key={i}
              d={d}
              retrieval={retrievalByFault.get(d.fault_id)}
              active={currentTime >= d.start_time && currentTime <= d.end_time}
              onSeek={onSeek}
            />
          ))
        )}
      </div>
    </div>
  );
}
