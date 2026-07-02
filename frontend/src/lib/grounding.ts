// Build the compact grounding blob the chat endpoint uses as its source of truth.
//
// The chat's whole credibility is groundedness: the LLM may only speak from the faults + retrieved
// knowledge the pipeline already produced. This derives exactly the causes / risks / corrections /
// evidence that ReasoningLog renders (same logic), so the coach and the visible feedback can never
// disagree — then ships them to the backend, which owns the system prompt.

import type {
  Analysis,
  ChatContext,
  ChatFaultContext,
  Detection,
  RagResult,
  Retrieval,
} from "../api";

interface SummaryNode {
  node_id: string;
  label: string;
}

// KG-mode retrievals expose grouped summaries (causes/risks/corrections) per seed node.
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

// RAG-mode retrievals fall back to a top text snippet instead of a structured subgraph.
function ragSnippet(retrieval: Retrieval | undefined): string | null {
  if (!retrieval || retrieval.retrieval_mode !== "rag") return null;
  const results = (retrieval.context.results as RagResult[]) || [];
  return results.length ? results[0].text.slice(0, 200) + "…" : null;
}

// The single most informative measured number attached to a detection (e.g. knee_valgus_ratio).
function keyEvidence(d: Detection): string | undefined {
  const ev = d.evidence || {};
  const entries = Object.entries(ev).filter(([, v]) => typeof v === "number");
  if (!entries.length) return undefined;
  const [k, v] = entries[0];
  return `${k.replace(/_/g, " ")} ${typeof v === "number" ? v.toFixed(2) : v}`;
}

export function buildChatContext(analysis: Analysis): ChatContext {
  const retrievalByFault = new Map<string, Retrieval>();
  for (const r of analysis.retrievals) {
    if (!retrievalByFault.has(r.fault_id)) retrievalByFault.set(r.fault_id, r);
  }

  const faults: ChatFaultContext[] = analysis.detections.map((d) => {
    const r = retrievalByFault.get(d.fault_id);
    return {
      fault_name: d.fault_name,
      phase: d.phase,
      severity: d.severity,
      start_time: d.start_time,
      end_time: d.end_time,
      evidence: keyEvidence(d),
      causes: summaryCategory(r, "causes"),
      risks: summaryCategory(r, "risks"),
      corrections: summaryCategory(r, "corrections"),
      rag_snippet: ragSnippet(r),
    };
  });

  const view = analysis.view;
  return {
    video_id: analysis.video_id,
    view_type: view?.view_type,
    view_confidence:
      typeof view?.view_confidence === "number" ? view.view_confidence : undefined,
    fault_count: analysis.detections.length,
    quality: analysis.quality || {},
    faults,
  };
}
