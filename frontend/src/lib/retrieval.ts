// Shared KG/RAG retrieval derivation — the single source of truth for turning an analysis's
// `retrievals` into the causes / risks / corrections / evidence that both the on-screen
// ReasoningLog and the chat grounding (lib/grounding.ts) present. Keeping ONE copy is what makes
// the coach and the visible feedback provably agree; forking this logic would let them drift.

import type { Detection, RagResult, Retrieval } from "../api";

export interface SummaryNode {
  node_id: string;
  label: string;
}

// First retrieval per fault_id (the analysis can carry several; the first is the primary).
export function retrievalByFault(retrievals: Retrieval[]): Map<string, Retrieval> {
  const m = new Map<string, Retrieval>();
  for (const r of retrievals) if (!m.has(r.fault_id)) m.set(r.fault_id, r);
  return m;
}

// KG-mode retrievals expose grouped summaries (causes/risks/corrections) per seed node.
export function summaryCategory(retrieval: Retrieval | undefined, key: string): string[] {
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
export function ragSnippet(retrieval: Retrieval | undefined): string | null {
  if (!retrieval || retrieval.retrieval_mode !== "rag") return null;
  const results = (retrieval.context.results as RagResult[]) || [];
  return results.length ? results[0].text.slice(0, 200) + "…" : null;
}

// The single most informative measured number on a detection (e.g. valgus_angle), or undefined.
export function keyEvidence(d: Detection): string | undefined {
  const ev = d.evidence || {};
  const entries = Object.entries(ev).filter(([, v]) => typeof v === "number");
  if (!entries.length) return undefined;
  const [k, v] = entries[0];
  return `${k.replace(/_/g, " ")} ${typeof v === "number" ? v.toFixed(2) : v}`;
}
