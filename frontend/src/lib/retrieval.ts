// Shared KG/RAG retrieval derivation — the single source of truth for turning an analysis's
// `retrievals` into the causes / risks / corrections / evidence that both the on-screen
// ReasoningLog and the chat grounding (lib/grounding.ts) present. Keeping ONE copy is what makes
// the coach and the visible feedback provably agree; forking this logic would let them drift.

import type { Detection, RagResult, Retrieval } from "../api";

export interface SummaryNode {
  node_id: string;
  // v3 namespaces scoped ids as "Movement:Name"; `name` is the bare display label. Optional so
  // pre-v3 payloads (node_id only) still type-check and fall back to node_id.
  name?: string;
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
    for (const n of summary[key] || []) out.push(n.name ?? n.node_id);
  }
  return Array.from(new Set(out)).slice(0, 4);
}

// RAG-mode retrievals fall back to a top text snippet instead of a structured subgraph.
export function ragSnippet(retrieval: Retrieval | undefined): string | null {
  if (!retrieval || retrieval.retrieval_mode !== "rag") return null;
  const results = (retrieval.context.results as RagResult[]) || [];
  return results.length ? results[0].text.slice(0, 200) + "…" : null;
}

// The single most informative measured number on a detection, formatted with the threshold it
// breached (e.g. "knee/ankle width 0.78 (< 0.82)"), or undefined when the detector authored no
// primary metric (e.g. a low-observability note that carries no fault measurement). The detector
// names the primary metric explicitly via primary_label/primary_value/primary_threshold, so this
// never has to guess it from key order.
export function keyEvidence(d: Detection): string | undefined {
  const ev = d.evidence || {};
  const label = ev.primary_label;
  const value = ev.primary_value;
  if (typeof label !== "string" || typeof value !== "number") return undefined;
  const threshold = ev.primary_threshold;
  const base = `${label} ${value.toFixed(2)}`;
  if (typeof threshold !== "number") return base;
  // The reported value is always the min/max on the violating side, so the numeric value-vs-
  // threshold relationship directly encodes the breach direction (below vs above the limit).
  return `${base} (${value < threshold ? "<" : ">"} ${threshold})`;
}
