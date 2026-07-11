// Build the compact grounding blob the chat endpoint uses as its source of truth.
//
// The chat's whole credibility is groundedness: the LLM may only speak from the faults + retrieved
// knowledge the pipeline already produced. This reuses the SAME derivation ReasoningLog renders
// (lib/retrieval.ts) so the coach and the visible feedback can never disagree, then ships the
// result to the backend, which owns the system prompt.

import type { Analysis, ChatContext, ChatFaultContext } from "../api";
import { keyEvidence, ragSnippet, retrievalByFault, summaryCategory } from "./retrieval";

export function buildChatContext(analysis: Analysis): ChatContext {
  const byFault = retrievalByFault(analysis.retrievals);

  const faults: ChatFaultContext[] = analysis.detections.map((d) => {
    const r = byFault.get(d.fault_id);
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
