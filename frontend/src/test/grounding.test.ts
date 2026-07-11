import { describe, it, expect } from "vitest";
import { buildChatContext } from "../lib/grounding";
import type { Analysis } from "../api";

// Minimal Analysis shaped just enough for buildChatContext; cast through unknown since we omit
// the heavy pose/metadata fields the grounding builder never touches.
function make(partial: Partial<Analysis>): Analysis {
  return {
    video_id: "v1",
    view: { view_type: "front", view_confidence: 0.9 },
    quality: { lower_body_visibility_mean: 0.88 },
    detections: [],
    retrievals: [],
    ...partial,
  } as unknown as Analysis;
}

describe("buildChatContext", () => {
  it("extracts KG causes/risks/corrections + evidence for a detected fault", () => {
    const ctx = buildChatContext(
      make({
        detections: [
          {
            fault_id: "f1",
            fault_name: "knees_inward",
            phase: "descent",
            severity: 0.8,
            start_time: 1,
            end_time: 2,
            evidence: { primary_label: "knee valgus ratio", primary_value: 0.82 },
          },
        ] as unknown as Analysis["detections"],
        retrievals: [
          {
            fault_id: "f1",
            retrieval_mode: "kg",
            context: {
              results: [
                {
                  summary: {
                    causes: [{ node_id: "weak glutes", label: "x" }],
                    risks: [{ node_id: "ACL strain", label: "y" }],
                    corrections: [{ node_id: "knees out", label: "z" }],
                  },
                },
              ],
            },
          },
        ] as unknown as Analysis["retrievals"],
      })
    );

    expect(ctx.fault_count).toBe(1);
    expect(ctx.view_type).toBe("front");
    expect(ctx.view_confidence).toBe(0.9);
    const f = ctx.faults[0];
    expect(f.fault_name).toBe("knees_inward");
    expect(f.causes).toContain("weak glutes");
    expect(f.risks).toContain("ACL strain");
    expect(f.corrections).toContain("knees out");
    expect(f.evidence).toBe("knee valgus ratio 0.82");
    expect(f.rag_snippet).toBeNull();
  });

  it("uses a RAG snippet when the retrieval is rag-mode, and copes with missing evidence", () => {
    const ctx = buildChatContext(
      make({
        detections: [
          { fault_id: "f2", fault_name: "shallow_depth", evidence: {} },
        ] as unknown as Analysis["detections"],
        retrievals: [
          {
            fault_id: "f2",
            retrieval_mode: "rag",
            context: { results: [{ rank: 1, score: 0.5, text: "sit below parallel", metadata: {} }] },
          },
        ] as unknown as Analysis["retrievals"],
      })
    );
    const f = ctx.faults[0];
    expect(f.evidence).toBeUndefined();
    expect(f.causes).toEqual([]);
    expect(f.rag_snippet).toContain("sit below parallel");
  });

  it("handles a clean rep and a non-numeric view confidence", () => {
    const ctx = buildChatContext(
      make({ view: { view_type: "side" } as unknown as Analysis["view"] })
    );
    expect(ctx.fault_count).toBe(0);
    expect(ctx.faults).toEqual([]);
    expect(ctx.view_confidence).toBeUndefined();
  });

  it("falls back to an empty quality object when the analysis has none", () => {
    const ctx = buildChatContext(make({ quality: undefined as unknown as Analysis["quality"] }));
    expect(ctx.quality).toEqual({});
  });
});
