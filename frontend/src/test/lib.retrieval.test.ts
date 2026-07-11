import { describe, it, expect } from "vitest";
import { keyEvidence, ragSnippet, retrievalByFault, summaryCategory } from "../lib/retrieval";
import type { Detection, Retrieval } from "../api";

function makeRetrieval(partial: Partial<Retrieval>): Retrieval {
  return {
    fault_id: "f1",
    fault_name: "knees_inward",
    query_text: "knee valgus",
    retrieval_mode: "kg",
    context: {},
    ...partial,
  };
}

describe("retrievalByFault", () => {
  it("keeps the first retrieval per fault_id and ignores later duplicates", () => {
    const first = makeRetrieval({ fault_id: "f1", query_text: "first" });
    const dup = makeRetrieval({ fault_id: "f1", query_text: "second" });
    const other = makeRetrieval({ fault_id: "f2", query_text: "third" });

    const byFault = retrievalByFault([first, dup, other]);

    expect(byFault.size).toBe(2);
    expect(byFault.get("f1")).toBe(first);
    expect(byFault.get("f2")).toBe(other);
  });

  it("returns an empty map for no retrievals", () => {
    expect(retrievalByFault([]).size).toBe(0);
  });
});

describe("summaryCategory", () => {
  it("returns [] when there is no retrieval", () => {
    expect(summaryCategory(undefined, "causes")).toEqual([]);
  });

  it("returns [] when the retrieval has no results", () => {
    const r = makeRetrieval({ context: {} });
    expect(summaryCategory(r, "causes")).toEqual([]);
  });

  it("returns [] when a result has no summary for the requested key", () => {
    const r = makeRetrieval({ context: { results: [{ summary: { risks: [] } }] } });
    expect(summaryCategory(r, "causes")).toEqual([]);
  });

  it("dedupes node_ids across seeds and caps at 4", () => {
    const r = makeRetrieval({
      context: {
        results: [
          { summary: { causes: [{ node_id: "a", label: "a" }, { node_id: "b", label: "b" }] } },
          {
            summary: {
              causes: [
                { node_id: "a", label: "a" },
                { node_id: "c", label: "c" },
                { node_id: "d", label: "d" },
                { node_id: "e", label: "e" },
              ],
            },
          },
        ],
      },
    });
    expect(summaryCategory(r, "causes")).toEqual(["a", "b", "c", "d"]);
  });
});

describe("ragSnippet", () => {
  it("returns null when there is no retrieval", () => {
    expect(ragSnippet(undefined)).toBeNull();
  });

  it("returns null for a kg-mode retrieval", () => {
    const r = makeRetrieval({ retrieval_mode: "kg" });
    expect(ragSnippet(r)).toBeNull();
  });

  it("returns null for rag-mode with no results", () => {
    const r = makeRetrieval({ retrieval_mode: "rag", context: {} });
    expect(ragSnippet(r)).toBeNull();
  });

  it("returns the first result's text, truncated to 200 chars, for rag-mode", () => {
    const longText = "a".repeat(250);
    const r = makeRetrieval({
      retrieval_mode: "rag",
      context: { results: [{ rank: 1, score: 0.9, text: longText, metadata: {} }] },
    });
    const snippet = ragSnippet(r);
    expect(snippet).toHaveLength(201); // 200 chars + ellipsis
    expect(snippet?.endsWith("…")).toBe(true);
  });
});

describe("keyEvidence", () => {
  it("returns undefined when evidence is missing", () => {
    const d = { evidence: undefined } as unknown as Detection;
    expect(keyEvidence(d)).toBeUndefined();
  });

  it("returns undefined when the detector authored no primary metric", () => {
    // e.g. a low-observability note whose only number is camera confidence, not a fault value.
    const d = { evidence: { reason: "side view required", view_confidence: 0.85 } } as unknown as Detection;
    expect(keyEvidence(d)).toBeUndefined();
  });

  it("formats the authored primary metric with the threshold it breached", () => {
    const d = {
      evidence: { primary_label: "knee/ankle width", primary_value: 0.7812, primary_threshold: 0.82 },
    } as unknown as Detection;
    // Value is below the threshold, so the derived comparator reads "<".
    expect(keyEvidence(d)).toBe("knee/ankle width 0.78 (< 0.82)");
  });

  it("derives '>' when the breached value sits above its threshold", () => {
    const d = {
      evidence: { primary_label: "torso lean angle", primary_value: 42.5, primary_threshold: 35 },
    } as unknown as Detection;
    expect(keyEvidence(d)).toBe("torso lean angle 42.50 (> 35)");
  });

  it("omits the parenthetical when no threshold was authored", () => {
    const d = { evidence: { primary_label: "valgus angle", primary_value: 0.35 } } as unknown as Detection;
    expect(keyEvidence(d)).toBe("valgus angle 0.35");
  });
});
