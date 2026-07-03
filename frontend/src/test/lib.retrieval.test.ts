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

  it("returns undefined when no evidence value is numeric", () => {
    const d = { evidence: { note: "not a number" } } as unknown as Detection;
    expect(keyEvidence(d)).toBeUndefined();
  });

  it("formats the first numeric evidence entry with underscores replaced", () => {
    const d = { evidence: { valgus_angle: 0.3456 } } as unknown as Detection;
    expect(keyEvidence(d)).toBe("valgus angle 0.35");
  });
});
