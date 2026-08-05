import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { I18nProvider } from "../lib/i18n";
import { ToolRunList } from "../components/ToolRunList";
import type { ToolSource } from "../api";

function renderRuns(runs: Array<{ name: string; query: string; sources?: ToolSource[]; pending?: boolean }>) {
  return render(
    <I18nProvider>
      <ToolRunList runs={runs} />
    </I18nProvider>
  );
}

describe("ToolRunList", () => {
  it("shows an animated marker while a run is pending, and none once it settles", async () => {
    // The dots are now the ONLY signal that anything is happening across a retrieval that can run
    // minutes — CoachTray's own dots are suppressed as soon as a tool record exists.
    const { rerender, container } = renderRuns([
      { name: "rag_search", query: "zzq-q", pending: true },
    ]);
    expect(container.querySelector(".lm-dots")).toBeTruthy();
    rerender(
      <I18nProvider>
        <ToolRunList runs={[{ name: "rag_search", query: "zzq-q", pending: false }]} />
      </I18nProvider>
    );
    expect(container.querySelector(".lm-dots")).toBeNull();
  });

  it("renders no source row at all for a settled run with nothing to cite", async () => {
    // get_analysis reads the user's own analysis — there is no outside source to credit, so a
    // "0 sources" row would be noise, and a pending marker would be a lie.
    const { container } = renderRuns([
      { name: "get_analysis", query: "Depth", pending: false },
    ]);
    expect(screen.queryByRole("button")).toBeNull();
    expect(container.querySelector(".lm-dots")).toBeNull();
  });

  it("collapses sources to a count and reveals the labels on click", async () => {
    renderRuns([
      {
        name: "rag_search",
        query: "ankle",
        sources: [
          { label: "zzq-Source-A", kind: "paper" },
          { label: "zzq-Source-B", kind: "encyclopedia" },
        ],
      },
    ]);
    const toggle = screen.getByRole("button", { name: /Sources · 2/ });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText("zzq-Source-A")).toBeNull();
    await userEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("zzq-Source-A")).toBeTruthy();
    expect(screen.getByText("zzq-Source-B")).toBeTruthy();
  });

  it("counts kg_query results as knowledge-graph concepts, not sources", async () => {
    // v3.1's red line survives the collapse: a graph node carries no citation anywhere in the
    // graph, so counting it must not rename it into a citation. Keyed off `kind`, not tool name.
    renderRuns([
      { name: "kg_query", query: "valgus", sources: [{ label: "zzq-Concept", kind: "concept" }] },
      { name: "rag_search", query: "ankle", sources: [{ label: "zzq-Paper", kind: "paper" }] },
    ]);
    expect(screen.getByRole("button", { name: /Knowledge-graph concepts · 1/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /Sources · 1/ })).toBeTruthy();
  });

  it("expands each row independently", async () => {
    renderRuns([
      { name: "rag_search", query: "a", sources: [{ label: "zzq-Row-One", kind: "paper" }] },
      { name: "rag_search", query: "b", sources: [{ label: "zzq-Row-Two", kind: "paper" }] },
    ]);
    const [first] = screen.getAllByRole("button");
    await userEvent.click(first);
    expect(screen.getByText("zzq-Row-One")).toBeTruthy();
    expect(screen.queryByText("zzq-Row-Two")).toBeNull();
  });
});
