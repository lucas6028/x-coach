import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import KnowledgeGraphWidget from "../components/KnowledgeGraphWidget";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis } from "./fixtures";

const emptyAnalysis = { ...mockCleanAnalysis, retrievals: [] };

describe("KnowledgeGraphWidget", () => {
  it("renders the 'Knowledge Graph' heading", () => {
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();
  });

  it("shows the GraphRAG badge", () => {
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    expect(screen.getByText("GraphRAG")).toBeInTheDocument();
  });

  it("shows 'No graph context' when there are no retrievals", () => {
    renderWithProviders(
      <KnowledgeGraphWidget analysis={emptyAnalysis} activeFaultId={null} />
    );
    expect(screen.getByText(/No graph context/i)).toBeInTheDocument();
  });

  it("renders the SVG graph when retrievals are present", () => {
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    expect(document.querySelector("svg")).toBeInTheDocument();
  });

  it("renders the 'expand to full screen' button when graph has data", () => {
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    expect(screen.getByRole("button", { name: /Expand to full screen/i })).toBeInTheDocument();
  });

  it("does not show expand button when there is no graph data", () => {
    renderWithProviders(
      <KnowledgeGraphWidget analysis={emptyAnalysis} activeFaultId={null} />
    );
    expect(screen.queryByRole("button", { name: /Expand/i })).not.toBeInTheDocument();
  });

  it("opens a fullscreen dialog when expand is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    await user.click(screen.getByRole("button", { name: /Expand to full screen/i }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("closes the fullscreen dialog when the collapse button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    await user.click(screen.getByRole("button", { name: /Expand to full screen/i }));
    await user.click(screen.getByRole("button", { name: /Close full screen/i }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("closes fullscreen on Escape key", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    await user.click(screen.getByRole("button", { name: /Expand to full screen/i }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders legend color keys when there is graph data", () => {
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    // legend shows cause / risk / correction
    expect(screen.getByText("Cause")).toBeInTheDocument();
    expect(screen.getByText("Risk")).toBeInTheDocument();
    expect(screen.getByText("Fix")).toBeInTheDocument();
  });

  it("uses the activeFaultId to select the retrieval", () => {
    renderWithProviders(
      <KnowledgeGraphWidget
        analysis={mockAnalysis}
        activeFaultId="knees_inward_1"
      />
    );
    // With the matching fault, the graph should still render
    expect(document.querySelector("svg")).toBeInTheDocument();
  });
});
