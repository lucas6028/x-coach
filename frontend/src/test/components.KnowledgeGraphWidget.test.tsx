import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import KnowledgeGraphWidget from "../components/KnowledgeGraphWidget";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis } from "./fixtures";

// The global setup freezes requestAnimationFrame, so framer-motion's
// AnimatePresence exit animation would never finish (the overlay would never
// unmount) under test. The close animation is declarative config; here we only
// need to assert the open/close *behavior*, so mock the animation primitives to
// mount/unmount synchronously and pass DOM props straight through.
vi.mock("motion/react", async () => {
  const React = await vi.importActual<typeof import("react")>("react");
  const cache: Record<string, React.ComponentType<Record<string, unknown>>> = {};
  const motion = new Proxy({} as Record<string, unknown>, {
    get: (_t, tag: string) => {
      if (!cache[tag]) {
        cache[tag] = React.forwardRef<unknown, Record<string, unknown>>((props, ref) => {
          // Drop framer-only props; forward everything else (role, className, onClick…).
          const { initial, animate, exit, transition, variants, children, ...rest } = props;
          void initial; void animate; void exit; void transition; void variants;
          return React.createElement(tag, { ...rest, ref }, children as React.ReactNode);
        }) as unknown as React.ComponentType<Record<string, unknown>>;
      }
      return cache[tag];
    },
  });
  return {
    __esModule: true,
    motion,
    AnimatePresence: ({ children }: { children: React.ReactNode }) =>
      React.createElement(React.Fragment, null, children),
    useReducedMotion: () => true,
  };
});

const emptyAnalysis = { ...mockCleanAnalysis, retrievals: [] };

describe("KnowledgeGraphWidget", () => {
  it("renders the 'Knowledge Graph' heading", () => {
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();
  });

  it("summarises the graph as a node count on the compact card", () => {
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    expect(screen.getByText(/nodes/i)).toBeInTheDocument();
  });

  it("shows the GraphRAG badge inside the fullscreen overlay", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    await user.click(screen.getByRole("button", { name: /Expand to full screen/i }));
    expect(screen.getByText("GraphRAG")).toBeInTheDocument();
  });

  it("shows 'No graph context' when there are no retrievals", () => {
    renderWithProviders(
      <KnowledgeGraphWidget analysis={emptyAnalysis} activeFaultId={null} />
    );
    expect(screen.getByText(/No graph context/i)).toBeInTheDocument();
  });

  it("renders the SVG graph in the fullscreen overlay when retrievals are present", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    await user.click(screen.getByRole("button", { name: /Expand to full screen/i }));
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

  it("closes the fullscreen dialog when clicking empty graph space", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    await user.click(screen.getByRole("button", { name: /Expand to full screen/i }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    // A click on the graph canvas background (not a node) bubbles up and closes.
    const svg = document.querySelector("svg.select-none") as SVGSVGElement;
    fireEvent.click(svg);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("keeps the dialog open when a graph node is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    await user.click(screen.getByRole("button", { name: /Expand to full screen/i }));
    // Clicking a draggable node stops propagation, so the overlay stays open.
    const node = document.querySelector("svg.select-none g g") as SVGGElement;
    expect(node).not.toBeNull();
    fireEvent.click(node);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
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

  it("renders legend color keys in the fullscreen overlay", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <KnowledgeGraphWidget analysis={mockAnalysis} activeFaultId={null} />
    );
    await user.click(screen.getByRole("button", { name: /Expand to full screen/i }));
    // legend shows cause / risk / correction
    expect(screen.getByText("Cause")).toBeInTheDocument();
    expect(screen.getByText("Risk")).toBeInTheDocument();
    expect(screen.getByText("Fix")).toBeInTheDocument();
  });

  it("uses the activeFaultId to select the retrieval", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <KnowledgeGraphWidget
        analysis={mockAnalysis}
        activeFaultId="knees_inward_1"
      />
    );
    // With the matching fault, the graph should still render once expanded
    await user.click(screen.getByRole("button", { name: /Expand to full screen/i }));
    expect(document.querySelector("svg")).toBeInTheDocument();
  });
});
