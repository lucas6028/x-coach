import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type RetrievalContext } from "../api";

// The global setup freezes requestAnimationFrame, so motion's animations would never settle under
// test. GraphScene's fade is declarative config; mock the primitives to mount synchronously and pass
// DOM props straight through (mirrors components.KnowledgeGraphWidget.test.tsx).
vi.mock("motion/react", async () => {
  const React = await vi.importActual<typeof import("react")>("react");
  const cache: Record<string, React.ComponentType<Record<string, unknown>>> = {};
  const motion = new Proxy({} as Record<string, unknown>, {
    get: (_t, tag: string) => {
      if (!cache[tag]) {
        cache[tag] = React.forwardRef<unknown, Record<string, unknown>>((props, ref) => {
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

import Explore from "../pages/Explore";

// The fault list. "Bar Drift" is intentionally FIRST and unlinked (connectivity 0): the page must
// auto-select the first fault WITH a graph, not blindly list[0].
const FAULTS = [
  { name: "Bar Drift", connectivity: 0 },
  { name: "Knee Valgus", connectivity: 3 },
  { name: "Butt Wink", connectivity: 2 },
];

const withCause: RetrievalContext = {
  results: [
    {
      summary: {
        // second cause has no `name` -> exercises the `name ?? node_id` fallback in collect().
        causes: [{ node_id: "Weak Hip Abductors", name: "Weak Hip Abductors" }, { node_id: "Bare Node Id" }],
        risks: [],
        corrections: [],
        evidence: [],
      },
    },
  ],
};
const noNeighbors: RetrievalContext = {
  results: [{ summary: { causes: [], risks: [], corrections: [], evidence: [] } }],
};

// A fault query returns that fault's summary; the unlinked "Bar Drift" returns no neighbours.
const graphImpl = async (query: string): Promise<RetrievalContext> =>
  query === "Bar Drift" ? noNeighbors : withCause;

let graphMock: ReturnType<typeof vi.spyOn>;
let faultsMock: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  faultsMock = vi
    .spyOn(api, "movementFaults")
    .mockImplementation(async (movement: string) => ({ movement, faults: FAULTS }));
  graphMock = vi.spyOn(api, "graph").mockImplementation(graphImpl as typeof api.graph);
});
afterEach(() => vi.restoreAllMocks());

describe("Explore", () => {
  it("defaults to Squat and loads the complete fault list as chips", async () => {
    renderWithProviders(<Explore />);
    expect(await screen.findByRole("button", { name: "Knee Valgus" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Butt Wink" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Bar Drift" })).toBeInTheDocument();
    expect(faultsMock).toHaveBeenCalledWith("Squat");
    expect(screen.getByRole("button", { name: "Squat" })).toBeInTheDocument();
  });

  it("auto-selects the first fault WITH a graph, skipping the unlinked first fault", async () => {
    renderWithProviders(<Explore />);
    // The legend/graph render for the auto-selected fault, and it is "Knee Valgus" (conn>0), not the
    // alphabetically/first-listed unlinked "Bar Drift".
    expect(await screen.findByText("Cause")).toBeInTheDocument();
    expect(document.querySelector("svg")).toBeInTheDocument();
    await waitFor(() => expect(graphMock).toHaveBeenCalledWith("Knee Valgus", "Squat"));
    expect(graphMock).not.toHaveBeenCalledWith("Bar Drift", "Squat");
  });

  it("shows the distinct 'no linked graph' message (not the empty message) for an unlinked fault", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    await user.click(await screen.findByRole("button", { name: "Bar Drift" }));
    expect(
      await screen.findByText("This fault has no linked causes, corrections or risks yet.")
    ).toBeInTheDocument();
    expect(screen.queryByText("No faults match.")).not.toBeInTheDocument();
  });

  it("refetches the subgraph when a different fault chip is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    await user.click(await screen.findByRole("button", { name: "Butt Wink" }));
    await waitFor(() => expect(graphMock).toHaveBeenCalledWith("Butt Wink", "Squat"));
  });

  it("opens the grouped selector and switches movement", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    await screen.findByRole("button", { name: "Knee Valgus" });
    await user.click(screen.getByRole("button", { name: "Squat" }));
    expect(screen.getByText("Flagship")).toBeInTheDocument();
    expect(screen.getByText("General")).toBeInTheDocument();
    await user.click(screen.getByRole("menuitemradio", { name: /Lunge/ }));
    await waitFor(() => expect(faultsMock).toHaveBeenCalledWith("Lunge"));
  });

  it("filters chips by search and shows the empty message on no match", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    await screen.findByRole("button", { name: "Knee Valgus" });
    const box = screen.getByRole("textbox", { name: /Search faults/i });
    await user.type(box, "zzz");
    expect(screen.getByText("No faults match.")).toBeInTheDocument();
  });

  it("does not render a stale graph when the search filters out the active fault", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    // "Knee Valgus" is auto-selected and its graph is showing.
    expect(await screen.findByText("Cause")).toBeInTheDocument();
    const box = screen.getByRole("textbox", { name: /Search faults/i });
    await user.type(box, "butt"); // leaves only "Butt Wink"; active "Knee Valgus" is filtered out
    expect(screen.getByText("Select a fault to view its knowledge.")).toBeInTheDocument();
    // The stale Knee-Valgus graph (its legend) is gone.
    expect(screen.queryByText("Cause")).not.toBeInTheDocument();
  });

  it("recovers from a graph-fetch error: Retry clears the skeleton and renders the graph", async () => {
    graphMock.mockReset();
    graphMock.mockRejectedValueOnce(new Error("boom")).mockImplementation(graphImpl as typeof api.graph);
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    expect(await screen.findByText("Could not load the knowledge graph.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    // Retry must re-run the graph fetch (not just the fault list) so the skeleton clears.
    expect(await screen.findByText("Cause")).toBeInTheDocument();
  });

  it("handles a movement with no faults (empty list)", async () => {
    faultsMock.mockImplementation(async (movement: string) => ({ movement, faults: [] }));
    renderWithProviders(<Explore />);
    expect(await screen.findByText("No faults match.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Knee Valgus" })).not.toBeInTheDocument();
    expect(graphMock).not.toHaveBeenCalled(); // no active fault -> the graph effect early-returns
  });

  it("falls back to the first fault when none are linked, showing the no-graph message", async () => {
    faultsMock.mockImplementation(async (movement: string) => ({
      movement,
      faults: [{ name: "Bar Drift", connectivity: 0 }, { name: "Sticking Point", connectivity: 0 }],
    }));
    renderWithProviders(<Explore />);
    expect(
      await screen.findByText("This fault has no linked causes, corrections or risks yet.")
    ).toBeInTheDocument();
    await waitFor(() => expect(graphMock).toHaveBeenCalledWith("Bar Drift", "Squat"));
  });

  it("recovers when the FAULT-LIST fetch (not the graph) fails", async () => {
    faultsMock.mockRejectedValueOnce(new Error("boom"));
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    expect(await screen.findByText("Could not load the knowledge graph.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("button", { name: "Knee Valgus" })).toBeInTheDocument();
  });

  it("re-selecting the current movement is a no-op (no refetch)", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    await screen.findByRole("button", { name: "Knee Valgus" });
    faultsMock.mockClear();
    await user.click(screen.getByRole("button", { name: "Squat" })); // open selector
    await user.click(screen.getByRole("menuitemradio", { name: /Squat/ })); // pick the same movement
    expect(faultsMock).not.toHaveBeenCalled();
  });

  it("closes the movement menu on Escape and on outside click", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    await screen.findByRole("button", { name: "Knee Valgus" });
    await user.click(screen.getByRole("button", { name: "Squat" }));
    expect(screen.getByText("Flagship")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByText("Flagship")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Squat" }));
    expect(screen.getByText("Flagship")).toBeInTheDocument();
    await user.click(document.body);
    expect(screen.queryByText("Flagship")).not.toBeInTheDocument();
  });
});
