import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type RetrievalContext } from "../api";
import { ALL_MOVEMENTS } from "../lib/movements";

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

const MOVEMENTS = new Set<string>(ALL_MOVEMENTS);

// A movement query (query === movement) returns the movement's Fault nodes; a fault query returns
// that fault's cause/correction/risk summary.
const defaultGraph = async (query: string): Promise<RetrievalContext> => {
  if (MOVEMENTS.has(query)) {
    return {
      subgraph: {
        nodes: [
          { node_id: "Squat:Knee Valgus", name: "Knee Valgus", label: "Fault" },
          { node_id: "Squat:Butt Wink", name: "Butt Wink", label: "Fault" },
        ],
        edges: [],
      },
    };
  }
  return {
    results: [
      {
        summary: {
          causes: [{ node_id: "Weak Hip Abductors", name: "Weak Hip Abductors" }],
          risks: [],
          corrections: [],
          evidence: [],
        },
      },
    ],
  };
};

let graphMock: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  graphMock = vi.spyOn(api, "graph").mockImplementation(defaultGraph as typeof api.graph);
});
afterEach(() => vi.restoreAllMocks());

describe("Explore", () => {
  it("defaults the movement selector to Squat and loads its fault chips", async () => {
    renderWithProviders(<Explore />);
    // Fault chips appear once the movement subgraph resolves.
    expect(await screen.findByRole("button", { name: "Knee Valgus" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Butt Wink" })).toBeInTheDocument();
    // Trigger control (a plain button, distinct from the menu's radios) shows the current movement.
    expect(screen.getByRole("button", { name: "Squat" })).toBeInTheDocument();
  });

  it("renders the graph scene and legend for the first fault", async () => {
    renderWithProviders(<Explore />);
    // The first fault is auto-selected -> its subgraph loads -> the legend + SVG render.
    expect(await screen.findByText("Cause")).toBeInTheDocument();
    expect(document.querySelector("svg")).toBeInTheDocument();
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
    await waitFor(() => expect(graphMock).toHaveBeenCalledWith("Lunge", "Lunge"));
  });

  it("filters fault chips by search and shows the empty state on no match", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    await screen.findByRole("button", { name: "Knee Valgus" });
    const box = screen.getByRole("textbox", { name: /Search faults/i });
    await user.type(box, "butt");
    expect(screen.getByRole("button", { name: "Butt Wink" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Knee Valgus" })).not.toBeInTheDocument();
    await user.clear(box);
    await user.type(box, "zzz");
    expect(screen.getByText("No faults match.")).toBeInTheDocument();
  });

  it("shows an error with a retry that reloads the faults", async () => {
    graphMock.mockReset();
    graphMock
      .mockRejectedValueOnce(new Error("boom"))
      .mockImplementation(defaultGraph as typeof api.graph);
    const user = userEvent.setup();
    renderWithProviders(<Explore />);
    expect(await screen.findByText("Could not load the knowledge graph.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("button", { name: "Knee Valgus" })).toBeInTheDocument();
  });
});
