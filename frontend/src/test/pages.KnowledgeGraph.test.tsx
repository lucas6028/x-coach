import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type FullGraphResponse } from "../api";

import KnowledgeGraph from "../pages/KnowledgeGraph";

// A small, hand-built fixture: 6 nodes across 3 movements (Squat, Overhead Press, shared) and 4
// labels (Fault, Cause, Cue, Action), 5 edges across 4 relations — including one edge whose BOTH
// endpoints are Squat-scoped, so the movement filter has something non-trivial to keep.
const NODES: FullGraphResponse["nodes"] = [
  { node_id: "Squat:Knee Valgus", name: "Knee Valgus", label: "Fault", movement: "Squat" },
  { node_id: "Squat:Shallow Depth", name: "Shallow Depth", label: "Fault", movement: "Squat" },
  { node_id: "Weak Glutes", name: "Weak Glutes", label: "Cause", movement: "shared" },
  { node_id: "Drive Knees Out", name: "Drive Knees Out", label: "Cue", movement: "shared" },
  { node_id: "OverheadPress:Knee Valgus", name: "Knee Valgus", label: "Fault", movement: "Overhead Press" },
  { node_id: "Overhead Press", name: "Overhead Press", label: "Action", movement: "Overhead Press" },
];
const EDGES: FullGraphResponse["edges"] = [
  { source: "Squat:Knee Valgus", target: "Squat:Shallow Depth", relation: "AFFECTS_QUALITY" },
  { source: "Squat:Knee Valgus", target: "Weak Glutes", relation: "CAUSED_BY" },
  { source: "Squat:Knee Valgus", target: "Drive Knees Out", relation: "CORRECTED_BY" },
  { source: "OverheadPress:Knee Valgus", target: "Weak Glutes", relation: "CAUSED_BY" },
  { source: "Overhead Press", target: "OverheadPress:Knee Valgus", relation: "HAS_FAULT" },
];

function makeFixture(nodes = NODES, edges = EDGES): FullGraphResponse {
  const labels: Record<string, number> = {};
  const movements: Record<string, number> = {};
  for (const n of nodes) {
    labels[n.label] = (labels[n.label] ?? 0) + 1;
    movements[n.movement] = (movements[n.movement] ?? 0) + 1;
  }
  const relations: Record<string, number> = {};
  for (const e of edges) relations[e.relation] = (relations[e.relation] ?? 0) + 1;
  return {
    graph_file: "kg.graphml",
    movement: null,
    label: null,
    counts: { nodes: nodes.length, edges: edges.length, labels, relations, movements },
    total: { nodes: nodes.length, edges: edges.length },
    nodes,
    edges,
  };
}

beforeEach(() => {
  vi.spyOn(api, "fullGraph").mockResolvedValue(makeFixture());
});
afterEach(() => vi.clearAllMocks());

describe("KnowledgeGraph page", () => {
  it("renders the counts strip once loaded", async () => {
    renderWithProviders(<KnowledgeGraph />);
    await waitFor(() => expect(screen.getByText("6 / 6 nodes")).toBeInTheDocument());
    expect(screen.getByText("5 / 5 edges")).toBeInTheDocument();
  });

  it("lists every node on the Nodes tab", async () => {
    renderWithProviders(<KnowledgeGraph />);
    await waitFor(() => expect(screen.getByText("6 / 6 nodes")).toBeInTheDocument());
    const table = screen.getByRole("table");
    // getAllByText (not getByText): a shared node's id and name are the same string, so it can
    // legitimately appear in both the id and name columns of its own row.
    for (const n of NODES) expect(within(table).getAllByText(n.node_id).length).toBeGreaterThan(0);
  });

  it("lists every edge on the Edges tab", async () => {
    renderWithProviders(<KnowledgeGraph />);
    await waitFor(() => expect(screen.getByText("6 / 6 nodes")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("tab", { name: "Edges" }));
    // Scoped to the table: the relation filter chips above it show the same relation strings
    // (as their own direct text, alongside a count), which a page-wide text query would also hit.
    const table = within(screen.getByRole("table"));
    expect(table.getAllByText("CAUSED_BY")).toHaveLength(2);
    expect(table.getByText("CORRECTED_BY")).toBeInTheDocument();
    expect(table.getByText("HAS_FAULT")).toBeInTheDocument();
    expect(table.getByText("AFFECTS_QUALITY")).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(1 + EDGES.length);
  });

  it("filters both tables by movement", async () => {
    renderWithProviders(<KnowledgeGraph />);
    await waitFor(() => expect(screen.getByText("6 / 6 nodes")).toBeInTheDocument());

    await userEvent.selectOptions(screen.getByLabelText(/Movement/), "Squat");
    // The two Squat-tagged nodes PLUS the shared Cause/Cue nodes one hop away from them
    // (same scope rule as the backend's movement filter); the Overhead Press side is excluded
    // even though it also touches "Weak Glutes".
    expect(screen.getByText("4 / 6 nodes")).toBeInTheDocument();
    expect(screen.getByText("3 / 5 edges")).toBeInTheDocument();
    let table = within(screen.getByRole("table"));
    expect(table.getByText("Squat:Knee Valgus")).toBeInTheDocument();
    expect(table.getAllByText("Weak Glutes").length).toBeGreaterThan(0); // id + name columns
    expect(table.queryByText("OverheadPress:Knee Valgus")).not.toBeInTheDocument();
    expect(table.queryByText("Overhead Press")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Edges" }));
    // Three edges have both endpoints in scope; the Overhead Press -> Weak Glutes edge does not.
    expect(screen.getAllByRole("row")).toHaveLength(4);
    table = within(screen.getByRole("table"));
    expect(table.getByText("AFFECTS_QUALITY")).toBeInTheDocument();
    expect(table.getByText("CORRECTED_BY")).toBeInTheDocument();
    expect(table.queryByText("HAS_FAULT")).not.toBeInTheDocument();
  });

  it("toggles a label chip to narrow the Nodes tab", async () => {
    renderWithProviders(<KnowledgeGraph />);
    await waitFor(() => expect(screen.getByText("6 / 6 nodes")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: /^Fault \(3\)$/ }));
    expect(screen.getByText("3 / 6 nodes")).toBeInTheDocument();
    expect(screen.getByText("Squat:Knee Valgus")).toBeInTheDocument();
    expect(screen.queryByText("Weak Glutes")).not.toBeInTheDocument();

    // Clicking again clears the filter.
    await userEvent.click(screen.getByRole("button", { name: /^Fault \(3\)$/ }));
    expect(screen.getByText("6 / 6 nodes")).toBeInTheDocument();
  });

  it("narrows both tables with the search box", async () => {
    renderWithProviders(<KnowledgeGraph />);
    await waitFor(() => expect(screen.getByText("6 / 6 nodes")).toBeInTheDocument());

    await userEvent.type(screen.getByRole("searchbox"), "shallow");
    expect(screen.getByText("1 / 6 nodes")).toBeInTheDocument();
    expect(screen.getByText("Squat:Shallow Depth")).toBeInTheDocument();
    expect(screen.queryByText("Squat:Knee Valgus")).not.toBeInTheDocument();
  });

  it("clicking a node row focuses the search on that node's id", async () => {
    renderWithProviders(<KnowledgeGraph />);
    await waitFor(() => expect(screen.getByText("6 / 6 nodes")).toBeInTheDocument());
    // "Weak Glutes" is a shared node, so its id and name columns show the same text — click
    // whichever cell matches first, the row's onClick handles either.
    await userEvent.click(screen.getAllByText("Weak Glutes")[0]);
    expect(screen.getByRole("searchbox")).toHaveValue("Weak Glutes");
  });

  it("draws one circle per filtered node and one line per filtered edge on the Graph tab", async () => {
    const { container } = renderWithProviders(<KnowledgeGraph />);
    await waitFor(() => expect(screen.getByText("6 / 6 nodes")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("tab", { name: "Graph" }));
    expect(container.querySelectorAll("svg circle")).toHaveLength(6);
    expect(container.querySelectorAll("svg line")).toHaveLength(5);
  });

  it("shows a too-many-nodes message instead of drawing past 300", async () => {
    const bigNodes = Array.from({ length: 301 }, (_, i) => ({
      node_id: `Squat:Node ${i}`,
      name: `Node ${i}`,
      label: "Fault",
      movement: "Squat",
    }));
    vi.spyOn(api, "fullGraph").mockResolvedValue(makeFixture(bigNodes, []));
    const { container } = renderWithProviders(<KnowledgeGraph />);
    await waitFor(() => expect(screen.getByText("301 / 301 nodes")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("tab", { name: "Graph" }));
    expect(screen.getByText(/Narrow the filters/)).toBeInTheDocument();
    expect(container.querySelectorAll("svg circle")).toHaveLength(0);
  });

  it("shows an error state with a working retry", async () => {
    vi.spyOn(api, "fullGraph").mockRejectedValueOnce(new Error("offline"));
    renderWithProviders(<KnowledgeGraph />);
    expect(await screen.findByText(/Could not load the knowledge graph/)).toBeInTheDocument();

    vi.spyOn(api, "fullGraph").mockResolvedValue(makeFixture());
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(screen.getByText("6 / 6 nodes")).toBeInTheDocument());
    expect(api.fullGraph).toHaveBeenCalled();
  });
});
