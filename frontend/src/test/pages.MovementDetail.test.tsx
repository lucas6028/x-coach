import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { api } from "../api";

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
import MovementDetail from "../pages/MovementDetail";

const navigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => navigate };
});

const mockUseAuth = vi.mocked(useAuth);

const LIVE = [
  { name: "Squat", validated: true },
  { name: "Overhead Press", validated: false },
];

// The route table in miniature: the real path pattern, so `useParams` sees a percent-encoded
// canonical movement name exactly as the router hands it over, plus the library it redirects to.
function renderAt(path: string) {
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/movements" element={<div>library page</div>} />
          <Route path="/movements/:movement" element={<MovementDetail />} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

beforeEach(() => {
  localStorage.clear();
  navigate.mockClear();
  mockUseAuth.mockReturnValue({ user: null } as unknown as ReturnType<typeof useAuth>);
  vi.spyOn(api, "getMovements").mockResolvedValue(LIVE);
  vi.spyOn(api, "movementFaults").mockResolvedValue({ movement: "Squat", faults: [] });
});
afterEach(() => vi.restoreAllMocks());

describe("MovementDetail — the movement itself", () => {
  it("names the movement, its body region and what it is", async () => {
    renderAt("/movements/Squat");
    expect(await screen.findByRole("heading", { name: "Squat", level: 1 })).toBeInTheDocument();
    expect(screen.getByText("Lower body")).toBeInTheDocument();
    expect(screen.getByText(/fundamental lower-body pattern/)).toBeInTheDocument();
  });

  it("lists the authored steps with their figures", async () => {
    renderAt("/movements/Squat");
    expect(await screen.findByText(/feet shoulder-width apart/)).toBeInTheDocument();
    expect(screen.getByText(/thighs are parallel/)).toBeInTheDocument();
  });

  it("names the trained muscles on both sides of the body map", async () => {
    renderAt("/movements/Squat");
    // Both the legend (Overview) and the map are driven by the same list, so naming the muscle is
    // what proves the page is about THIS movement and not a fixed picture.
    expect(await screen.findByText("Quadriceps")).toBeInTheDocument();
    expect(screen.getByText("Glutes")).toBeInTheDocument();
    expect(screen.getByText("Hamstrings")).toBeInTheDocument();
  });

  it("reports the taxonomy and whether the pipeline can analyse it", async () => {
    renderAt("/movements/Squat");
    expect(await screen.findByText("Bodyweight")).toBeInTheDocument();
    expect(screen.getByText("Compound")).toBeInTheDocument();
    // The row that is a fact about this app rather than about the exercise.
    await waitFor(() => expect(screen.getByText("Live")).toBeInTheDocument());
  });

  it("renders the whole page in Traditional Chinese", async () => {
    localStorage.setItem("lang", "zh-Hant");
    renderAt("/movements/Squat");
    expect(await screen.findByRole("heading", { name: "深蹲", level: 1 })).toBeInTheDocument();
    expect(screen.getByText(/最基本的下肢動作/)).toBeInTheDocument();
    expect(screen.getByText("股四頭肌")).toBeInTheDocument();
  });

  it("sends an unknown movement back to the library instead of erroring", async () => {
    renderAt("/movements/Burpee");
    expect(await screen.findByText("library page")).toBeInTheDocument();
  });

  it("resolves a percent-encoded name with a space", async () => {
    renderAt("/movements/Overhead%20Press");
    expect(
      await screen.findByRole("heading", { name: "Overhead Press", level: 1 })
    ).toBeInTheDocument();
  });
});

describe("MovementDetail — starting an analysis", () => {
  it("opens the studio on the dropzone, or on the camera, per the card picked", async () => {
    renderAt("/movements/Squat");
    await waitFor(() => expect(api.getMovements).toHaveBeenCalled());

    await userEvent.click(await screen.findByRole("button", { name: /Upload a video/ }));
    expect(navigate).toHaveBeenCalledWith("/app?movement=Squat");

    await userEvent.click(screen.getByRole("button", { name: /Start recording/ }));
    // `capture=record` is the seam that makes the two cards do different things — without it
    // "Start recording" would land on the upload tab.
    expect(navigate).toHaveBeenCalledWith("/app?movement=Squat&capture=record");
  });

  it("says why a movement cannot be analysed instead of offering a button that would fail", async () => {
    renderAt("/movements/Bicep%20Curl");
    expect(await screen.findByText(/No detector is registered/)).toBeInTheDocument();
    await waitFor(() => expect(api.getMovements).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: /Start recording/ })).toBeNull();
    expect(screen.getByText("Not yet")).toBeInTheDocument();
  });

  it("tags a movement whose rules are unvalidated as Beta", async () => {
    renderAt("/movements/Overhead%20Press");
    expect(await screen.findByText("Beta")).toBeInTheDocument();
  });
});

describe("MovementDetail — common mistakes", () => {
  it("cards the faults this movement's DETECTOR reports, not the graph's fault index", async () => {
    const faults = vi.spyOn(api, "movementFaults");
    renderAt("/movements/Squat");
    await userEvent.click(screen.getByRole("tab", { name: "Common mistakes" }));

    // Squat's five detector rules, in the order src/pose/movements/squat.py defines them.
    expect(await screen.findByRole("heading", { name: "Knees caving in", level: 3 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Heels lifting off the floor", level: 3 })).toBeInTheDocument();
    const cards = screen.getAllByRole("heading", { level: 3 });
    expect(cards.map((h) => h.textContent)).toEqual([
      "Knees caving in",
      "Knees travelling too far forward",
      "Not squatting deep enough",
      "Leaning too far forward",
      "Heels lifting off the floor",
    ]);
    // The list is authored locally, so opening the tab costs no request at all — the endpoint that
    // used to drive it is not called even once.
    expect(faults).not.toHaveBeenCalled();
  });

  it("states each fault, argues it, and lists its cues", async () => {
    renderAt("/movements/Squat");
    await userEvent.click(screen.getByRole("tab", { name: "Common mistakes" }));

    expect(await screen.findByText("Knees collapse inward during the squat movement.")).toBeInTheDocument();
    expect(
      screen.getByText("This puts excessive stress on your knee joints and can lead to injury over time.")
    ).toBeInTheDocument();
    expect(screen.getAllByText("Why it's a problem")).toHaveLength(5);
    expect(screen.getAllByText("How to fix it")).toHaveLength(5);
    expect(screen.getByText("Think about spreading the floor apart.")).toBeInTheDocument();
  });

  it("fetches a fault's causes, risks and cues only when it is opened, by its detector's kg_query", async () => {
    const graph = vi.spyOn(api, "graph").mockResolvedValue({
      results: [
        {
          summary: {
            causes: [{ node_id: "c1", name: "Weak Gluteus Medius", label: "Cause" }],
            corrections: [{ node_id: "q1", name: "Knees out cue", label: "Cue" }],
          },
        },
      ],
    });

    renderAt("/movements/Squat");
    await userEvent.click(screen.getByRole("tab", { name: "Common mistakes" }));
    expect(await screen.findByRole("heading", { name: "Knees caving in", level: 3 })).toBeInTheDocument();
    // Opening the tab is free: the traversal is one request per fault the reader actually opens.
    expect(graph).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole("button", { name: /Knees caving in/ }));
    expect(await screen.findByText("Weak Gluteus Medius")).toBeInTheDocument();
    expect(screen.getByText("Knees out cue")).toBeInTheDocument();
    // "Knee Valgus", not "Knees caving in": the card's title is written for a reader, the query has
    // to be the string squat.py's rule sends.
    expect(graph).toHaveBeenCalledWith("Knee Valgus", "Squat");
  });

  it("says so when the traversal fails, instead of a card that expands into nothing", async () => {
    vi.spyOn(api, "graph").mockRejectedValue(new Error("offline"));
    renderAt("/movements/Squat");
    await userEvent.click(screen.getByRole("tab", { name: "Common mistakes" }));
    await userEvent.click(await screen.findByRole("button", { name: /Knees caving in/ }));
    expect(await screen.findByText("Couldn't load the linked concepts.")).toBeInTheDocument();
  });

  it("retries a failed traversal in place, without shutting the card on the retry click", async () => {
    const graph = vi
      .spyOn(api, "graph")
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue({
        results: [{ summary: { causes: [{ node_id: "c1", name: "Weak Gluteus Medius", label: "Cause" }] } }],
      });
    renderAt("/movements/Squat");
    await userEvent.click(screen.getByRole("tab", { name: "Common mistakes" }));
    const expander = await screen.findByRole("button", { name: /Knees caving in/ });

    await userEvent.click(expander);
    expect(await screen.findByText("Couldn't load the linked concepts.")).toBeInTheDocument();

    // The second click is the retry, and its result has to be visible immediately. Toggling the
    // card shut here would put the answer behind a third click.
    await userEvent.click(expander);
    expect(await screen.findByText("Weak Gluteus Medius")).toBeInTheDocument();
    expect(graph).toHaveBeenCalledTimes(2);

    // Only once it is showing something does a click close it again.
    await userEvent.click(expander);
    expect(screen.queryByText("Weak Gluteus Medius")).not.toBeInTheDocument();
    expect(graph).toHaveBeenCalledTimes(2);
  });

  it("draws all five squat faults as their own wrong/correct pair", async () => {
    renderAt("/movements/Squat");
    await userEvent.click(screen.getByRole("tab", { name: "Common mistakes" }));
    await screen.findByRole("heading", { name: "Knees caving in", level: 3 });
    // Squat is fully illustrated, so no slot falls back to its placeholder. The placeholders are
    // still the behaviour every other movement takes, which lib.movementMistakes covers.
    expect(screen.queryByText("Common mistake")).not.toBeInTheDocument();
    expect(screen.queryByText("Correct form")).not.toBeInTheDocument();
    // Ten images, in detector order, and every one of them a DIFFERENT file. Asserting the whole
    // list rather than a count is what catches the failure this design exists to prevent: one
    // drawing captioned both ways, or a pair pointing at the neighbouring fault's art.
    const drawn = screen.getAllByRole("img").map((img) => img.getAttribute("src"));
    expect(drawn).toEqual([
      "/movements/mistakes/knees-inward-wrong.webp",
      "/movements/mistakes/knees-inward-correct.webp",
      "/movements/mistakes/knees-forward-wrong.webp",
      "/movements/mistakes/knees-forward-correct.webp",
      "/movements/mistakes/shallow-depth-wrong.webp",
      "/movements/mistakes/shallow-depth-correct.webp",
      "/movements/mistakes/excessive-forward-lean-wrong.webp",
      "/movements/mistakes/excessive-forward-lean-correct.webp",
      "/movements/mistakes/heel-rise-wrong.webp",
      "/movements/mistakes/heel-rise-correct.webp",
    ]);
    expect(new Set(drawn).size).toBe(drawn.length);
  });

  it("says a movement with no detector has nothing to watch for", async () => {
    // Jumping Jacks is in the catalog but has no registered detector — every rule of its detector
    // is permanently silent or withdrawn — so the honest answer is that there is nothing to list.
    renderAt("/movements/Jumping%20Jacks");
    await userEvent.click(screen.getByRole("tab", { name: "Common mistakes" }));
    expect(await screen.findByText(/no fault checks yet/)).toBeInTheDocument();
  });
});

describe("MovementDetail — my records", () => {
  it("asks a signed-out visitor to sign in, without gating the rest of the page", async () => {
    renderAt("/movements/Squat");
    await userEvent.click(screen.getByRole("tab", { name: "My records" }));
    expect(await screen.findByText(/Sign in to keep a history/)).toBeInTheDocument();
  });

  it("shows only this movement's saved analyses", async () => {
    mockUseAuth.mockReturnValue({
      user: { email: "ada@x.com" },
    } as unknown as ReturnType<typeof useAuth>);
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 3,
      items: [
        {
          id: "a1",
          video_id: "v1",
          source: "upload",
          view_type: "side",
          fault_count: 2,
          created_at: "2026-06-20T10:00:00.000Z",
          movement: "Squat",
        },
        {
          id: "a2",
          video_id: "v2",
          source: "upload",
          view_type: "side",
          fault_count: 0,
          created_at: "2026-06-21T10:00:00.000Z",
          movement: "Push-up",
        },
        // No `movement` at all: saved before per-movement selection existed, and counted as Squat
        // the same way History counts it.
        {
          id: "a3",
          video_id: "v3",
          source: "upload",
          view_type: "side",
          fault_count: 0,
          created_at: "2026-06-22T10:00:00.000Z",
        },
      ],
    });

    renderAt("/movements/Squat");
    await userEvent.click(screen.getByRole("tab", { name: "My records" }));

    const rows = await screen.findAllByRole("link", { name: /2026/ });
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByText("2 faults")).toBeInTheDocument();
    expect(screen.getByText("Clean")).toBeInTheDocument();
  });

  it("says so when the account holds more sessions than the fetched window", async () => {
    mockUseAuth.mockReturnValue({
      user: { email: "ada@x.com" },
    } as unknown as ReturnType<typeof useAuth>);
    // The filter runs client-side over one window, so a heavier account is showing a subset —
    // the alternative to saying so is presenting that subset as the whole history.
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 240,
      items: [
        {
          id: "a1",
          video_id: "v1",
          source: "upload",
          view_type: "side",
          fault_count: 1,
          created_at: "2026-06-20T10:00:00.000Z",
          movement: "Squat",
        },
      ],
    });

    renderAt("/movements/Squat");
    await userEvent.click(screen.getByRole("tab", { name: "My records" }));
    expect(await screen.findByText(/1 most recent sessions/)).toBeInTheDocument();
    expect(screen.getByText(/240 saved in all/)).toBeInTheDocument();
  });
});
