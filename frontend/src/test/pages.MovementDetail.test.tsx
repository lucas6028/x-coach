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
  it("lists the knowledge graph's faults for this movement", async () => {
    vi.spyOn(api, "movementFaults").mockResolvedValue({
      movement: "Squat",
      faults: [
        { name: "Knee Valgus", connectivity: 4 },
        { name: "Heel Rise", connectivity: 0 },
      ],
    });
    renderAt("/movements/Squat");
    await userEvent.click(screen.getByRole("tab", { name: "Common mistakes" }));

    expect(await screen.findByText("Knee Valgus")).toBeInTheDocument();
    expect(api.movementFaults).toHaveBeenCalledWith("Squat");
    expect(screen.getByText("4 linked concepts")).toBeInTheDocument();
    // A fault with nothing linked says so rather than showing an empty expander promise.
    expect(screen.getByText("No linked concepts yet")).toBeInTheDocument();
  });

  it("fetches a fault's causes, risks and cues only when it is opened", async () => {
    vi.spyOn(api, "movementFaults").mockResolvedValue({
      movement: "Squat",
      faults: [{ name: "Knee Valgus", connectivity: 3 }],
    });
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
    expect(await screen.findByText("Knee Valgus")).toBeInTheDocument();
    // Opening the tab is free: the traversal is one request per fault the reader actually opens.
    expect(graph).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole("button", { name: /Knee Valgus/ }));
    expect(await screen.findByText("Weak Gluteus Medius")).toBeInTheDocument();
    expect(screen.getByText("Knees out cue")).toBeInTheDocument();
    expect(graph).toHaveBeenCalledWith("Knee Valgus", "Squat");
  });

  it("says the graph has nothing rather than showing an empty grid", async () => {
    renderAt("/movements/Torso%20Twist");
    await userEvent.click(screen.getByRole("tab", { name: "Common mistakes" }));
    expect(await screen.findByText(/no faults authored/)).toBeInTheDocument();
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
