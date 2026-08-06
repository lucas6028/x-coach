import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

// The card's whole job is the signed-in path, so the session is mocked rather than driven through
// the real AuthProvider (which would need a live Supabase exchange). Same approach as
// components.Header.auth.test.tsx.
const { mockUseAuth } = vi.hoisted(() => ({ mockUseAuth: vi.fn() }));
vi.mock("../lib/auth", () => ({ useAuth: mockUseAuth }));

import { api, type HistoryItem } from "../api";
import PreviousSessionsCard from "../components/studio/PreviousSessionsCard";

const row = (over: Partial<HistoryItem> = {}): HistoryItem => ({
  id: "a1",
  video_id: "v1",
  source: "upload",
  view_type: "side",
  fault_count: 2,
  created_at: "2026-08-02T10:00:00Z",
  movement: "Squat",
  ...over,
});

function renderCard(currentVideoId?: string) {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <PreviousSessionsCard currentVideoId={currentVideoId} />
      </I18nProvider>
    </MemoryRouter>
  );
}

beforeEach(() => {
  mockUseAuth.mockReturnValue({ user: { email: "ada@example.com" } });
});
afterEach(() => vi.restoreAllMocks());

describe("PreviousSessionsCard — signed in", () => {
  it("lists earlier sessions with their real fault count", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 2,
      items: [row(), row({ id: "a2", video_id: "v2", fault_count: 0 })],
    });
    vi.spyOn(api, "uploadMediaBatch").mockResolvedValue({});

    renderCard();
    // The count comes from the row itself — the list endpoint promotes fault_count but not the
    // per-fault severities the form-score ring needs, so this card must not print a score.
    expect(await screen.findByText("2 faults")).toBeInTheDocument();
    expect(screen.getByText("Clean")).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: /Squat/ })[0]).toHaveAttribute(
      "href",
      "/app?analysis=a1"
    );
  });

  it("excludes the analysis already on screen — these are PREVIOUS sessions", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 2,
      items: [row({ id: "current", video_id: "here" }), row({ id: "a2", video_id: "v2" })],
    });
    vi.spyOn(api, "uploadMediaBatch").mockResolvedValue({});

    renderCard("here");
    await waitFor(() => expect(screen.getAllByRole("link")).toHaveLength(2)); // "View all" + one row
    expect(screen.queryByRole("link", { name: /2 faults/ })).not.toHaveAttribute(
      "href",
      "/app?analysis=current"
    );
  });

  it("renders the thumbnail when storage returns one", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 1, items: [row()] });
    vi.spyOn(api, "uploadMediaBatch").mockResolvedValue({
      v1: { video_url: "v", thumbnail_url: "https://cdn.test/t.jpg" },
    });

    const { container } = renderCard();
    await waitFor(() =>
      expect(container.querySelector("img")).toHaveAttribute("src", "https://cdn.test/t.jpg")
    );
  });

  // Thumbnails are decoration: a storage failure must leave a readable list, not blank it.
  it("still lists the rows when the thumbnail batch fails", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 1, items: [row()] });
    vi.spyOn(api, "uploadMediaBatch").mockRejectedValue(new Error("no storage"));

    renderCard();
    expect(await screen.findByText("2 faults")).toBeInTheDocument();
  });

  it("shows the empty note when this is the user's first session", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 0, items: [] });
    vi.spyOn(api, "uploadMediaBatch").mockResolvedValue({});

    renderCard();
    expect(
      await screen.findByText("No earlier sessions yet — this is your first.")
    ).toBeInTheDocument();
  });

  it("reports a failed fetch instead of spinning forever", async () => {
    vi.spyOn(api, "listAnalyses").mockRejectedValue(new Error("500"));

    renderCard();
    expect(await screen.findByText("Couldn't load your earlier sessions.")).toBeInTheDocument();
  });
});
