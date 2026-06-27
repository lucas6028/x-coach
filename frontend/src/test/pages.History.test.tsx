import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { api, type HistoryItem } from "../api";

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
import History from "../pages/History";

const mockUseAuth = vi.mocked(useAuth);
const signOut = vi.fn().mockResolvedValue(undefined);

function item(over: Partial<HistoryItem> = {}): HistoryItem {
  return {
    id: "a1",
    video_id: "upload_1",
    source: "upload",
    view_type: "side",
    fault_count: 2,
    created_at: "2026-06-20T10:00:00.000Z",
    ...over,
  };
}

function renderHistory() {
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={["/history"]}>
        <Routes>
          <Route path="/history" element={<History />} />
          <Route path="/app" element={<div>app studio</div>} />
          <Route path="/" element={<div>home page</div>} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

beforeEach(() => {
  mockUseAuth.mockReturnValue({
    user: { email: "ada@x.com" },
    signOut,
  } as unknown as ReturnType<typeof useAuth>);
});
afterEach(() => vi.restoreAllMocks());

describe("History", () => {
  it("lists saved analyses", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 1, items: [item()] });
    renderHistory();
    expect(await screen.findByText("Side squat")).toBeInTheDocument();
    expect(screen.getByText("2 faults")).toBeInTheDocument();
  });

  it("renders a clean badge for fault-free reps", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 1,
      items: [item({ fault_count: 0 })],
    });
    renderHistory();
    expect(await screen.findByText("clean rep")).toBeInTheDocument();
  });

  it("shows the empty state with no analyses", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 0, items: [] });
    renderHistory();
    expect(await screen.findByText("No saved analyses yet.")).toBeInTheDocument();
  });

  it("shows an error and retries", async () => {
    const spy = vi
      .spyOn(api, "listAnalyses")
      .mockRejectedValueOnce(new Error("401 Unauthorized"))
      .mockResolvedValueOnce({ total: 0, items: [] });
    renderHistory();
    expect(await screen.findByText("Couldn't load your history")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("No saved analyses yet.")).toBeInTheDocument();
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("signs out from the header", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 0, items: [] });
    renderHistory();
    await screen.findByText("No saved analyses yet.");
    await userEvent.click(screen.getByRole("button", { name: /Sign out/i }));
    await waitFor(() => expect(signOut).toHaveBeenCalled());
  });
});
