import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
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
    expect(await screen.findByText("Side Squat")).toBeInTheDocument();
    expect(screen.getByText("2 faults")).toBeInTheDocument();
  });

  it("groups rows under one date header per day", async () => {
    // Two rows on the same day + one 10 days earlier -> two day groups. Midday-UTC and a 10-day gap
    // keep the day boundaries stable across timezones so the header count isn't TZ-dependent.
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 3,
      items: [
        item({ id: "a", created_at: "2026-06-20T12:00:00.000Z" }),
        item({ id: "b", created_at: "2026-06-20T13:00:00.000Z" }),
        item({ id: "c", created_at: "2026-06-10T12:00:00.000Z" }),
      ],
    });
    renderHistory();
    await screen.findAllByText("2 faults"); // wait for the ready state (3 rows share the badge)

    // One <h2> date separator per distinct day (the page title is the only <h1>).
    expect(screen.getAllByRole("heading", { level: 2 })).toHaveLength(2);
    // All three rows still render, each replayable.
    expect(screen.getAllByRole("link", { name: /Side squat/i })).toHaveLength(3);
  });

  it("labels a single-fault rep and groups an unparseable date under a fallback header", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 2,
      items: [
        item({ id: "one", fault_count: 1, created_at: "2026-06-20T12:00:00.000Z" }),
        item({ id: "bad", fault_count: 3, created_at: "not-a-date" }),
      ],
    });
    renderHistory();
    expect(await screen.findByText("1 fault")).toBeInTheDocument(); // faultOne branch
    // The row with an unparseable timestamp still renders (grouped under its own fallback header).
    expect(screen.getByText("3 faults")).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 2 })).toHaveLength(2);
  });

  it("renders a clean badge for fault-free reps", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 1,
      items: [item({ fault_count: 0 })],
    });
    renderHistory();
    expect(await screen.findByText("clean rep")).toBeInTheDocument();
  });

  it("badges a row with the movement whose rules produced it", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 1,
      items: [item({ movement: "Push-up" })],
    });
    renderHistory();
    expect(await screen.findByText("Push-up")).toBeInTheDocument();
  });

  it("titles a row with its own movement, not a hardcoded Squat", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 1,
      items: [item({ movement: "Push-up" })],
    });
    renderHistory();
    expect(await screen.findByText("Side Push-up")).toBeInTheDocument();
    expect(screen.queryByText("Side Squat")).not.toBeInTheDocument();
  });

  it("badges a row predating the movement column as Squat", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 1,
      items: [item()], // no `movement` — the column didn't exist yet for this row
    });
    renderHistory();
    expect(await screen.findByText("Squat")).toBeInTheDocument();
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

  it("deletes a row after confirmation and removes it from the list", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 2,
      items: [
        item({ id: "a", created_at: "2026-06-20T12:00:00.000Z" }),
        item({ id: "b", fault_count: 0, created_at: "2026-06-20T13:00:00.000Z" }),
      ],
    });
    const del = vi.spyOn(api, "deleteAnalysis").mockResolvedValue({ deleted: 1 });
    renderHistory();
    await screen.findByText("2 faults");

    // Each row carries its own delete control; rows render in array order (not sorted by
    // recency here), so the first row in the DOM is "a" regardless of its timestamp.
    await userEvent.click(screen.getAllByRole("button", { name: "Delete this record" })[0]);
    // The icon button's accessible name comes from its aria-label ("Delete this record"), so a
    // full-string match on "Delete" hits only the confirm button.
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(del).toHaveBeenCalledWith("a"));
    // The deleted row is spliced out locally; the surviving row stays.
    await waitFor(() => expect(screen.queryByText("2 faults")).not.toBeInTheDocument());
    expect(screen.getByText("clean rep")).toBeInTheDocument();
  });

  it("keeps the row and shows an error when the delete fails", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 1, items: [item()] });
    vi.spyOn(api, "deleteAnalysis").mockRejectedValue(new Error("500 Internal Server Error"));
    renderHistory();
    await screen.findByText("2 faults");

    await userEvent.click(screen.getByRole("button", { name: "Delete this record" }));
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));

    expect(
      await screen.findByText("Couldn't delete this record. Please try again.")
    ).toBeInTheDocument();
    expect(screen.getByText("2 faults")).toBeInTheDocument(); // row survives, retryable
  });

  it("does not clear a different row's still-unresolved error when another row's trash is clicked", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 2,
      items: [
        item({ id: "a", created_at: "2026-06-20T13:00:00.000Z" }),
        item({ id: "b", fault_count: 0, created_at: "2026-06-20T12:00:00.000Z" }),
      ],
    });
    // Row "b" always fails; row "a" succeeds -- lets the second half of this test confirm A's
    // delete without touching B's still-unresolved error.
    vi.spyOn(api, "deleteAnalysis").mockImplementation((id) =>
      id === "b"
        ? Promise.reject(new Error("500 Internal Server Error"))
        : Promise.resolve({ deleted: 1 })
    );
    renderHistory();
    await screen.findByText("2 faults");

    // Newest first: row "a" (2 faults) is first, row "b" (clean) is second.
    const [trashA, trashB] = screen.getAllByRole("button", { name: "Delete this record" });

    // Fail row B's delete -> its error message appears, and its confirm auto-closes (see the
    // "keeps the row and shows an error" test above for that behavior).
    await userEvent.click(trashB);
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(
      await screen.findByText("Couldn't delete this record. Please try again.")
    ).toBeInTheDocument();

    // Opening a DIFFERENT row's confirm (row A) must not touch row B's still-unresolved error --
    // the user never retried or dismissed it.
    await userEvent.click(trashA);
    expect(
      screen.getByText("Couldn't delete this record. Please try again.")
    ).toBeInTheDocument();

    // Confirming row A's delete (which succeeds) must ALSO not touch row B's still-unresolved
    // error -- only an action on row B itself may clear row B's error.
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(screen.queryByText("2 faults")).not.toBeInTheDocument());
    expect(
      screen.getByText("Couldn't delete this record. Please try again.")
    ).toBeInTheDocument();

    // Clicking row B's own trash again *does* clear its own error.
    await userEvent.click(trashB);
    expect(
      screen.queryByText("Couldn't delete this record. Please try again.")
    ).not.toBeInTheDocument();
  });

  it("names the clicked row in the confirm dialog", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 2,
      items: [
        item({ id: "a", created_at: "2026-06-20T12:00:00.000Z" }),
        item({ id: "b", movement: "Push-up", created_at: "2026-06-20T13:00:00.000Z" }),
      ],
    });
    renderHistory();
    await screen.findAllByText("2 faults");

    // The dialog is page-level, so the only thing tying it to a row is the echoed label. Click the
    // SECOND row's trash and the dialog must describe that row, not the first.
    await userEvent.click(screen.getAllByRole("button", { name: "Delete this record" })[1]);
    const dialog = await screen.findByRole("dialog", { name: "Delete this record?" });
    expect(within(dialog).getByText(/Side Push-up ·/)).toBeInTheDocument();
    expect(within(dialog).queryByText(/Side Squat ·/)).not.toBeInTheDocument();
  });

  it("cancelling the confirmation calls no API and restores the row", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 1, items: [item()] });
    const del = vi.spyOn(api, "deleteAnalysis").mockResolvedValue({ deleted: 1 });
    renderHistory();
    await screen.findByText("2 faults");

    await userEvent.click(screen.getByRole("button", { name: "Delete this record" }));
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(del).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
    expect(screen.getByText("2 faults")).toBeInTheDocument();
  });

  it("signs out from the account menu in the shared navbar", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 0, items: [] });
    renderHistory();
    await screen.findByText("No saved analyses yet.");
    // Sign-out now lives in the unified navbar's account menu, not a page-local header button.
    await userEvent.click(screen.getByRole("button", { name: /Account menu/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /Sign out/i }));
    await waitFor(() => expect(signOut).toHaveBeenCalled());
  });
});
