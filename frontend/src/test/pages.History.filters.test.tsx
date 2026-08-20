import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { api, type HistoryItem } from "../api";

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
import History from "../pages/History";

const mockUseAuth = vi.mocked(useAuth);

// Dates are built relative to "now" rather than pinned: the range presets and the Today/Yesterday
// separators are both defined against the local midnight of the machine running the test, so a
// fixed ISO string would make them timezone- and calendar-dependent.
function daysAgo(n: number, hour = 12): string {
  const d = new Date();
  d.setDate(d.getDate() - n);
  d.setHours(hour, 0, 0, 0);
  return d.toISOString();
}

function item(over: Partial<HistoryItem> = {}): HistoryItem {
  return {
    id: "a1",
    video_id: "upload_1",
    source: "upload",
    view_type: "side",
    fault_count: 2,
    created_at: daysAgo(0),
    ...over,
  };
}

// Four rows spanning today / yesterday / three days back / forty days back, two movements, one
// clean — enough for every filter and both relative day labels to be exercised.
const ROWS: HistoryItem[] = [
  item({ id: "today-squat", movement: "Squat", fault_count: 2, created_at: daysAgo(0) }),
  item({ id: "yday-pushup", movement: "Push-up", fault_count: 0, created_at: daysAgo(1) }),
  item({ id: "d3-squat", movement: "Squat", fault_count: 1, created_at: daysAgo(3) }),
  item({ id: "d40-pushup", movement: "Push-up", fault_count: 4, created_at: daysAgo(40) }),
];

function renderHistory() {
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={["/history"]}>
        <Routes>
          <Route path="/history" element={<History />} />
          <Route path="/app" element={<div>app studio</div>} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

/** The record cards currently rendered, by their accessible link name. */
const cards = () => screen.queryAllByRole("link", { name: /Side (Squat|Push-up)/i });

/**
 * The filter controls are the studio's MenuCard, not native selects: the trigger is a button named
 * "<caption>: <current value>", and choosing is a click on a `menuitemradio`. Driving them by
 * their visible caption keeps these tests readable and independent of which value is showing.
 */
async function pick(caption: string, option: string | RegExp) {
  await userEvent.click(screen.getByRole("button", { name: new RegExp(`^${caption}:`) }));
  await userEvent.click(screen.getByRole("menuitemradio", { name: option }));
}

beforeEach(() => {
  mockUseAuth.mockReturnValue({
    user: { email: "ada@x.com" },
    signOut: vi.fn(),
  } as unknown as ReturnType<typeof useAuth>);
  vi.spyOn(api, "uploadMediaBatch").mockResolvedValue({});
  vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: ROWS.length, items: ROWS });
});
afterEach(() => vi.restoreAllMocks());

describe("History — summary strip", () => {
  it("reports the API's all-time total, not the number of rows on the page", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 128, items: ROWS });
    renderHistory();
    const tile = (await screen.findByText("Total analyses")).closest("div")!;
    expect(within(tile).getByText("128")).toBeInTheDocument();
  });

  // The point of the whole "label the scope" decision: rates come from one page, so when the page
  // is not the whole history the strip has to say which window it is describing.
  it("says which window the derived rates cover when the page is partial", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 128, items: ROWS });
    renderHistory();
    expect(
      await screen.findByText("Rates below are from the 4 most recent of 128 analyses.")
    ).toBeInTheDocument();
  });

  it("drops the scope note once the page holds every record", async () => {
    renderHistory();
    await screen.findByText("Total analyses");
    expect(screen.queryByText(/Rates below are from/)).not.toBeInTheDocument();
  });

  it("computes the clean-rep rate and the most-trained movement from the loaded rows", async () => {
    renderHistory();
    // One of four rows is fault-free.
    expect(await screen.findByText("25%")).toBeInTheDocument();
    // Squat and Push-up tie at two each; the tile shows one of them with its count.
    const tile = screen.getByText("Most trained").closest("div")!;
    expect(within(tile).getByText("2 analyses")).toBeInTheDocument();
  });
});

describe("History — filters", () => {
  it("filters by movement", async () => {
    renderHistory();
    await screen.findByText("Total analyses");
    expect(cards()).toHaveLength(4);

    await pick("Movement", "Push-up");
    expect(cards()).toHaveLength(2);
    expect(screen.queryByRole("link", { name: /Side Squat/i })).not.toBeInTheDocument();
  });

  it("filters by result", async () => {
    renderHistory();
    await screen.findByText("Total analyses");

    await pick("Result", "Clean reps");
    expect(cards()).toHaveLength(1);

    await pick("Result", "Needs work");
    expect(cards()).toHaveLength(3);
  });

  it("filters by period, counting from local midnight", async () => {
    renderHistory();
    await screen.findByText("Total analyses");

    await pick("Period", "Today");
    expect(cards()).toHaveLength(1);

    // 7 days covers today / yesterday / 3-days-ago but not the 40-day-old row.
    await pick("Period", "Last 7 days");
    expect(cards()).toHaveLength(3);
  });

  it("searches by movement name", async () => {
    renderHistory();
    await screen.findByText("Total analyses");

    await userEvent.type(screen.getByLabelText("Search movements"), "push");
    expect(cards()).toHaveLength(2);
  });

  // "Nothing matches your filters" and "you have no records" are different situations; showing the
  // empty state here would read as data loss.
  it("distinguishes an empty filter result from an empty history", async () => {
    renderHistory();
    await screen.findByText("Total analyses");

    await userEvent.type(screen.getByLabelText("Search movements"), "zzzz");
    expect(await screen.findByText("No records match these filters.")).toBeInTheDocument();
    expect(screen.queryByText("No saved analyses yet.")).not.toBeInTheDocument();
  });

  it("offers a reset only while something is filtered, and it restores every row", async () => {
    renderHistory();
    await screen.findByText("Total analyses");
    expect(screen.queryByRole("button", { name: "Clear filters" })).not.toBeInTheDocument();

    await pick("Movement", "Push-up");
    await userEvent.click(screen.getByRole("button", { name: "Clear filters" }));

    expect(cards()).toHaveLength(4);
    expect(screen.queryByRole("button", { name: "Clear filters" })).not.toBeInTheDocument();
  });

  // The menu must not offer a movement the user has never recorded, and must lead with the
  // "no filter" entry so one control can undo itself.
  it("builds the movement menu from the rows that exist", async () => {
    renderHistory();
    await screen.findByText("Total analyses");
    await userEvent.click(screen.getByRole("button", { name: /^Movement:/ }));
    const menu = screen.getByRole("menu", { name: "Movement" });
    const options = within(menu)
      .getAllByRole("menuitemradio")
      .map((o) => o.textContent);
    expect(options).toEqual(["All movements", "Push-up", "Squat"]);
  });

  // Dismissal is the part that goes wrong when a dropdown is reimplemented per site, which is why
  // the studio's control is now shared rather than copied.
  it("closes an open filter menu on Escape", async () => {
    renderHistory();
    await screen.findByText("Total analyses");
    await userEvent.click(screen.getByRole("button", { name: /^Period:/ }));
    expect(screen.getByRole("menu", { name: "Period" })).toBeInTheDocument();

    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("menu", { name: "Period" })).not.toBeInTheDocument();
  });
});

// On the phone the three menus fold behind a funnel button. Driven by `useIsMobile`, so this
// forces its media query to match — the global setup stub answers `false` to everything, which is
// what keeps every describe above on the desktop row.
describe("History — filters on the phone", () => {
  beforeEach(() => {
    vi.spyOn(window, "matchMedia").mockImplementation(
      (query: string) =>
        ({
          matches: query === "(max-width: 1023px)",
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn(),
        }) as unknown as MediaQueryList
    );
  });

  it("folds the three menus away, keeping search reachable in one tap", async () => {
    renderHistory();
    await screen.findByText("Total analyses");

    expect(screen.getByLabelText("Search movements")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Filters" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Movement:/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Result:/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Period:/ })).not.toBeInTheDocument();
  });

  it("reveals all three when the funnel is tapped, and filters from there", async () => {
    renderHistory();
    await screen.findByText("Total analyses");

    await userEvent.click(screen.getByRole("button", { name: "Filters" }));
    expect(screen.getByRole("button", { name: /^Movement:/ })).toBeInTheDocument();

    await pick("Movement", "Push-up");
    expect(cards()).toHaveLength(2);
  });

  // With the menus folded away, this badge is the only thing telling the user that the list in
  // front of them is being narrowed by something they cannot see.
  it("counts the applied filters on the button, ignoring the visible search", async () => {
    renderHistory();
    await screen.findByText("Total analyses");

    await userEvent.click(screen.getByRole("button", { name: "Filters" }));
    await pick("Movement", "Push-up");
    await pick("Result", "Clean reps");
    expect(screen.getByRole("button", { name: "Filters (2 active)" })).toBeInTheDocument();

    // Search is visible either way, so it must not add to the count of what is hidden.
    await userEvent.type(screen.getByLabelText("Search movements"), "push");
    expect(screen.getByRole("button", { name: "Filters (2 active)" })).toBeInTheDocument();
  });
});

describe("History — day separators", () => {
  it("labels the two most recent days relatively and dates the rest", async () => {
    renderHistory();
    await screen.findByText("Total analyses");
    const heads = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
    expect(heads[0]).toContain("Today");
    expect(heads[1]).toContain("Yesterday");
    // The older two groups fall back to a written date, so they carry neither relative label.
    expect(heads.slice(2).join(" ")).not.toMatch(/Today|Yesterday/);
  });
});
