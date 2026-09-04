import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Plan, type PlanItem } from "../api";

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useParams: () => ({ planId: "p1" }) };
});

import PlanDetail from "../pages/PlanDetail";

function item(overrides: Partial<PlanItem> & { id: string; day_index: number }): PlanItem {
  return {
    plan_id: "p1",
    position: 0,
    movement: "Squat",
    sets: 3,
    reps: 10,
    notes: null,
    completed_at: null,
    analysis_id: null,
    created_at: "2026-08-13T00:00:00Z",
    ...overrides,
  };
}

function plan(overrides: Partial<Plan> = {}): Plan {
  return {
    id: "p1",
    name: "Upper week",
    notes: null,
    template_key: null,
    started_at: null,
    created_at: "2026-08-13T00:00:00Z",
    updated_at: "2026-08-13T00:00:00Z",
    items: [item({ id: "i1", day_index: 1 })],
    ...overrides,
  };
}

beforeEach(() => {
  vi.spyOn(api, "getMovements").mockResolvedValue([
    { name: "Squat", validated: true },
    { name: "Row", validated: false },
  ]);
  vi.spyOn(api, "getPlan").mockResolvedValue(plan());
});
afterEach(() => vi.restoreAllMocks());

describe("PlanDetail — rendering", () => {
  it("shows every day slot, rest days included", async () => {
    // Hiding empty days would make "Day 3" mean a different position in different plans.
    // Asserted through the strip's tabs rather than by counting the words "rest day" loose on the
    // page: the week is now seven tiles, and "there are seven of them, six marked rest" is the
    // same claim stated against the structure that carries it.
    renderWithProviders(<PlanDetail />);
    const strip = await screen.findByRole("tablist");
    const tabs = within(strip).getAllByRole("tab");
    expect(tabs).toHaveLength(7);
    expect(tabs[0]).toHaveTextContent("Day 1");
    expect(tabs[6]).toHaveTextContent("Day 7");
    expect(within(strip).getAllByText("Rest")).toHaveLength(6);
  });

  it("opens on the day the user is on", async () => {
    // Default selection: currentDay, else the first day with items, else day 1. Day 1 holds the
    // only (unticked) exercise here, so that is the day the panel must be showing.
    renderWithProviders(<PlanDetail />);
    const panel = await screen.findByRole("tabpanel");
    expect(within(panel).getByRole("heading", { name: "Day 1" })).toBeInTheDocument();
    expect(within(panel).getByText("Squat")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /day 1/i })).toHaveAttribute("aria-selected", "true");
  });

  it("opens on the current day when earlier days are already done", async () => {
    vi.mocked(api.getPlan).mockResolvedValue(
      plan({
        items: [
          item({ id: "i1", day_index: 1, completed_at: "2026-08-13T01:00:00Z" }),
          item({ id: "i2", day_index: 4, movement: "Row" }),
        ],
      })
    );
    renderWithProviders(<PlanDetail />);
    const panel = await screen.findByRole("tabpanel");
    expect(within(panel).getByRole("heading", { name: "Day 4" })).toBeInTheDocument();
    expect(within(panel).getByText("Row")).toBeInTheDocument();
  });

  it("shows another day's exercises when its tile is picked", async () => {
    vi.mocked(api.getPlan).mockResolvedValue(
      plan({
        items: [
          item({ id: "i1", day_index: 1 }),
          item({ id: "i2", day_index: 5, movement: "Row" }),
        ],
      })
    );
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("tab", { name: /day 5/i }));

    const panel = screen.getByRole("tabpanel");
    expect(within(panel).getByText("Row")).toBeInTheDocument();
    expect(within(panel).queryByText("Squat")).toBeNull();
    expect(screen.getByRole("tab", { name: /day 5/i })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: /day 1/i })).toHaveAttribute("aria-selected", "false");
  });

  it("shows a composed rest-day state with a way out of it", async () => {
    const add = vi
      .spyOn(api, "addPlanItem")
      .mockResolvedValue(item({ id: "i2", day_index: 2, movement: "Row" }));
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("tab", { name: /day 2/i }));

    const panel = screen.getByRole("tabpanel");
    expect(within(panel).getByText(/nothing planned for this day/i)).toBeInTheDocument();
    // The empty state's own action is the only "Add exercise" on a rest day, and it works.
    await userEvent.click(within(panel).getByRole("button", { name: /add exercise/i }));
    await userEvent.selectOptions(within(panel).getByLabelText(/movement/i), "Row");
    await userEvent.click(within(panel).getByRole("button", { name: /^add$/i }));

    await waitFor(() =>
      expect(add).toHaveBeenCalledWith("p1", { day_index: 2, movement: "Row", sets: 3, reps: 10 })
    );
    expect(await within(panel).findByText("Row")).toBeInTheDocument();
  });

  it("renders an exercise with its sets and reps", async () => {
    renderWithProviders(<PlanDetail />);
    expect(await screen.findByText("Squat")).toBeInTheDocument();
    expect(screen.getByText(/3 × 10/)).toBeInTheDocument();
  });

  it("shows the not-found state for a missing plan", async () => {
    vi.mocked(api.getPlan).mockRejectedValue(new Error("404"));
    renderWithProviders(<PlanDetail />);
    expect(await screen.findByText(/no longer exists/i)).toBeInTheDocument();
  });
});

describe("PlanDetail — the analysis seam", () => {
  it("links an analysable exercise into the studio carrying both plan ids", async () => {
    // Both ids: the item PATCH is scoped by plan, so an item id alone cannot be written back.
    renderWithProviders(<PlanDetail />);
    const link = await screen.findByRole("link", { name: /record & analyse/i });
    expect(link).toHaveAttribute("href", "/app?movement=Squat&plan=p1&plan_item=i1");
  });

  it("draws a short action label but keeps the full phrase as its accessible name", async () => {
    // jsdom has no layout engine, so the overflow this guards against cannot be asserted here
    // directly — it was measured in a real browser (the row needed 241px inside a 147px column and
    // the movement name was squeezed to zero). What IS testable is the contract that fixed it: the
    // drawn label is short so the name has room, while the full wording survives for assistive
    // tech. Restoring the long visible label would fail this.
    renderWithProviders(<PlanDetail />);
    const link = await screen.findByRole("link", { name: /record & analyse/i });
    expect(link).toHaveTextContent("Analyse");
    expect(link).not.toHaveTextContent(/record/i);
  });

  it("offers no studio link for a movement with no detector", async () => {
    vi.mocked(api.getPlan).mockResolvedValue(
      plan({ items: [item({ id: "i1", day_index: 1, movement: "Jumping Jacks" })] })
    );
    renderWithProviders(<PlanDetail />);
    await screen.findByText("Jumping Jacks");
    expect(screen.queryByRole("link", { name: /record & analyse/i })).not.toBeInTheDocument();
    // The chip says only the ANALYSIS is missing, not the movement — "Soon" (the movements menu's
    // word for an unavailable movement) would contradict the fact that it is sitting in a plan.
    expect(screen.getByText("No analysis")).toBeInTheDocument();
    expect(screen.queryByText("Soon")).not.toBeInTheDocument();
    // Still tickable by hand — the plan is a schedule first.
    expect(screen.getByRole("button", { name: /mark jumping jacks as done/i })).toBeInTheDocument();
  });

  it("replaces the studio link with a report link once the item carries an analysis", async () => {
    vi.mocked(api.getPlan).mockResolvedValue(
      plan({
        items: [item({ id: "i1", day_index: 1, completed_at: "2026-08-13T01:00:00Z", analysis_id: "an-9" })],
      })
    );
    renderWithProviders(<PlanDetail />);
    const report = await screen.findByRole("link", { name: /view report/i });
    expect(report).toHaveAttribute("href", "/app?analysis=an-9");
    expect(screen.queryByRole("link", { name: /record & analyse/i })).not.toBeInTheDocument();
  });
});

describe("PlanDetail — editing", () => {
  it("ticks an exercise off and keeps the row without refetching the plan", async () => {
    const patch = vi
      .spyOn(api, "updatePlanItem")
      .mockResolvedValue(item({ id: "i1", day_index: 1, completed_at: "2026-08-13T01:00:00Z" }));
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /mark squat as done/i }));

    await waitFor(() => expect(patch).toHaveBeenCalledWith("p1", "i1", { completed: true }));
    expect(await screen.findByRole("button", { name: /mark squat as not done/i })).toBeInTheDocument();
    // One fetch, at mount. A refetch per tick would blank the columns each time.
    expect(api.getPlan).toHaveBeenCalledTimes(1);
  });

  it("unticks a completed exercise", async () => {
    vi.mocked(api.getPlan).mockResolvedValue(
      plan({ items: [item({ id: "i1", day_index: 1, completed_at: "2026-08-13T01:00:00Z" })] })
    );
    const patch = vi.spyOn(api, "updatePlanItem").mockResolvedValue(item({ id: "i1", day_index: 1 }));
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /mark squat as not done/i }));
    await waitFor(() => expect(patch).toHaveBeenCalledWith("p1", "i1", { completed: false }));
  });

  it("reports a failed tick without losing the row", async () => {
    vi.spyOn(api, "updatePlanItem").mockRejectedValue(new Error("offline"));
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /mark squat as done/i }));
    expect(await screen.findByText("offline")).toBeInTheDocument();
    expect(screen.getByText("Squat")).toBeInTheDocument();
  });

  it("adds an exercise to the day the panel is showing", async () => {
    // The form is still per-day and the day is still not a field — but the day it belongs to is
    // now the SELECTED one, so the day is chosen in the strip first. Scoped to the panel rather
    // than to `.closest("section")` on the text "Day 3", which now matches a tile as well.
    const add = vi
      .spyOn(api, "addPlanItem")
      .mockResolvedValue(item({ id: "i2", day_index: 3, movement: "Row" }));
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("tab", { name: /day 3/i }));

    const panel = screen.getByRole("tabpanel");
    await userEvent.click(within(panel).getByRole("button", { name: /add exercise/i }));
    await userEvent.selectOptions(within(panel).getByLabelText(/movement/i), "Row");
    await userEvent.click(within(panel).getByRole("button", { name: /^add$/i }));

    await waitFor(() =>
      expect(add).toHaveBeenCalledWith("p1", { day_index: 3, movement: "Row", sets: 3, reps: 10 })
    );
    expect(await within(panel).findByText("Row")).toBeInTheDocument();
  });

  it("closes the add form when another day is selected", async () => {
    // A form left open across a selection change would let someone fill it in while looking at
    // Wednesday and land the exercise on Monday.
    renderWithProviders(<PlanDetail />);
    const panel = await screen.findByRole("tabpanel");
    await userEvent.click(within(panel).getByRole("button", { name: /add exercise/i }));
    expect(within(panel).getByLabelText(/movement/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /day 6/i }));
    expect(screen.queryByLabelText(/^movement$/i)).toBeNull();
  });

  it("offers every catalog movement in the picker, not just the analysable ones", async () => {
    renderWithProviders(<PlanDetail />);
    const panel = await screen.findByRole("tabpanel");
    await userEvent.click(within(panel).getByRole("button", { name: /add exercise/i }));
    const select = within(panel).getByLabelText(/movement/i);
    expect(within(select).getAllByRole("option")).toHaveLength(16);
    expect(within(select).getByRole("option", { name: "Jumping Jacks" })).toBeInTheDocument();
  });

  it("removes an exercise", async () => {
    const del = vi.spyOn(api, "deletePlanItem").mockResolvedValue({ deleted: 1 });
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /remove squat/i }));
    await waitFor(() => expect(del).toHaveBeenCalledWith("p1", "i1"));
    expect(screen.queryByText("Squat")).not.toBeInTheDocument();
  });
});

describe("PlanDetail — starting a run", () => {
  it("starts an unstarted plan without a confirmation", async () => {
    // Nothing is destroyed on a first start; a dialog would just be a click in the way.
    const start = vi
      .spyOn(api, "startPlan")
      .mockResolvedValue(plan({ started_at: "2026-08-13T09:00:00Z" }));
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /start this plan/i }));
    await waitFor(() => expect(start).toHaveBeenCalledWith("p1"));
  });

  it("confirms before restarting, because a restart clears progress", async () => {
    const start = vi.spyOn(api, "startPlan").mockResolvedValue(plan());
    vi.mocked(api.getPlan).mockResolvedValue(
      plan({
        started_at: "2026-08-10T00:00:00Z",
        items: [item({ id: "i1", day_index: 1, completed_at: "2026-08-11T00:00:00Z" })],
      })
    );
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /start again/i }));

    expect(start).not.toHaveBeenCalled();
    const dialog = await screen.findByRole("dialog");
    // The dialog says what is lost AND what is kept — the analyses survive in 我的紀錄.
    expect(within(dialog).getByText(/analyses themselves stay/i)).toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole("button", { name: /start again/i }));
    await waitFor(() => expect(start).toHaveBeenCalledWith("p1"));
  });

  it("cannot be started while it holds no exercises", async () => {
    vi.mocked(api.getPlan).mockResolvedValue(plan({ items: [] }));
    renderWithProviders(<PlanDetail />);
    expect(await screen.findByRole("button", { name: /start this plan/i })).toBeDisabled();
  });
});

describe("PlanDetail — deleting", () => {
  it("confirms, deletes, and then shows the gone state", async () => {
    const del = vi.spyOn(api, "deletePlan").mockResolvedValue({ deleted: 1 });
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /delete plan/i }));
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(within(dialog).getByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(del).toHaveBeenCalledWith("p1"));
    expect(await screen.findByText(/no longer exists/i)).toBeInTheDocument();
  });
});

describe("PlanDetail — what the plan trains", () => {
  it("names the day's prime movers on the focused day's header line", async () => {
    renderWithProviders(<PlanDetail />);
    // Scoped to the PANEL, where `.closest("section")` on a day label used to do the scoping: the
    // same names now appear in three places — this header, the day's tile in the strip, and the
    // coverage summary below the week — so an unscoped query would match several nodes.
    const panel = await screen.findByRole("tabpanel");
    expect(within(panel).getByText("Glutes")).toBeInTheDocument();
    expect(within(panel).getByText("Quadriceps")).toBeInTheDocument();
  });

  it("names the day's top group on its tile in the strip", async () => {
    renderWithProviders(<PlanDetail />);
    const tab = await screen.findByRole("tab", { name: /day 1/i });
    // One name and a count, not the whole map: a ~150px tile has room for about eleven characters.
    expect(within(tab).getByText("Glutes")).toBeInTheDocument();
    expect(within(tab).queryByText("Quadriceps")).toBeNull();
    expect(within(tab).getByText("+1")).toBeInTheDocument();
    expect(within(tab).getByText("0/1")).toBeInTheDocument();
  });

  it("caps a busy day at three groups and counts the rest", async () => {
    vi.spyOn(api, "getPlan").mockResolvedValue(
      plan({
        items: [
          item({ id: "i1", day_index: 1, movement: "Squat" }),
          item({ id: "i2", day_index: 1, movement: "Push-up" }),
          item({ id: "i3", day_index: 1, movement: "Row" }),
        ],
      })
    );
    renderWithProviders(<PlanDetail />);
    const panel = await screen.findByRole("tabpanel");
    expect(within(panel).getByText("Chest")).toBeInTheDocument();
    expect(within(panel).getByText("+3")).toBeInTheDocument();
    // Ranked fourth, so it belongs to the "+3" and must not be drawn as a fourth chip.
    expect(within(panel).queryByText("Lats")).toBeNull();
  });

  it("says nothing about muscles on a rest day", async () => {
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("tab", { name: /day 2/i }));
    const panel = screen.getByRole("tabpanel");
    expect(within(panel).getByText(/rest day/i)).toBeInTheDocument();
    expect(within(panel).queryByText("Glutes")).toBeNull();
    expect(within(panel).queryByText("Quadriceps")).toBeNull();
    // And its tile says nothing either — a rest day has no prime movers to name.
    const tab = screen.getByRole("tab", { name: /day 2/i });
    expect(within(tab).queryByText("Glutes")).toBeNull();
  });

  it("summarises the whole week below the day bands, gaps included", async () => {
    renderWithProviders(<PlanDetail />);
    const card = (await screen.findByRole("heading", {
      name: /what this plan trains/i,
    })).closest("section") as HTMLElement;
    expect(within(card).getByText("Quadriceps")).toBeInTheDocument();
    // A week of nothing but squats misses most of the upper body, and the card says which parts.
    expect(within(card).getByText(/not trained this week/i)).toHaveTextContent(/chest/i);
  });
});
