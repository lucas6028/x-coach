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
    renderWithProviders(<PlanDetail />);
    expect(await screen.findByText("Day 1")).toBeInTheDocument();
    expect(screen.getByText("Day 7")).toBeInTheDocument();
    expect(screen.getAllByText(/rest day/i).length).toBe(6);
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

  it("adds an exercise to the day whose form was opened", async () => {
    const add = vi
      .spyOn(api, "addPlanItem")
      .mockResolvedValue(item({ id: "i2", day_index: 3, movement: "Row" }));
    renderWithProviders(<PlanDetail />);
    // Day 3's own "Add exercise" button — the form is per-day, so the day is not a field.
    const dayThree = (await screen.findByText("Day 3")).closest("section")!;
    await userEvent.click(within(dayThree).getByRole("button", { name: /add exercise/i }));
    await userEvent.selectOptions(within(dayThree).getByLabelText(/movement/i), "Row");
    await userEvent.click(within(dayThree).getByRole("button", { name: /^add$/i }));

    await waitFor(() =>
      expect(add).toHaveBeenCalledWith("p1", { day_index: 3, movement: "Row", sets: 3, reps: 10 })
    );
    expect(await within(dayThree).findByText("Row")).toBeInTheDocument();
  });

  it("offers every catalog movement in the picker, not just the analysable ones", async () => {
    renderWithProviders(<PlanDetail />);
    const dayOne = (await screen.findByText("Day 1")).closest("section")!;
    await userEvent.click(within(dayOne).getByRole("button", { name: /add exercise/i }));
    const select = within(dayOne).getByLabelText(/movement/i);
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
