import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Plan, type PlanItem, type PlanSummary, type PlanTemplate } from "../api";

const navigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => navigate };
});

import Plans from "../pages/Plans";

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
    created_at: "2026-09-01T00:00:00Z",
    ...overrides,
  };
}

const STARTED: PlanSummary = {
  id: "p1",
  name: "Upper week",
  notes: null,
  template_key: null,
  started_at: "2026-09-01T00:00:00Z",
  created_at: "2026-09-01T00:00:00Z",
  updated_at: "2026-09-01T00:00:00Z",
  item_count: 4,
  completed_count: 1,
  day_count: 2,
  movements: ["Squat", "Row"],
};

const FULL: Plan = {
  id: "p1",
  name: "Upper week",
  notes: null,
  template_key: null,
  started_at: "2026-09-01T00:00:00Z",
  created_at: "2026-09-01T00:00:00Z",
  updated_at: "2026-09-01T00:00:00Z",
  items: [
    // Day 1 is finished, so the day the user is ON is day 2 — and only its unfinished exercises
    // belong on a card whose whole job is "what is left".
    item({ id: "done", day_index: 1, completed_at: "2026-09-01T10:00:00Z" }),
    item({ id: "i2", day_index: 2, position: 0, movement: "Row" }),
    item({ id: "i3", day_index: 2, position: 1, movement: "Push-up" }),
    item({ id: "i4", day_index: 2, position: 2, completed_at: "2026-09-02T10:00:00Z" }),
  ],
};

const template: PlanTemplate = {
  key: "quick_core",
  name: "Quick core session",
  description: "One 15-minute session.",
  items: [{ day_index: 1, movement: "Sit-up", sets: 3, reps: 15 }],
};

beforeEach(() => {
  navigate.mockReset();
  vi.spyOn(api, "listPlans").mockResolvedValue([STARTED]);
  vi.spyOn(api, "planTemplates").mockResolvedValue([template]);
  vi.spyOn(api, "getPlan").mockResolvedValue(FULL);
});
afterEach(() => vi.restoreAllMocks());

/** The "continue" card, addressed by its own heading rather than by the plan name — the plan is
 *  also in the grid below, under the same name. */
async function continueCard(): Promise<HTMLElement> {
  const heading = await screen.findByText(/pick up where you left off/i);
  return heading.closest("section") as HTMLElement;
}

describe("Plans — continuing a started plan", () => {
  it("leads with the run already under way, and says how far in it is", async () => {
    renderWithProviders(<Plans />);
    const card = await continueCard();
    expect(within(card).getByText("Upper week")).toBeInTheDocument();
    // "Day N" needs the plan's items, which arrive after the list row the rest of the card is
    // drawn from — so this is the one line on the card that has to be awaited.
    expect(await within(card).findByText(/day 2 of 2/i)).toBeInTheDocument();
    expect(within(card).getByText(/1 of 4 done/i)).toBeInTheDocument();
  });

  it("offers today's unfinished exercises as links into the studio", async () => {
    renderWithProviders(<Plans />);
    const card = await continueCard();
    // Both plan ids ride along: the item PATCH is scoped by plan, so an item id alone could not be
    // written back when the analysis lands.
    expect(await within(card).findByRole("link", { name: /Row/ })).toHaveAttribute(
      "href",
      "/app?movement=Row&plan=p1&plan_item=i2"
    );
    expect(within(card).getByRole("link", { name: /Push-up/ })).toBeInTheDocument();
    // The exercise already ticked off today is not re-offered.
    expect(within(card).queryByRole("link", { name: /Squat/ })).not.toBeInTheDocument();
  });

  it("still shows the card when the plan's own fetch fails, minus the exercises", async () => {
    // The headline facts come from the list row, so a failed detail fetch costs the chips and
    // nothing else — never the card, and never the page.
    vi.mocked(api.getPlan).mockRejectedValue(new Error("offline"));
    renderWithProviders(<Plans />);
    const card = await continueCard();
    expect(within(card).getByText("Upper week")).toBeInTheDocument();
    await waitFor(() =>
      expect(within(card).queryByText(/today's exercises/i)).not.toBeInTheDocument()
    );
  });

  it("shows no continue card once every exercise is done", async () => {
    vi.mocked(api.listPlans).mockResolvedValue([{ ...STARTED, completed_count: 4 }]);
    renderWithProviders(<Plans />);
    await screen.findByText("Upper week");
    expect(screen.queryByText(/pick up where you left off/i)).not.toBeInTheDocument();
    expect(api.getPlan).not.toHaveBeenCalled();
  });

  it("shows no continue card for a plan that was never started", async () => {
    vi.mocked(api.listPlans).mockResolvedValue([
      { ...STARTED, started_at: null, completed_count: 0 },
    ]);
    renderWithProviders(<Plans />);
    await screen.findByText("Upper week");
    expect(screen.queryByText(/pick up where you left off/i)).not.toBeInTheDocument();
  });
});

describe("Plans — handing a template to Lumen", () => {
  it("creates the plan first, then opens it with the coach up", async () => {
    // Two steps on purpose: the plan is the user's before any conversation happens, so an
    // abandoned chat still leaves them something usable.
    const create = vi.spyOn(api, "createPlan").mockResolvedValue({
      id: "new-plan",
      name: "Quick core session",
      notes: null,
      template_key: "quick_core",
      started_at: null,
      created_at: "",
      updated_at: "",
      items: [],
    });
    renderWithProviders(<Plans />);
    await userEvent.click(await screen.findByRole("button", { name: /customise with lumen/i }));

    await waitFor(() =>
      expect(create).toHaveBeenCalledWith({
        name: "Quick core session",
        template_key: "quick_core",
      })
    );
    expect(navigate).toHaveBeenCalledWith("/plans/new-plan?coach=1");
  });

  it("reports a failed create instead of navigating", async () => {
    vi.spyOn(api, "createPlan").mockRejectedValue(new Error("Unknown template 'nope'."));
    renderWithProviders(<Plans />);
    await userEvent.click(await screen.findByRole("button", { name: /customise with lumen/i }));

    expect(await screen.findByText("Unknown template 'nope'.")).toBeInTheDocument();
    expect(navigate).not.toHaveBeenCalled();
  });

  it("keeps the plain copy-the-template route beside it", async () => {
    renderWithProviders(<Plans />);
    await userEvent.click(await screen.findByRole("button", { name: /use this/i }));
    expect((screen.getByLabelText(/plan name/i) as HTMLInputElement).value).toBe(
      "Quick core session"
    );
  });
});
