import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Plan, type PlanSummary, type PlanTemplate } from "../api";

const navigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => navigate };
});

import Plans from "../pages/Plans";

function plan(overrides: Partial<PlanSummary> = {}): PlanSummary {
  return {
    id: "p1",
    name: "Upper week",
    notes: null,
    template_key: null,
    started_at: null,
    created_at: "2026-08-13T00:00:00Z",
    updated_at: "2026-08-13T00:00:00Z",
    item_count: 3,
    completed_count: 0,
    day_count: 2,
    movements: ["Squat", "Row"],
    ...overrides,
  };
}

const template: PlanTemplate = {
  key: "quick_core",
  name: "Quick core session",
  description: "One 15-minute session.",
  items: [
    { day_index: 1, movement: "Sit-up", sets: 3, reps: 15 },
    { day_index: 1, movement: "Torso Twist", sets: 3, reps: 20 },
  ],
};

// The "continue" card fetches the one started plan to learn today's exercises, which the list rows
// do not carry. Stubbed for every test so a started fixture does not reach the network.
const full: Plan = {
  id: "p1",
  name: "Upper week",
  notes: null,
  template_key: null,
  started_at: "2026-08-10T00:00:00Z",
  created_at: "2026-08-13T00:00:00Z",
  updated_at: "2026-08-13T00:00:00Z",
  items: [],
};

beforeEach(() => {
  navigate.mockReset();
  vi.spyOn(api, "listPlans").mockResolvedValue([]);
  vi.spyOn(api, "planTemplates").mockResolvedValue([template]);
  vi.spyOn(api, "getPlan").mockResolvedValue(full);
});
afterEach(() => vi.restoreAllMocks());

describe("Plans — listing", () => {
  it("renders the user's plans with their progress", async () => {
    vi.mocked(api.listPlans).mockResolvedValue([
      plan({ started_at: "2026-08-10T00:00:00Z", completed_count: 1 }),
    ]);
    renderWithProviders(<Plans />);
    // A started, unfinished plan now appears TWICE: once in the "continue" card that leads the
    // page and once in the grid below it. Both state the same progress, on purpose — the card is
    // the shortcut into today, the grid row is the plan among the others.
    expect(await screen.findAllByText("Upper week")).toHaveLength(2);
    expect(screen.getAllByText(/1 of 3 done/i)).toHaveLength(2);
  });

  it("shows a plan that was never started as 'not started', not as 0% done", async () => {
    // Those are different things, and only one of them is the user's fault.
    vi.mocked(api.listPlans).mockResolvedValue([plan()]);
    renderWithProviders(<Plans />);
    expect(await screen.findByText(/not started/i)).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("shows the empty state when the user has no plans", async () => {
    renderWithProviders(<Plans />);
    expect(await screen.findByText(/no plans yet/i)).toBeInTheDocument();
  });

  it("surfaces a load failure with a retry that refetches", async () => {
    vi.mocked(api.listPlans)
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce([plan()]);
    renderWithProviders(<Plans />);
    await screen.findByText(/could not load your plans/i);
    await userEvent.click(screen.getByRole("button", { name: /try again/i }));
    expect(await screen.findByText("Upper week")).toBeInTheDocument();
  });
});

describe("Plans — templates", () => {
  it("lists the built-in templates", async () => {
    renderWithProviders(<Plans />);
    expect(await screen.findByText("Quick core session")).toBeInTheDocument();
  });

  it("still renders the user's own plans when the template fetch fails", async () => {
    // The templates are a suggestion, not the page.
    vi.mocked(api.listPlans).mockResolvedValue([plan()]);
    vi.mocked(api.planTemplates).mockRejectedValue(new Error("boom"));
    renderWithProviders(<Plans />);
    expect(await screen.findByText("Upper week")).toBeInTheDocument();
    expect(screen.queryByText(/could not load your plans/i)).not.toBeInTheDocument();
  });

  it("opens the create dialog with the template's name prefilled", async () => {
    renderWithProviders(<Plans />);
    await userEvent.click(await screen.findByRole("button", { name: /use this/i }));
    const field = screen.getByLabelText(/plan name/i) as HTMLInputElement;
    // Prefilled so the common case is one more click, not a naming decision.
    expect(field.value).toBe("Quick core session");
  });

  it("gives every template its own card, each carrying both actions", async () => {
    // Mocked locally rather than in `beforeEach`: the other tests address "Use this" and
    // "Customise with Lumen" in the singular, and a second template would make those ambiguous.
    vi.mocked(api.planTemplates).mockResolvedValue([
      template,
      {
        key: "full_body_starter",
        name: "Full-body starter",
        description: "Three sessions a week covering the whole body.",
        items: [
          { day_index: 1, movement: "Squat", sets: 3, reps: 10 },
          { day_index: 1, movement: "Push-up", sets: 3, reps: 8 },
          { day_index: 3, movement: "Row", sets: 3, reps: 10 },
        ],
      },
    ]);
    renderWithProviders(<Plans />);

    for (const [name, meta] of [
      // A one-day template takes the singular key, so this reads "in one day" rather than the
      // "over 1 days" the plural string produced.
      ["Quick core session", /2 exercises in one day/i],
      ["Full-body starter", /3 exercises over 2 days/i],
    ] as const) {
      const card = (await screen.findByText(name)).closest("li") as HTMLElement;
      // Each card stands on its own: what it is, how big it is, and both ways in.
      expect(within(card).getByText(meta)).toBeInTheDocument();
      expect(within(card).getByRole("button", { name: /use this/i })).toBeInTheDocument();
      expect(within(card).getByRole("button", { name: /customise with lumen/i })).toBeInTheDocument();
      // The actions are real buttons in the card, not a link wrapping the card — a card with two
      // destinations cannot be one click target, and nothing here may nest interactives.
      expect(card.querySelector("a")).toBeNull();
    }
  });

  it("names the movements a template trains, capped with a +n", async () => {
    vi.mocked(api.planTemplates).mockResolvedValue([
      {
        key: "full_body_starter",
        name: "Full-body starter",
        description: "Everything.",
        items: ["Squat", "Push-up", "Row", "Lunge", "Deadlift", "Sit-up"].map((movement, i) => ({
          day_index: 1,
          movement,
          sets: 3,
          reps: 10 + i,
        })),
      },
    ]);
    renderWithProviders(<Plans />);
    const card = (await screen.findByText("Full-body starter")).closest("li") as HTMLElement;
    // Scoped to the PILL row, not the card: the muscle line below carries its own "+n" overflow,
    // so a bare getByText("+2") would be asserting "somewhere on this card" and would go
    // ambiguous the moment either limit changes.
    const pills = card.querySelector("ul") as HTMLElement;
    expect(within(pills).getByText("Squat")).toBeInTheDocument();
    // Four pills then a count, the same ending PlanCard uses — nine distinct movements would
    // otherwise be the whole card.
    expect(within(pills).getAllByRole("listitem")).toHaveLength(5);
    expect(within(pills).getByText("+2")).toBeInTheDocument();
  });

  it("disables only the Lumen action while that template is being created", async () => {
    // Held open deliberately: the busy state exists to stop a double-fire, so it has to be
    // observable while the request is still in flight.
    let release!: (plan: Plan) => void;
    vi.spyOn(api, "createPlan").mockReturnValue(
      new Promise<Plan>((resolve) => {
        release = resolve;
      })
    );
    renderWithProviders(<Plans />);
    const card = (await screen.findByText("Quick core session")).closest("li") as HTMLElement;
    await userEvent.click(within(card).getByRole("button", { name: /customise with lumen/i }));

    const busy = await within(card).findByRole("button", { name: /creating/i });
    expect(busy).toBeDisabled();
    // The plain copy route stays open: it does not go through the same request.
    expect(within(card).getByRole("button", { name: /use this/i })).toBeEnabled();

    release({
      id: "new-plan",
      name: "Quick core session",
      notes: null,
      template_key: "quick_core",
      started_at: null,
      created_at: "",
      updated_at: "",
      items: [],
    });
    await waitFor(() => expect(navigate).toHaveBeenCalledWith("/plans/new-plan?coach=1"));
  });

  it("creates from the template and opens the new plan", async () => {
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
    await userEvent.click(await screen.findByRole("button", { name: /use this/i }));
    await userEvent.click(screen.getByRole("button", { name: /^create$/i }));

    await waitFor(() =>
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({ name: "Quick core session", template_key: "quick_core" })
      )
    );
    // Straight into the plan: a blank plan needs exercises and a template plan needs starting —
    // either way the next action is inside the plan, never in the list.
    expect(navigate).toHaveBeenCalledWith("/plans/new-plan");
  });
});

describe("Plans — blank create", () => {
  it("sends no template_key when started from the Build it myself button", async () => {
    const create = vi.spyOn(api, "createPlan").mockResolvedValue({
      id: "blank",
      name: "Leg day",
      notes: null,
      template_key: null,
      started_at: null,
      created_at: "",
      updated_at: "",
      items: [],
    });
    renderWithProviders(<Plans />);
    await userEvent.click(await screen.findByRole("button", { name: /build it myself/i }));
    await userEvent.type(screen.getByLabelText(/plan name/i), "Leg day");
    await userEvent.click(screen.getByRole("button", { name: /^create$/i }));

    await waitFor(() => expect(create).toHaveBeenCalled());
    expect(create.mock.calls[0][0]).not.toHaveProperty("template_key");
  });

  it("refuses to submit an empty name", async () => {
    const create = vi.spyOn(api, "createPlan");
    renderWithProviders(<Plans />);
    await userEvent.click(await screen.findByRole("button", { name: /build it myself/i }));
    expect(screen.getByRole("button", { name: /^create$/i })).toBeDisabled();
    expect(create).not.toHaveBeenCalled();
  });

  it("shows the server's message when the create fails, and stays open", async () => {
    vi.spyOn(api, "createPlan").mockRejectedValue(new Error("Unknown template 'nope'."));
    renderWithProviders(<Plans />);
    await userEvent.click(await screen.findByRole("button", { name: /build it myself/i }));
    await userEvent.type(screen.getByLabelText(/plan name/i), "Leg day");
    await userEvent.click(screen.getByRole("button", { name: /^create$/i }));

    expect(await screen.findByText("Unknown template 'nope'.")).toBeInTheDocument();
    // Still open, so the user can fix the input rather than starting over.
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(navigate).not.toHaveBeenCalled();
  });

  it("clears the previous name when the dialog is reopened", async () => {
    // The dialog stays mounted between uses; without a reset the second plan starts life named
    // after the first.
    renderWithProviders(<Plans />);
    await userEvent.click(await screen.findByRole("button", { name: /build it myself/i }));
    await userEvent.type(screen.getByLabelText(/plan name/i), "Leg day");
    await userEvent.click(screen.getByRole("button", { name: /cancel/i }));
    await userEvent.click(screen.getByRole("button", { name: /build it myself/i }));
    expect((screen.getByLabelText(/plan name/i) as HTMLInputElement).value).toBe("");
  });
});
