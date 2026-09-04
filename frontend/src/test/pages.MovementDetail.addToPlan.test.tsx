import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { api, type PlanItem, type PlanSummary } from "../api";

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
import MovementDetail from "../pages/MovementDetail";

const mockUseAuth = vi.mocked(useAuth);

function signedIn(yes: boolean) {
  mockUseAuth.mockReturnValue({ user: yes ? { id: "u1" } : null } as unknown as ReturnType<
    typeof useAuth
  >);
}

const PLANS: PlanSummary[] = [
  {
    id: "p1",
    name: "Upper week",
    notes: null,
    template_key: null,
    started_at: null,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    item_count: 2,
    completed_count: 0,
    day_count: 1,
    movements: ["Row"],
  },
  {
    id: "p2",
    name: "Leg day",
    notes: null,
    template_key: null,
    started_at: null,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    item_count: 1,
    completed_count: 0,
    day_count: 1,
    movements: ["Squat"],
  },
];

const CREATED: PlanItem = {
  id: "new-item",
  plan_id: "p2",
  day_index: 2,
  position: 0,
  movement: "Squat",
  sets: 3,
  reps: 10,
  notes: null,
  completed_at: null,
  analysis_id: null,
  created_at: "2026-09-03T00:00:00Z",
};

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

/** Open the menu and hand back its `role="menu"` container. */
async function openMenu(): Promise<HTMLElement> {
  await userEvent.click(await screen.findByRole("button", { name: /add to a plan/i }));
  return await screen.findByRole("menu");
}

beforeEach(() => {
  signedIn(true);
  vi.spyOn(api, "getMovements").mockResolvedValue([{ name: "Squat", validated: true }]);
  vi.spyOn(api, "movementFaults").mockResolvedValue({ movement: "Squat", faults: [] });
  vi.spyOn(api, "listPlans").mockResolvedValue(PLANS);
});
afterEach(() => vi.restoreAllMocks());

describe("MovementDetail — add to a plan", () => {
  it("offers nothing to a signed-out visitor", async () => {
    // This is a public page, and a control that can only ever answer "sign in first" is a worse
    // invitation than no control.
    signedIn(false);
    renderAt("/movements/Squat");
    await screen.findByRole("heading", { name: "Squat", level: 1 });
    expect(screen.queryByRole("button", { name: /add to a plan/i })).not.toBeInTheDocument();
    expect(api.listPlans).not.toHaveBeenCalled();
  });

  it("fetches the plans only once the menu is opened", async () => {
    renderAt("/movements/Squat");
    await screen.findByRole("button", { name: /add to a plan/i });
    // An eager listPlans on a public page is a request nobody asked for.
    expect(api.listPlans).not.toHaveBeenCalled();

    await openMenu();
    await waitFor(() => expect(api.listPlans).toHaveBeenCalled());
    expect(await screen.findByText("Upper week")).toBeInTheDocument();
    expect(screen.getByText("Leg day")).toBeInTheDocument();
  });

  it("adds the movement to the chosen plan and day", async () => {
    const add = vi.spyOn(api, "addPlanItem").mockResolvedValue(CREATED);
    renderAt("/movements/Squat");
    const menu = await openMenu();

    await userEvent.click(await within(menu).findByRole("menuitemradio", { name: /leg day/i }));
    await userEvent.click(within(menu).getByRole("button", { name: "2" }));
    await userEvent.click(within(menu).getByRole("button", { name: /^add$/i }));

    await waitFor(() =>
      expect(add).toHaveBeenCalledWith("p2", {
        day_index: 2,
        movement: "Squat",
        sets: 3,
        reps: 10,
      })
    );
  });

  it("confirms where the movement landed, with a way into that plan", async () => {
    vi.spyOn(api, "addPlanItem").mockResolvedValue(CREATED);
    renderAt("/movements/Squat");
    const menu = await openMenu();
    await userEvent.click(await within(menu).findByRole("menuitemradio", { name: /leg day/i }));
    await userEvent.click(within(menu).getByRole("button", { name: "2" }));
    await userEvent.click(within(menu).getByRole("button", { name: /^add$/i }));

    expect(await screen.findByText(/added to.*leg day.*day 2/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /^open$/i })).toHaveAttribute("href", "/plans/p2");
    // The menu is done with — the confirmation replaces it.
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("defaults to the first plan and day 1, so one click is enough", async () => {
    const add = vi.spyOn(api, "addPlanItem").mockResolvedValue(CREATED);
    renderAt("/movements/Squat");
    const menu = await openMenu();
    await userEvent.click(await within(menu).findByRole("button", { name: /^add$/i }));

    await waitFor(() =>
      expect(add).toHaveBeenCalledWith("p1", expect.objectContaining({ day_index: 1 }))
    );
  });

  it("reports a failed add inside the menu, without closing it", async () => {
    vi.spyOn(api, "addPlanItem").mockRejectedValue(new Error("That plan is full."));
    renderAt("/movements/Squat");
    const menu = await openMenu();
    await userEvent.click(await within(menu).findByRole("button", { name: /^add$/i }));

    expect(await within(menu).findByText("That plan is full.")).toBeInTheDocument();
    expect(screen.getByRole("menu")).toBeInTheDocument();
  });

  it("points someone with no plans at the builder", async () => {
    vi.mocked(api.listPlans).mockResolvedValue([]);
    renderAt("/movements/Squat");
    const menu = await openMenu();

    expect(await within(menu).findByText(/you have no plans yet/i)).toBeInTheDocument();
    expect(within(menu).getByRole("link", { name: /create one with lumen/i })).toHaveAttribute(
      "href",
      "/plans/new"
    );
  });

  it("closes on Escape", async () => {
    renderAt("/movements/Squat");
    await openMenu();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });
});
