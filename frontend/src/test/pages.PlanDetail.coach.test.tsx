import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Plan, type PlanItem } from "../api";

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useParams: () => ({ planId: "p1" }) };
});

// PlanCoach is stubbed on purpose. Its own conversation — streaming, persistence, the three
// composer states — is covered in components.PlanCoach.test.tsx against a hoisted API mock. What
// is new HERE is the wiring: whether the panel appears, what opens it, and what the page does with
// a plan the coach hands back. Rendering the real component would test the former again through
// the real AuthProvider, which has no user, so the composer would be the sign-in invite and
// nothing could be sent anyway.
vi.mock("../components/plans/PlanCoach", () => ({
  default: ({ planId, onPlan }: { planId: string | null; onPlan?: (p: Plan) => void }) => (
    <div>
      <p>coach bound to {planId}</p>
      <button type="button" onClick={() => onPlan?.(rewritten)}>
        rewrite the plan
      </button>
    </div>
  ),
}));

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
    created_at: "2026-09-01T00:00:00Z",
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
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    items: [item({ id: "i1", day_index: 1 })],
    ...overrides,
  };
}

// What a coach tool returns: the whole fresh plan, not a patch.
const rewritten: Plan = plan({
  name: "Upper week",
  items: [item({ id: "i2", day_index: 2, movement: "Row" })],
});

beforeEach(() => {
  vi.spyOn(api, "getMovements").mockResolvedValue([
    { name: "Squat", validated: true },
    { name: "Row", validated: false },
  ]);
  vi.spyOn(api, "getPlan").mockResolvedValue(plan());
});
afterEach(() => vi.restoreAllMocks());

describe("PlanDetail — Lumen's panel", () => {
  it("stays closed until asked for", async () => {
    renderWithProviders(<PlanDetail />);
    await screen.findByText("Squat");
    expect(screen.queryByText(/coach bound to/)).not.toBeInTheDocument();
  });

  it("opens from the header button, bound to this plan", async () => {
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /ask lumen to adjust/i }));

    expect(screen.getByText("coach bound to p1")).toBeInTheDocument();
    // The same control closes it again, and says so.
    expect(screen.getByRole("button", { name: /close lumen/i })).toBeInTheDocument();
  });

  it("closes again from the same button", async () => {
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /ask lumen to adjust/i }));
    await userEvent.click(screen.getByRole("button", { name: /close lumen/i }));
    expect(screen.queryByText(/coach bound to/)).not.toBeInTheDocument();
  });

  it("opens already up when the URL asks for it", async () => {
    // How "請 Lumen 客製" lands here from the plans list: create, then arrive mid-conversation.
    renderWithProviders(<PlanDetail />, { route: "/plans/p1?coach=1" });
    expect(await screen.findByText("coach bound to p1")).toBeInTheDocument();
  });

  it("stays closed once dismissed, even though ?coach=1 is still in the URL", async () => {
    // The param is read once, at mount. Kept as an effect it would re-open the panel every time
    // the user closed it.
    renderWithProviders(<PlanDetail />, { route: "/plans/p1?coach=1" });
    await userEvent.click(await screen.findByRole("button", { name: /close lumen/i }));
    expect(screen.queryByText(/coach bound to/)).not.toBeInTheDocument();
  });

  it("reaches the same coach from a floating button and a sheet on a phone", async () => {
    // `useIsMobile` renders a SEPARATE tree rather than hiding one with CSS, so none of the phone
    // affordances execute at all under the default jsdom viewport.
    const real = window.matchMedia;
    window.matchMedia = vi.fn().mockReturnValue({
      matches: true,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }) as unknown as typeof window.matchMedia;
    try {
      renderWithProviders(<PlanDetail />);
      // No header toggle on a phone — a fifth control in that row would wrap onto its own line.
      await screen.findByText("Squat");
      expect(screen.queryByRole("button", { name: /close lumen/i })).not.toBeInTheDocument();

      await userEvent.click(screen.getByRole("button", { name: /ask lumen to adjust/i }));
      const sheet = await screen.findByRole("dialog");
      expect(within(sheet).getByText("coach bound to p1")).toBeInTheDocument();

      await userEvent.keyboard("{Escape}");
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    } finally {
      window.matchMedia = real;
    }
  });

  it("replaces the plan in place when a coach tool rewrites it", async () => {
    // No refetch: the tool already handed back the whole fresh plan, and refetching would blank
    // the day bands to learn what we were just told.
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /ask lumen to adjust/i }));
    await userEvent.click(screen.getByRole("button", { name: /rewrite the plan/i }));

    expect(await screen.findByText("Row")).toBeInTheDocument();
    expect(screen.queryByText("Squat")).not.toBeInTheDocument();
    expect(api.getPlan).toHaveBeenCalledTimes(1);
  });
});
