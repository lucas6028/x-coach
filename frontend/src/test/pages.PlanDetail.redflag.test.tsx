import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Checkin, type Plan, type PlanItem } from "../api";

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

function checkin(overrides: Partial<Checkin> = {}): Checkin {
  return {
    id: "c1",
    user_id: "u1",
    plan_id: "p1",
    plan_item_id: "i1",
    analysis_id: null,
    pain_nrs: 8,
    rpe: null,
    note: null,
    form_score: null,
    flagged: true,
    flag_reasons: ["pain_high"],
    acknowledged_at: null,
    created_at: "2026-09-15T00:00:00Z",
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

describe("PlanDetail — red-flag banner", () => {
  it("shows the banner and its reasons when the latest check-in is flagged", async () => {
    vi.spyOn(api, "listCheckins").mockResolvedValue([checkin()]);
    renderWithProviders(<PlanDetail />);
    expect(await screen.findByText(/your last check-in needs attention/i)).toBeInTheDocument();
    expect(screen.getByText("Pain score is high")).toBeInTheDocument();
    expect(api.listCheckins).toHaveBeenCalledWith("p1", 1);
  });

  it("shows no banner when the latest check-in is not flagged", async () => {
    vi.spyOn(api, "listCheckins").mockResolvedValue([checkin({ flagged: false, flag_reasons: [] })]);
    renderWithProviders(<PlanDetail />);
    await screen.findByRole("tabpanel");
    expect(screen.queryByText(/needs attention/i)).not.toBeInTheDocument();
  });

  it("shows no banner when there are no check-ins yet", async () => {
    vi.spyOn(api, "listCheckins").mockResolvedValue([]);
    renderWithProviders(<PlanDetail />);
    await screen.findByRole("tabpanel");
    expect(screen.queryByText(/needs attention/i)).not.toBeInTheDocument();
  });

  it("opens the check-in dialog from the item's report button", async () => {
    vi.spyOn(api, "listCheckins").mockResolvedValue([]);
    renderWithProviders(<PlanDetail />);
    await userEvent.click(await screen.findByRole("button", { name: /report today's status/i }));
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
  });
});
