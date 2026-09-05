import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { I18nProvider } from "../lib/i18n";
import PlanMuscleCoverage from "../components/plans/PlanMuscleCoverage";
import type { PlanItem } from "../api";

// Every one of the sixteen catalog movements ships a plate today, so the "no artwork" branch is
// unreachable through real data -- and a card that renders an empty framed box for a movement
// added to the catalog before its illustration arrives is exactly the regression worth pinning.
// Stripping `plate` and nothing else keeps the muscle lists intact, so the card still has groups
// to name and the only thing missing is the picture.
vi.mock("../lib/movementDetail", async () => {
  const actual = await vi.importActual<typeof import("../lib/movementDetail")>(
    "../lib/movementDetail"
  );
  return {
    ...actual,
    movementDetail: (name: string) => {
      const detail = actual.movementDetail(name);
      if (!detail) return undefined;
      const { plate: _plate, ...withoutPlate } = detail;
      return withoutPlate;
    },
  };
});

function item(day: number, movement: string): PlanItem {
  return {
    id: `${movement}-${day}`,
    plan_id: "p1",
    day_index: day,
    position: 0,
    movement,
    sets: 3,
    reps: 10,
    notes: null,
    completed_at: null,
    analysis_id: null,
    created_at: "2026-01-01T00:00:00Z",
  };
}

describe("PlanMuscleCoverage without artwork", () => {
  it("keeps the named groups but draws no image box when no movement has a plate", () => {
    const { container } = render(
      <I18nProvider>
        <PlanMuscleCoverage items={[item(1, "Squat"), item(2, "Push-up")]} />
      </I18nProvider>
    );

    expect(screen.getByRole("heading", { name: /what this plan trains/i })).toBeInTheDocument();
    expect(screen.getByText("Quadriceps")).toBeInTheDocument();
    expect(screen.getByText("Chest")).toBeInTheDocument();
    expect(container.querySelectorAll("img").length).toBe(0);
  });
});
