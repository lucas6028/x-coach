import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { I18nProvider } from "../lib/i18n";
import PlanPreview from "../components/plans/PlanPreview";
import type { Plan, PlanItem } from "../api";

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

function plan(items: PlanItem[]): Plan {
  return {
    id: "p1",
    name: "zzq-plan",
    notes: null,
    template_key: null,
    started_at: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    items,
  };
}

function renderPreview(items: PlanItem[]) {
  return render(
    <I18nProvider>
      <PlanPreview plan={plan(items)} />
    </I18nProvider>
  );
}

describe("PlanPreview rest days", () => {
  it("folds the trailing empty days of a three-day plan into one line", () => {
    // Four of seven bands saying nothing reads as a plan that failed to finish, not as a week
    // with rest in it. The days that DO have work still get their own band.
    renderPreview([item(1, "Squat"), item(2, "Push-up"), item(3, "Plank")]);

    expect(screen.getByText(/Day 4–7 · Rest/)).toBeInTheDocument();
    expect(screen.queryByText("Day 4")).toBeNull();
    expect(screen.queryByText("Day 7")).toBeNull();
    expect(screen.getByText("Day 1")).toBeInTheDocument();
    expect(screen.getByText("Day 3")).toBeInTheDocument();
  });

  it("leaves a lone empty day between two working days as its own band", () => {
    // "Day 2–2 · Rest" would be worse than the band that already says "Rest day", so a run of one
    // is not a run.
    renderPreview([item(1, "Squat"), item(3, "Plank")]);

    expect(screen.getByText("Day 2")).toBeInTheDocument();
    expect(screen.getByText("Rest day")).toBeInTheDocument();
    expect(screen.queryByText(/Day 2–/)).toBeNull();
    // The trailing run 4–7 still collapses, so exactly one range line exists.
    expect(screen.getByText(/Day 4–7 · Rest/)).toBeInTheDocument();
  });
});
