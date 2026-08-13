import { describe, it, expect } from "vitest";
import {
  PLAN_DAYS,
  currentDay,
  isAnalyzable,
  itemsByDay,
  progressRatio,
  templateText,
  usedDays,
} from "../lib/plans";
import type { PlanItem } from "../api";
import type { TFunc } from "../lib/i18n";

function item(overrides: Partial<PlanItem> & { day_index: number }): PlanItem {
  return {
    id: `item-${overrides.day_index}-${overrides.movement ?? "Squat"}`,
    plan_id: "plan-1",
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

describe("itemsByDay", () => {
  it("returns one bucket per day slot, empty ones included", () => {
    // A rest day is part of a plan. Dropping empty days would make "Day 3" mean a different
    // position in different plans.
    const days = itemsByDay([item({ day_index: 2 })]);
    expect(days).toHaveLength(PLAN_DAYS.length);
    expect(days[0]).toEqual([]);
    expect(days[1]).toHaveLength(1);
  });

  it("drops an item outside the seven-day week rather than crashing", () => {
    // The API and the DB both reject one; a stale client reading a future schema should still
    // render the page.
    expect(itemsByDay([item({ day_index: 9 }), item({ day_index: 0 })]).flat()).toEqual([]);
  });
});

describe("usedDays", () => {
  it("lists only the days that hold exercises, ascending", () => {
    expect(usedDays([item({ day_index: 4 }), item({ day_index: 1 })])).toEqual([1, 4]);
  });

  it("is empty for a plan with no exercises", () => {
    expect(usedDays([])).toEqual([]);
  });
});

describe("currentDay", () => {
  it("is the first used day that still has something unticked", () => {
    const items = [
      item({ day_index: 1, completed_at: "2026-08-13T00:00:00Z" }),
      item({ day_index: 3, movement: "Row" }),
    ];
    expect(currentDay(items)).toBe(3);
  });

  it("skips over days with no exercises", () => {
    // Day 2 is a rest day: "the day you are on" must be a day with work on it.
    expect(currentDay([item({ day_index: 1, completed_at: "x" }), item({ day_index: 5 })])).toBe(5);
  });

  it("is null once every exercise is done", () => {
    expect(currentDay([item({ day_index: 1, completed_at: "x" })])).toBeNull();
  });

  it("is null for an empty plan", () => {
    expect(currentDay([])).toBeNull();
  });

  it("moves BACK when a completed item is unticked", () => {
    // The whole reason this is derived rather than stored: a stored pointer would have to be
    // rewound by whoever unticked, and would otherwise disagree with the ticks.
    const done = [
      item({ day_index: 1, completed_at: "x" }),
      item({ day_index: 2, completed_at: "x" }),
    ];
    expect(currentDay(done)).toBeNull();
    const undone = [done[0], { ...done[1], completed_at: null }];
    expect(currentDay(undone)).toBe(2);
  });
});

describe("progressRatio", () => {
  it("is completed over total", () => {
    expect(progressRatio(1, 4)).toBe(0.25);
  });

  it("is 0 — not NaN — for an empty plan", () => {
    // A NaN width would render the bar at its full width in some browsers, i.e. "100% done".
    expect(progressRatio(0, 0)).toBe(0);
  });
});

describe("isAnalyzable", () => {
  const analyzable = [{ name: "Squat" }, { name: "Push-up" }];

  it("accepts a movement with a registered detector", () => {
    expect(isAnalyzable("Squat", analyzable)).toBe(true);
  });

  it("matches case- and whitespace-insensitively, as the backend registry does", () => {
    expect(isAnalyzable("  push-UP ", analyzable)).toBe(true);
  });

  it("rejects a catalog movement with no detector", () => {
    // Jumping Jacks and High Knee are plannable but tick-only.
    expect(isAnalyzable("Jumping Jacks", analyzable)).toBe(false);
  });
});

describe("templateText", () => {
  // `t` returns the KEY when there is no entry — the same contract movementLabel relies on.
  const t = ((key: string) =>
    key === "plans.template.known.name" ? "已知範本" : key) as TFunc;

  it("prefers the translation when one exists", () => {
    expect(templateText(t, "known", "name", "Known template")).toBe("已知範本");
  });

  it("falls back to the server's English string for an untranslated template", () => {
    // A template added to the backend before its i18n keys exist must render its name, not
    // "plans.template.brand_new.name".
    expect(templateText(t, "brand_new", "name", "Brand new")).toBe("Brand new");
  });
});
