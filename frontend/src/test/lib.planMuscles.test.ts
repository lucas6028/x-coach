import { describe, it, expect } from "vitest";
import { coverageOf, dayMuscles, planCoverage } from "../lib/planMuscles";
import type { PlanItem } from "../api";

// The whole gap list, in the order a gap line names them. Pinned here so a change to the order is
// a deliberate edit of this file rather than something that quietly reorders the copy.
const ALL_GAPS = [
  "quads",
  "hamstrings",
  "glutes",
  "chest",
  "upperBack",
  "lats",
  "shoulders",
  "abs",
];

let seq = 0;
function item(movement: string, day = 1): PlanItem {
  seq += 1;
  return {
    id: `i${seq}`,
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

describe("planCoverage ranking", () => {
  it("orders by times primary, then the Muscle union's own order", () => {
    // Two squats (quads+glutes primary) and a deadlift (glutes+hamstrings primary): glutes is a
    // prime mover three times, quads twice, hamstrings once.
    const items = [item("Squat"), item("Squat"), item("Deadlift")];
    const { primary, secondary } = planCoverage(items);

    expect(primary).toEqual(["glutes", "quads", "hamstrings"]);
    // None of these is ever a prime mover here, so they are all tied at zero and fall straight to
    // the union's declaration order — forearms (4), abs (5), upperBack (7), lats (8), lowerBack
    // (9), calves (15). How often each is a SUPPORTING muscle deliberately does not reorder them.
    expect(secondary).toEqual(["forearms", "abs", "upperBack", "lats", "lowerBack", "calves"]);
  });

  it("never lets a supporting count promote a muscle above an equally-driven one", () => {
    // Squat + Push-up + Sit-up drive quads, glutes, chest, triceps and abs once EACH, and abs is
    // additionally a supporting muscle twice (squat, push-up). Counting that used to put abs at
    // the head of the list — the regression this pins.
    const day = [item("Squat"), item("Push-up"), item("Sit-up")];
    const { primary } = planCoverage(day);
    expect(primary).toEqual(["chest", "triceps", "abs", "glutes", "quads"]);
  });

  it("puts a muscle in primary even when another movement only supports it", () => {
    // The squat drives the quads; the deadlift merely uses them. One primary appearance is enough.
    const { primary, secondary } = planCoverage([item("Squat"), item("Deadlift")]);
    expect(primary).toContain("quads");
    expect(secondary).not.toContain("quads");
  });

  it("ranks a single day's prime movers", () => {
    // Squat lists quads then glutes, but with one appearance each the tie-break is the union's
    // declaration order — glutes is declared first.
    expect(dayMuscles([item("Squat")])).toEqual(["glutes", "quads"]);
  });
});

describe("planCoverage gaps", () => {
  it("names the untrained major groups in the gap list's own order, not the ranking order", () => {
    // A curl trains biceps (primary) and forearms (secondary) — neither is a gap group — so every
    // gap group is missing, which pins the order exactly.
    expect(planCoverage([item("Bicep Curl")]).gaps).toEqual(ALL_GAPS);
  });

  it("drops a group that is trained either way", () => {
    // Push-up: chest and triceps primary, shoulders/abs/obliques secondary. Shoulders and abs are
    // supporting muscles here and still count as trained.
    const { gaps } = planCoverage([item("Push-up")]);
    expect(gaps).not.toContain("chest");
    expect(gaps).not.toContain("shoulders");
    expect(gaps).not.toContain("abs");
    expect(gaps).toEqual(["quads", "hamstrings", "glutes", "upperBack", "lats"]);
  });
});

describe("planCoverage edge cases", () => {
  it("skips a movement outside the catalog rather than throwing", () => {
    const withUnknown = planCoverage([item("Squat"), item("Plank")]);
    expect(withUnknown).toEqual(planCoverage([item("Squat")]));
  });

  it("returns nothing trained, and every group a gap, for an empty plan", () => {
    expect(planCoverage([])).toEqual({ primary: [], secondary: [], gaps: ALL_GAPS });
    expect(dayMuscles([])).toEqual([]);
  });

  it("reads bare movement names the same way it reads items", () => {
    // What a plan CARD has: distinct names, no items.
    expect(coverageOf(["Squat", "Deadlift"])).toEqual(
      planCoverage([item("Squat"), item("Deadlift")])
    );
  });
});
