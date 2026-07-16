import { describe, it, expect, beforeEach } from "vitest";
import { loadCalories, addCalories, clearCalories } from "../lib/calorieStore";

const KEY = "xcoach.calories.v1";

describe("calorie store", () => {
  beforeEach(() => clearCalories());

  it("starts at zero", () => {
    expect(loadCalories()).toEqual({
      total: 0,
      byGame: { sixseven: 0, ninja: 0, blast: 0 },
      sessions: 0,
    });
  });

  it("accumulates per game and in total, counting sessions", () => {
    addCalories("ninja", 12);
    addCalories("sixseven", 3);
    const t = addCalories("ninja", 8);
    expect(t.byGame).toEqual({ ninja: 20, sixseven: 3, blast: 0 });
    expect(t.total).toBe(23);
    expect(t.sessions).toBe(3);
    // Persisted across a fresh read.
    expect(loadCalories().total).toBe(23);
  });

  it("coerces bad kcal to zero but still counts the session", () => {
    const t = addCalories("ninja", Number.NaN);
    expect(t.total).toBe(0);
    expect(t.sessions).toBe(1);
  });

  it("degrades corrupt data to zeros", () => {
    localStorage.setItem(KEY, "not json {");
    expect(loadCalories().total).toBe(0);
    localStorage.setItem(KEY, JSON.stringify({ total: "lots" }));
    expect(loadCalories()).toEqual({
      total: 0,
      byGame: { sixseven: 0, ninja: 0, blast: 0 },
      sessions: 0,
    });
  });

  it("re-derives an inconsistent total from the breakdown", () => {
    localStorage.setItem(
      KEY,
      JSON.stringify({ total: 999, byGame: { ninja: 10, sixseven: 5 }, sessions: 2 })
    );
    expect(loadCalories().total).toBe(15);
  });

  it("clears the ledger", () => {
    addCalories("ninja", 20);
    clearCalories();
    expect(loadCalories().total).toBe(0);
  });
});
