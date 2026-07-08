import { describe, it, expect } from "vitest";
import {
  gradeFor,
  comboMultiplier,
  hitPoints,
  GRADE_POINTS,
  HIT_THRESHOLD,
} from "../lib/game/scoring";

describe("gradeFor", () => {
  it("maps quality bands to grades", () => {
    expect(gradeFor(0.95)).toBe("perfect");
    expect(gradeFor(0.9)).toBe("perfect");
    expect(gradeFor(0.8)).toBe("great");
    expect(gradeFor(0.7)).toBe("good");
    expect(gradeFor(HIT_THRESHOLD)).toBe("good");
    expect(gradeFor(0.5)).toBe("miss");
  });
});

describe("comboMultiplier", () => {
  it("adds 10% per combo and caps at 3x", () => {
    expect(comboMultiplier(0)).toBeCloseTo(1);
    expect(comboMultiplier(1)).toBeCloseTo(1.1);
    expect(comboMultiplier(10)).toBeCloseTo(2);
    expect(comboMultiplier(100)).toBe(3);
  });

  it("never drops below 1 for a negative combo", () => {
    expect(comboMultiplier(-5)).toBe(1);
  });
});

describe("hitPoints", () => {
  it("awards base points times the combo multiplier", () => {
    // Perfect at combo 1 → 100 * 1.1 = 110.
    expect(hitPoints(0.95, 1)).toBe(Math.round(GRADE_POINTS.perfect * 1.1));
  });

  it("awards nothing for a miss regardless of combo", () => {
    expect(hitPoints(0.3, 5)).toBe(0);
  });

  it("scales up with the combo", () => {
    expect(hitPoints(0.95, 10)).toBeGreaterThan(hitPoints(0.95, 1));
  });
});
