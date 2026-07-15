import { describe, it, expect } from "vitest";
import { comboMultiplier, sliceScore, BASE_POINTS } from "../lib/ninja/scoring";

describe("comboMultiplier", () => {
  it("is 1× for the first cut", () => {
    expect(comboMultiplier(1)).toBe(1);
    expect(comboMultiplier(0)).toBe(1);
  });

  it("grows 10% per fruit in the streak", () => {
    expect(comboMultiplier(3)).toBeCloseTo(1.2, 6);
  });

  it("caps at 4×", () => {
    expect(comboMultiplier(100)).toBe(4);
  });
});

describe("sliceScore", () => {
  it("is zero when nothing is cut", () => {
    expect(sliceScore(0, 5)).toBe(0);
  });

  it("scores a single cut at the base value", () => {
    expect(sliceScore(1, 1)).toBe(BASE_POINTS);
  });

  it("pays a multi-slice bonus for cutting 3+ in one swipe", () => {
    // 3 fruits at combo 3: per = round(10 * 1.2) = 12; 12*3 + (3-2)*20 = 36 + 20
    expect(sliceScore(3, 3)).toBe(12 * 3 + 20);
  });

  it("gives no multi bonus for a two-fruit swipe", () => {
    expect(sliceScore(2, 2)).toBe(Math.round(BASE_POINTS * 1.1) * 2);
  });
});
