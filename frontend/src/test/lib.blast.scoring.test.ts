import { describe, it, expect } from "vitest";
import {
  blastPoints,
  comboMultiplier,
  multiKillMultiplier,
  difficulty,
  BASE_POINTS,
} from "../lib/blast/scoring";

describe("comboMultiplier", () => {
  it("adds 10% per combo, capped at 3x", () => {
    expect(comboMultiplier(0)).toBeCloseTo(1);
    expect(comboMultiplier(10)).toBeCloseTo(2);
    expect(comboMultiplier(100)).toBe(3);
    expect(comboMultiplier(-3)).toBe(1);
  });
});

describe("multiKillMultiplier", () => {
  it("is 1 for a single kill and grows +50% per extra orb", () => {
    expect(multiKillMultiplier(1)).toBe(1);
    expect(multiKillMultiplier(2)).toBe(1.5);
    expect(multiKillMultiplier(3)).toBe(2);
  });
});

describe("blastPoints", () => {
  it("scores zero for a whiff", () => {
    expect(blastPoints(0, 5)).toBe(0);
  });

  it("scores base points for a single kill at combo 1", () => {
    // 100 * 1 * 1 * 1.1
    expect(blastPoints(1, 1)).toBe(Math.round(BASE_POINTS * 1.1));
  });

  it("rewards multi-kills more than the same orbs split up", () => {
    const double = blastPoints(2, 1);
    const twoSingles = blastPoints(1, 1) + blastPoints(1, 2);
    expect(double).toBeGreaterThan(0);
    expect(twoSingles).toBeGreaterThan(0);
    // A 2-kill gets the multi-kill bonus on both orbs at once.
    expect(double).toBeGreaterThan(blastPoints(1, 1) * 2);
  });
});

describe("difficulty", () => {
  it("ramps speed up and spawn interval down over the round", () => {
    const start = difficulty(0);
    const end = difficulty(1);
    expect(end.speed).toBeGreaterThan(start.speed);
    expect(end.spawnMs).toBeLessThan(start.spawnMs);
  });

  it("clamps the elapsed fraction", () => {
    expect(difficulty(-1)).toEqual(difficulty(0));
    expect(difficulty(2)).toEqual(difficulty(1));
  });
});
