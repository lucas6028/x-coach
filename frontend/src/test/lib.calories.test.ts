import { describe, it, expect } from "vitest";
import { estimateKcal, metForRate, EFFORT, DEFAULT_WEIGHT_KG } from "../lib/calories";

describe("calorie estimate", () => {
  it("burns nothing for a non-positive duration", () => {
    expect(estimateKcal({ durationSec: 0, moves: 50, effort: EFFORT.ninja })).toBe(0);
    expect(estimateKcal({ durationSec: -5, moves: 50, effort: EFFORT.ninja })).toBe(0);
  });

  it("uses the floor MET when the player is idle", () => {
    const e = EFFORT.sixseven;
    expect(metForRate(e, 0)).toBe(e.floorMet);
    // 60s at floor MET, default weight → MET × kg × (s/3600), rounded.
    const expected = Math.round(e.floorMet * DEFAULT_WEIGHT_KG * (60 / 3600));
    expect(estimateKcal({ durationSec: 60, moves: 0, effort: e })).toBe(expected);
  });

  it("saturates at the peak MET beyond the peak rate", () => {
    const e = EFFORT.ninja;
    expect(metForRate(e, e.movesForPeakPerMin)).toBe(e.peakMet);
    // Way above the peak rate clamps to peakMet, not higher.
    expect(metForRate(e, e.movesForPeakPerMin * 10)).toBe(e.peakMet);
  });

  it("interpolates MET linearly between floor and peak", () => {
    const e = EFFORT.sixseven;
    const mid = metForRate(e, e.movesForPeakPerMin / 2);
    expect(mid).toBeCloseTo((e.floorMet + e.peakMet) / 2, 5);
  });

  it("burns more the faster the player moves", () => {
    const slow = estimateKcal({ durationSec: 60, moves: 10, effort: EFFORT.sixseven });
    const fast = estimateKcal({ durationSec: 60, moves: 200, effort: EFFORT.sixseven });
    expect(fast).toBeGreaterThan(slow);
  });

  it("scales with body weight", () => {
    const light = estimateKcal({ durationSec: 300, moves: 100, effort: EFFORT.ninja, weightKg: 50 });
    const heavy = estimateKcal({ durationSec: 300, moves: 100, effort: EFFORT.ninja, weightKg: 90 });
    expect(heavy).toBeGreaterThan(light);
  });

  it("returns a whole non-negative number", () => {
    const k = estimateKcal({ durationSec: 137, moves: 42, effort: EFFORT.ninja });
    expect(Number.isInteger(k)).toBe(true);
    expect(k).toBeGreaterThanOrEqual(0);
  });
});
