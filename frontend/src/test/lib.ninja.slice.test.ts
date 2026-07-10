import { describe, it, expect } from "vitest";
import { segmentPointDist, bladeHits, sliceEntities, type Blade } from "../lib/ninja/slice";
import type { Entity } from "../lib/ninja/physics";

function fruit(over: Partial<Entity> = {}): Entity {
  return { id: 1, kind: "fruit", emoji: "🍉", x: 0.5, y: 0.5, vx: 0, vy: 0, radius: 0.075, ...over };
}

const through: Blade = { x1: 0.4, y1: 0.5, x2: 0.6, y2: 0.5 }; // passes through (0.5, 0.5)
const miss: Blade = { x1: 0.0, y1: 0.0, x2: 0.1, y2: 0.0 };

describe("segmentPointDist", () => {
  it("is zero for a point on the segment", () => {
    expect(segmentPointDist(0, 0, 1, 0, 0.5, 0)).toBeCloseTo(0, 6);
  });

  it("measures perpendicular distance", () => {
    expect(segmentPointDist(0, 0, 1, 0, 0.5, 0.3)).toBeCloseTo(0.3, 6);
  });

  it("clamps to the nearest endpoint past the segment", () => {
    expect(segmentPointDist(0, 0, 1, 0, 2, 0)).toBeCloseTo(1, 6);
  });

  it("handles a degenerate (zero-length) segment", () => {
    expect(segmentPointDist(1, 1, 1, 1, 1, 4)).toBeCloseTo(3, 6);
  });
});

describe("bladeHits", () => {
  it("hits a fruit the blade passes through", () => {
    expect(bladeHits(through, fruit())).toBe(true);
  });

  it("misses a distant fruit", () => {
    expect(bladeHits(miss, fruit())).toBe(false);
  });
});

describe("sliceEntities", () => {
  it("cuts fruit the blades pass through and keeps the rest", () => {
    const hit = fruit({ id: 1 });
    const safe = fruit({ id: 2, x: 0.9 });
    const r = sliceEntities([hit, safe], [through]);
    expect(r.slicedFruits.map((e) => e.id)).toEqual([1]);
    expect(r.remaining.map((e) => e.id)).toEqual([2]);
    expect(r.bombHit).toBe(false);
  });

  it("flags a bomb hit and removes the bomb", () => {
    const bomb = fruit({ id: 3, kind: "bomb", emoji: "💣" });
    const r = sliceEntities([bomb], [through]);
    expect(r.bombHit).toBe(true);
    expect(r.slicedFruits).toHaveLength(0);
    expect(r.remaining).toHaveLength(0);
  });

  it("keeps everything when no blade connects", () => {
    const r = sliceEntities([fruit()], [miss]);
    expect(r.slicedFruits).toHaveLength(0);
    expect(r.remaining).toHaveLength(1);
  });
});
