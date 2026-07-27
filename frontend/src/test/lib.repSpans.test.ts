import { describe, it, expect } from "vitest";
import { CANONICAL_FPS, frameIndexAt } from "../lib/repSpans";

describe("frameIndexAt", () => {
  it("puts every sample on the canonical 30fps grid", () => {
    expect(CANONICAL_FPS).toBe(30);
    expect(frameIndexAt(0)).toBe(0);
    expect(frameIndexAt(1 / 30)).toBe(1);
    expect(frameIndexAt(1)).toBe(30);
  });

  it("gives a coarse pass the SAME indices a dense pass would give", () => {
    // The bug this pins: an incrementing counter makes coarse sample k index k, not 3k.
    const dense = Array.from({ length: 10 }, (_, i) => frameIndexAt(i / 30));
    const coarse = [0, 3, 6, 9].map((i) => frameIndexAt(i / 30));
    expect(coarse).toEqual([0, 3, 6, 9]);
    expect(coarse.every((c) => dense.includes(c))).toBe(true);
  });
});
