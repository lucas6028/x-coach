import { describe, it, expect } from "vitest";
import { centroidX, assignPlayers, type Landmarks } from "../lib/duel/assign";
import { LM } from "../lib/pose";

type P = { x: number; y: number; visibility?: number };

// A body whose torso landmarks all sit at x = `x`.
function body(x: number, visibility = 1): Landmarks {
  const arr: (P | undefined)[] = new Array(33).fill(undefined);
  for (const i of [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP]) {
    arr[i] = { x, y: 0.5, visibility };
  }
  return arr;
}

describe("centroidX", () => {
  it("averages the visible torso landmarks", () => {
    expect(centroidX(body(0.3))).toBeCloseTo(0.3, 6);
  });

  it("is null when the torso isn't visible enough", () => {
    expect(centroidX(body(0.3, 0.1))).toBeNull();
    expect(centroidX(new Array(33).fill(undefined))).toBeNull();
  });
});

describe("assignPlayers", () => {
  it("returns both null for no bodies", () => {
    expect(assignPlayers([])).toEqual({ a: null, b: null });
  });

  it("assigns the left-of-frame body to A and the right to B, regardless of input order", () => {
    const left = body(0.2);
    const right = body(0.8);
    const { a, b } = assignPlayers([right, left]); // deliberately out of order
    expect(a).toBe(left);
    expect(b).toBe(right);
  });

  it("puts a lone left-half body in slot A", () => {
    const only = body(0.3);
    expect(assignPlayers([only])).toEqual({ a: only, b: null });
  });

  it("puts a lone right-half body in slot B", () => {
    const only = body(0.7);
    expect(assignPlayers([only])).toEqual({ a: null, b: only });
  });

  it("drops bodies with no locatable torso", () => {
    const ghost = body(0.5, 0.0);
    const real = body(0.7);
    expect(assignPlayers([ghost, real])).toEqual({ a: null, b: real });
  });
});
