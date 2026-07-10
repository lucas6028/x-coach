import { describe, it, expect } from "vitest";
import { angleAt, poseSignature, JOINT_TRIPLES } from "../lib/game/angles";
import { LM } from "../lib/pose";

// Build a full 33-slot landmark array, then override the points a test cares about.
function frame(overrides: Record<number, { x: number; y: number; visibility?: number }>) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));
  for (const [i, p] of Object.entries(overrides)) lm[Number(i)] = { visibility: 1, ...p };
  return lm;
}

describe("angleAt", () => {
  it("measures a right angle", () => {
    // Vertex at origin, arms along +x and +y → 90°.
    const a = { x: 1, y: 0 };
    const b = { x: 0, y: 0 };
    const c = { x: 0, y: 1 };
    expect(angleAt(a, b, c)).toBeCloseTo(90, 5);
  });

  it("measures a straight line as 180°", () => {
    expect(angleAt({ x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 })).toBeCloseTo(180, 5);
  });

  it("measures a fully closed angle as 0°", () => {
    expect(angleAt({ x: 1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 })).toBeCloseTo(0, 5);
  });

  it("returns NaN when an arm has zero length", () => {
    expect(Number.isNaN(angleAt({ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 }))).toBe(true);
  });
});

describe("poseSignature", () => {
  it("computes an elbow angle from a landmark frame", () => {
    // Right angle at the left elbow.
    const lm = frame({
      [LM.LEFT_SHOULDER]: { x: 0, y: 0 },
      [LM.LEFT_ELBOW]: { x: 1, y: 0 },
      [LM.LEFT_WRIST]: { x: 1, y: 1 },
    });
    const sig = poseSignature(lm);
    expect(sig.leftElbow).toBeCloseTo(90, 1);
  });

  it("drops a joint when a landmark is below the visibility gate", () => {
    const lm = frame({
      [LM.LEFT_SHOULDER]: { x: 0, y: 0, visibility: 0.1 },
      [LM.LEFT_ELBOW]: { x: 1, y: 0 },
      [LM.LEFT_WRIST]: { x: 1, y: 1 },
    });
    expect(poseSignature(lm).leftElbow).toBeUndefined();
  });

  it("skips joints whose landmarks are missing entirely", () => {
    const lm: (null | { x: number; y: number })[] = new Array(33).fill(null);
    const sig = poseSignature(lm);
    expect(Object.keys(sig)).toHaveLength(0);
  });

  it("defines a triple for every named joint", () => {
    for (const triple of Object.values(JOINT_TRIPLES)) {
      expect(triple).toHaveLength(3);
    }
  });
});
