import { describe, it, expect } from "vitest";
import { handState, dist } from "../lib/blast/gestures";
import { LM } from "../lib/pose";

// Full 33-slot frame with overridable landmarks.
function frame(overrides: Record<number, { x: number; y: number; visibility?: number }>) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));
  for (const [i, p] of Object.entries(overrides)) lm[Number(i)] = { visibility: 1, ...p };
  return lm;
}

describe("dist", () => {
  it("is the euclidean distance", () => {
    expect(dist({ x: 0, y: 0 }, { x: 3, y: 4 })).toBe(5);
  });
});

describe("handState", () => {
  it("reports a small gap when the wrists are together", () => {
    const lm = frame({
      [LM.LEFT_SHOULDER]: { x: 0.4, y: 0.3 },
      [LM.RIGHT_SHOULDER]: { x: 0.6, y: 0.3 }, // shoulder width 0.2
      [LM.LEFT_WRIST]: { x: 0.49, y: 0.5 },
      [LM.RIGHT_WRIST]: { x: 0.51, y: 0.5 }, // wrist gap 0.02 → 0.1 shoulder-widths
    });
    const hs = handState(lm);
    expect(hs.valid).toBe(true);
    expect(hs.gap).toBeCloseTo(0.1, 5);
    expect(hs.aimY).toBeCloseTo(0.5, 5);
  });

  it("reports a large gap when the arms are thrown apart", () => {
    const lm = frame({
      [LM.LEFT_SHOULDER]: { x: 0.4, y: 0.3 },
      [LM.RIGHT_SHOULDER]: { x: 0.6, y: 0.3 }, // width 0.2
      [LM.LEFT_WRIST]: { x: 0.1, y: 0.4 },
      [LM.RIGHT_WRIST]: { x: 0.9, y: 0.4 }, // gap 0.8 → 4 shoulder-widths
    });
    expect(handState(lm).gap).toBeCloseTo(4, 5);
  });

  it("normalises the gap by shoulder width (distance-invariant)", () => {
    // Same pose twice as far from the camera → all coords scaled, gap ratio unchanged.
    const near = handState(
      frame({
        [LM.LEFT_SHOULDER]: { x: 0.3, y: 0.3 },
        [LM.RIGHT_SHOULDER]: { x: 0.7, y: 0.3 },
        [LM.LEFT_WRIST]: { x: 0.35, y: 0.5 },
        [LM.RIGHT_WRIST]: { x: 0.65, y: 0.5 },
      })
    );
    const far = handState(
      frame({
        [LM.LEFT_SHOULDER]: { x: 0.4, y: 0.4 },
        [LM.RIGHT_SHOULDER]: { x: 0.6, y: 0.4 },
        [LM.LEFT_WRIST]: { x: 0.425, y: 0.5 },
        [LM.RIGHT_WRIST]: { x: 0.575, y: 0.5 },
      })
    );
    expect(near.gap).toBeCloseTo(far.gap, 5);
  });

  it("is invalid when a wrist is missing or low-visibility", () => {
    const lm = frame({
      [LM.LEFT_SHOULDER]: { x: 0.4, y: 0.3 },
      [LM.RIGHT_SHOULDER]: { x: 0.6, y: 0.3 },
      [LM.LEFT_WRIST]: { x: 0.49, y: 0.5, visibility: 0.1 },
      [LM.RIGHT_WRIST]: { x: 0.51, y: 0.5 },
    });
    expect(handState(lm).valid).toBe(false);
  });

  it("is invalid when shoulders coincide (zero width)", () => {
    const lm = frame({
      [LM.LEFT_SHOULDER]: { x: 0.5, y: 0.3 },
      [LM.RIGHT_SHOULDER]: { x: 0.5, y: 0.3 },
      [LM.LEFT_WRIST]: { x: 0.4, y: 0.5 },
      [LM.RIGHT_WRIST]: { x: 0.6, y: 0.5 },
    });
    expect(handState(lm).valid).toBe(false);
  });
});
