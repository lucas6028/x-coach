import { describe, it, expect } from "vitest";
import { handLead, dist, LEAD_THRESHOLD, MIN_VIS } from "../lib/sixseven/gesture";
import { LM } from "../lib/pose";

type P = { x: number; y: number; visibility?: number };

// Shoulders 0.2 wide; wrists at the given heights.
function frame(leftY: number, rightY: number, over: Record<number, P | undefined> = {}) {
  const arr: (P | undefined)[] = new Array(33).fill(undefined);
  arr[LM.LEFT_SHOULDER] = { x: 0.4, y: 0.5 };
  arr[LM.RIGHT_SHOULDER] = { x: 0.6, y: 0.5 };
  arr[LM.LEFT_WRIST] = { x: 0.4, y: leftY };
  arr[LM.RIGHT_WRIST] = { x: 0.6, y: rightY };
  for (const [i, p] of Object.entries(over)) arr[Number(i)] = p;
  return arr;
}

describe("dist", () => {
  it("is Euclidean", () => {
    expect(dist({ x: 0, y: 0 }, { x: 3, y: 4 })).toBe(5);
  });
});

describe("handLead", () => {
  it("reads the left hand as up when it's clearly higher", () => {
    const s = handLead(frame(0.3, 0.7)); // left wrist higher (smaller y)
    expect(s.valid).toBe(true);
    expect(s.lead).toBe("left");
    expect(s.diff).toBeGreaterThan(LEAD_THRESHOLD);
  });

  it("reads the right hand as up when it's clearly higher", () => {
    const s = handLead(frame(0.7, 0.3));
    expect(s.lead).toBe("right");
    expect(s.diff).toBeLessThan(-LEAD_THRESHOLD);
  });

  it("is neutral inside the dead-zone", () => {
    const s = handLead(frame(0.5, 0.53)); // gap 0.03 / 0.2 = 0.15 < threshold
    expect(s.lead).toBe("neutral");
  });

  it("is invalid when a wrist isn't visible", () => {
    const arr = frame(0.3, 0.7, { [LM.RIGHT_WRIST]: undefined });
    expect(handLead(arr).valid).toBe(false);
  });

  it("treats a low-visibility wrist as unseen", () => {
    const arr = frame(0.3, 0.7, { [LM.LEFT_WRIST]: { x: 0.4, y: 0.3, visibility: MIN_VIS - 0.1 } });
    expect(handLead(arr).valid).toBe(false);
  });

  it("is invalid when the shoulders coincide (zero width)", () => {
    const arr = frame(0.3, 0.7, { [LM.RIGHT_SHOULDER]: { x: 0.4, y: 0.5 } });
    expect(handLead(arr).valid).toBe(false);
  });
});
