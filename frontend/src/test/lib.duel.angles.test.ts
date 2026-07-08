import { describe, it, expect } from "vitest";
import { angleAt, poseSignature, JOINT_TRIPLES, MIN_VISIBILITY } from "../lib/duel/angles";
import { LM } from "../lib/pose";

type P = { x: number; y: number; visibility?: number };

// Build a 33-slot landmark array; only the indices in `pts` are set (rest undefined).
function frame(pts: Record<number, P>): (P | undefined)[] {
  const arr: (P | undefined)[] = new Array(33).fill(undefined);
  for (const [i, p] of Object.entries(pts)) arr[Number(i)] = p;
  return arr;
}

describe("angleAt", () => {
  it("measures a right angle", () => {
    expect(angleAt({ x: 1, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 1 })).toBeCloseTo(90, 4);
  });

  it("measures a straight angle", () => {
    expect(angleAt({ x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 })).toBeCloseTo(180, 4);
  });

  it("returns 0 for a degenerate triple", () => {
    expect(angleAt({ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 })).toBe(0);
  });
});

describe("poseSignature", () => {
  it("computes joint angles for a visible left arm held out flat", () => {
    const sig = poseSignature(
      frame({
        [LM.LEFT_SHOULDER]: { x: 0.5, y: 0.5 },
        [LM.LEFT_ELBOW]: { x: 0.4, y: 0.5 },
        [LM.LEFT_WRIST]: { x: 0.3, y: 0.5 },
        [LM.LEFT_HIP]: { x: 0.5, y: 0.8 },
        [LM.LEFT_KNEE]: { x: 0.5, y: 1.0 },
      })
    );
    // Straight arm -> elbow ~180; arm out to the side -> shoulder ~90.
    expect(sig.leftElbow).toBeCloseTo(180, 0);
    expect(sig.leftShoulder).toBeCloseTo(90, 0);
  });

  it("is null for a joint whose triple isn't fully visible", () => {
    const sig = poseSignature(
      frame({
        [LM.LEFT_SHOULDER]: { x: 0.5, y: 0.5 },
        [LM.LEFT_ELBOW]: { x: 0.4, y: 0.5 },
        // wrist missing -> leftElbow can't be computed
      })
    );
    expect(sig.leftElbow).toBeNull();
  });

  it("treats a low-visibility landmark as unseen", () => {
    const sig = poseSignature(
      frame({
        [LM.RIGHT_SHOULDER]: { x: 0.5, y: 0.5, visibility: MIN_VISIBILITY - 0.1 },
        [LM.RIGHT_ELBOW]: { x: 0.6, y: 0.5 },
        [LM.RIGHT_WRIST]: { x: 0.7, y: 0.5 },
      })
    );
    expect(sig.rightElbow).toBeNull();
  });

  it("covers every joint in the catalogue", () => {
    const sig = poseSignature(frame({}));
    expect(Object.keys(sig).sort()).toEqual(Object.keys(JOINT_TRIPLES).sort());
  });
});
