import { describe, it, expect } from "vitest";
import {
  avgKneeAngle,
  centeredMedian,
  TS_REP_SIGNALS,
  type SignalLandmark,
} from "../lib/repSignal";

// A 33-point skeleton where every landmark is visible and at the origin, so a test only has to
// place the four points avgKneeAngle reads (hips 23/24, knees 25/26, ankles 27/28).
function skeleton(overrides: Record<number, [number, number, number]>): SignalLandmark[] {
  const lms: SignalLandmark[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 1 }));
  for (const [index, [x, y, z]] of Object.entries(overrides)) {
    lms[Number(index)] = { x, y, z, visibility: 1 };
  }
  return lms;
}

// hip directly above knee, ankle directly below knee => a perfectly straight leg => 180 degrees.
const STRAIGHT = skeleton({
  23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 2, 0],
  24: [1, 0, 0], 26: [1, 1, 0], 28: [1, 2, 0],
});
// hip above knee, ankle horizontally out from knee => a right angle => 90 degrees.
const BENT = skeleton({
  23: [0, 0, 0], 25: [0, 1, 0], 27: [1, 1, 0],
  24: [1, 0, 0], 26: [1, 1, 0], 28: [2, 1, 0],
});

describe("avgKneeAngle", () => {
  it("measures a straight leg as 180 degrees", () => {
    expect(avgKneeAngle(STRAIGHT)).toBeCloseTo(180, 4);
  });

  it("measures a right-angled knee as 90 degrees", () => {
    expect(avgKneeAngle(BENT)).toBeCloseTo(90, 4);
  });

  it("averages the two sides", () => {
    const mixed = skeleton({
      23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 2, 0],   // left straight  => 180
      24: [1, 0, 0], 26: [1, 1, 0], 28: [2, 1, 0],   // right bent     => 90
    });
    expect(avgKneeAngle(mixed)).toBeCloseTo(135, 4);
  });

  it("is NaN when a required point is below the visibility threshold", () => {
    const hidden = skeleton({
      23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 2, 0],
      24: [1, 0, 0], 26: [1, 1, 0], 28: [1, 2, 0],
    });
    hidden[25] = { ...hidden[25], visibility: 0.49 };
    expect(Number.isNaN(avgKneeAngle(hidden))).toBe(true);
  });

  it("is NaN for a missing or short landmark list", () => {
    expect(Number.isNaN(avgKneeAngle(null))).toBe(true);
    expect(Number.isNaN(avgKneeAngle([]))).toBe(true);
  });

  it("uses z, matching Python's 3-D angle_degrees", () => {
    // Same x/y as STRAIGHT but the ankle pushed out in z: a 3-D measure must see the bend, a
    // 2-D one would still report 180.
    const inZ = skeleton({
      23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 1, 1],
      24: [1, 0, 0], 26: [1, 1, 0], 28: [1, 1, 1],
    });
    expect(avgKneeAngle(inZ)).toBeCloseTo(90, 4);
  });
});

describe("centeredMedian", () => {
  it("smooths with a centred window and shrinks it at the edges", () => {
    expect(centeredMedian([1, 100, 1, 1, 1], 5)).toEqual([1, 1, 1, 1, 1]);
  });

  it("skips non-finite values instead of poisoning the window", () => {
    // The hole matters: RS-SP2 sends null frames, and geometry.py:117 skips them the same way.
    const out = centeredMedian([2, NaN, 4, NaN, 6], 3);
    expect(out[0]).toBe(2);
    expect(out[2]).toBe(4);
  });

  it("returns NaN where the whole window is non-finite", () => {
    expect(Number.isNaN(centeredMedian([NaN], 3)[0])).toBe(true);
  });

  it("returns an empty array for empty input", () => {
    expect(centeredMedian([], 5)).toEqual([]);
  });
});

describe("TS_REP_SIGNALS", () => {
  it("covers Squat and nothing else in SP2", () => {
    expect(Object.keys(TS_REP_SIGNALS)).toEqual(["Squat"]);
    expect(TS_REP_SIGNALS.Squat(STRAIGHT)).toBeCloseTo(180, 4);
  });
});
