import { describe, it, expect } from "vitest";
import {
  avgKneeAngle,
  bicepCurlAvgElbowAngle,
  centeredMedian,
  lungeMinKneeAngle,
  pushUpMinElbowAngle,
  rowMinElbowAngle,
  sitUpHipAngle,
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

  it("allows NaN z during the validity gate, computing the metric from the other side", () => {
    // Right ankle has NaN z but finite x/y/visibility. The validity gate (dims=2) must pass
    // because x/y are visible, and the right knee angle must be NaN so mean_finite drops it.
    // Python returns 180.0 here (the left side alone), not NaN. This is the critical
    // divergence the reviewer caught: a naive 3-D gate would reject this frame entirely.
    const rightAnkleZisNaN = skeleton({
      23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 2, 0],   // left: straight
      24: [1, 0, 0], 26: [1, 1, 0], 28: [1, 2, NaN], // right: z is NaN
    });
    expect(avgKneeAngle(rightAnkleZisNaN)).toBeCloseTo(180, 4);
  });

  it("includes landmarks at visibility exactly 0.5 (boundary case)", () => {
    // visibility >= VISIBILITY_THRESHOLD must be true, so 0.5 is included.
    const atBoundary = skeleton({
      23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 2, 0],
      24: [1, 0, 0], 26: [1, 1, 0], 28: [1, 2, 0],
    });
    atBoundary[27] = { ...atBoundary[27], visibility: 0.5 };
    expect(avgKneeAngle(atBoundary)).toBeCloseTo(180, 4);
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

// Left arm straight (shoulder 11 -> elbow 13 -> wrist 15 in a line => 180), right arm bent at a
// right angle (12 -> 14 -> 16 => 90). Min and mean differ, so a swapped reducer cannot pass.
const ARMS = skeleton({
  11: [0, 0, 0], 13: [0, 1, 0], 15: [0, 2, 0],
  12: [1, 0, 0], 14: [1, 1, 0], 16: [2, 1, 0],
});
// Left leg straight (180), right knee bent (90) -- `mixed` from avgKneeAngle's tests.
const LEGS = skeleton({
  23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 2, 0],
  24: [1, 0, 0], 26: [1, 1, 0], 28: [2, 1, 0],
});
// Left trunk/thigh in a line (shoulder 11 -> hip 23 -> knee 25 => 180), right hip flexed to 90.
const TRUNK = skeleton({
  11: [0, 0, 0], 23: [0, 1, 0], 25: [0, 2, 0],
  12: [1, 0, 0], 24: [1, 1, 0], 26: [2, 1, 0],
});

function hide(lms: SignalLandmark[], index: number): SignalLandmark[] {
  const copy = lms.map((lm) => ({ ...lm }));
  copy[index] = { ...copy[index], visibility: 0.49 };
  return copy;
}

describe("elbow-angle rep signals", () => {
  it("Push-up and Row take the more-flexed arm, Bicep Curl the mean", () => {
    expect(pushUpMinElbowAngle(ARMS)).toBeCloseTo(90, 4);
    expect(rowMinElbowAngle(ARMS)).toBeCloseTo(90, 4);
    expect(bicepCurlAvgElbowAngle(ARMS)).toBeCloseTo(135, 4);
  });

  it("gate on their own module's landmarks: only Push-up requires the ankles", () => {
    const noAnkle = hide(ARMS, 27);
    expect(Number.isNaN(pushUpMinElbowAngle(noAnkle))).toBe(true);
    expect(rowMinElbowAngle(noAnkle)).toBeCloseTo(90, 4);
    expect(bicepCurlAvgElbowAngle(noAnkle)).toBeCloseTo(135, 4);
  });

  it("all require the hips", () => {
    const noHip = hide(ARMS, 24);
    expect(Number.isNaN(pushUpMinElbowAngle(noHip))).toBe(true);
    expect(Number.isNaN(rowMinElbowAngle(noHip))).toBe(true);
    expect(Number.isNaN(bicepCurlAvgElbowAngle(noHip))).toBe(true);
  });

  it("fall back to the other arm when one wrist has a NaN z (valid frame, one NaN angle)", () => {
    const bentZ = ARMS.map((lm) => ({ ...lm }));
    bentZ[16] = { ...bentZ[16], z: NaN };
    expect(pushUpMinElbowAngle(bentZ)).toBeCloseTo(180, 4);
    expect(rowMinElbowAngle(bentZ)).toBeCloseTo(180, 4);
    expect(bicepCurlAvgElbowAngle(bentZ)).toBeCloseTo(180, 4);
  });
});

describe("lungeMinKneeAngle", () => {
  it("takes the more-flexed knee", () => {
    expect(lungeMinKneeAngle(LEGS)).toBeCloseTo(90, 4);
  });

  it("requires the toes, which Squat's signal does not", () => {
    const noToe = hide(LEGS, 31);
    expect(Number.isNaN(lungeMinKneeAngle(noToe))).toBe(true);
    expect(avgKneeAngle(noToe)).toBeCloseTo(135, 4);
  });

  it("requires the shoulders", () => {
    expect(Number.isNaN(lungeMinKneeAngle(hide(LEGS, 12)))).toBe(true);
  });
});

describe("sitUpHipAngle", () => {
  it("averages the two shoulder-hip-knee angles", () => {
    expect(sitUpHipAngle(TRUNK)).toBeCloseTo(135, 4);
  });

  it("does not need the arms or the feet", () => {
    expect(sitUpHipAngle(hide(hide(TRUNK, 15), 27))).toBeCloseTo(135, 4);
  });

  it("is NaN when a knee is hidden", () => {
    expect(Number.isNaN(sitUpHipAngle(hide(TRUNK, 25)))).toBe(true);
  });
});

describe("TS_REP_SIGNALS", () => {
  it("covers the movements ported so far, by registry name", () => {
    expect(Object.keys(TS_REP_SIGNALS)).toEqual(
      ["Squat", "Push-up", "Lunge", "Row", "Bicep Curl", "Sit-up"]
    );
    expect(TS_REP_SIGNALS.Squat(STRAIGHT)).toBeCloseTo(180, 4);
    expect(TS_REP_SIGNALS["Bicep Curl"](ARMS)).toBeCloseTo(135, 4);
  });

  it("is NaN for every movement on a missing or short landmark list", () => {
    for (const signal of Object.values(TS_REP_SIGNALS)) {
      expect(Number.isNaN(signal(null))).toBe(true);
      expect(Number.isNaN(signal([]))).toBe(true);
    }
  });
});
