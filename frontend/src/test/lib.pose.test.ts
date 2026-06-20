import { describe, it, expect } from "vitest";
import { LM, POSE_CONNECTIONS, FAULT_LANDMARKS, edgeIsFaulty } from "../lib/pose";

describe("LM landmark indices", () => {
  it("has expected key indices", () => {
    expect(LM.NOSE).toBe(0);
    expect(LM.LEFT_SHOULDER).toBe(11);
    expect(LM.RIGHT_SHOULDER).toBe(12);
    expect(LM.LEFT_HIP).toBe(23);
    expect(LM.RIGHT_HIP).toBe(24);
    expect(LM.LEFT_KNEE).toBe(25);
    expect(LM.RIGHT_KNEE).toBe(26);
    expect(LM.LEFT_ANKLE).toBe(27);
    expect(LM.RIGHT_ANKLE).toBe(28);
  });

  it("indices are unique", () => {
    const vals = Object.values(LM);
    expect(new Set(vals).size).toBe(vals.length);
  });

  it("all indices are non-negative integers ≤ 32", () => {
    for (const v of Object.values(LM)) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(32);
      expect(Number.isInteger(v)).toBe(true);
    }
  });
});

describe("POSE_CONNECTIONS", () => {
  it("is a non-empty array of pairs", () => {
    expect(POSE_CONNECTIONS.length).toBeGreaterThan(0);
    for (const [a, b] of POSE_CONNECTIONS) {
      expect(typeof a).toBe("number");
      expect(typeof b).toBe("number");
    }
  });

  it("covers torso edges", () => {
    const has = (a: number, b: number) =>
      POSE_CONNECTIONS.some(([x, y]) => (x === a && y === b) || (x === b && y === a));
    expect(has(LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)).toBe(true);
    expect(has(LM.LEFT_HIP, LM.RIGHT_HIP)).toBe(true);
  });
});

describe("FAULT_LANDMARKS", () => {
  it("defines knees_inward", () => {
    expect(FAULT_LANDMARKS.knees_inward).toContain(LM.LEFT_KNEE);
    expect(FAULT_LANDMARKS.knees_inward).toContain(LM.RIGHT_KNEE);
  });

  it("defines excessive_forward_lean as torso group", () => {
    expect(FAULT_LANDMARKS.excessive_forward_lean).toContain(LM.LEFT_SHOULDER);
    expect(FAULT_LANDMARKS.excessive_forward_lean).toContain(LM.RIGHT_HIP);
  });

  it("defines heel_rise covering ankle group", () => {
    expect(FAULT_LANDMARKS.heel_rise).toContain(LM.LEFT_ANKLE);
    expect(FAULT_LANDMARKS.heel_rise).toContain(LM.RIGHT_HEEL);
  });
});

describe("edgeIsFaulty", () => {
  it("returns true when both endpoints are in the active set", () => {
    const active = new Set([LM.LEFT_KNEE, LM.LEFT_HIP]);
    expect(edgeIsFaulty(LM.LEFT_HIP, LM.LEFT_KNEE, active)).toBe(true);
  });

  it("returns false when only one endpoint is in the set", () => {
    const active = new Set([LM.LEFT_KNEE]);
    expect(edgeIsFaulty(LM.LEFT_HIP, LM.LEFT_KNEE, active)).toBe(false);
  });

  it("returns false when neither endpoint is in the set", () => {
    const active = new Set([LM.NOSE]);
    expect(edgeIsFaulty(LM.LEFT_HIP, LM.LEFT_KNEE, active)).toBe(false);
  });

  it("returns false for an empty set", () => {
    expect(edgeIsFaulty(LM.LEFT_HIP, LM.LEFT_KNEE, new Set())).toBe(false);
  });
});
