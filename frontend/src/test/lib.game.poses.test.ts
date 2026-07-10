import { describe, it, expect } from "vitest";
import { POSES, scorePose, poseById, TOLERANCE, type GamePose } from "../lib/game/poses";
import type { PoseSignature } from "../lib/game/angles";

// A signature that exactly hits a pose's target angles.
function exact(pose: GamePose): PoseSignature {
  return { ...pose.angles };
}

describe("POSES catalogue", () => {
  it("has unique ids and at least one constrained joint each", () => {
    const ids = POSES.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const p of POSES) {
      expect(Object.keys(p.angles).length).toBeGreaterThan(0);
    }
  });

  it("includes the squat hold that ties back to X-Coach", () => {
    expect(POSES.some((p) => p.id === "squat")).toBe(true);
  });
});

describe("poseById", () => {
  it("finds a known pose and returns undefined otherwise", () => {
    expect(poseById("t_pose")?.id).toBe("t_pose");
    expect(poseById("nope")).toBeUndefined();
  });
});

describe("scorePose", () => {
  const tPose = POSES.find((p) => p.id === "t_pose")!;

  it("scores a perfect match as 1", () => {
    const { score, matched, total } = scorePose(exact(tPose), tPose);
    expect(score).toBeCloseTo(1, 5);
    expect(matched).toBe(total);
  });

  it("penalises missing joints (they count as zero)", () => {
    const partial: PoseSignature = { leftShoulder: 90, rightShoulder: 90 };
    const { score, matched, total } = scorePose(partial, tPose);
    expect(matched).toBe(2);
    expect(total).toBe(4);
    // Two perfect joints out of four constrained → 0.5.
    expect(score).toBeCloseTo(0.5, 5);
  });

  it("falls to zero once error reaches the tolerance", () => {
    const off: PoseSignature = {
      leftShoulder: 90 + TOLERANCE,
      rightShoulder: 90 + TOLERANCE,
      leftElbow: 172 + TOLERANCE,
      rightElbow: 172 + TOLERANCE,
    };
    expect(scorePose(off, tPose).score).toBeCloseTo(0, 5);
  });

  it("gives partial credit within tolerance", () => {
    const half: PoseSignature = {
      leftShoulder: 90 + TOLERANCE / 2,
      rightShoulder: 90,
      leftElbow: 172,
      rightElbow: 172,
    };
    const { score } = scorePose(half, tPose);
    expect(score).toBeGreaterThan(0.8);
    expect(score).toBeLessThan(1);
  });

  it("returns zero for a pose with no constrained joints", () => {
    const empty: GamePose = { id: "x", nameKey: "x", emoji: "", angles: {} };
    expect(scorePose({ leftElbow: 90 }, empty)).toEqual({ score: 0, matched: 0, total: 0 });
  });
});
