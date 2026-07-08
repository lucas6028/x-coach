import { describe, it, expect } from "vitest";
import { POSES, poseById, scorePose, pickPose, TOLERANCE, type DuelPose } from "../lib/duel/poses";
import type { PoseSignature } from "../lib/duel/angles";

describe("POSES catalogue", () => {
  it("has unique ids and non-empty angle sets", () => {
    const ids = POSES.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const p of POSES) expect(Object.keys(p.angles).length).toBeGreaterThan(0);
  });

  it("poseById finds and misses", () => {
    expect(poseById("t_pose")?.id).toBe("t_pose");
    expect(poseById("nope")).toBeUndefined();
  });
});

const tPose = poseById("t_pose") as DuelPose;

describe("scorePose", () => {
  it("scores a perfect match at 1 with every joint matched", () => {
    const sig: PoseSignature = { ...tPose.angles };
    const r = scorePose(sig, tPose);
    expect(r.score).toBeCloseTo(1, 5);
    expect(r.matched).toBe(r.total);
    expect(r.total).toBe(Object.keys(tPose.angles).length);
  });

  it("counts a joint just inside tolerance as matched", () => {
    const j = Object.keys(tPose.angles)[0];
    const sig: PoseSignature = { ...tPose.angles, [j]: tPose.angles[j] + (TOLERANCE - 1) };
    const r = scorePose(sig, tPose);
    expect(r.matched).toBe(r.total);
  });

  it("drops a joint out of tolerance from the matched count", () => {
    const j = Object.keys(tPose.angles)[0];
    const sig: PoseSignature = { ...tPose.angles, [j]: tPose.angles[j] + (TOLERANCE + 30) };
    const r = scorePose(sig, tPose);
    expect(r.matched).toBe(r.total - 1);
    expect(r.score).toBeLessThan(1);
  });

  it("treats a null (unseen) joint as a zero contribution, not a match", () => {
    const j = Object.keys(tPose.angles)[0];
    const sig: PoseSignature = { ...tPose.angles, [j]: null };
    const r = scorePose(sig, tPose);
    expect(r.matched).toBe(r.total - 1);
    expect(r.score).toBeLessThan(1);
  });

  it("returns a zero score for a pose with no angles", () => {
    const empty: DuelPose = { id: "x", emoji: "", nameKey: "x", angles: {} };
    expect(scorePose({}, empty)).toEqual({ score: 0, matched: 0, total: 0 });
  });
});

describe("pickPose", () => {
  it("never returns the excluded pose", () => {
    for (let r = 0; r < 1; r += 0.07) {
      expect(pickPose("t_pose", r).id).not.toBe("t_pose");
    }
  });

  it("returns a valid pose when nothing is excluded", () => {
    const ids = new Set(POSES.map((p) => p.id));
    expect(ids.has(pickPose(null, 0).id)).toBe(true);
    expect(ids.has(pickPose(null, 0.999).id)).toBe(true);
  });

  it("clamps r = 1 to the last pose rather than overflowing", () => {
    expect(pickPose(null, 1).id).toBe(POSES[POSES.length - 1].id);
  });
});
