// The target-pose catalogue for Pose Duel. Each pose is a set of target joint angles
// (degrees); a pose only constrains the joints that matter, leaving the rest "don't care".
// Pure logic — scoreable against a live PoseSignature and unit-tested in isolation.
import type { PoseSignature } from "./angles";

export type DuelPose = {
  id: string;
  emoji: string;
  // i18n key suffix -> "duelPose.<id>".
  nameKey: string;
  // joint -> target interior angle in degrees.
  angles: Record<string, number>;
};

// A joint counts as "matched" when it's within this many degrees of target.
export const TOLERANCE = 40;

// Six striking, camera-legible poses. Arms carry most of the signal (legs are often cropped
// or occluded when two people share one frame), so every pose is arm-distinguishable.
export const POSES: DuelPose[] = [
  {
    id: "t_pose",
    emoji: "🧍",
    nameKey: "t_pose",
    // Arms straight out to the sides.
    angles: { leftShoulder: 90, rightShoulder: 90, leftElbow: 175, rightElbow: 175 },
  },
  {
    id: "cactus",
    emoji: "🌵",
    nameKey: "cactus",
    // Goalpost arms: upper arm out, forearm up.
    angles: { leftShoulder: 90, rightShoulder: 90, leftElbow: 90, rightElbow: 90 },
  },
  {
    id: "cheer",
    emoji: "🙌",
    nameKey: "cheer",
    // Arms thrown up in a Y.
    angles: { leftShoulder: 155, rightShoulder: 155, leftElbow: 172, rightElbow: 172 },
  },
  {
    id: "flex",
    emoji: "💪",
    nameKey: "flex",
    // Double-biceps: upper arm out, forearm fully curled.
    angles: { leftShoulder: 85, rightShoulder: 85, leftElbow: 40, rightElbow: 40 },
  },
  {
    id: "star",
    emoji: "⭐",
    nameKey: "star",
    // Star jump: arms high-and-wide, legs apart.
    angles: {
      leftShoulder: 130,
      rightShoulder: 130,
      leftElbow: 172,
      rightElbow: 172,
      leftHip: 155,
      rightHip: 155,
    },
  },
  {
    id: "sumo",
    emoji: "🦵",
    nameKey: "sumo",
    // Deep athletic stance: hips and knees bent, elbows tucked.
    angles: { leftHip: 95, rightHip: 95, leftKnee: 95, rightKnee: 95, leftElbow: 80, rightElbow: 80 },
  },
];

export const poseById = (id: string): DuelPose | undefined => POSES.find((p) => p.id === id);

export type PoseScore = { score: number; matched: number; total: number };

// How well a signature matches a pose: soft 0..1 average over the pose's joints (a joint that
// isn't visible contributes 0), plus a hard matched/total tally for display.
export function scorePose(sig: PoseSignature, pose: DuelPose, tolerance = TOLERANCE): PoseScore {
  const joints = Object.keys(pose.angles);
  if (joints.length === 0) return { score: 0, matched: 0, total: 0 };
  let sum = 0;
  let matched = 0;
  for (const j of joints) {
    const actual = sig[j];
    if (actual == null) continue; // not visible -> contributes 0
    const diff = Math.abs(actual - pose.angles[j]);
    if (diff <= tolerance) matched += 1;
    sum += Math.max(0, 1 - diff / (tolerance * 1.6));
  }
  return { score: sum / joints.length, matched, total: joints.length };
}

// Pick a pose other than `excludeId`. `r` is a random float in [0, 1).
export function pickPose(excludeId: string | null, r: number): DuelPose {
  const pool = POSES.filter((p) => p.id !== excludeId);
  const list = pool.length ? pool : POSES;
  const idx = Math.min(list.length - 1, Math.floor(r * list.length));
  return list[idx];
}
