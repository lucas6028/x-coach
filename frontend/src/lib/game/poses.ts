// The catalogue of target poses the game asks players to strike, and the angle-match
// scorer. Each pose is defined by the joint angles (degrees) it expects; a frame's
// pose signature is scored against those with a linear tolerance falloff.
import type { JointName, PoseSignature } from "./angles";

export type GamePose = {
  id: string;
  // i18n key for the display name (see lib/i18n game.pose.*).
  nameKey: string;
  emoji: string;
  // Target angle per joint the pose cares about. Joints omitted here are unconstrained.
  angles: Partial<Record<JointName, number>>;
};

// Degrees of error at which a joint scores 0. ~42° means "roughly the right shape"
// still earns partial credit, keeping the game forgiving on a phone camera.
export const TOLERANCE = 42;

// Ordered catalogue. Mixed upper- and lower-body shapes; the squat hold ties the game
// back to X-Coach's squat-analysis core.
export const POSES: GamePose[] = [
  {
    id: "t_pose",
    nameKey: "game.pose.tPose",
    emoji: "🧍",
    angles: { leftShoulder: 90, rightShoulder: 90, leftElbow: 172, rightElbow: 172 },
  },
  {
    id: "cactus",
    nameKey: "game.pose.cactus",
    emoji: "🌵",
    angles: { leftShoulder: 90, rightShoulder: 90, leftElbow: 90, rightElbow: 90 },
  },
  {
    id: "cheer",
    nameKey: "game.pose.cheer",
    emoji: "🙌",
    angles: { leftShoulder: 158, rightShoulder: 158, leftElbow: 168, rightElbow: 168 },
  },
  {
    id: "flex",
    nameKey: "game.pose.flex",
    emoji: "💪",
    angles: { leftShoulder: 88, rightShoulder: 88, leftElbow: 48, rightElbow: 48 },
  },
  {
    id: "stand",
    nameKey: "game.pose.stand",
    emoji: "🕴️",
    angles: { leftShoulder: 12, rightShoulder: 12, leftElbow: 172, rightElbow: 172 },
  },
  {
    id: "squat",
    nameKey: "game.pose.squat",
    emoji: "🏋️",
    angles: { leftKnee: 95, rightKnee: 95, leftHip: 95, rightHip: 95 },
  },
];

export type PoseScore = {
  // 0..1 overall match quality.
  score: number;
  // How many of the pose's joints were actually visible this frame.
  matched: number;
  // How many joints the pose constrains in total.
  total: number;
};

// Score a signature against a target pose. Every constrained joint contributes
// `max(0, 1 - err/tolerance)`; unseen joints contribute 0, so the player must show the
// whole shape — not just the joints that happen to be in frame.
export function scorePose(
  signature: PoseSignature,
  pose: GamePose,
  tolerance: number = TOLERANCE
): PoseScore {
  const names = Object.keys(pose.angles) as JointName[];
  const total = names.length;
  if (total === 0) return { score: 0, matched: 0, total: 0 };
  let sum = 0;
  let matched = 0;
  for (const name of names) {
    const target = pose.angles[name]!;
    const actual = signature[name];
    if (actual == null) continue;
    matched++;
    const err = Math.abs(actual - target);
    sum += Math.max(0, 1 - err / tolerance);
  }
  return { score: sum / total, matched, total };
}

// Look up a pose by id (used when replaying a saved game or a deep link).
export function poseById(id: string): GamePose | undefined {
  return POSES.find((p) => p.id === id);
}
