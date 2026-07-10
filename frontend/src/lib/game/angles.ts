// Joint-angle helpers for the Pose Match game. Given a MediaPipe 33-landmark frame,
// distil it to a small "pose signature" of the joint angles the game scores against.
// Pure functions only (no DOM / MediaPipe imports) so they unit-test in isolation.
import { LM } from "../pose";

// A single 2-D landmark. MediaPipe returns normalized (0..1) x/y plus a visibility
// score; we only need x/y for angles and visibility to gate low-confidence joints.
export type Keypoint = { x: number; y: number; visibility?: number };

// The joints the game measures. Symmetric upper/lower-body pairs keep target poses
// reliable from a single phone camera.
export type JointName =
  | "leftElbow"
  | "rightElbow"
  | "leftShoulder"
  | "rightShoulder"
  | "leftHip"
  | "rightHip"
  | "leftKnee"
  | "rightKnee";

// Each joint is the interior angle at the middle landmark of a [a, vertex, c] triple.
export const JOINT_TRIPLES: Record<JointName, [number, number, number]> = {
  leftElbow: [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST],
  rightElbow: [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST],
  leftShoulder: [LM.LEFT_ELBOW, LM.LEFT_SHOULDER, LM.LEFT_HIP],
  rightShoulder: [LM.RIGHT_ELBOW, LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
  leftHip: [LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE],
  rightHip: [LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE],
  leftKnee: [LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE],
  rightKnee: [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
};

export const JOINT_NAMES = Object.keys(JOINT_TRIPLES) as JointName[];

// Below this visibility a landmark is treated as unseen — the joint drops out of the
// signature rather than contributing a garbage angle.
export const MIN_VISIBILITY = 0.5;

// Interior angle (degrees, 0..180) at vertex b for the corner a-b-c. NaN if either
// arm has zero length (coincident points).
export function angleAt(a: Keypoint, b: Keypoint, c: Keypoint): number {
  const v1x = a.x - b.x;
  const v1y = a.y - b.y;
  const v2x = c.x - b.x;
  const v2y = c.y - b.y;
  const m1 = Math.hypot(v1x, v1y);
  const m2 = Math.hypot(v2x, v2y);
  if (m1 === 0 || m2 === 0) return NaN;
  const cos = Math.min(1, Math.max(-1, (v1x * v2x + v1y * v2y) / (m1 * m2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

// A frame's measurable joint angles. Joints that are missing or low-visibility are
// simply absent (not zero), so scoring can tell "wrong angle" from "can't see it".
export type PoseSignature = Partial<Record<JointName, number>>;

// Distil a landmark array into the game's joint-angle signature.
export function poseSignature(landmarks: (Keypoint | null | undefined)[]): PoseSignature {
  const sig: PoseSignature = {};
  for (const name of JOINT_NAMES) {
    const [ia, ib, ic] = JOINT_TRIPLES[name];
    const a = landmarks[ia];
    const b = landmarks[ib];
    const c = landmarks[ic];
    if (!a || !b || !c) continue;
    if (
      (a.visibility ?? 1) < MIN_VISIBILITY ||
      (b.visibility ?? 1) < MIN_VISIBILITY ||
      (c.visibility ?? 1) < MIN_VISIBILITY
    )
      continue;
    const ang = angleAt(a, b, c);
    if (!Number.isNaN(ang)) sig[name] = ang;
  }
  return sig;
}
