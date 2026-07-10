// Turn a MediaPipe landmark frame into a compact "pose signature" — the interior angle at
// each joint we care about. Pure (no DOM / MediaPipe imports) so it unit-tests in isolation.
// Shared by both duelling players; the same joint geometry x-coach uses to read a squat is
// reused here to score whether a player is holding the target pose.
import { LM } from "../pose";

export type Point = { x: number; y: number; visibility?: number };

export const MIN_VISIBILITY = 0.5;

// Joint -> the [a, b, c] landmark triple whose angle-at-b defines it.
export const JOINT_TRIPLES: Record<string, [number, number, number]> = {
  leftElbow: [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST],
  rightElbow: [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST],
  leftShoulder: [LM.LEFT_ELBOW, LM.LEFT_SHOULDER, LM.LEFT_HIP],
  rightShoulder: [LM.RIGHT_ELBOW, LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
  leftHip: [LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE],
  rightHip: [LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE],
  leftKnee: [LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE],
  rightKnee: [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
};

// Interior angle ABC at vertex b, in degrees (0..180). Returns 0 for a degenerate triple.
export function angleAt(a: Point, b: Point, c: Point): number {
  const v1x = a.x - b.x;
  const v1y = a.y - b.y;
  const v2x = c.x - b.x;
  const v2y = c.y - b.y;
  const n1 = Math.hypot(v1x, v1y);
  const n2 = Math.hypot(v2x, v2y);
  if (n1 === 0 || n2 === 0) return 0;
  const cos = (v1x * v2x + v1y * v2y) / (n1 * n2);
  return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
}

// null for a joint whose triple isn't fully visible, so scoring can ignore it.
export type PoseSignature = Record<string, number | null>;

function seen(p: Point | null | undefined): p is Point {
  return !!p && (p.visibility ?? 1) >= MIN_VISIBILITY;
}

export function poseSignature(landmarks: (Point | null | undefined)[]): PoseSignature {
  const sig: PoseSignature = {};
  for (const joint of Object.keys(JOINT_TRIPLES)) {
    const [a, b, c] = JOINT_TRIPLES[joint];
    const pa = landmarks[a];
    const pb = landmarks[b];
    const pc = landmarks[c];
    sig[joint] = seen(pa) && seen(pb) && seen(pc) ? angleAt(pa, pb, pc) : null;
  }
  return sig;
}
