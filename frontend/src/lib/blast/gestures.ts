// Read the player's "energy charge" gesture from a MediaPipe landmark frame: how far
// apart the two wrists are (normalised by shoulder width, so it's robust to how close
// the player stands) and how high they're holding them (the vertical aim). Pure — no
// DOM / MediaPipe imports — so it unit-tests in isolation.
import { LM } from "../pose";

// A 2-D landmark; MediaPipe reports normalised (0..1) x/y plus a visibility score.
export type Point = { x: number; y: number; visibility?: number };

const MIN_VIS = 0.5;

export function dist(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export type HandState = {
  // Both wrists and both shoulders were visible this frame.
  valid: boolean;
  // Wrist separation in shoulder-widths: ~0 when hands touch, ~3+ at full T-pose.
  gap: number;
  // Average wrist height, 0 (top of frame) .. 1 (bottom) — the beam's aim.
  aimY: number;
};

const INVALID: HandState = { valid: false, gap: 0, aimY: 0.5 };

function seen(p: Point | null | undefined): p is Point {
  return !!p && (p.visibility ?? 1) >= MIN_VIS;
}

// Distil a landmark frame into the charge gesture's inputs.
export function handState(landmarks: (Point | null | undefined)[]): HandState {
  const lw = landmarks[LM.LEFT_WRIST];
  const rw = landmarks[LM.RIGHT_WRIST];
  const ls = landmarks[LM.LEFT_SHOULDER];
  const rs = landmarks[LM.RIGHT_SHOULDER];
  if (!seen(lw) || !seen(rw) || !seen(ls) || !seen(rs)) return INVALID;
  const shoulderWidth = dist(ls, rs);
  if (shoulderWidth <= 0) return INVALID;
  return {
    valid: true,
    gap: dist(lw, rw) / shoulderWidth,
    aimY: (lw.y + rw.y) / 2,
  };
}
