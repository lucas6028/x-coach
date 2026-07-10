// Read the "6-7" meme gesture from a MediaPipe landmark frame: two hands out, palms up,
// bobbing alternately up and down (the "six... seven..." balance-scale motion). We only need
// which wrist is currently held higher, normalised by shoulder width so it's robust to how far
// the player stands from the camera. Pure — no DOM / MediaPipe imports — so it unit-tests alone.
import { LM } from "../pose";

export type Point = { x: number; y: number; visibility?: number };

export const MIN_VIS = 0.5;
// Vertical wrist gap (in shoulder-widths) beyond which one hand counts as "clearly up". The
// dead-zone below it keeps small wobbles from registering as a switch.
export const LEAD_THRESHOLD = 0.35;

// Which hand is raised this frame: "left"/"right", or "neutral" inside the dead-zone.
export type Lead = "left" | "right" | "neutral";

export type HandsState = {
  valid: boolean;
  lead: Lead;
  // Signed height gap in shoulder-widths (positive → left hand higher).
  diff: number;
};

const INVALID: HandsState = { valid: false, lead: "neutral", diff: 0 };

// A landmark counts as present when MediaPipe is at least MIN_VIS confident (absent visibility
// → assume 1). Shared with the canvas overlay so the counter and the on-screen dot agree about
// whether a wrist is on-frame. Structural param so both Point and MediaPipe's NormalizedLandmark fit.
export function visible(p: { visibility?: number } | null | undefined): boolean {
  return !!p && (p.visibility ?? 1) >= MIN_VIS;
}

function seen(p: Point | null | undefined): p is Point {
  return visible(p);
}

export function dist(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function handLead(landmarks: (Point | null | undefined)[]): HandsState {
  const lw = landmarks[LM.LEFT_WRIST];
  const rw = landmarks[LM.RIGHT_WRIST];
  const ls = landmarks[LM.LEFT_SHOULDER];
  const rs = landmarks[LM.RIGHT_SHOULDER];
  if (!seen(lw) || !seen(rw) || !seen(ls) || !seen(rs)) return INVALID;
  const shoulderWidth = dist(ls, rs);
  if (shoulderWidth <= 0) return INVALID;
  // y grows downward, so a higher hand has the smaller y; diff > 0 means the left hand is up.
  const diff = (rw.y - lw.y) / shoulderWidth;
  let lead: Lead = "neutral";
  if (diff > LEAD_THRESHOLD) lead = "left";
  else if (diff < -LEAD_THRESHOLD) lead = "right";
  return { valid: true, lead, diff };
}
