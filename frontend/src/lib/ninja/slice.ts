// Blade → fruit collision for Fruit Ninja. A "blade" is the segment a wrist swept between two
// frames; a fruit is cut when the segment passes within its radius. Pure geometry, unit-tested.
import type { Entity } from "./physics";

export type Blade = { x1: number; y1: number; x2: number; y2: number };

// A wrist must move at least this fast (normalised units / second) for its segment to cut —
// so resting a hand on a fruit doesn't slice it, only a real swipe does. The page computes wrist
// speed and only forwards blades at or above this.
export const MIN_SLICE_SPEED = 1.0;
// A real swipe also has to *cover ground*: this is the smallest per-frame travel (normalised
// units) that counts. Landmark jitter on a still hand is tiny, and — unlike a speed derived from a
// frame-timing-sensitive dt — this displacement floor rejects it regardless of frame rate.
export const MIN_SLICE_DIST = 0.035;
// Above this, a single-frame jump is a tracking glitch (the pose snapping), not a swipe — ignore
// it so a spurious segment can't rake across the whole board and hit a distant bomb.
export const MAX_SLICE_DIST = 0.5;

// Did the wrist actually swipe between two frames? It must travel a real distance (not jitter, not
// a glitch teleport) and move fast enough. A resting hand clears none of these, so it never forms
// a cutting blade — the guard against "I didn't swing but it sliced / it hit a bomb".
export function isSwipe(
  prev: { x: number; y: number },
  cur: { x: number; y: number },
  dtSec: number
): boolean {
  if (dtSec <= 0) return false;
  const dist = Math.hypot(cur.x - prev.x, cur.y - prev.y);
  if (dist < MIN_SLICE_DIST || dist > MAX_SLICE_DIST) return false;
  return dist / dtSec >= MIN_SLICE_SPEED;
}

// Shortest distance from point C to segment AB.
export function segmentPointDist(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 === 0 ? 0 : ((cx - ax) * dx + (cy - ay) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  const px = ax + t * dx;
  const py = ay + t * dy;
  return Math.hypot(cx - px, cy - py);
}

export function bladeHits(blade: Blade, e: Entity): boolean {
  return segmentPointDist(blade.x1, blade.y1, blade.x2, blade.y2, e.x, e.y) <= e.radius;
}

// Apply every active blade to the entities. Returns the fruits cut, whether a bomb was struck,
// and the entities left untouched (cut fruits and the struck bomb are removed).
export function sliceEntities(
  entities: Entity[],
  blades: Blade[]
): { slicedFruits: Entity[]; bombHit: boolean; remaining: Entity[] } {
  const slicedFruits: Entity[] = [];
  const remaining: Entity[] = [];
  let bombHit = false;
  for (const e of entities) {
    const hit = blades.some((b) => bladeHits(b, e));
    if (!hit) {
      remaining.push(e);
    } else if (e.kind === "bomb") {
      bombHit = true;
    } else {
      slicedFruits.push(e);
    }
  }
  return { slicedFruits, bombHit, remaining };
}
