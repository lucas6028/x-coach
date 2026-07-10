// Blade → fruit collision for Fruit Ninja. A "blade" is the segment a wrist swept between two
// frames; a fruit is cut when the segment passes within its radius. Pure geometry, unit-tested.
import type { Entity } from "./physics";

export type Blade = { x1: number; y1: number; x2: number; y2: number };

// A wrist must move at least this fast (normalised units / second) for its segment to cut —
// so resting a hand on a fruit doesn't slice it, only a real swipe does. The page computes wrist
// speed and only forwards blades at or above this.
export const MIN_SLICE_SPEED = 0.9;

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
