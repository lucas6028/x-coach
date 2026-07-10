// Cosmetic fruit-split pieces for Fruit Ninja: when a fruit is cut it bursts into two halves that
// fly apart perpendicular to the blade, arc down under gravity, spin, and fade out. Purely visual,
// but the motion is a pure simulation (unit-tested); the page holds the array and the detector
// clips each emoji in half to draw it.
import { GRAVITY } from "./physics";

export type Half = "left" | "right";

export type Piece = {
  id: number;
  emoji: string;
  // Centre position (normalised, same space as entities). vy < 0 is upward.
  x: number;
  y: number;
  vx: number;
  vy: number;
  // Current rotation and spin (radians, radians/sec). rot0 aligns the cut seam with the blade.
  rot: number;
  spin: number;
  radius: number;
  // Which half of the emoji this piece shows (clipped in the piece's local frame).
  half: Half;
  // Remaining life, 1 → 0.
  life: number;
};

// How fast the two halves push apart (normalised units/sec) and how long they live (seconds).
export const SPLIT_SPEED = 0.25;
export const PIECE_LIFE = 0.9;

// Split a cut fruit into two halves. `dirX`/`dirY` is the blade direction (need not be unit): the
// halves separate along its normal and the seam is rotated to lie along it. `rng` (injected for
// determinism) adds a small upward pop and spin.
export function spawnPieces(
  nextId: number,
  fruit: { emoji: string; x: number; y: number; vx: number; vy: number; radius: number },
  dirX: number,
  dirY: number,
  rng: () => number
): { pieces: Piece[]; nextId: number } {
  // Separation is perpendicular to the blade; fall back to horizontal for a zero-length blade.
  let nx = -dirY;
  let ny = dirX;
  const len = Math.hypot(nx, ny);
  if (len < 1e-6) {
    nx = 1;
    ny = 0;
  } else {
    nx /= len;
    ny /= len;
  }
  // Seam runs along the blade so the two halves look genuinely cut apart.
  const rot0 = dirX === 0 && dirY === 0 ? 0 : Math.atan2(-dirX, dirY);
  const pop = -0.15 - rng() * 0.1;

  const halves: Half[] = ["left", "right"];
  const pieces: Piece[] = halves.map((half, i) => {
    const sign = i === 0 ? 1 : -1;
    return {
      id: nextId + i,
      emoji: fruit.emoji,
      x: fruit.x,
      y: fruit.y,
      vx: fruit.vx + sign * nx * SPLIT_SPEED,
      vy: fruit.vy + sign * ny * SPLIT_SPEED + pop,
      rot: rot0,
      spin: sign * (2 + rng() * 3),
      radius: fruit.radius,
      half,
      life: 1,
    };
  });
  return { pieces, nextId: nextId + 2 };
}

// Advance every piece by dt seconds under gravity, spinning and fading; drop the expired ones.
export function advancePieces(pieces: Piece[], dt: number): Piece[] {
  const next: Piece[] = [];
  for (const p of pieces) {
    const life = p.life - dt / PIECE_LIFE;
    if (life <= 0) continue;
    const vy = p.vy + GRAVITY * dt;
    next.push({
      ...p,
      x: p.x + p.vx * dt,
      y: p.y + vy * dt,
      vy,
      rot: p.rot + p.spin * dt,
      life,
    });
  }
  return next;
}
