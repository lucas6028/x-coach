// Projectile physics for Fruit Ninja: fruits (and the occasional bomb) launch up from just
// below the bottom edge and arc back down under gravity. All coordinates are normalised (0..1)
// in the raw camera-image space, so the same numbers drive collision and the mirrored overlay.
// Pure — no DOM — so the whole simulation unit-tests without a canvas.

export type Kind = "fruit" | "bomb";

export type Entity = {
  id: number;
  kind: Kind;
  emoji: string;
  // Centre position (normalised). vy < 0 is upward (y grows downward).
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
};

// Downward acceleration (per second²) and the y past which a falling entity is gone.
export const GRAVITY = 1.35;
export const OFFSCREEN_Y = 1.2;
// Fruits are generous to slice; bombs are a touch smaller so they're easier to steer clear of.
export const FRUIT_RADIUS = 0.09;
export const BOMB_RADIUS = 0.07;
// Fraction of spawns that are bombs.
export const BOMB_CHANCE = 0.1;

export const FRUITS = ["🍉", "🍎", "🍊", "🍋", "🍓", "🍇", "🥝", "🍑", "🍍"];

// Advance every entity by dt seconds under gravity, dropping those that have fallen off the
// bottom. Returns the survivors and how many *fruits* were missed (fell past the edge uncut) —
// bombs falling away are harmless.
export function advanceEntities(
  entities: Entity[],
  dt: number
): { entities: Entity[]; droppedFruits: number } {
  const next: Entity[] = [];
  let droppedFruits = 0;
  for (const e of entities) {
    const vy = e.vy + GRAVITY * dt;
    const x = e.x + e.vx * dt;
    const y = e.y + vy * dt;
    if (y > OFFSCREEN_Y && vy > 0) {
      if (e.kind === "fruit") droppedFruits += 1;
    } else {
      next.push({ ...e, x, y, vy });
    }
  }
  return { entities: next, droppedFruits };
}

// Launch a wave of 1..3 entities from just below the bottom edge. `rng` returns floats in
// [0, 1); it's injected so the whole spawn sequence is deterministic in tests.
export function spawnWave(nextId: number, rng: () => number): { entities: Entity[]; nextId: number } {
  const count = 1 + Math.floor(rng() * 3);
  const entities: Entity[] = [];
  let id = nextId;
  for (let i = 0; i < count; i += 1) {
    const isBomb = rng() < BOMB_CHANCE;
    const x = 0.15 + rng() * 0.7;
    const vx = (rng() - 0.5) * 0.3;
    const vy = -(1.25 + rng() * 0.3);
    const emoji = isBomb ? "💣" : FRUITS[Math.floor(rng() * FRUITS.length)];
    entities.push({
      id: id,
      kind: isBomb ? "bomb" : "fruit",
      emoji,
      x,
      y: 1.08,
      vx,
      vy,
      radius: isBomb ? BOMB_RADIUS : FRUIT_RADIUS,
    });
    id += 1;
  }
  return { entities, nextId: id };
}
