// The drifting meme-emoji targets and the beam collision test. All coordinates are
// normalised (0..1) in screen space and independent of the mirrored camera feed, so the
// beam aims purely by height. Pure helpers — the game loop owns spawning cadence & RNG.

// Trending-meme emoji deck the orbs are drawn from.
export const MEME_EMOJIS = ["💀", "🗿", "🔥", "😭", "🤡", "🧠", "👽", "🐸", "💅", "🥶"];

export type Target = {
  id: number;
  emoji: string;
  // Normalised position; x starts >1 (off the right edge) and drifts toward 0.
  x: number;
  y: number;
  // Horizontal speed in screen-widths per second.
  speed: number;
};

// Half-height (normalised) of the beam's kill band around the aim line.
export const BEAM_HALF_HEIGHT = 0.1;

// Spawn a target just off the right edge at height `y`.
export function makeTarget(id: number, y: number, speed: number, emoji: string): Target {
  return { id, emoji, x: 1.12, y, speed };
}

// Move every target left by dt (seconds) and drop any that have left the screen.
// Returns the survivors and how many drifted off (a missed target).
export function advanceTargets(
  targets: Target[],
  dt: number
): { targets: Target[]; escaped: number } {
  const next: Target[] = [];
  let escaped = 0;
  for (const t of targets) {
    const x = t.x - t.speed * dt;
    if (x < -0.15) escaped++;
    else next.push({ ...t, x });
  }
  return { targets: next, escaped };
}

// Split targets into those the beam at `aimY` destroys and those it misses.
export function beamHits(
  targets: Target[],
  aimY: number,
  half: number = BEAM_HALF_HEIGHT
): { hit: Target[]; remaining: Target[] } {
  const hit: Target[] = [];
  const remaining: Target[] = [];
  for (const t of targets) {
    // Only on-screen targets (x <= 1) are hittable.
    if (t.x <= 1 && Math.abs(t.y - aimY) <= half) hit.push(t);
    else remaining.push(t);
  }
  return { hit, remaining };
}
