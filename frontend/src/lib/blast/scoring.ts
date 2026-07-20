// Scoring for Meme Blaster: points for a beam that lands, with a bonus for wiping out
// several orbs in one shot and a rising combo multiplier. Pure helpers.

// Seconds in a round.
export const ROUND_SECONDS = 45;

// Base points per orb destroyed.
export const BASE_POINTS = 100;

// +10% per combo, capped at 3×. `combo` is the streak length after this shot.
export function comboMultiplier(combo: number): number {
  return Math.min(3, 1 + Math.max(0, combo) * 0.1);
}

// Wiping N orbs in a single beam pays a widening multi-kill bonus: +50% per extra orb.
export function multiKillMultiplier(hitCount: number): number {
  return hitCount > 1 ? 1 + (hitCount - 1) * 0.5 : 1;
}

// Points for a beam that destroys `hitCount` orbs at the given combo length. A shot that
// hits nothing scores 0 (and, in the loop, breaks the combo).
export function blastPoints(hitCount: number, combo: number): number {
  if (hitCount <= 0) return 0;
  return Math.round(
    BASE_POINTS * hitCount * multiKillMultiplier(hitCount) * comboMultiplier(combo)
  );
}

// How fast/often orbs come as the round progresses (0..1 of the round elapsed).
// Difficulty ramps so the finish is frantic.
export function difficulty(elapsedFraction: number): { speed: number; spawnMs: number } {
  const f = Math.min(1, Math.max(0, elapsedFraction));
  return {
    speed: 0.16 + f * 0.14, // screen-widths / second
    spawnMs: 1100 - f * 550, // gap between spawns
  };
}
