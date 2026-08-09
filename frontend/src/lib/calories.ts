// Rough calorie estimate for the pose mini-games. A camera can't measure real exertion, so this is
// a deliberately simple MET model: energy = MET × bodyweight(kg) × hours, with the MET scaled by
// how fast the player is actually moving (moves per minute) between a resting floor and an active
// ceiling. The UI always labels the number an estimate — the goal is a fun, directionally-honest
// "you burned about this much", not clinical accuracy. Pure + unit-tested.

export type GameId = "sixseven" | "ninja" | "webslinger";

// Assumed body weight — the app has no weight field yet, so every estimate uses this default.
export const DEFAULT_WEIGHT_KG = 65;

export interface GameEffort {
  // MET when the player is barely moving.
  floorMet: number;
  // MET once the player is moving at `movesForPeakPerMin` or faster.
  peakMet: number;
  // Moves per minute at which the MET saturates to `peakMet`.
  movesForPeakPerMin: number;
}

// Per-game effort profiles. The METs sit in the "standing arm activity" band (~3–5) from the
// Compendium of Physical Activities; the ceiling is only reached when the player is genuinely busy.
export const EFFORT: Record<GameId, GameEffort> = {
  // The 67 bob: continuous alternating arm raises. `moves` = 67s completed.
  sixseven: { floorMet: 2.5, peakMet: 5, movesForPeakPerMin: 120 },
  // Fruit Ninja: larger whole-arm swipes, so a slightly higher ceiling. `moves` = fruit sliced.
  ninja: { floorMet: 2.5, peakMet: 5.5, movesForPeakPerMin: 90 },
  // Web Slinger: repeated arm extensions from both sides. `moves` = targets webbed.
  webslinger: { floorMet: 2.5, peakMet: 5.2, movesForPeakPerMin: 70 },
};

// MET for a given movement rate, linearly interpolated from floor to peak and clamped.
export function metForRate(effort: GameEffort, movesPerMin: number): number {
  const frac = Math.max(0, Math.min(1, movesPerMin / effort.movesForPeakPerMin));
  return effort.floorMet + (effort.peakMet - effort.floorMet) * frac;
}

export interface EstimateInput {
  durationSec: number;
  moves: number;
  effort: GameEffort;
  weightKg?: number;
}

// Whole-kcal estimate for one round. Non-positive durations (e.g. a round that never really
// started) burn nothing.
export function estimateKcal({
  durationSec,
  moves,
  effort,
  weightKg = DEFAULT_WEIGHT_KG,
}: EstimateInput): number {
  if (durationSec <= 0) return 0;
  const movesPerMin = (Math.max(0, moves) / durationSec) * 60;
  const met = metForRate(effort, movesPerMin);
  const kcal = met * weightKg * (durationSec / 3600);
  return Math.max(0, Math.round(kcal));
}
