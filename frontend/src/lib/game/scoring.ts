// Game scoring: turn a per-frame pose-match quality into a grade, and a grade + combo
// into points. Pure helpers so the reducer-like game loop stays trivially testable.

export type Grade = "perfect" | "great" | "good" | "miss";

// A hold that scores at or above this quality counts as a successful pose.
export const HIT_THRESHOLD = 0.62;

// How long (ms) the player must hold a pose above HIT_THRESHOLD for it to lock in.
export const HOLD_MS = 550;

// Seconds in a round.
export const ROUND_SECONDS = 60;

export function gradeFor(score: number): Grade {
  if (score >= 0.9) return "perfect";
  if (score >= 0.78) return "great";
  if (score >= HIT_THRESHOLD) return "good";
  return "miss";
}

// Base points per grade, before the combo multiplier.
export const GRADE_POINTS: Record<Grade, number> = {
  perfect: 100,
  great: 60,
  good: 30,
  miss: 0,
};

// +10% per pose in the current streak, capped at 3×. `combo` is the streak length
// *including* the pose being scored (first hit → combo 1 → 1.1×).
export function comboMultiplier(combo: number): number {
  return Math.min(3, 1 + Math.max(0, combo) * 0.1);
}

// Points awarded for locking in a pose at the given quality and combo length.
export function hitPoints(score: number, combo: number): number {
  const base = GRADE_POINTS[gradeFor(score)];
  return Math.round(base * comboMultiplier(combo));
}
