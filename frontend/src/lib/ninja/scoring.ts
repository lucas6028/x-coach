// Scoring for Fruit Ninja: points per cut, a combo multiplier for keeping the blade moving, and
// a bonus for cutting several fruits in one swipe. Pure helpers.

// Fruits you can miss before it's game over.
export const START_LIVES = 5;
// Base points per fruit before the combo multiplier.
export const BASE_POINTS = 10;
// Cut another fruit within this window to keep the combo alive.
export const COMBO_WINDOW_MS = 700;

// +10% per fruit in the current streak, capped at 4×. `combo` is the streak length including
// the fruits being scored (first cut → combo 1 → 1×).
export function comboMultiplier(combo: number): number {
  return Math.min(4, 1 + Math.max(0, combo - 1) * 0.1);
}

// Points for cutting `n` fruits in one frame at the running combo (which already includes n),
// plus a multi-slice bonus for a 3+ fruit swipe.
export function sliceScore(n: number, combo: number): number {
  if (n <= 0) return 0;
  const per = Math.round(BASE_POINTS * comboMultiplier(combo));
  const multiBonus = n >= 3 ? (n - 2) * 20 : 0;
  return per * n + multiBonus;
}
