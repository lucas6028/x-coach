// The head-to-head round/match rules for Pose Duel. Each round both players race to hold the
// target pose; the first to accumulate HOLD_MS of matching takes the round, and the first to
// MATCH_POINTS round wins takes the match. Pure reducers — all timing lives in the caller's
// rAF loop, which just feeds these functions the per-frame match scores and delta time.

export type Side = "a" | "b";

// First to this many round wins takes the match (best-of-5).
export const MATCH_POINTS = 3;
// A player must hold the pose for this long (ms of accumulated matching) to win a round.
export const HOLD_MS = 900;
// Per-frame pose score at/above which a player counts as "holding" the pose this frame.
export const MATCH_THRESHOLD = 0.7;
// After a round is won, freeze scoring this long to show the result before the next pose.
export const ROUND_BREAK_MS = 1600;
// Holds bleed away faster than they build, so a wobble costs you.
export const DECAY_FACTOR = 1.6;

// Advance a player's hold accumulator (ms toward HOLD_MS), clamped to [0, HOLD_MS].
export function advanceHold(hold: number, matched: boolean, dtMs: number): number {
  const next = matched ? hold + dtMs : hold - dtMs * DECAY_FACTOR;
  return Math.max(0, Math.min(HOLD_MS, next));
}

// Which player (if any) has completed the hold this frame. If both cross in the same frame the
// one strictly further along wins; an exact tie yields null so the round simply continues.
export function roundWinner(aHold: number, bHold: number): Side | null {
  const aDone = aHold >= HOLD_MS;
  const bDone = bHold >= HOLD_MS;
  if (aDone && bDone) return aHold === bHold ? null : aHold > bHold ? "a" : "b";
  if (aDone) return "a";
  if (bDone) return "b";
  return null;
}

// Which player (if any) has won the match.
export function matchWinner(aWins: number, bWins: number): Side | null {
  if (aWins >= MATCH_POINTS) return "a";
  if (bWins >= MATCH_POINTS) return "b";
  return null;
}
