// The per-frame round/match transition for Pose Duel, as a pure reducer. The page's rAF loop
// owns only the impure edges — reading the camera frame, scoring each player's pose (via the
// already-tested scorePose), drawing, and throttling React state — and hands stepRound the two
// "is this player matching now?" booleans plus dt. Everything about advancing holds, awarding a
// round, picking the next pose, entering the break, and ending the match lives here so it's
// unit-tested without a camera. Randomness (next pose) is injected via `rng` for determinism.
import { advanceHold, roundWinner, matchWinner, ROUND_BREAK_MS, type Side } from "./match";
import { pickPose } from "./poses";

export type RoundState = {
  poseId: string;
  winsA: number;
  winsB: number;
  // ms of accumulated matching toward HOLD_MS, per player.
  holdA: number;
  holdB: number;
  // Timestamp (ms) until which scoring is frozen after a round is taken.
  breakUntil: number;
  // Who took the most recent round (shown during the break), or null.
  roundFlash: Side | null;
};

// A fresh match with the first target pose already chosen.
export function createRoundState(rng: () => number): RoundState {
  return {
    poseId: pickPose(null, rng()).id,
    winsA: 0,
    winsB: 0,
    holdA: 0,
    holdB: 0,
    breakUntil: 0,
    roundFlash: null,
  };
}

export type RoundInput = {
  // Whether each player is holding the target pose this frame (page computes via scorePose).
  matchedA: boolean;
  matchedB: boolean;
  dtMs: number;
  now: number;
  // Random source in [0, 1) for the next pose; injected for deterministic tests.
  rng: () => number;
};

export type RoundResult = {
  state: RoundState;
  // Set when this frame ends the whole match.
  matchOver: Side | null;
};

// Advance the match by one frame. During the post-round break scoring is frozen. Returns a new
// state (the input is not mutated) and, when the match is decided, the winning side.
export function stepRound(state: RoundState, input: RoundInput): RoundResult {
  const { matchedA, matchedB, dtMs, now, rng } = input;
  const s: RoundState = { ...state };

  if (now < s.breakUntil) return { state: s, matchOver: null };

  s.holdA = advanceHold(s.holdA, matchedA, dtMs);
  s.holdB = advanceHold(s.holdB, matchedB, dtMs);

  const w = roundWinner(s.holdA, s.holdB);
  if (w) {
    if (w === "a") s.winsA += 1;
    else s.winsB += 1;
    s.holdA = 0;
    s.holdB = 0;
    s.roundFlash = w;
    const mw = matchWinner(s.winsA, s.winsB);
    if (mw) return { state: s, matchOver: mw };
    s.poseId = pickPose(s.poseId, rng()).id;
    s.breakUntil = now + ROUND_BREAK_MS;
  }

  return { state: s, matchOver: null };
}
