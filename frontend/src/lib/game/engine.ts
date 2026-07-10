// The per-frame scoring loop for Pose Match Rush, as a pure reducer. The page's rAF callback
// owns only the impure edges — running detection, drawing the skeleton, and throttling React
// state — and hands stepGame the frame's match quality. Everything about holding a pose,
// locking it in (combo/score/next target), and dropping the streak lives here so it's
// unit-tested without a camera. The next-target RNG is injected for determinism.
import { POSES } from "./poses";
import { gradeFor, hitPoints, HIT_THRESHOLD, HOLD_MS, type Grade } from "./scoring";

// How long (ms) the grade badge stays up after a lock-in.
export const GRADE_MS = 850;
// Below HIT_THRESHOLD * this, a mid-hold drop is decisive enough to break the combo.
export const DROP_FACTOR = 0.6;

export type GameState = {
  score: number;
  combo: number;
  bestCombo: number;
  // Poses locked in this round.
  poses: number;
  // Current target pose id.
  targetId: string;
  // Timestamp (ms) the current above-threshold hold began, or 0.
  holdStart: number;
  // Timestamp (ms) until which the last grade badge shows, or 0.
  gradeUntil: number;
};

export function createGameState(): GameState {
  return {
    score: 0,
    combo: 0,
    bestCombo: 0,
    poses: 0,
    targetId: POSES[0].id,
    holdStart: 0,
    gradeUntil: 0,
  };
}

// Pick a target other than `currentId`. `r` is a random float in [0, 1).
export function nextPoseId(currentId: string, r: number): string {
  const pool = POSES.filter((p) => p.id !== currentId);
  const list = pool.length ? pool : POSES;
  return list[Math.min(list.length - 1, Math.floor(r * list.length))].id;
}

export type FrameInput = {
  // scorePose(...).score for this frame against the current target.
  quality: number;
  // Whether a body was detected this frame.
  hasLandmarks: boolean;
  now: number;
  // Random source in [0, 1) for the next target; injected for deterministic tests.
  rng: () => number;
};

export type StepResult = {
  state: GameState;
  // The grade to flash when a pose locks in this frame, else null.
  grade: Grade | null;
};

// Advance the game by one detected frame. Returns a new state (the input is not mutated) and,
// on a lock-in, the grade to flash. Should be called only on frames where detection ran.
export function stepGame(state: GameState, input: FrameInput): StepResult {
  const { quality, hasLandmarks, now, rng } = input;
  const s: GameState = { ...state };
  let grade: Grade | null = null;

  if (!hasLandmarks) {
    s.holdStart = 0;
    return { state: s, grade };
  }

  if (quality >= HIT_THRESHOLD) {
    if (s.holdStart === 0) s.holdStart = now;
    if (now - s.holdStart >= HOLD_MS) {
      // Lock the pose in.
      s.combo += 1;
      s.bestCombo = Math.max(s.bestCombo, s.combo);
      s.poses += 1;
      s.score += hitPoints(quality, s.combo);
      s.gradeUntil = now + GRADE_MS;
      grade = gradeFor(quality);
      s.holdStart = 0;
      s.targetId = nextPoseId(s.targetId, rng());
    }
  } else {
    // Dropping the pose mid-hold (clearly, not a flicker) resets the streak.
    if (s.holdStart !== 0 && quality < HIT_THRESHOLD * DROP_FACTOR) s.combo = 0;
    s.holdStart = 0;
  }

  return { state: s, grade };
}
