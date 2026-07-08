// The multiplayer core: MediaPipe returns up to two detected bodies per frame with no stable
// identity, so we assign them to player A / player B by torso position. Sorting by torso
// centroid-x keeps each player's slot stable frame-to-frame as long as they don't cross over.
// Pure — no DOM — so the assignment rules are unit-tested directly.
import { LM } from "../pose";
import { MIN_VISIBILITY, type Point } from "./angles";

export type Landmarks = (Point | null | undefined)[];

const TORSO = [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP];

// Mean x of the visible torso landmarks (0..1), or null if the torso isn't visible enough to
// place the body on one side of the frame.
export function centroidX(landmarks: Landmarks): number | null {
  let sum = 0;
  let n = 0;
  for (const i of TORSO) {
    const p = landmarks[i];
    if (p && (p.visibility ?? 1) >= MIN_VISIBILITY) {
      sum += p.x;
      n += 1;
    }
  }
  return n ? sum / n : null;
}

export type Players = { a: Landmarks | null; b: Landmarks | null };

// Assign 0..2 detected poses to the two player slots. Two bodies -> the left-of-frame one is A,
// the right one is B. A lone body takes the slot for the half it stands in, so a single player
// can still warm up (and test both slots by stepping across the frame). Bodies with no locatable
// torso are dropped.
export function assignPlayers(poses: Landmarks[]): Players {
  const scored = poses
    .map((lm) => ({ lm, cx: centroidX(lm) }))
    .filter((p): p is { lm: Landmarks; cx: number } => p.cx !== null)
    .sort((p, q) => p.cx - q.cx);

  if (scored.length === 0) return { a: null, b: null };
  if (scored.length === 1) {
    return scored[0].cx < 0.5 ? { a: scored[0].lm, b: null } : { a: null, b: scored[0].lm };
  }
  return { a: scored[0].lm, b: scored[1].lm };
}
