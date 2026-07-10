// The per-frame game loop for Fruit Ninja, as a pure reducer. The page's rAF callback owns only
// the impure edges — detecting the wrists, building blade segments, drawing, and throttling React
// state — and hands stepGame the active blades and dt. Slicing, scoring, life loss, spawning, and
// game-over all live here so they're unit-tested without a camera. Randomness (spawns) is
// injected via `rng` for determinism.
import { advanceEntities, spawnWave, type Entity } from "./physics";
import { sliceEntities, type Blade } from "./slice";
import { START_LIVES, COMBO_WINDOW_MS, sliceScore } from "./scoring";

// Spawn cadence: starts slow, ramps toward the floor as the player racks up cuts.
export const SPAWN_BASE_MS = 1300;
export const SPAWN_MIN_MS = 520;
export const SPAWN_RAMP_MS = 8;

export type GameState = {
  entities: Entity[];
  nextId: number;
  // Timestamp (ms) the next wave launches.
  spawnAt: number;
  score: number;
  combo: number;
  bestCombo: number;
  // Total fruits cut this round.
  sliced: number;
  lives: number;
  // Timestamp (ms) of the last cut, for the combo window.
  lastSliceAt: number;
  over: boolean;
};

export function createGameState(now: number): GameState {
  return {
    entities: [],
    nextId: 1,
    spawnAt: now,
    score: 0,
    combo: 0,
    bestCombo: 0,
    sliced: 0,
    lives: START_LIVES,
    lastSliceAt: 0,
    over: false,
  };
}

export type FrameInput = {
  // Active (fast-enough) wrist blades this frame.
  blades: Blade[];
  dtMs: number;
  now: number;
  rng: () => number;
};

export type StepResult = {
  state: GameState;
  // Fruits cut this frame (drives the "+N" pop), and whether a bomb just went off.
  sliceFlash: number;
  bombFlash: boolean;
  // The actual fruits cut this frame, so the page can burst each into flying halves.
  slicedFruits: Entity[];
};

// Advance the game by one frame. Returns a new state (the input is not mutated).
export function stepGame(state: GameState, input: FrameInput): StepResult {
  const { blades, dtMs, now, rng } = input;
  const s: GameState = { ...state };
  if (s.over) return { state: s, sliceFlash: 0, bombFlash: false, slicedFruits: [] };

  const dt = dtMs / 1000;

  // 1) Slice against current positions.
  let sliceFlash = 0;
  let cutFruits: Entity[] = [];
  if (blades.length > 0) {
    const { slicedFruits, bombHit, remaining } = sliceEntities(s.entities, blades);
    s.entities = remaining;
    if (slicedFruits.length > 0) {
      cutFruits = slicedFruits;
      const n = slicedFruits.length;
      const inRhythm = now - s.lastSliceAt <= COMBO_WINDOW_MS;
      s.combo = (inRhythm ? s.combo : 0) + n;
      s.bestCombo = Math.max(s.bestCombo, s.combo);
      s.score += sliceScore(n, s.combo);
      s.sliced += n;
      s.lastSliceAt = now;
      sliceFlash = n;
    }
    // A swipe can cut a bomb and adjacent fruits in the same frame — award/burst those fruits
    // before ending the round, instead of discarding them.
    if (bombHit) {
      s.over = true;
      return { state: s, sliceFlash, bombFlash: true, slicedFruits: cutFruits };
    }
  }

  // 2) Advance physics; missed fruits cost lives (and break the combo).
  const adv = advanceEntities(s.entities, dt);
  s.entities = adv.entities;
  if (adv.droppedFruits > 0) {
    s.lives -= adv.droppedFruits;
    s.combo = 0;
    if (s.lives <= 0) {
      s.lives = 0;
      s.over = true;
    }
  }

  // 3) Launch the next wave on cadence.
  if (!s.over && now >= s.spawnAt) {
    const wave = spawnWave(s.nextId, rng);
    s.entities = [...s.entities, ...wave.entities];
    s.nextId = wave.nextId;
    const interval = Math.max(SPAWN_MIN_MS, SPAWN_BASE_MS - s.sliced * SPAWN_RAMP_MS);
    s.spawnAt = now + interval;
  }

  return { state: s, sliceFlash, bombFlash: false, slicedFruits: cutFruits };
}
