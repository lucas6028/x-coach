// The per-frame game loop for Meme Blaster, as a pure reducer. The page's rAF callback owns
// only the impure edges — reading the camera frame, drawing the canvas, and throttling React
// state — and delegates every state transition (charge/fire, spawning, physics, scoring, and
// the fleeting beam/flash timers) to stepFrame here, so all of it is unit-tested without a
// camera or GPU. Randomness (spawn height/emoji) is injected via `rng` so tests stay
// deterministic.
import { stepCharge, DECAY_MS, initialCharge, type ChargeState } from "./charger";
import type { HandState } from "./gestures";
import { advanceTargets, beamHits, makeTarget, MEME_EMOJIS, type Target } from "./targets";
import { blastPoints, difficulty, ROUND_SECONDS } from "./scoring";

// How long (ms) a fired beam stays drawn, and a hit/miss flash stays on screen.
export const BEAM_MS = 220;
export const FLASH_MS = 700;

export type Beam = { y: number; until: number };
export type Flash = { hits: number; points: number };

export type GameState = {
  charge: ChargeState;
  targets: Target[];
  nextId: number;
  // Timestamp (ms) at/after which the next orb spawns.
  spawnAt: number;
  score: number;
  combo: number;
  bestCombo: number;
  hits: number;
  beam: Beam | null;
  flash: Flash | null;
  // Timestamp (ms) at which the current flash expires.
  flashUntil: number;
  // Round start timestamp (ms) — drives the difficulty ramp.
  roundStart: number;
};

// A fresh game state for a round starting at `now`.
export function createGameState(now: number): GameState {
  return {
    charge: initialCharge,
    targets: [],
    nextId: 1,
    spawnAt: now,
    score: 0,
    combo: 0,
    bestCombo: 0,
    hits: 0,
    beam: null,
    flash: null,
    flashUntil: 0,
    roundStart: now,
  };
}

export type FrameInput = {
  // The player's hand geometry this frame (valid=false when no fresh/usable detection).
  hand: HandState;
  // Milliseconds since the previous frame.
  dtMs: number;
  // Current timestamp (ms).
  now: number;
  // Random source in [0, 1) for spawn height + emoji; injected for deterministic tests.
  rng: () => number;
};

// Advance the whole game by one frame. Returns a new GameState; the input is not mutated.
export function stepFrame(state: GameState, input: FrameInput): GameState {
  const { hand, dtMs, now, rng } = input;
  const s: GameState = { ...state };

  // Charge / fire.
  if (hand.valid) {
    const stepped = stepCharge(s.charge, hand.gap, dtMs);
    s.charge = stepped.state;
    if (stepped.fired) {
      const { hit, remaining } = beamHits(s.targets, hand.aimY);
      if (hit.length > 0) {
        s.combo += 1;
        s.bestCombo = Math.max(s.bestCombo, s.combo);
        s.hits += hit.length;
        const pts = blastPoints(hit.length, s.combo);
        s.score += pts;
        s.targets = remaining;
        s.flash = { hits: hit.length, points: pts };
      } else {
        s.combo = 0;
        s.flash = { hits: 0, points: 0 };
      }
      s.beam = { y: hand.aimY, until: now + BEAM_MS };
      s.flashUntil = now + FLASH_MS;
    }
  } else {
    s.charge = { charge: Math.max(0, s.charge.charge - dtMs / DECAY_MS) };
  }

  // Spawn + advance orbs, ramping with round progress.
  const frac = (now - s.roundStart) / (ROUND_SECONDS * 1000);
  const diff = difficulty(frac);
  if (now >= s.spawnAt) {
    const y = 0.15 + rng() * 0.7;
    const emoji = MEME_EMOJIS[Math.floor(rng() * MEME_EMOJIS.length)];
    s.targets = [...s.targets, makeTarget(s.nextId, y, diff.speed, emoji)];
    s.nextId += 1;
    s.spawnAt = now + diff.spawnMs;
  }
  s.targets = advanceTargets(s.targets, dtMs / 1000).targets;

  // Expire the transient beam + flash.
  if (s.beam && now >= s.beam.until) s.beam = null;
  if (s.flashUntil && now >= s.flashUntil) {
    s.flash = null;
    s.flashUntil = 0;
  }

  return s;
}
