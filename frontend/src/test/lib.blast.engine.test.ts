import { describe, it, expect } from "vitest";
import {
  createGameState,
  stepFrame,
  BEAM_MS,
  FLASH_MS,
  type GameState,
} from "../lib/blast/engine";
import { CHARGE_MS, DECAY_MS } from "../lib/blast/charger";
import { MEME_EMOJIS, type Target } from "../lib/blast/targets";
import type { HandState } from "../lib/blast/gestures";

const noHand: HandState = { valid: false, gap: 0, aimY: 0.5 };
const chargingHand: HandState = { valid: true, gap: 0.2, aimY: 0.5 };
const fireHand = (aimY = 0.5): HandState => ({ valid: true, gap: 2.5, aimY });

function orb(over: Partial<Target> = {}): Target {
  return { id: 1, emoji: "💀", x: 0.5, y: 0.5, speed: 0.2, ...over };
}

// A state that won't spawn during the frame under test.
function idle(now: number, over: Partial<GameState> = {}): GameState {
  return { ...createGameState(now), spawnAt: now + 1e9, ...over };
}

const input = (hand: HandState, over: Partial<{ dtMs: number; now: number; rng: () => number }> = {}) => ({
  hand,
  dtMs: 16,
  now: 1000,
  rng: () => 0,
  ...over,
});

describe("createGameState", () => {
  it("starts empty at the given round start", () => {
    const s = createGameState(500);
    expect(s.score).toBe(0);
    expect(s.combo).toBe(0);
    expect(s.targets).toEqual([]);
    expect(s.spawnAt).toBe(500);
    expect(s.roundStart).toBe(500);
    expect(s.beam).toBeNull();
  });
});

describe("stepFrame — charge/decay", () => {
  it("builds charge while hands are together", () => {
    const out = stepFrame(idle(1000), input(chargingHand, { dtMs: CHARGE_MS / 2 }));
    expect(out.charge.charge).toBeCloseTo(0.5, 5);
  });

  it("decays charge when no valid hand this frame", () => {
    const out = stepFrame(idle(1000, { charge: { charge: 0.5 } }), input(noHand, { dtMs: DECAY_MS / 5 }));
    expect(out.charge.charge).toBeCloseTo(0.3, 5);
  });
});

describe("stepFrame — fire", () => {
  it("destroys an orb in the beam band and scores with a combo bump", () => {
    const state = idle(1000, { charge: { charge: 1 }, targets: [orb({ y: 0.5 })] });
    const out = stepFrame(state, input(fireHand(0.5)));
    expect(out.combo).toBe(1);
    expect(out.hits).toBe(1);
    expect(out.score).toBeGreaterThan(0);
    expect(out.targets).toEqual([]);
    expect(out.flash).toEqual({ hits: 1, points: out.score });
    expect(out.beam).toEqual({ y: 0.5, until: 1000 + BEAM_MS });
    expect(out.flashUntil).toBe(1000 + FLASH_MS);
    expect(out.charge.charge).toBe(0);
  });

  it("wipes several orbs in one beam (multi-kill)", () => {
    const state = idle(1000, {
      charge: { charge: 1 },
      targets: [orb({ id: 1, y: 0.5 }), orb({ id: 2, y: 0.52 })],
    });
    const out = stepFrame(state, input(fireHand(0.5)));
    expect(out.hits).toBe(2);
    expect(out.flash).toEqual({ hits: 2, points: out.score });
    expect(out.targets).toEqual([]);
  });

  it("breaks the combo and scores nothing on a whiff", () => {
    const state = idle(1000, { charge: { charge: 1 }, combo: 4, targets: [orb({ y: 0.95 })] });
    const out = stepFrame(state, input(fireHand(0.2)));
    expect(out.combo).toBe(0);
    expect(out.score).toBe(0);
    expect(out.flash).toEqual({ hits: 0, points: 0 });
    expect(out.targets).toHaveLength(1); // the missed orb survives
  });

  it("tracks the best combo reached", () => {
    const state = idle(1000, {
      charge: { charge: 1 },
      combo: 2,
      bestCombo: 2,
      targets: [orb({ y: 0.5 })],
    });
    const out = stepFrame(state, input(fireHand(0.5)));
    expect(out.combo).toBe(3);
    expect(out.bestCombo).toBe(3);
  });
});

describe("stepFrame — spawning & physics", () => {
  it("spawns an orb once the spawn timer elapses, using injected rng", () => {
    const state = createGameState(0); // spawnAt = 0
    const out = stepFrame(state, input(noHand, { now: 100, rng: () => 0.5 }));
    expect(out.targets).toHaveLength(1);
    expect(out.nextId).toBe(2);
    expect(out.spawnAt).toBeGreaterThan(100);
    // y = 0.15 + 0.5*0.7; emoji index = floor(0.5 * 10)
    expect(out.targets[0].y).toBeCloseTo(0.5, 5);
    expect(out.targets[0].emoji).toBe(MEME_EMOJIS[5]);
  });

  it("advances existing orbs and drops those that leave the screen", () => {
    const state = idle(1000, { targets: [orb({ x: -0.1, speed: 1 })] });
    const out = stepFrame(state, input(noHand, { dtMs: 1000 }));
    expect(out.targets).toHaveLength(0);
  });
});

describe("stepFrame — transient timers", () => {
  it("clears a beam past its lifetime", () => {
    const state = idle(1000, { beam: { y: 0.5, until: 500 } });
    const out = stepFrame(state, input(noHand, { now: 600 }));
    expect(out.beam).toBeNull();
  });

  it("clears an expired flash", () => {
    const state = idle(1000, { flash: { hits: 1, points: 100 }, flashUntil: 500 });
    const out = stepFrame(state, input(noHand, { now: 600 }));
    expect(out.flash).toBeNull();
    expect(out.flashUntil).toBe(0);
  });
});

describe("stepFrame — purity", () => {
  it("does not mutate the input state", () => {
    const state = idle(1000, { charge: { charge: 0.4 }, targets: [orb()] });
    const chargeRef = state.charge;
    const targetsRef = state.targets;
    stepFrame(state, input(chargingHand));
    expect(state.charge).toBe(chargeRef);
    expect(state.charge.charge).toBe(0.4);
    expect(state.targets).toBe(targetsRef);
    expect(state.targets).toHaveLength(1);
  });
});
