import { describe, it, expect } from "vitest";
import {
  createGameState,
  stepGame,
  SPAWN_BASE_MS,
  type GameState,
} from "../lib/ninja/engine";
import { START_LIVES, sliceScore } from "../lib/ninja/scoring";
import type { Entity } from "../lib/ninja/physics";
import type { Blade } from "../lib/ninja/slice";

function fruit(over: Partial<Entity> = {}): Entity {
  return { id: 1, kind: "fruit", emoji: "🍉", x: 0.5, y: 0.5, vx: 0, vy: 0, radius: 0.075, ...over };
}

const through: Blade = { x1: 0.4, y1: 0.5, x2: 0.6, y2: 0.5 };

// A state that won't spawn during the frame under test.
function idle(now: number, over: Partial<GameState> = {}): GameState {
  return { ...createGameState(now), spawnAt: now + 1e9, ...over };
}

const input = (
  over: Partial<{ blades: Blade[]; dtMs: number; now: number; rng: () => number }> = {}
) => ({ blades: [], dtMs: 16, now: 1000, rng: () => 0, ...over });

describe("createGameState", () => {
  it("starts empty with full lives", () => {
    const s = createGameState(500);
    expect(s.score).toBe(0);
    expect(s.lives).toBe(START_LIVES);
    expect(s.over).toBe(false);
    expect(s.spawnAt).toBe(500);
  });
});

describe("stepGame", () => {
  it("is a no-op once the game is over", () => {
    const s = idle(1000, { over: true, score: 42 });
    const out = stepGame(s, input({ blades: [through], now: 1000 }));
    expect(out.state.score).toBe(42);
    expect(out.sliceFlash).toBe(0);
  });

  it("cuts a fruit: scores, bumps combo, and flashes", () => {
    const s = idle(1000, { entities: [fruit()] });
    const out = stepGame(s, input({ blades: [through] }));
    expect(out.state.score).toBe(sliceScore(1, 1));
    expect(out.state.combo).toBe(1);
    expect(out.state.sliced).toBe(1);
    expect(out.state.entities).toHaveLength(0);
    expect(out.sliceFlash).toBe(1);
    // The cut fruit is surfaced so the page can burst it into halves.
    expect(out.slicedFruits.map((e) => e.id)).toEqual([1]);
  });

  it("reports no cut fruits on an empty swing or a bomb hit", () => {
    const empty = stepGame(idle(1000, { entities: [fruit()] }), input({ blades: [] }));
    expect(empty.slicedFruits).toEqual([]);
    const bomb = stepGame(
      idle(1000, { entities: [fruit({ kind: "bomb", emoji: "💣" })] }),
      input({ blades: [through] })
    );
    expect(bomb.slicedFruits).toEqual([]);
  });

  it("cuts several fruits in one swipe for a multi bonus", () => {
    const s = idle(1000, {
      entities: [fruit({ id: 1, x: 0.45 }), fruit({ id: 2, x: 0.5 }), fruit({ id: 3, x: 0.55 })],
    });
    const blade: Blade = { x1: 0.4, y1: 0.5, x2: 0.6, y2: 0.5 };
    const out = stepGame(s, input({ blades: [blade] }));
    expect(out.state.sliced).toBe(3);
    expect(out.state.combo).toBe(3);
    expect(out.sliceFlash).toBe(3);
    expect(out.state.score).toBe(sliceScore(3, 3));
  });

  it("ends the game and flashes when a bomb is struck", () => {
    const s = idle(1000, { entities: [fruit({ kind: "bomb", emoji: "💣" })] });
    const out = stepGame(s, input({ blades: [through] }));
    expect(out.state.over).toBe(true);
    expect(out.bombFlash).toBe(true);
    expect(out.state.score).toBe(0);
  });

  it("loses a life and breaks the combo when a fruit drops", () => {
    const s = idle(1000, { entities: [fruit({ y: 1.15, vy: 1 })], combo: 5, lives: 3 });
    const out = stepGame(s, input({ dtMs: 300 }));
    expect(out.state.lives).toBe(2);
    expect(out.state.combo).toBe(0);
    expect(out.state.over).toBe(false);
  });

  it("ends the game when the last life is lost", () => {
    const s = idle(1000, { entities: [fruit({ y: 1.15, vy: 1 })], lives: 1 });
    const out = stepGame(s, input({ dtMs: 300 }));
    expect(out.state.lives).toBe(0);
    expect(out.state.over).toBe(true);
  });

  it("launches a wave once the spawn timer elapses", () => {
    const s = createGameState(0); // spawnAt = 0
    const out = stepGame(s, input({ now: 100, rng: () => 0 }));
    expect(out.state.entities.length).toBeGreaterThanOrEqual(1);
    expect(out.state.spawnAt).toBe(100 + SPAWN_BASE_MS);
  });

  it("does not mutate the input state", () => {
    const s = idle(1000, { entities: [fruit()], score: 5 });
    const before = s.score;
    const entitiesRef = s.entities;
    stepGame(s, input({ blades: [through] }));
    expect(s.score).toBe(before);
    expect(s.entities).toBe(entitiesRef);
    expect(s.entities).toHaveLength(1);
  });
});
