import { describe, expect, it } from "vitest";
import {
  advanceWorld,
  createWebGameState,
  detectWebFlick,
  fireWeb,
  type WebGameState,
} from "../lib/webslinger/engine";

describe("web slinger engine", () => {
  it("detects a fast outward hand flick", () => {
    const ray = detectWebFlick(
      { x: 0.5, y: 0.7 },
      { x: 0.5, y: 0.35 },
      { x: 0.5, y: 0.39 },
      40
    );
    expect(ray).not.toBeNull();
    expect(ray?.y2).toBeLessThan(ray?.y ?? 1);
  });

  it("ignores a slow or inward wrist movement", () => {
    expect(
      detectWebFlick({ x: 0.5, y: 0.7 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.351 }, 40)
    ).toBeNull();
    expect(
      detectWebFlick({ x: 0.5, y: 0.7 }, { x: 0.5, y: 0.4 }, { x: 0.5, y: 0.3 }, 40)
    ).toBeNull();
  });

  it("hits the closest target on the web ray and awards combo points", () => {
    const state: WebGameState = {
      targets: [
        { id: 1, x: 0.5, y: 0.5, vx: 0, vy: 0, radius: 0.05 },
        { id: 2, x: 0.5, y: 0.25, vx: 0, vy: 0, radius: 0.05 },
      ],
      traces: [],
      score: 0,
      combo: 0,
      bestCombo: 0,
      hits: 0,
      nextId: 3,
      nextSpawnAt: 1000,
    };
    const first = fireWeb(state, { x: 0.5, y: 0.8, x2: 0.5, y2: 0 });
    expect(first.targets.map((target) => target.id)).toEqual([2]);
    expect(first.score).toBe(100);
    expect(first.combo).toBe(1);
    expect(first.traces[0]).toMatchObject({ hit: true, progress: 0, hand: 0 });
    const second = fireWeb(first, { x: 0.5, y: 0.8, x2: 0.5, y2: 0 });
    expect(second.score).toBe(225);
    expect(second.bestCombo).toBe(2);
  });

  it("resets the combo when a shot misses", () => {
    const state = createWebGameState(0, () => 0.5);
    const hit = { ...state, combo: 4, bestCombo: 4 };
    const missed = fireWeb(hit, { x: 0, y: 0, x2: 0.05, y2: 0.05 });
    expect(missed.combo).toBe(0);
    expect(missed.bestCombo).toBe(4);
  });

  it("moves targets, fades traces, and replenishes the board", () => {
    const state = createWebGameState(0, () => 0.5);
    const thinned = { ...state, targets: state.targets.slice(0, 1), nextSpawnAt: 10 };
    const next = advanceWorld(thinned, 50, 20, () => 0.5);
    expect(next.targets).toHaveLength(2);
    expect(next.nextId).toBe(thinned.nextId + 1);
  });

  it("advances the silk head before fading an attached web", () => {
    const state = createWebGameState(0, () => 0.5);
    const fired = fireWeb(
      {
        ...state,
        targets: [{ id: 7, x: 0.5, y: 0.3, vx: 0, vy: 0, radius: 0.05 }],
      },
      { x: 0.5, y: 0.8, x2: 0.5, y2: 0, hand: 1 }
    );
    const advanced = advanceWorld(fired, 50, 50, () => 0.5);
    expect(advanced.traces[0].progress).toBeGreaterThan(0);
    expect(advanced.traces[0].progress).toBeLessThan(1);
    expect(advanced.traces[0].life).toBeGreaterThan(0.95);
    expect(advanced.traces[0].hand).toBe(1);
  });
});
