import { describe, it, expect } from "vitest";
import {
  advanceEntities,
  spawnWave,
  GRAVITY,
  FRUITS,
  type Entity,
} from "../lib/ninja/physics";

function fruit(over: Partial<Entity> = {}): Entity {
  return { id: 1, kind: "fruit", emoji: "🍉", x: 0.5, y: 0.5, vx: 0, vy: 0, radius: 0.075, ...over };
}

// A deterministic rng that walks a fixed list (clamped at the end).
function seq(values: number[]): () => number {
  let i = 0;
  return () => values[Math.min(i++, values.length - 1)];
}

describe("advanceEntities", () => {
  it("applies gravity and integrates position", () => {
    const { entities } = advanceEntities([fruit({ vy: -1 })], 0.1);
    expect(entities).toHaveLength(1);
    expect(entities[0].vy).toBeCloseTo(-1 + GRAVITY * 0.1, 6);
    expect(entities[0].y).toBeCloseTo(0.5 + (-1 + GRAVITY * 0.1) * 0.1, 6);
  });

  it("drops a fruit that falls off the bottom and counts it", () => {
    const { entities, droppedFruits } = advanceEntities([fruit({ y: 1.15, vy: 1 })], 0.3);
    expect(entities).toHaveLength(0);
    expect(droppedFruits).toBe(1);
  });

  it("does not count a bomb that falls away", () => {
    const bomb = fruit({ kind: "bomb", emoji: "💣", y: 1.15, vy: 1 });
    const { entities, droppedFruits } = advanceEntities([bomb], 0.3);
    expect(entities).toHaveLength(0);
    expect(droppedFruits).toBe(0);
  });

  it("keeps an entity still on screen", () => {
    const { entities, droppedFruits } = advanceEntities([fruit({ y: 0.5, vy: -1 })], 0.1);
    expect(entities).toHaveLength(1);
    expect(droppedFruits).toBe(0);
  });
});

describe("spawnWave", () => {
  it("spawns a single fruit with the injected parameters", () => {
    // count→1, isBomb(0.5→fruit), x(0.5), vx(0.5→0), vy(0.5), emoji(0)
    const { entities, nextId } = spawnWave(7, seq([0, 0.5, 0.5, 0.5, 0.5, 0]));
    expect(entities).toHaveLength(1);
    expect(nextId).toBe(8);
    const e = entities[0];
    expect(e.kind).toBe("fruit");
    expect(e.id).toBe(7);
    expect(e.emoji).toBe(FRUITS[0]);
    expect(e.x).toBeCloseTo(0.5, 6);
    expect(e.vx).toBeCloseTo(0, 6);
    expect(e.vy).toBeLessThan(0); // launched upward
    expect(e.y).toBeCloseTo(1.08, 6);
  });

  it("spawns a bomb when the roll is below the bomb chance", () => {
    // count→1, isBomb(0.05<0.1→bomb), x, vx, vy
    const { entities } = spawnWave(1, seq([0, 0.05, 0.5, 0.5, 0.5]));
    expect(entities[0].kind).toBe("bomb");
    expect(entities[0].emoji).toBe("💣");
  });

  it("can spawn a wave of several and increments ids", () => {
    // count→3 (0.9*3=2.7→2, +1=3); then 3 fruit blocks
    const fruitBlock = [0.5, 0.5, 0.5, 0.5, 0];
    const { entities, nextId } = spawnWave(1, seq([0.9, ...fruitBlock, ...fruitBlock, ...fruitBlock]));
    expect(entities).toHaveLength(3);
    expect(entities.map((e) => e.id)).toEqual([1, 2, 3]);
    expect(nextId).toBe(4);
  });
});
