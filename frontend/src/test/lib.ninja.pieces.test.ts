import { describe, it, expect } from "vitest";
import {
  spawnPieces,
  advancePieces,
  PIECE_LIFE,
  SPLIT_SPEED,
  type Piece,
} from "../lib/ninja/pieces";
import { GRAVITY } from "../lib/ninja/physics";

const fruit = { emoji: "🍉", x: 0.5, y: 0.5, vx: 0, vy: 0, radius: 0.075 };

describe("spawnPieces", () => {
  it("makes a left and a right half at the fruit, with a fresh id run", () => {
    const { pieces, nextId } = spawnPieces(7, fruit, 1, 0, () => 0);
    expect(pieces).toHaveLength(2);
    expect(pieces.map((p) => p.half)).toEqual(["left", "right"]);
    expect(pieces.map((p) => p.id)).toEqual([7, 8]);
    expect(nextId).toBe(9);
    pieces.forEach((p) => {
      expect(p.x).toBe(fruit.x);
      expect(p.y).toBe(fruit.y);
      expect(p.emoji).toBe("🍉");
      expect(p.life).toBe(1);
    });
  });

  it("pushes the two halves apart along the blade normal", () => {
    // Horizontal blade (dir +x) → halves separate vertically (±y) by SPLIT_SPEED.
    const { pieces } = spawnPieces(1, fruit, 1, 0, () => 0);
    expect(pieces[0].vy - pieces[1].vy).toBeCloseTo(2 * SPLIT_SPEED, 6);
    expect(pieces[0].vx).toBeCloseTo(0, 6);
    expect(pieces[1].vx).toBeCloseTo(0, 6);
  });

  it("falls back to a horizontal split for a zero-length blade", () => {
    const { pieces } = spawnPieces(1, fruit, 0, 0, () => 0);
    // Separation along x, seam unrotated.
    expect(pieces[0].vx - pieces[1].vx).toBeCloseTo(2 * SPLIT_SPEED, 6);
    expect(pieces[0].rot).toBe(0);
  });

  it("gives the halves opposite spin", () => {
    const { pieces } = spawnPieces(1, fruit, 1, 0, () => 0.5);
    expect(Math.sign(pieces[0].spin)).toBe(-Math.sign(pieces[1].spin));
  });
});

describe("advancePieces", () => {
  const base: Piece = {
    id: 1,
    emoji: "🍎",
    x: 0.5,
    y: 0.5,
    vx: 0.1,
    vy: 0,
    rot: 0,
    spin: 2,
    radius: 0.075,
    half: "left",
    life: 1,
  };

  it("moves, spins, and applies gravity", () => {
    const [p] = advancePieces([base], 0.1);
    expect(p.x).toBeCloseTo(0.5 + 0.1 * 0.1, 6);
    expect(p.vy).toBeCloseTo(GRAVITY * 0.1, 6);
    expect(p.rot).toBeCloseTo(0.2, 6);
    expect(p.life).toBeCloseTo(1 - 0.1 / PIECE_LIFE, 6);
  });

  it("drops pieces once their life runs out", () => {
    expect(advancePieces([{ ...base, life: 0.05 }], 0.1)).toHaveLength(0);
  });

  it("keeps living pieces", () => {
    expect(advancePieces([base], 0.1)).toHaveLength(1);
  });
});
