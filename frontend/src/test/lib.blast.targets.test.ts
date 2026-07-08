import { describe, it, expect } from "vitest";
import {
  makeTarget,
  advanceTargets,
  beamHits,
  MEME_EMOJIS,
  BEAM_HALF_HEIGHT,
  type Target,
} from "../lib/blast/targets";

describe("makeTarget", () => {
  it("spawns just off the right edge", () => {
    const t = makeTarget(1, 0.5, 0.2, "💀");
    expect(t.x).toBeGreaterThan(1);
    expect(t).toMatchObject({ id: 1, y: 0.5, speed: 0.2, emoji: "💀" });
  });
});

describe("advanceTargets", () => {
  it("moves targets left by speed × dt", () => {
    const { targets } = advanceTargets([makeTarget(1, 0.5, 0.2, "🔥")], 0.5);
    expect(targets[0].x).toBeCloseTo(1.12 - 0.1, 5);
  });

  it("drops and counts targets that leave the screen", () => {
    const t: Target = { id: 1, x: -0.1, y: 0.5, speed: 0.2, emoji: "🗿" };
    const { targets, escaped } = advanceTargets([t], 0.5);
    expect(targets).toHaveLength(0);
    expect(escaped).toBe(1);
  });
});

describe("beamHits", () => {
  const onScreen = (y: number): Target => ({ id: 1, x: 0.5, y, speed: 0.2, emoji: "😭" });

  it("destroys on-screen targets within the beam band", () => {
    const aimY = 0.5;
    const targets = [onScreen(0.5), onScreen(0.5 + BEAM_HALF_HEIGHT - 0.01), onScreen(0.9)];
    const { hit, remaining } = beamHits(targets, aimY);
    expect(hit).toHaveLength(2);
    expect(remaining).toHaveLength(1);
    expect(remaining[0].y).toBe(0.9);
  });

  it("misses targets outside the band", () => {
    const { hit } = beamHits([onScreen(0.1)], 0.9);
    expect(hit).toHaveLength(0);
  });

  it("cannot hit a target that is still off-screen", () => {
    const offscreen: Target = { id: 1, x: 1.1, y: 0.5, speed: 0.2, emoji: "🤡" };
    const { hit } = beamHits([offscreen], 0.5);
    expect(hit).toHaveLength(0);
  });

  it("has a non-empty emoji deck", () => {
    expect(MEME_EMOJIS.length).toBeGreaterThan(3);
  });
});
