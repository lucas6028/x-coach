import { describe, it, expect, beforeEach } from "vitest";
import {
  loadLeaderboard,
  saveScore,
  bestScore,
  clearLeaderboard,
  MAX_ENTRIES,
  type NinjaEntry,
} from "../lib/ninja/leaderboard";

const KEY = "xcoach.fruitNinja.leaderboard.v1";

function entry(over: Partial<NinjaEntry> = {}): NinjaEntry {
  return { name: "Ann", score: 200, bestCombo: 6, ts: 1000, ...over };
}

describe("fruit ninja leaderboard", () => {
  beforeEach(() => clearLeaderboard());

  it("starts empty", () => {
    expect(loadLeaderboard()).toEqual([]);
    expect(bestScore()).toBe(0);
  });

  it("ranks by score and reports rank + best", () => {
    saveScore(entry({ name: "Low", score: 100 }));
    const { rank } = saveScore(entry({ name: "High", score: 300 }));
    expect(rank).toBe(1);
    expect(loadLeaderboard().map((e) => e.name)).toEqual(["High", "Low"]);
    expect(bestScore()).toBe(300);
  });

  it("breaks ties by best combo", () => {
    saveScore(entry({ name: "A", score: 200, bestCombo: 3, ts: 1 }));
    saveScore(entry({ name: "B", score: 200, bestCombo: 9, ts: 2 }));
    expect(loadLeaderboard().map((e) => e.name)).toEqual(["B", "A"]);
  });

  it("trims to MAX_ENTRIES", () => {
    for (let i = 0; i < MAX_ENTRIES + 5; i++) saveScore(entry({ score: i, ts: i }));
    expect(loadLeaderboard()).toHaveLength(MAX_ENTRIES);
  });

  it("ignores corrupt or foreign data", () => {
    localStorage.setItem(KEY, "not json {");
    expect(loadLeaderboard()).toEqual([]);
    localStorage.setItem(KEY, JSON.stringify({ nope: true }));
    expect(loadLeaderboard()).toEqual([]);
    localStorage.setItem(KEY, JSON.stringify([entry(), { name: "X" }]));
    expect(loadLeaderboard()).toHaveLength(1);
  });

  it("clears the board", () => {
    saveScore(entry());
    clearLeaderboard();
    expect(loadLeaderboard()).toEqual([]);
  });
});
