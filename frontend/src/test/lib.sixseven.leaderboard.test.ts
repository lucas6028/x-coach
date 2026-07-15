import { describe, it, expect, beforeEach } from "vitest";
import {
  loadLeaderboard,
  saveScore,
  clearLeaderboard,
  MAX_ENTRIES,
  type SixSevenEntry,
} from "../lib/sixseven/leaderboard";

const KEY = "xcoach.sixSeven.leaderboard.v1";

function entry(over: Partial<SixSevenEntry> = {}): SixSevenEntry {
  return { name: "Ann", count: 20, bestCombo: 5, ts: 1000, ...over };
}

describe("67 leaderboard", () => {
  beforeEach(() => clearLeaderboard());

  it("starts empty", () => {
    expect(loadLeaderboard()).toEqual([]);
  });

  it("saves, ranks by count, and reports rank", () => {
    saveScore(entry({ name: "Low", count: 10 }));
    const { rank } = saveScore(entry({ name: "High", count: 30 }));
    expect(rank).toBe(1);
    expect(loadLeaderboard().map((e) => e.name)).toEqual(["High", "Low"]);
  });

  it("breaks ties by best combo", () => {
    saveScore(entry({ name: "A", count: 20, bestCombo: 3, ts: 1 }));
    saveScore(entry({ name: "B", count: 20, bestCombo: 8, ts: 2 }));
    expect(loadLeaderboard().map((e) => e.name)).toEqual(["B", "A"]);
  });

  it("trims to MAX_ENTRIES", () => {
    for (let i = 0; i < MAX_ENTRIES + 4; i++) saveScore(entry({ count: i, ts: i }));
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
