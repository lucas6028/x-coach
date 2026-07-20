import { describe, it, expect, beforeEach } from "vitest";
import {
  loadLeaderboard,
  saveScore,
  bestScore,
  clearLeaderboard,
  MAX_ENTRIES,
  type BlastEntry,
} from "../lib/blast/leaderboard";

const KEY = "xcoach.memeBlast.leaderboard.v1";

function entry(score: number, extra: Partial<BlastEntry> = {}): BlastEntry {
  return { name: "P", score, hits: 1, bestCombo: 1, ts: 1000, ...extra };
}

describe("blast leaderboard", () => {
  beforeEach(() => clearLeaderboard());

  it("starts empty", () => {
    expect(loadLeaderboard()).toEqual([]);
    expect(bestScore()).toBe(0);
  });

  it("ranks by score descending", () => {
    saveScore(entry(100, { name: "A" }));
    const { board, rank } = saveScore(entry(300, { name: "B" }));
    expect(board.map((e) => e.name)).toEqual(["B", "A"]);
    expect(rank).toBe(1);
    expect(bestScore()).toBe(300);
  });

  it("breaks ties by hits then earlier timestamp", () => {
    saveScore(entry(200, { name: "Older", hits: 2, ts: 1 }));
    saveScore(entry(200, { name: "Fewer", hits: 1, ts: 2 }));
    saveScore(entry(200, { name: "Newer", hits: 2, ts: 5 }));
    expect(loadLeaderboard().map((e) => e.name)).toEqual(["Older", "Newer", "Fewer"]);
  });

  it("caps at MAX_ENTRIES and reports off-board as rank -1", () => {
    for (let i = 0; i < MAX_ENTRIES; i++) saveScore(entry(1000 + i, { ts: i }));
    const { rank } = saveScore(entry(1, { name: "Last" }));
    expect(rank).toBe(-1);
    expect(loadLeaderboard()).toHaveLength(MAX_ENTRIES);
  });

  it("tolerates corrupt or malformed storage", () => {
    localStorage.setItem(KEY, "not json");
    expect(loadLeaderboard()).toEqual([]);
    localStorage.setItem(KEY, JSON.stringify([entry(50), { name: "bad" }, null]));
    expect(loadLeaderboard()).toHaveLength(1);
  });
});
