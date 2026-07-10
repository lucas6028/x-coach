import { describe, it, expect, beforeEach } from "vitest";
import {
  loadLeaderboard,
  saveScore,
  bestScore,
  clearLeaderboard,
  MAX_ENTRIES,
  type ScoreEntry,
} from "../lib/game/leaderboard";

const KEY = "xcoach.poseGame.leaderboard.v1";

function entry(score: number, extra: Partial<ScoreEntry> = {}): ScoreEntry {
  return { name: "P", score, poses: 1, bestCombo: 1, ts: 1000, ...extra };
}

describe("leaderboard", () => {
  beforeEach(() => clearLeaderboard());

  it("starts empty", () => {
    expect(loadLeaderboard()).toEqual([]);
    expect(bestScore()).toBe(0);
  });

  it("saves and ranks by score descending", () => {
    saveScore(entry(100, { name: "A" }));
    const { board, rank } = saveScore(entry(300, { name: "B" }));
    expect(board.map((e) => e.name)).toEqual(["B", "A"]);
    expect(rank).toBe(1);
    expect(bestScore()).toBe(300);
  });

  it("breaks ties by poses then earlier timestamp", () => {
    saveScore(entry(200, { name: "Older", poses: 2, ts: 1 }));
    saveScore(entry(200, { name: "Fewer", poses: 1, ts: 2 }));
    saveScore(entry(200, { name: "Newer", poses: 2, ts: 5 }));
    expect(loadLeaderboard().map((e) => e.name)).toEqual(["Older", "Newer", "Fewer"]);
  });

  it("caps the board at MAX_ENTRIES and reports off-board scores as rank -1", () => {
    for (let i = 0; i < MAX_ENTRIES; i++) saveScore(entry(1000 + i, { ts: i }));
    const { rank } = saveScore(entry(1, { name: "Last" }));
    expect(rank).toBe(-1);
    expect(loadLeaderboard()).toHaveLength(MAX_ENTRIES);
    expect(loadLeaderboard().some((e) => e.name === "Last")).toBe(false);
  });

  it("ignores corrupt storage", () => {
    localStorage.setItem(KEY, "not json");
    expect(loadLeaderboard()).toEqual([]);
    localStorage.setItem(KEY, JSON.stringify({ not: "an array" }));
    expect(loadLeaderboard()).toEqual([]);
  });

  it("filters out malformed entries", () => {
    localStorage.setItem(
      KEY,
      JSON.stringify([entry(50), { name: "bad" }, { score: 10 }, null])
    );
    const board = loadLeaderboard();
    expect(board).toHaveLength(1);
    expect(board[0].score).toBe(50);
  });
});
