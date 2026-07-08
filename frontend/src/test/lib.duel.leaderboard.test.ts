import { describe, it, expect, beforeEach } from "vitest";
import {
  loadResults,
  saveResult,
  clearResults,
  MAX_ENTRIES,
  type DuelEntry,
} from "../lib/duel/leaderboard";

const KEY = "xcoach.poseDuel.results.v1";

function entry(over: Partial<DuelEntry> = {}): DuelEntry {
  return { winner: "Ann", loser: "Bo", winnerPoints: 3, loserPoints: 1, ts: 1000, ...over };
}

describe("duel leaderboard", () => {
  beforeEach(() => clearResults());

  it("starts empty", () => {
    expect(loadResults()).toEqual([]);
  });

  it("saves and reloads a result", () => {
    saveResult(entry());
    const board = loadResults();
    expect(board).toHaveLength(1);
    expect(board[0].winner).toBe("Ann");
  });

  it("orders most-recent first", () => {
    saveResult(entry({ winner: "Old", ts: 100 }));
    saveResult(entry({ winner: "New", ts: 200 }));
    expect(loadResults().map((e) => e.winner)).toEqual(["New", "Old"]);
  });

  it("trims to MAX_ENTRIES", () => {
    for (let i = 0; i < MAX_ENTRIES + 5; i++) saveResult(entry({ ts: i }));
    expect(loadResults()).toHaveLength(MAX_ENTRIES);
  });

  it("ignores corrupt JSON", () => {
    localStorage.setItem(KEY, "not json {");
    expect(loadResults()).toEqual([]);
  });

  it("ignores a non-array payload", () => {
    localStorage.setItem(KEY, JSON.stringify({ nope: true }));
    expect(loadResults()).toEqual([]);
  });

  it("filters out malformed entries", () => {
    localStorage.setItem(
      KEY,
      JSON.stringify([entry(), { winner: "X" /* missing fields */ }])
    );
    expect(loadResults()).toHaveLength(1);
  });

  it("clears the board", () => {
    saveResult(entry());
    clearResults();
    expect(loadResults()).toEqual([]);
  });
});
