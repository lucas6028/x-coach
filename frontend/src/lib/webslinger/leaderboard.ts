import { createLocalLeaderboard } from "../localLeaderboard";

export type WebSlingerEntry = {
  name: string;
  score: number;
  bestCombo: number;
  ts: number;
};

function isEntry(value: unknown): value is WebSlingerEntry {
  if (!value || typeof value !== "object") return false;
  const entry = value as Record<string, unknown>;
  return (
    typeof entry.name === "string" &&
    typeof entry.score === "number" &&
    typeof entry.bestCombo === "number" &&
    typeof entry.ts === "number"
  );
}

const board = createLocalLeaderboard<WebSlingerEntry>(
  "xcoach.webSlinger.leaderboard.v1",
  10,
  isEntry,
  (a, b) => b.score - a.score || b.bestCombo - a.bestCombo || a.ts - b.ts
);

export const loadLeaderboard = board.loadLeaderboard;
export const saveScore = board.saveScore;
export const clearLeaderboard = board.clearLeaderboard;
export const bestScore = () => loadLeaderboard()[0]?.score ?? 0;
