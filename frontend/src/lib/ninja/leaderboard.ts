// localStorage-backed local leaderboard for Fruit Ninja. Client-only so the demo runs on a phone
// with no backend/auth. Corrupt or foreign data degrades to an empty board. Ranked by score, then
// best combo, then earliest submission.
import { createLocalLeaderboard } from "../localLeaderboard";

export type NinjaEntry = {
  name: string;
  score: number;
  bestCombo: number;
  ts: number;
};

const STORAGE_KEY = "xcoach.fruitNinja.leaderboard.v1";
export const MAX_ENTRIES = 10;

function isEntry(v: unknown): v is NinjaEntry {
  if (!v || typeof v !== "object") return false;
  const e = v as Record<string, unknown>;
  return (
    typeof e.name === "string" &&
    typeof e.score === "number" &&
    typeof e.bestCombo === "number" &&
    typeof e.ts === "number"
  );
}

function byRank(a: NinjaEntry, b: NinjaEntry): number {
  return b.score - a.score || b.bestCombo - a.bestCombo || a.ts - b.ts;
}

const board = createLocalLeaderboard(STORAGE_KEY, MAX_ENTRIES, isEntry, byRank);
export const loadLeaderboard = board.loadLeaderboard;
export const saveScore = board.saveScore;
export const clearLeaderboard = board.clearLeaderboard;

export function bestScore(): number {
  const b = loadLeaderboard();
  return b.length ? b[0].score : 0;
}
