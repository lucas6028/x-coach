// localStorage-backed local leaderboard for the 67 game. Client-only so the demo runs on a
// phone with no backend/auth. Corrupt or foreign data degrades to an empty board. Ranked by
// most 67s, then best rhythm combo, then earliest submission.
import { createLocalLeaderboard } from "../localLeaderboard";

export type SixSevenEntry = {
  name: string;
  // 67s completed in the round.
  count: number;
  // Best rhythm streak reached.
  bestCombo: number;
  // Epoch ms — stable tie-break (earlier submission ranks higher).
  ts: number;
};

const STORAGE_KEY = "xcoach.sixSeven.leaderboard.v1";
export const MAX_ENTRIES = 10;

function isEntry(v: unknown): v is SixSevenEntry {
  if (!v || typeof v !== "object") return false;
  const e = v as Record<string, unknown>;
  return (
    typeof e.name === "string" &&
    typeof e.count === "number" &&
    typeof e.bestCombo === "number" &&
    typeof e.ts === "number"
  );
}

function byRank(a: SixSevenEntry, b: SixSevenEntry): number {
  return b.count - a.count || b.bestCombo - a.bestCombo || a.ts - b.ts;
}

const board = createLocalLeaderboard(STORAGE_KEY, MAX_ENTRIES, isEntry, byRank);
export const loadLeaderboard = board.loadLeaderboard;
export const saveScore = board.saveScore;
export const clearLeaderboard = board.clearLeaderboard;
