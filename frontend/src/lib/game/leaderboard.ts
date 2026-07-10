// A tiny localStorage-backed leaderboard for the Pose Match game. Deliberately
// client-only so the demo works with nothing but a phone and a browser — no auth,
// no backend round-trip. Corrupt / foreign data is tolerated (returns an empty board).

export type ScoreEntry = {
  name: string;
  score: number;
  // Poses locked in this round.
  poses: number;
  // Best combo streak reached.
  bestCombo: number;
  // Epoch ms, used as a stable tie-break (earlier submission ranks higher).
  ts: number;
};

const STORAGE_KEY = "xcoach.poseGame.leaderboard.v1";
export const MAX_ENTRIES = 10;

function isEntry(v: unknown): v is ScoreEntry {
  if (!v || typeof v !== "object") return false;
  const e = v as Record<string, unknown>;
  return (
    typeof e.name === "string" &&
    typeof e.score === "number" &&
    typeof e.poses === "number" &&
    typeof e.bestCombo === "number" &&
    typeof e.ts === "number"
  );
}

// Higher score first; ties broken by more poses, then earlier submission.
function byRank(a: ScoreEntry, b: ScoreEntry): number {
  return b.score - a.score || b.poses - a.poses || a.ts - b.ts;
}

export function loadLeaderboard(): ScoreEntry[] {
  try {
    const raw = typeof localStorage !== "undefined" && localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isEntry).sort(byRank).slice(0, MAX_ENTRIES);
  } catch {
    return [];
  }
}

// Insert an entry, persist the trimmed top-N, and report where it landed.
// A rank of -1 means the score didn't make the board.
export function saveScore(entry: ScoreEntry): { board: ScoreEntry[]; rank: number } {
  const merged = [...loadLeaderboard(), entry].sort(byRank);
  const board = merged.slice(0, MAX_ENTRIES);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(board));
  } catch {
    // Ignore quota / private-mode write failures — the round still ended cleanly.
  }
  const rank = board.indexOf(entry);
  return { board, rank: rank === -1 ? -1 : rank + 1 };
}

// Highest score on record, or 0 for a fresh board.
export function bestScore(): number {
  const board = loadLeaderboard();
  return board.length ? board[0].score : 0;
}

export function clearLeaderboard(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
