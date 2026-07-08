// localStorage-backed local leaderboard for Meme Blaster. Client-only so the demo runs
// on a phone with no backend/auth. Corrupt or foreign data degrades to an empty board.

export type BlastEntry = {
  name: string;
  score: number;
  // Orbs destroyed this round.
  hits: number;
  // Best combo streak reached.
  bestCombo: number;
  // Epoch ms — stable tie-break (earlier submission ranks higher).
  ts: number;
};

const STORAGE_KEY = "xcoach.memeBlast.leaderboard.v1";
export const MAX_ENTRIES = 10;

function isEntry(v: unknown): v is BlastEntry {
  if (!v || typeof v !== "object") return false;
  const e = v as Record<string, unknown>;
  return (
    typeof e.name === "string" &&
    typeof e.score === "number" &&
    typeof e.hits === "number" &&
    typeof e.bestCombo === "number" &&
    typeof e.ts === "number"
  );
}

function byRank(a: BlastEntry, b: BlastEntry): number {
  return b.score - a.score || b.hits - a.hits || a.ts - b.ts;
}

export function loadLeaderboard(): BlastEntry[] {
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

// Insert an entry, persist the trimmed top-N, and report its 1-based rank (or -1 if it
// didn't make the board).
export function saveScore(entry: BlastEntry): { board: BlastEntry[]; rank: number } {
  const board = [...loadLeaderboard(), entry].sort(byRank).slice(0, MAX_ENTRIES);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(board));
  } catch {
    // Ignore quota / private-mode write failures.
  }
  const idx = board.indexOf(entry);
  return { board, rank: idx === -1 ? -1 : idx + 1 };
}

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
