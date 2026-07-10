// localStorage-backed local leaderboard for Fruit Ninja. Client-only so the demo runs on a phone
// with no backend/auth. Corrupt or foreign data degrades to an empty board. Ranked by score, then
// best combo, then earliest submission.

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

export function loadLeaderboard(): NinjaEntry[] {
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

// Insert an entry, persist the trimmed top-N, and report its 1-based rank (or -1 if it missed).
export function saveScore(entry: NinjaEntry): { board: NinjaEntry[]; rank: number } {
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
