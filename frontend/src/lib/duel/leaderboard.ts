// localStorage-backed log of recent duels — the "records + leaderboard" for Pose Duel. Client
// only, so the demo runs on a phone with no backend/auth. Corrupt or foreign data degrades to
// an empty board. Ordered most-recent-first (a duel is an event, not a high score).

export type DuelEntry = {
  // Winning player's name.
  winner: string;
  // Losing player's name.
  loser: string;
  // Round wins for each side, e.g. 3 and 1.
  winnerPoints: number;
  loserPoints: number;
  // Epoch ms — recency ordering + stable identity.
  ts: number;
};

const STORAGE_KEY = "xcoach.poseDuel.results.v1";
export const MAX_ENTRIES = 10;

function isEntry(v: unknown): v is DuelEntry {
  if (!v || typeof v !== "object") return false;
  const e = v as Record<string, unknown>;
  return (
    typeof e.winner === "string" &&
    typeof e.loser === "string" &&
    typeof e.winnerPoints === "number" &&
    typeof e.loserPoints === "number" &&
    typeof e.ts === "number"
  );
}

// Newest first; ts is unique enough in practice, id as a final tiebreak by winner name.
function byRecency(a: DuelEntry, b: DuelEntry): number {
  return b.ts - a.ts || a.winner.localeCompare(b.winner);
}

export function loadResults(): DuelEntry[] {
  try {
    const raw = typeof localStorage !== "undefined" && localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isEntry).sort(byRecency).slice(0, MAX_ENTRIES);
  } catch {
    return [];
  }
}

// Prepend a duel result, persist the trimmed most-recent-N, and return the new board.
export function saveResult(entry: DuelEntry): DuelEntry[] {
  const board = [entry, ...loadResults()].sort(byRecency).slice(0, MAX_ENTRIES);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(board));
  } catch {
    // Ignore quota / private-mode write failures.
  }
  return board;
}

export function clearResults(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
