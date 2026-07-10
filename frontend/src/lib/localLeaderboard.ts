// Generic localStorage-backed leaderboard, shared by the pose mini-games (Fruit Ninja, 67).
// Client-only so each demo runs on a phone with no backend/auth. Corrupt or foreign data degrades
// to an empty board.
export function createLocalLeaderboard<T>(
  storageKey: string,
  maxEntries: number,
  isEntry: (v: unknown) => v is T,
  byRank: (a: T, b: T) => number
) {
  function loadLeaderboard(): T[] {
    try {
      const raw = typeof localStorage !== "undefined" && localStorage.getItem(storageKey);
      if (!raw) return [];
      const parsed: unknown = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(isEntry).sort(byRank).slice(0, maxEntries);
    } catch {
      return [];
    }
  }

  // Insert an entry, persist the trimmed top-N, and report its 1-based rank (or -1 if it missed).
  function saveScore(entry: T): { board: T[]; rank: number } {
    const board = [...loadLeaderboard(), entry].sort(byRank).slice(0, maxEntries);
    try {
      localStorage.setItem(storageKey, JSON.stringify(board));
    } catch {
      // Ignore quota / private-mode write failures.
    }
    const idx = board.indexOf(entry);
    return { board, rank: idx === -1 ? -1 : idx + 1 };
  }

  function clearLeaderboard(): void {
    try {
      localStorage.removeItem(storageKey);
    } catch {
      // ignore
    }
  }

  return { loadLeaderboard, saveScore, clearLeaderboard };
}
