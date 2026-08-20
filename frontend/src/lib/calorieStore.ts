// Cumulative calorie ledger across the pose mini-games, kept in localStorage so the games hub can
// show a lifetime total with no backend/auth. Corrupt or foreign data degrades to zeros, mirroring
// the local leaderboards (lib/localLeaderboard.ts).
import type { GameId } from "./calories";

export interface CalorieTotals {
  // Total estimated kcal across every game.
  total: number;
  // Per-game breakdown.
  byGame: Record<GameId, number>;
  // Number of rounds recorded.
  sessions: number;
}

const STORAGE_KEY = "xcoach.calories.v1";
const GAME_IDS: GameId[] = ["sixseven", "ninja", "webslinger"];

function empty(): CalorieTotals {
  return { total: 0, byGame: { sixseven: 0, ninja: 0, webslinger: 0 }, sessions: 0 };
}

const finite = (v: unknown): number =>
  typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : 0;

export function loadCalories(): CalorieTotals {
  try {
    const raw = typeof localStorage !== "undefined" && localStorage.getItem(STORAGE_KEY);
    if (!raw) return empty();
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (!parsed || typeof parsed !== "object") return empty();
    const byGameRaw = (parsed.byGame ?? {}) as Record<string, unknown>;
    const totals = empty();
    for (const id of GAME_IDS) totals.byGame[id] = finite(byGameRaw[id]);
    totals.sessions = finite(parsed.sessions);
    // Trust the stored total only if it's self-consistent; otherwise re-derive from the breakdown.
    const derived = GAME_IDS.reduce((s, id) => s + totals.byGame[id], 0);
    const stored = finite(parsed.total);
    totals.total = Math.abs(stored - derived) < 0.5 ? stored : derived;
    return totals;
  } catch {
    return empty();
  }
}

// Record one round's estimate and return the updated totals. `kcal` is coerced to a non-negative
// number so a bad caller can never corrupt the ledger.
export function addCalories(game: GameId, kcal: number): CalorieTotals {
  const safe = finite(kcal);
  const totals = loadCalories();
  totals.byGame[game] += safe;
  totals.total += safe;
  totals.sessions += 1;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(totals));
  } catch {
    // Ignore quota / private-mode write failures.
  }
  return totals;
}

export function clearCalories(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
