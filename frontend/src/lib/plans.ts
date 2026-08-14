// Derived views over a training plan. Everything here is a pure function of the rows the API
// returns — none of it is stored, and that is deliberate: a plan's "length" and "which day am I
// on" are answers to the items, so storing them would just be two more things to keep in sync.

import type { PlanItem } from "../api";
import type { TFunc } from "./i18n";

/**
 * The localized name or description of a built-in template, falling back to the English string the
 * server sent.
 *
 * Same idiom as `movementLabel`: `t` returns the KEY when there is no entry, so comparing against
 * the key is how "untranslated" is detected. The fallback matters — a template added to
 * `backend/app/routers/plans.py` before its i18n keys exist must render its English name, not
 * `plans.template.foo.name`.
 */
export function templateText(
  t: TFunc,
  templateKey: string,
  field: "name" | "desc",
  fallback: string
): string {
  const key = `plans.template.${templateKey}.${field}`;
  const value = t(key);
  return value === key ? fallback : value;
}

/** The seven relative day slots a plan can use. Matches the backend's `day_index between 1 and 7`
 *  and the migration's check constraint — one decision ("a plan covers at most a week"). */
export const PLAN_DAYS = [1, 2, 3, 4, 5, 6, 7] as const;

/** Items grouped into the seven slots. Index 0 is Day 1; an empty array is a rest day. */
export function itemsByDay(items: PlanItem[]): PlanItem[][] {
  const days: PlanItem[][] = PLAN_DAYS.map(() => []);
  for (const item of items) {
    // Defensive against a day_index outside the week: the API and the DB both reject one, but a
    // stale client reading a future schema should drop the row rather than crash the page.
    if (item.day_index >= 1 && item.day_index <= PLAN_DAYS.length) {
      days[item.day_index - 1].push(item);
    }
  }
  return days;
}

/** Which days actually hold exercises, ascending. This is the plan's real length. */
export function usedDays(items: PlanItem[]): number[] {
  return PLAN_DAYS.filter((day) => items.some((it) => it.day_index === day));
}

/**
 * The day the user is on: the first day WITH exercises that still has something unticked, or
 * `null` when every exercise is done (or the plan is empty).
 *
 * Derived, not stored. A stored pointer would have to be advanced by whoever ticked the last item
 * of a day, and would then disagree with the ticks the moment one was undone.
 */
export function currentDay(items: PlanItem[]): number | null {
  for (const day of usedDays(items)) {
    if (items.some((it) => it.day_index === day && !it.completed_at)) return day;
  }
  return null;
}

/** Completed / total, as a 0..1 ratio. An empty plan is 0, not NaN. */
export function progressRatio(completed: number, total: number): number {
  return total > 0 ? completed / total : 0;
}

/** Whether a movement can be sent to the studio, given the analysable list from GET /api/movements.
 *
 *  Compared case-insensitively for the same reason App.tsx canonicalizes its `?movement=`: the
 *  backend's registry lowercases its lookup key, so a spelling that differs only in case is one
 *  the API would happily analyse. */
export function isAnalyzable(movement: string, analyzable: { name: string }[]): boolean {
  return analyzable.some((m) => m.name.toLowerCase() === movement.trim().toLowerCase());
}
