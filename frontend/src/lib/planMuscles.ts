// What a set of plan items TRAINS, derived from the movement catalog.
//
// WHY DERIVED AND NOT STORED, same reasoning as the rest of lib/plans.ts: a plan's coverage is an
// answer to its items, so a stored copy would be one more thing to keep in sync — wrong the moment
// an exercise is added, removed or swapped, and wrong in a way nobody would notice until the map
// lit the wrong muscle. The only authored source is `MOVEMENT_DETAIL`; everything here is a pure
// function over it, so a movement whose muscle list is corrected updates every plan at once.

import type { PlanItem } from "../api";
import { movementDetail, type Muscle } from "./movementDetail";

/** Declaration order of the `Muscle` union, as the final tie-break. A `Record` rather than an
 *  array on purpose: under strict mode a muscle added to the union without a rank here is a
 *  COMPILE error, instead of silently sorting last behind everything else. */
const MUSCLE_RANK: Record<Muscle, number> = {
  shoulders: 0,
  chest: 1,
  biceps: 2,
  triceps: 3,
  forearms: 4,
  abs: 5,
  obliques: 6,
  upperBack: 7,
  lats: 8,
  lowerBack: 9,
  glutes: 10,
  quads: 11,
  hamstrings: 12,
  hipFlexors: 13,
  adductors: 14,
  calves: 15,
};

/**
 * The major groups a balanced week should hit, IN THE ORDER A GAP LINE NAMES THEM — big movers
 * first, then the smaller ones. Deliberately eight of the sixteen: nobody needs "forearms" flagged
 * as a hole in their week, and a list that flags everything flags nothing.
 *
 * This order is its own thing and is NOT the ranking order — gaps have no counts to rank by.
 */
const GAP_GROUPS: Muscle[] = [
  "quads",
  "hamstrings",
  "glutes",
  "chest",
  "upperBack",
  "lats",
  "shoulders",
  "abs",
];

export interface PlanCoverage {
  /** Groups that are `primary` for at least one movement, ranked. */
  primary: Muscle[];
  /** Groups that are only ever `secondary`, ranked. Disjoint from `primary`. */
  secondary: Muscle[];
  /** Major groups the set touches neither way, in `GAP_GROUPS` order. */
  gaps: Muscle[];
}

interface Tally {
  primary: number;
  secondary: number;
}

/** How often each group appears as a prime mover and as a supporting one. A movement outside the
 *  catalog contributes nothing rather than throwing: the plan API accepts any of the 16 catalog
 *  names, but a plan written against a newer catalog than this build must still render. */
function tally(movements: string[]): Map<Muscle, Tally> {
  const counts = new Map<Muscle, Tally>();
  const bump = (muscle: Muscle, key: keyof Tally) => {
    const seen = counts.get(muscle) ?? { primary: 0, secondary: 0 };
    seen[key] += 1;
    counts.set(muscle, seen);
  };
  for (const movement of movements) {
    const detail = movementDetail(movement);
    if (!detail) continue;
    for (const muscle of detail.primary) bump(muscle, "primary");
    for (const muscle of detail.secondary) bump(muscle, "secondary");
  }
  return counts;
}

/**
 * Times primary desc, then the union's declaration order. Stable and language-independent — the
 * list must not reorder itself when the user switches to 中文.
 *
 * WHY THE SECONDARY COUNT IS NOT A TIEBREAK: it let a stabiliser outrank a prime mover. A day of
 * Squat + Push-up + Sit-up makes quads, glutes, chest, triceps and abs each primary exactly once,
 * but abs is ALSO secondary twice — so counting that promoted abs to the head of a list whose whole
 * point is "what drives this session", and the three chips a day header has room for named the
 * supporting muscle first. Among movements that drive a group equally often, nothing about how
 * often something else merely braces it should reorder them; the fixed declaration order does.
 */
function rank(counts: Map<Muscle, Tally>): (a: Muscle, b: Muscle) => number {
  const of = (m: Muscle): Tally => counts.get(m) ?? { primary: 0, secondary: 0 };
  return (a, b) => of(b).primary - of(a).primary || MUSCLE_RANK[a] - MUSCLE_RANK[b];
}

/**
 * Coverage for a bare list of movement names.
 *
 * The plans LIST only carries `PlanSummary.movements` — DISTINCT movement names, no items — so the
 * card ranks over distinct movements while the plan page ranks over items. For the same plan the
 * two can therefore order the groups differently (three squat sessions count once on the card and
 * three times on the page). That is intended: a card says what a plan trains, a page says how much.
 */
export function coverageOf(movements: string[]): PlanCoverage {
  const counts = tally(movements);
  const byRank = rank(counts);
  const trained = [...counts.keys()];
  const primary = trained.filter((m) => (counts.get(m)?.primary ?? 0) > 0).sort(byRank);
  const secondary = trained.filter((m) => (counts.get(m)?.primary ?? 0) === 0).sort(byRank);
  return {
    primary,
    secondary,
    gaps: GAP_GROUPS.filter((m) => !counts.has(m)),
  };
}

/** Coverage for a plan (or any subset of its items). */
export function planCoverage(items: PlanItem[]): PlanCoverage {
  return coverageOf(items.map((it) => it.movement));
}

/** The ranked prime movers of one day — what that day's chips name. Secondary groups are dropped
 *  here on purpose: a day label has room for "what this session is", not for its whole map. */
export function dayMuscles(items: PlanItem[]): Muscle[] {
  return planCoverage(items).primary;
}
