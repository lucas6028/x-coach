import type { Plan, PlanItem } from "../../api";
import { itemsByDay, PLAN_DAYS } from "../../lib/plans";
import { movementLabel, useI18n } from "../../lib/i18n";
import MovementArt from "../movements/MovementArt";
import PlanMuscleCoverage from "./PlanMuscleCoverage";

/** One row of the preview: a real day, or a run of consecutive rest days folded into a single line. */
type Band = { kind: "day"; day: number } | { kind: "rest"; from: number; to: number };

/**
 * Fold RUNS of empty days into one band each. A three-day plan otherwise spends four of its seven
 * rows saying nothing, which reads as a plan that failed to finish rather than a week with rest in
 * it. A lone empty day stays a band of its own — "Day 4–4 · Rest" is worse than "Rest day", and the
 * editable detail page keeps all seven regardless, because there every day is a drop target.
 */
function bandsFor(days: PlanItem[][]): Band[] {
  const bands: Band[] = [];
  for (let i = 0; i < PLAN_DAYS.length; i++) {
    if (days[i].length > 0) {
      bands.push({ kind: "day", day: PLAN_DAYS[i] });
      continue;
    }
    let last = i;
    while (last + 1 < PLAN_DAYS.length && days[last + 1].length === 0) last++;
    if (last > i) {
      bands.push({ kind: "rest", from: PLAN_DAYS[i], to: PLAN_DAYS[last] });
      i = last;
    } else {
      bands.push({ kind: "day", day: PLAN_DAYS[i] });
    }
  }
  return bands;
}

/**
 * A plan, read-only: what Lumen has built so far, shown beside the conversation that is building
 * it. No ticks, no studio links, no remove buttons — every one of those is an action on a plan the
 * user owns, and on the builder page the plan is still being drafted. The editable version of this
 * is `PlanItemRow` on the detail page.
 *
 * ONE FULL-WIDTH BAND PER DAY, exercises 1/2/3-across inside it — never seven day columns. Seven
 * columns broke the detail page at >=1280px (a day column got ~147px against a ~241px row) and the
 * same arithmetic applies here.
 */
export default function PlanPreview({ plan, compact }: { plan: Plan; compact?: boolean }) {
  const { t } = useI18n();
  const days = itemsByDay(plan.items);

  return (
    <div className="flex flex-col gap-3">
      {bandsFor(days).map((band) => {
        if (band.kind === "rest") {
          return (
            <section
              key={`rest-${band.from}`}
              className="rounded-2xl border border-border-dark/60 bg-surface/60 px-4 py-3"
            >
              <div className="flex items-center gap-3">
                <span className="shrink-0 text-xs font-medium tabular-nums text-faint">
                  {t("plans.restRange", { from: band.from, to: band.to })}
                </span>
                <span className="h-px flex-1 bg-border-dark" />
              </div>
            </section>
          );
        }
        const day = band.day;
        const items = days[day - 1];
        return (
          <section
            key={day}
            className={`rounded-2xl border p-4 ${
              items.length ? "border-border-dark bg-surface" : "border-border-dark/60 bg-surface/60"
            }`}
          >
            <div className="flex items-center gap-3">
              <h3 className="shrink-0 text-xs font-semibold uppercase tracking-wider text-faint">
                {t("plans.day", { n: day })}
              </h3>
              {items.length === 0 && (
                <span className="shrink-0 text-[11px] text-faint">{t("plans.rest")}</span>
              )}
              <span className="h-px flex-1 bg-border-dark" />
            </div>

            {items.length > 0 && (
              <ul
                className={
                  // `sm:`/`xl:` ask how wide the WINDOW is, and this list's width comes from its
                  // container -- in the builder's preview COLUMN two across at a 1024px window is
                  // two ~100px cells, every movement name an ellipsis. `compact` is the container
                  // signal the caller passes; without it the preview owns its full width.
                  // Three across waits for `2xl`, not `xl`: the 64px figure stage takes width the
                  // name column used to have, and at a 1280px window three cells left the movement
                  // name truncated to about three characters ("Pu…", "Sit…"). Two across keeps the
                  // name readable, which is the point of the row.
                  compact
                    ? "mt-3 grid grid-cols-1 gap-2"
                    : "mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 2xl:grid-cols-3"
                }
              >
                {items.map((item) => (
                  <li
                    // `min-w-0` so the row shrinks to its grid cell instead of flooring at its
                    // intrinsic width and pushing out of the card.
                    key={item.id}
                    className="flex min-w-0 items-center gap-2.5 rounded-xl border border-border-dark bg-surface px-3 py-2.5"
                  >
                    {/* The library figure, not the 17px glyph: this column is where the user
                        watches a plan appear, and a name plus a monochrome pictogram does not tell
                        them what Lumen just added. A thumbnail rather than the detail page's
                        full-width stage — that card is a square, and a square per exercise in a
                        ~400px preview column would make a nine-exercise week metres tall.
                        The stage is the library's own: the art is an opaque pre-matted square, so
                        it needs a square, full-bleed, rounded frame or its matte shows as a hard
                        white box. */}
                    <span className="h-16 w-16 shrink-0 overflow-hidden rounded-xl bg-gradient-to-b from-[#f7f5ff] to-[#eceefb]">
                      <MovementArt movement={item.movement} />
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-[13px] font-medium text-content">
                        {movementLabel(t, item.movement)}
                      </p>
                      <p className="text-[11px] tabular-nums text-faint">
                        {t("plans.setsReps", { sets: item.sets, reps: item.reps })}
                        {item.notes ? ` · ${item.notes}` : ""}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        );
      })}

      {/* Under the week, so it fills in live as Lumen writes the plan. It renders nothing at all
          while the plan has no exercises, which is also the only state this page opens in. */}
      <PlanMuscleCoverage items={plan.items} compact={compact} />
    </div>
  );
}
