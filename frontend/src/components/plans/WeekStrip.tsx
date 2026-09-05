import { useEffect, useRef } from "react";
import { Check } from "@phosphor-icons/react";
import { useReducedMotion } from "motion/react";
import type { PlanItem } from "../../api";
import { useI18n } from "../../lib/i18n";
import { PLAN_DAYS } from "../../lib/plans";
import { dayMuscles } from "../../lib/planMuscles";

interface Props {
  /** The seven day slots, from `itemsByDay`. Index 0 is Day 1; an empty slot is a rest day. */
  days: PlanItem[][];
  /** The day the focused panel is showing, 1..7. */
  selected: number;
  /** `currentDay(plan.items)` — the day the user is actually on, or null on a finished plan. */
  today: number | null;
  onSelect: (day: number) => void;
  /** The id of the `tabpanel` these tabs drive. */
  panelId: string;
  /** How a day's tab id is built, so the panel can name its own tab back. */
  tabId: (day: number) => string;
}

/**
 * THE WEEK AS ONE OBJECT: seven tiles in a single row, each a SUMMARY of its day.
 *
 * WHY A SUMMARY AND NOTHING MORE — this is the constraint that killed two earlier layouts, so it
 * is written down rather than left to be rediscovered. An exercise ROW carries a checkbox, a
 * movement icon, a name, an action chip and a delete button: its fixed parts alone need ~241px,
 * measured. Seven columns of a 1152px page are ~147px each, so a row put inside one overflowed its
 * column by ~120px and squeezed the movement NAME to zero width — the name simply disappeared. No
 * amount of truncation fixes that, because none of the overflow is in the truncatable part.
 * A tile is legal at 147px precisely because it holds no row: a label, a count, a bar and one
 * muscle name all truncate or wrap happily. The exercise rows live in the focused day panel below,
 * which has the WHOLE page width — the arithmetic that broke is not even attempted here.
 *
 * States, all visually distinct because they answer different questions: SELECTED is "what am I
 * looking at" (filled and ringed), TODAY is "where am I in the plan" (a badge), and the two are
 * therefore readable at once on the same tile. FULLY DONE gets the green tick, and REST is drawn
 * lighter than everything else — a dashed, unfilled outline — because four rest days should never
 * out-shout three training days, which is exactly what the old full-width bands did.
 *
 * Width: the tiles flex, but floor at ~7.5rem. When seven of those do not fit — the phone, and the
 * desktop column with Lumen's panel open — the strip SCROLLS with scroll-snap instead of crushing
 * the tiles below legibility or wrapping into a ragged second row that would stop reading as one
 * week.
 */
export default function WeekStrip({ days, selected, today, onSelect, panelId, tabId }: Props) {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  const tabs = useRef(new Map<number, HTMLButtonElement>());

  // The selected tile is brought into view whenever selection changes — including when it changed
  // because the PLAN changed (Lumen rewrote the week), not just because the user clicked.
  // `block: nearest` on purpose: the app shell is a fixed box with its own scroll region, and a
  // default scrollIntoView would drag that region as well as the strip.
  useEffect(() => {
    tabs.current.get(selected)?.scrollIntoView?.({
      block: "nearest",
      inline: "nearest",
      behavior: reduce ? "auto" : "smooth",
    });
  }, [selected, reduce]);

  // Arrow keys move selection AND focus together, which is the `tablist` pattern users expect:
  // the panel follows the arrow rather than waiting for a second Enter. Wraps at both ends because
  // a week is a cycle. `preventScroll` because the effect above owns the scrolling.
  const move = (from: number, delta: number) => {
    const next = ((from - 1 + delta + PLAN_DAYS.length) % PLAN_DAYS.length) + 1;
    onSelect(next);
    tabs.current.get(next)?.focus({ preventScroll: true });
  };

  const onKeyDown = (e: React.KeyboardEvent, day: number) => {
    if (e.key === "ArrowRight") move(day, 1);
    else if (e.key === "ArrowLeft") move(day, -1);
    else if (e.key === "Home") move(1, 0);
    else if (e.key === "End") move(PLAN_DAYS.length, 0);
    else return;
    // Only after one of the four matched: the page must still scroll on Space and PageDown.
    e.preventDefault();
  };

  return (
    // The negative margin pays for the padding, which exists so a focus ring on the first or last
    // tile is not clipped by the scroll container it lives in.
    <div
      role="tablist"
      aria-label={t("plans.weekLabel")}
      aria-orientation="horizontal"
      className="-mx-1 flex snap-x snap-mandatory gap-2 overflow-x-auto scroll-px-1 px-1 pb-1"
    >
      {PLAN_DAYS.map((day) => {
        const items = days[day - 1];
        const done = items.filter((it) => it.completed_at).length;
        const isRest = items.length === 0;
        const isSelected = selected === day;
        const isToday = today === day;
        const allDone = items.length > 0 && done === items.length;
        const muscles = dayMuscles(items);

        return (
          <button
            key={day}
            ref={(el) => {
              if (el) tabs.current.set(day, el);
              else tabs.current.delete(day);
            }}
            type="button"
            role="tab"
            id={tabId(day)}
            aria-controls={panelId}
            aria-selected={isSelected}
            // Roving tabindex: one stop for the whole strip, then arrows inside it. Seven tab
            // stops would put the muscle-coverage card six presses further away for a keyboard
            // user than it is for anyone else.
            tabIndex={isSelected ? 0 : -1}
            onClick={() => onSelect(day)}
            onKeyDown={(e) => onKeyDown(e, day)}
            // `flex flex-col` for one reason worth writing down: a <button> centres its content
            // vertically, and the tiles are stretched to a common height, so a rest tile's two
            // lines floated to the middle while a training tile's four filled the box — seven day
            // labels at two different heights, which stops the row reading as one strip. Column
            // flow pins every tile's label to the top. NOT `items-start`, which would shrink the
            // progress bar (a span with no intrinsic width) to nothing.
            className={`flex min-h-[76px] min-w-[7.5rem] flex-1 basis-0 flex-col snap-start rounded-2xl border px-3 py-2.5 text-left transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary ${
              isSelected
                ? "border-primary bg-primary/[0.07] ring-2 ring-primary/20"
                : isRest
                  ? "border-dashed border-border-dark bg-transparent hover:border-primary/30"
                  : allDone
                    ? "border-secondary/40 bg-secondary/[0.05] hover:border-secondary/60"
                    : isToday
                      ? "border-primary/40 bg-surface hover:border-primary/60"
                      : "border-border-dark bg-surface hover:border-primary/30"
            }`}
          >
            <span className="flex items-center gap-1.5">
              <span
                className={`text-[11px] font-semibold uppercase tracking-wider ${
                  isSelected ? "text-primary" : isRest ? "text-faint" : "text-muted"
                }`}
              >
                {t("plans.day", { n: day })}
              </span>
              {isToday && (
                // TODAY is a badge rather than a colour, so it survives being the SELECTED tile
                // too — colour alone would have made "the day I am on" and "the day I am reading"
                // indistinguishable the moment they coincided, which is most of the time.
                <span className="rounded-full bg-primary px-1.5 py-px text-[9.5px] font-semibold uppercase tracking-wide text-primary-content">
                  {t("plans.todayMarker")}
                </span>
              )}
            </span>

            {isRest ? (
              <span className="mt-1.5 block text-[11px] text-faint">{t("plans.restShort")}</span>
            ) : (
              <>
                <span
                  className="mt-1.5 flex items-center gap-1 text-[13px] font-semibold tabular-nums text-content"
                  title={t("plans.progress", { done, total: items.length })}
                >
                  {allDone && (
                    <>
                      <Check size={12} weight="bold" className="text-secondary" aria-hidden="true" />
                      <span className="sr-only">{t("plans.dayDone")}</span>
                    </>
                  )}
                  {done}/{items.length}
                </span>

                <span className="mt-1.5 block h-1 overflow-hidden rounded-full bg-content/[0.07]">
                  <span
                    className={`block h-full rounded-full ${allDone ? "bg-secondary" : "bg-primary"}`}
                    style={{ width: `${Math.round((done / items.length) * 100)}%` }}
                  />
                </span>

                {/* One group and a count, not the whole map: a tile this wide fits about eleven
                    characters before it truncates, and the day's prime mover is the thing worth
                    those characters. The panel header below names up to three, and the coverage
                    card names all of them. */}
                {muscles.length > 0 && (
                  <span className="mt-1.5 flex items-baseline gap-1 text-[10.5px] font-medium text-faint">
                    <span className="truncate">{t(`muscle.${muscles[0]}`)}</span>
                    {muscles.length > 1 && (
                      <span className="shrink-0 tabular-nums">
                        {t("plans.muscles.more", { n: muscles.length - 1 })}
                      </span>
                    )}
                  </span>
                )}
              </>
            )}
          </button>
        );
      })}
    </div>
  );
}
