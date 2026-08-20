import { CalendarBlank, CaretRight, CheckCircle } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import MovementIcon from "../movements/MovementIcon";
import { movementLabel, useI18n } from "../../lib/i18n";
import { progressRatio } from "../../lib/plans";
import type { PlanSummary } from "../../api";

interface Props {
  plan: PlanSummary;
}

// One plan on the /plans list: name, how long it is, how far through the current run the user is,
// and which movements it trains.
//
// The whole card is one link, for MovementCard's reason: it has a single destination, so a nested
// button would be a second tab stop to the same place and would leave the rest of the card dead to
// the keyboard.
export default function PlanCard({ plan }: Props) {
  const { t, lang } = useI18n();

  const ratio = progressRatio(plan.completed_count, plan.item_count);
  const done = plan.item_count > 0 && plan.completed_count === plan.item_count;

  // What the run is doing right now. Three states, in the order they actually occur — a plan that
  // was never started reads "not started" rather than "0% done", because those are different
  // things and only one of them is the user's fault.
  const status = !plan.started_at
    ? t("plans.notStarted")
    : done
      ? t("plans.finished")
      : t("plans.progress", { done: plan.completed_count, total: plan.item_count });

  // Only the first few movements, then the rest as a count: a seven-day plan can hold twenty
  // items and the chips would then be the whole card.
  const shown = plan.movements.slice(0, 4);
  const overflow = plan.movements.length - shown.length;

  return (
    <Link
      to={`/plans/${plan.id}`}
      className="group flex h-full flex-col rounded-2xl border border-border-dark bg-surface p-4 shadow-card transition-all hover:-translate-y-0.5 hover:border-primary/35 hover:shadow-card-hover"
    >
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <h3 className="truncate font-display text-[15px] font-semibold text-content">
            {plan.name}
          </h3>
          <p className="mt-1 flex items-center gap-1.5 text-xs text-muted">
            <CalendarBlank size={13} weight="duotone" className="shrink-0" />
            {plan.day_count === 1
              ? t("plans.dayCountOne")
              : t("plans.daysCount", { n: plan.day_count })}
            <span aria-hidden="true">·</span>
            {done ? (
              <span className="inline-flex items-center gap-1 font-medium text-secondary">
                <CheckCircle size={13} weight="fill" />
                {status}
              </span>
            ) : (
              status
            )}
          </p>
        </div>
        <CaretRight
          size={16}
          className="mt-0.5 shrink-0 text-faint transition-transform group-hover:translate-x-0.5"
        />
      </div>

      {/* The progress bar is hidden for a plan that was never started: a full-width empty track
          under "not started" reads as "you are 0% of the way through", which is a judgement about
          a run that has not begun. */}
      {plan.started_at && plan.item_count > 0 && (
        <div
          className="mt-3 h-1.5 overflow-hidden rounded-full bg-content/[0.06]"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={plan.item_count}
          aria-valuenow={plan.completed_count}
          aria-label={t("plans.progress", {
            done: plan.completed_count,
            total: plan.item_count,
          })}
        >
          <div
            className={`h-full rounded-full transition-[width] ${done ? "bg-secondary" : "bg-primary"}`}
            style={{ width: `${Math.round(ratio * 100)}%` }}
          />
        </div>
      )}

      <ul className="mt-3 flex flex-wrap gap-1.5">
        {shown.map((movement) => (
          <li
            key={movement}
            className="inline-flex items-center gap-1 rounded-full bg-content/[0.04] px-2 py-1 text-[11px] font-medium text-muted"
          >
            <MovementIcon movement={movement} size={13} />
            {movementLabel(t, movement)}
          </li>
        ))}
        {overflow > 0 && (
          <li className="inline-flex items-center rounded-full bg-content/[0.04] px-2 py-1 text-[11px] font-medium text-faint">
            +{overflow}
          </li>
        )}
      </ul>

      {plan.started_at && (
        <p className="mt-auto pt-3 text-[11px] text-faint">
          {t("plans.startedOn", {
            date: new Date(plan.started_at).toLocaleDateString(lang),
          })}
        </p>
      )}
    </Link>
  );
}
