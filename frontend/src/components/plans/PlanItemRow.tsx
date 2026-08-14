import { Check, ChartBar, Lock, Trash, VideoCamera } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import MovementIcon from "../movements/MovementIcon";
import { movementLabel, useI18n } from "../../lib/i18n";
import type { PlanItem } from "../../api";

interface Props {
  item: PlanItem;
  planId: string;
  /** False when no detector is registered for this movement (Jumping Jacks, High Knee): the row
   *  offers only the manual tick, and says why. */
  analyzable: boolean;
  /** A write for this row is in flight — the controls lock rather than queueing a second one. */
  busy: boolean;
  onToggle: () => void;
  onRemove: () => void;
}

// One exercise inside a day. Three things live on the row and they are deliberately separate
// controls, not one: the tick (I did this), the studio link (record it and let the coach look), and
// the remove button. Folding the tick into the studio link would make "I trained but didn't film
// it" unrecordable, which is most sessions.
export default function PlanItemRow({
  item,
  planId,
  analyzable,
  busy,
  onToggle,
  onRemove,
}: Props) {
  const { t } = useI18n();
  const label = movementLabel(t, item.movement);
  const done = !!item.completed_at;

  // Carries the plan item through to the studio, which ticks it off and links the analysis once the
  // upload persists (see App.tsx). `plan` rides along so the studio can offer a way back.
  const studioHref =
    `/app?movement=${encodeURIComponent(item.movement)}` +
    `&plan=${encodeURIComponent(planId)}&plan_item=${encodeURIComponent(item.id)}`;

  return (
    <li
      // `min-w-0` so the row can actually shrink to its grid cell. Without it a flex/grid item
      // floors at its intrinsic content width and pushes out of the card instead of letting the
      // label truncate — which is exactly how this row used to overflow its day.
      className={`flex min-w-0 items-center gap-2.5 rounded-xl border px-3 py-2.5 transition-colors ${
        done ? "border-secondary/30 bg-secondary/[0.06]" : "border-border-dark bg-surface"
      }`}
    >
      <button
        type="button"
        onClick={onToggle}
        disabled={busy}
        aria-pressed={done}
        aria-label={done ? t("plans.markUndone", { movement: label }) : t("plans.markDone", { movement: label })}
        className={`flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-md border transition-colors disabled:opacity-50 ${
          done
            ? "border-secondary bg-secondary text-white"
            : "border-border-dark text-transparent hover:border-primary/50"
        }`}
      >
        <Check size={13} weight="bold" />
      </button>

      <MovementIcon movement={item.movement} size={17} dim={done} />

      <div className="min-w-0 flex-1">
        <p
          className={`truncate text-[13px] font-medium ${done ? "text-muted line-through" : "text-content"}`}
        >
          {label}
        </p>
        <p className="text-[11px] text-faint">
          {t("plans.setsReps", { sets: item.sets, reps: item.reps })}
          {item.notes ? ` · ${item.notes}` : ""}
        </p>
      </div>

      {/* Only ever ONE of these three: the report (this item produced an analysis), the studio link
          (it can produce one), or the tick-only note (it never can). */}
      {/* The VISIBLE label is the short one; the full phrase is the accessible name.
          "Record & analyse" / 錄影並分析 renders ~128px wide, and the row's fixed parts already
          come to ~100px — together they left nothing for the movement name, which is how the name
          vanished entirely. Screen readers and tooltips still get the full wording, so nothing is
          actually lost by shortening what is drawn. */}
      {item.analysis_id ? (
        <Link
          to={`/app?analysis=${encodeURIComponent(item.analysis_id)}`}
          aria-label={t("plans.viewReport")}
          title={t("plans.viewReport")}
          className="inline-flex shrink-0 items-center gap-1 rounded-full bg-content/[0.04] px-2.5 py-1 text-[11px] font-semibold text-content transition-colors hover:bg-primary hover:text-primary-content"
        >
          <ChartBar size={12} weight="duotone" />
          {t("plans.viewReportShort")}
        </Link>
      ) : analyzable ? (
        <Link
          to={studioHref}
          aria-label={t("plans.analyze")}
          title={t("plans.analyze")}
          className="inline-flex shrink-0 items-center gap-1 rounded-full bg-content/[0.04] px-2.5 py-1 text-[11px] font-semibold text-content transition-colors hover:bg-primary hover:text-primary-content"
        >
          <VideoCamera size={12} weight="fill" />
          {t("plans.analyzeShort")}
        </Link>
      ) : (
        // Not a disabled button: there is no action behind it. It is a labelled note explaining
        // why this row has no studio link, which is the same choice MovementCard makes for the
        // movements its "Soon" tile cannot open.
        <span
          title={t("plans.tickOnly")}
          className="inline-flex shrink-0 items-center gap-1 rounded-full bg-content/[0.04] px-2.5 py-1 text-[11px] font-medium text-faint"
        >
          {/* Deliberately NOT `movements.soon`. On the movement menu "Soon" means the movement
              itself is unavailable; here the movement is perfectly plannable and only its VIDEO
              ANALYSIS is missing, so it gets a label that says that. */}
          <Lock size={12} weight="duotone" />
          {t("plans.tickOnlyLabel")}
        </span>
      )}

      <button
        type="button"
        onClick={onRemove}
        disabled={busy}
        aria-label={t("plans.removeItem", { movement: label })}
        className="shrink-0 rounded-lg p-1 text-faint transition-colors hover:bg-danger/10 hover:text-danger disabled:opacity-50"
      >
        <Trash size={14} weight="duotone" />
      </button>
    </li>
  );
}
