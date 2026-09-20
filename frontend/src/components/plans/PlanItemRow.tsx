import { Check, ChartBar, ClipboardText, Lock, Trash, VideoCamera } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import MovementArt from "../movements/MovementArt";
import { movementLabel, useI18n } from "../../lib/i18n";
import type { PlanItem } from "../../api";

interface Props {
  item: PlanItem;
  planId: string;
  /** False when no detector is registered for this movement (Jumping Jacks, High Knee): the card
   *  offers only the manual tick, and says why. */
  analyzable: boolean;
  /** A write for this card is in flight — the controls lock rather than queueing a second one. */
  busy: boolean;
  onToggle: () => void;
  onRemove: () => void;
  /** Opens the check-in dialog for THIS item. Present on every card, assigned or self-made —
   *  reporting a session's pain/effort is useful whether or not a therapist is watching. Absent
   *  callers (e.g. an older story that hasn't wired it up yet) simply get no button. */
  onCheckin?: () => void;
}

// One exercise inside a day, as a CARD led by the movements-library figure. It was a thin
// horizontal band with a 17px glyph; at that size the figure said nothing a reader could use, so
// the exercise is now recognised by its picture the way it is on the library page, and the card is
// deliberately shaped as a sibling of `MovementCard` rather than a second visual language.
//
// THE STAGE. Square and full-bleed, for the reason recorded at length in MovementCard.tsx: the
// 1254px source PNGs are OPAQUE, PRE-MATTED squares, not trimmed transparent cutouts, so insetting
// them — or dropping them into a wide, short band — leaves a hard-edged white box floating over
// the gradient. Square also holds both the tall figures (a standing press) and the wide ones (a
// push-up plank) at a usable size, which a 16:9 band cannot.
//
// THREE CONTROLS, STILL SEPARATE. The tick (I did this), the studio link (record it and let the
// coach look), and remove are deliberately three controls, not one. Folding the tick into the
// studio link would make "I trained but didn't film it" unrecordable, which is most sessions. The
// tick moved onto the stage as a 36px overlay so the art can have the card's full width; it is
// still its own button with its own `aria-pressed` and its own labels.
export default function PlanItemRow({
  item,
  planId,
  analyzable,
  busy,
  onToggle,
  onRemove,
  onCheckin,
}: Props) {
  const { t } = useI18n();
  const label = movementLabel(t, item.movement);
  const done = !!item.completed_at;

  // Carries the plan item through to the studio, which ticks it off and links the analysis once the
  // upload persists (see App.tsx). `plan` rides along so the studio can offer a way back.
  const studioHref =
    `/app?movement=${encodeURIComponent(item.movement)}` +
    `&plan=${encodeURIComponent(planId)}&plan_item=${encodeURIComponent(item.id)}`;

  const FOCUS =
    "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary";

  // The action slot shares a shape across all three of its states so the bottom of every card in
  // the grid lines up. `min-w-0 flex-1` (not `shrink-0`) is what lets it yield to the remove
  // button in a ~131px phone cell instead of pushing out of the card.
  const SLOT =
    `inline-flex min-h-[36px] min-w-0 flex-1 items-center justify-center gap-1 rounded-full px-2.5 text-[11px] transition-colors ${FOCUS}`;

  return (
    <li
      // `min-w-0` so the card can actually shrink to its grid cell. Without it a flex/grid item
      // floors at its intrinsic content width and pushes out of the day panel instead of letting
      // the label truncate — which is exactly how this used to overflow.
      className={`flex h-full min-w-0 flex-col rounded-2xl border p-2.5 transition-colors ${
        done
          ? "border-secondary/45 bg-secondary/[0.07]"
          : "border-border-dark bg-surface hover:border-primary/35"
      }`}
    >
      <div className="relative">
        <span className="block aspect-square overflow-hidden rounded-xl bg-gradient-to-b from-[#f7f5ff] to-[#eceefb]">
          <MovementArt movement={item.movement} />
        </span>

        {/* The tick sits ON the stage because the stage is the card's full width and there is no
            room beside it for a control. It carries its own solid backdrop rather than relying on
            the art behind it: the figures are drawn light, and a translucent badge over a pale
            limb is not readable as either state. */}
        <button
          type="button"
          onClick={onToggle}
          disabled={busy}
          aria-pressed={done}
          aria-label={done ? t("plans.markUndone", { movement: label }) : t("plans.markDone", { movement: label })}
          title={done ? t("plans.markUndone", { movement: label }) : t("plans.markDone", { movement: label })}
          // `motion-safe:` on the press, because a transform is the one thing here that a reader
          // who asked for reduced motion should not get; the colour transitions are fine either way.
          className={`absolute left-1.5 top-1.5 flex h-9 w-9 items-center justify-center rounded-full shadow-sm ring-1 transition-colors motion-safe:active:scale-95 disabled:opacity-50 ${FOCUS} ${
            done
              ? // Unticking is a real action and this badge is its only affordance, so the done
                // state gets a hover of its own rather than sitting there looking inert.
                "bg-secondary text-white ring-secondary/60 hover:bg-secondary/85 active:bg-secondary/75"
              : "bg-white text-content/30 ring-black/15 hover:text-secondary hover:ring-secondary/40 active:bg-secondary/10"
          }`}
        >
          <Check size={18} weight="bold" />
        </button>
      </div>

      <p
        className={`mt-2 truncate text-[13px] font-medium ${done ? "text-muted line-through" : "text-content"}`}
      >
        {label}
      </p>
      <p className="truncate text-[11px] text-faint">
        {t("plans.setsReps", { sets: item.sets, reps: item.reps })}
      </p>
      {item.notes && <p className="line-clamp-2 text-[11px] text-faint">{item.notes}</p>}

      {/* `mt-auto` so the action row sits on the card's floor whatever the note above it does —
          grid cells stretch to the tallest card in their row, and a floating action row would put
          three cards' buttons at three different heights. */}
      <div className="mt-auto flex items-center gap-1.5 pt-2">
        {/* Only ever ONE of these three: the report (this item produced an analysis), the studio
            link (it can produce one), or the tick-only note (it never can). */}
        {/* The VISIBLE label is the short one; the full phrase is the accessible name. The card is
            as narrow as half a phone, and the full "Record & analyse" / 錄影並分析 does not fit
            beside the remove button. Screen readers and tooltips still get the full wording. */}
        {item.analysis_id ? (
          <Link
            to={`/app?analysis=${encodeURIComponent(item.analysis_id)}`}
            aria-label={t("plans.viewReport")}
            title={t("plans.viewReport")}
            className={`${SLOT} bg-content/[0.04] font-semibold text-content hover:bg-primary hover:text-primary-content active:bg-primary/85 motion-safe:active:scale-[0.98]`}
          >
            <ChartBar size={13} weight="duotone" className="shrink-0" />
            <span className="truncate">{t("plans.viewReportShort")}</span>
          </Link>
        ) : analyzable ? (
          <Link
            to={studioHref}
            aria-label={t("plans.analyze")}
            title={t("plans.analyze")}
            className={`${SLOT} bg-content/[0.04] font-semibold text-content hover:bg-primary hover:text-primary-content active:bg-primary/85 motion-safe:active:scale-[0.98]`}
          >
            <VideoCamera size={13} weight="fill" className="shrink-0" />
            <span className="truncate">{t("plans.analyzeShort")}</span>
          </Link>
        ) : (
          // Not a disabled button: there is no action behind it. It is a labelled note explaining
          // why this card has no studio link, which is the same choice MovementCard makes for the
          // movements its "Soon" tile cannot open.
          <span
            title={t("plans.tickOnly")}
            className={`${SLOT} bg-content/[0.04] font-medium text-faint`}
          >
            {/* Deliberately NOT `movements.soon`. On the movement menu "Soon" means the movement
                itself is unavailable; here the movement is perfectly plannable and only its VIDEO
                ANALYSIS is missing, so it gets a label that says that. */}
            <Lock size={13} weight="duotone" className="shrink-0" />
            <span className="truncate">{t("plans.tickOnlyLabel")}</span>
          </span>
        )}

        {/* Present on every card, assigned or self-made — reporting how a session felt is not
            gated on having a therapist watching. A 36px icon button, same footprint as remove
            below it, rather than a fourth SLOT competing with the primary action for width. */}
        {onCheckin && (
          <button
            type="button"
            onClick={onCheckin}
            aria-label={t("plans.checkin")}
            title={t("plans.checkin")}
            className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-faint transition-colors hover:bg-primary/10 hover:text-primary motion-safe:active:scale-95 ${FOCUS}`}
          >
            <ClipboardText size={15} weight="duotone" />
          </button>
        )}

        {/* Quiet, but a real 36px target — it was a 22px `p-1` glyph, which is under the floor for
            anything meant to be tapped. */}
        <button
          type="button"
          onClick={onRemove}
          disabled={busy}
          aria-label={t("plans.removeItem", { movement: label })}
          className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-faint transition-colors hover:bg-danger/10 hover:text-danger active:bg-danger/20 motion-safe:active:scale-95 disabled:opacity-50 ${FOCUS}`}
        >
          <Trash size={15} weight="duotone" />
        </button>
      </div>
    </li>
  );
}
