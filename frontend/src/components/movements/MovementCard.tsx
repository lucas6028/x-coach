import { Lock, VideoCamera } from "@phosphor-icons/react";
import MovementArt from "./MovementArt";
import MovementIcon from "./MovementIcon";
import { movementLabel, useI18n } from "../../lib/i18n";

interface Props {
  /** Canonical English movement name — the data key. Never rendered; see `movementLabel`. */
  movement: string;
  /** Absent when the pipeline has no detector for this movement, so the card is inert. */
  live?: { validated: boolean };
  onPick: () => void;
}

// One movement, as the exercise_library_muse-spark card: a titled tile with a preview stage and a
// single action along the bottom.
//
// THREE DEPARTURES FROM THE REFERENCE CARD, all forced by facts the mock does not have:
//
//  * Its "View details" button is dropped rather than wired. There is no movement-detail view in
//    this app, and shipping a control that opens nothing is the one thing a static mock can afford
//    and a running app cannot.
//  * Its category chip is dropped. The chip exists in the mock BECAUSE the mock has no sections —
//    it filters with pills over one flat grid. This page keeps its body-region sections, so a chip
//    would restate the heading directly above it on every card in the section.
//  * The whole live card is ONE button, not a div with a button in its footer. The card has a
//    single destination, so a nested control would be a second tab stop to the same place, and the
//    surrounding 200px of card would be dead to the keyboard. The violet bar along the bottom is
//    the affordance for the card, not a control of its own.
export default function MovementCard({ movement, live, onPick }: Props) {
  const { t } = useI18n();
  const name = movementLabel(t, movement);

  // The preview stage. Square, because the figures are a mix of tall (a standing press) and wide
  // (a push-up plank) and only a square holds both at a usable size — in a 16:9 band the standing
  // ones scale to its height and end up a third of the tile wide, with dead space either side.
  // Pale, because the illustrations are drawn as black-outlined figures for a white page and that
  // outline disappears against anything dark.
  const stage = (
    <span className="mt-3 block aspect-square overflow-hidden rounded-xl bg-gradient-to-b from-[#f7f5ff] to-[#eceefb] p-2.5">
      <MovementArt movement={movement} />
    </span>
  );

  const title = (
    <span className="flex min-w-0 items-center gap-1.5">
      <MovementIcon movement={movement} dim={!live} />
      <span className="truncate font-medium text-content">{name}</span>
      {live && !live.validated && (
        <span
          title={t("movements.betaNote")}
          className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-warning ring-1 ring-warning/40"
        >
          {t("movements.beta")}
        </span>
      )}
    </span>
  );

  if (!live) {
    // Not a disabled <button>: there is no action behind it, so it is listed as plain content
    // rather than as a control that pretends to be one.
    return (
      <div className="flex h-full flex-col rounded-2xl border border-border-dark bg-surface p-3">
        {title}
        {stage}
        <span className="mt-3 flex items-center justify-center gap-1.5 rounded-xl bg-content/[0.04] py-2 text-xs font-medium text-faint">
          <Lock size={14} weight="duotone" />
          {t("movements.soon")}
        </span>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={onPick}
      className="group flex h-full w-full flex-col rounded-2xl border border-border-dark bg-surface p-3 text-left shadow-card transition-all hover:-translate-y-0.5 hover:border-primary/35 hover:shadow-card-hover active:translate-y-0"
    >
      {title}
      {stage}
      <span className="mt-3 flex items-center justify-center gap-1.5 rounded-xl bg-primary py-2 text-xs font-semibold text-primary-content shadow-accent transition-colors group-hover:bg-primary/90">
        <VideoCamera size={14} weight="fill" />
        {t("movements.analyze")}
      </span>
    </button>
  );
}
