import { Info, Lock, VideoCamera } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import MovementArt from "./MovementArt";
import MovementIcon from "./MovementIcon";
import { movementLabel, useI18n } from "../../lib/i18n";

interface Props {
  /** Canonical English movement name — the data key. Never rendered; see `movementLabel`. */
  movement: string;
  /** Absent when the pipeline has no detector for this movement, so the card cannot be analysed.
   *  It still links to its detail page — that is what a locked movement DOES have. */
  live?: { validated: boolean };
  onPick: () => void;
}

// One movement, as the exercise_library_muse-spark card: a titled tile with a preview stage and
// its actions along the bottom.
//
// TWO DEPARTURES FROM THE REFERENCE CARD:
//
//  * Its category chip is dropped. The chip exists in the mock BECAUSE the mock has no sections —
//    it filters with pills over one flat grid. This page keeps its body-region sections, so a chip
//    would restate the heading directly above it on every card in the section.
//  * The reference's single "View details" button became a pair, side by side along the bottom:
//    "View details" on the left, the primary action on the right. The mock's library has only the
//    first because its detail page is where you start an analysis; here the library IS the launcher
//    for the analyzable movements, and demoting that to a detail page would put a click in front of
//    the main task.
//
// The card is NOT one big button any more. It has two destinations now — the studio and the detail
// page — and a card-sized control with a nested link inside it is a control whose hit area lies
// about where it goes.
// The two halves of the action row share a shape, so the pair reads as one control strip rather
// than two things that happen to be adjacent. `min-w` is what makes them wrap instead of squeezing.
//
// The icons fit again now that the primary action is the bare verb: "Analyze a video" was ~92px of
// the ~116px each half gets at the four-column breakpoint, which left the label touching both edges
// of its pill. "Analyze" is half that.
const SLOT =
  "flex min-w-[104px] flex-1 items-center justify-center gap-1.5 rounded-xl px-2 py-2 text-xs transition-colors";
const ACTIVE =
  "bg-primary font-semibold text-primary-content shadow-accent hover:bg-primary/90 active:scale-[0.99]";

export default function MovementCard({ movement, live, onPick }: Props) {
  const { t } = useI18n();
  const name = movementLabel(t, movement);

  // The preview stage. Square, because the figures are a mix of tall (a standing press) and wide
  // (a push-up plank) and only a square holds both at a usable size — in a 16:9 band the standing
  // ones scale to its height and end up a third of the tile wide, with dead space either side.
  // Pale, because the illustrations are drawn as black-outlined figures for a white page and that
  // outline disappears against anything dark.
  //
  // No padding around the art itself: the 512px source PNGs are opaque, pre-matted squares (not
  // the old trimmed transparent cutouts), so insetting them left a hard-edged white box floating
  // over the gradient. Full-bleed lets the square art fill the tile and the rounded corners clip
  // it; the gradient still shows through as backdrop for the icon-only fallback below.
  const stage = (
    <span className="mt-3 block aspect-square overflow-hidden rounded-xl bg-gradient-to-b from-[#f7f5ff] to-[#eceefb]">
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

  // The detail page exists for all sixteen, analysable or not — it is knowledge about the
  // movement, not a result of analysing one. It sits on the LEFT of the action row, the quieter
  // half of the pair.
  const details = (
    <Link
      to={`/movements/${encodeURIComponent(movement)}`}
      className={`${SLOT} border border-border-dark font-medium text-muted hover:border-primary/40 hover:text-content`}
    >
      <Info size={14} weight="duotone" className="shrink-0" />
      <span className="truncate">{t("movements.viewDetails")}</span>
    </Link>
  );

  // The right-hand half: the analysis, or the reason there isn't one.
  const action = live ? (
    <button type="button" onClick={onPick} className={`${SLOT} ${ACTIVE}`}>
      <VideoCamera size={14} weight="fill" className="shrink-0" />
      <span className="truncate">{t("movements.analyze")}</span>
    </button>
  ) : (
    // Not a disabled <button>: there is no analysis behind it, so it is content rather than a
    // control pretending to be one.
    <span className={`${SLOT} bg-content/[0.04] font-medium text-faint`}>
      <Lock size={14} weight="duotone" className="shrink-0" />
      <span className="truncate">{t("movements.soon")}</span>
    </span>
  );

  return (
    <div
      className={`flex h-full flex-col rounded-2xl border border-border-dark bg-surface p-3 ${
        live
          ? "shadow-card transition-all hover:-translate-y-0.5 hover:border-primary/35 hover:shadow-card-hover"
          : ""
      }`}
    >
      {title}
      {stage}
      {/* One row, details first. `flex-wrap` with a min width on each half is the concession to a
          narrow phone: two 2-column cards at 375px leave ~70px a side, which would truncate both
          labels to nothing. Where they fit — which is every card from `sm` up — they sit side by
          side; below that they stack rather than lie about what they say. */}
      <div className="mt-3 flex flex-wrap gap-2">
        {details}
        {action}
      </div>
    </div>
  );
}
