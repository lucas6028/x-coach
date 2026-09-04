import type { PlanItem } from "../../api";
import { useI18n } from "../../lib/i18n";
import type { Muscle } from "../../lib/movementDetail";
import { planCoverage } from "../../lib/planMuscles";
import MuscleMap from "../movements/MuscleMap";

interface Props {
  items: PlanItem[];
  className?: string;
  /** Drop the mannequins and keep the named groups. For a column too narrow to show two bodies
   *  side by side at a size where the highlights are readable — under about 320px they are two
   *  grey smudges, which is worse than not drawing them. */
  compact?: boolean;
}

/**
 * What a plan trains, shown the same way a single movement's page shows it: the two bodies with the
 * worked groups lit, and the groups named beside them. The difference is what feeds it — a whole
 * week's items aggregated by `lib/planMuscles.ts` rather than one movement's authored lists — so a
 * user can see their week's shape without opening sixteen movement pages.
 *
 * Deliberately static: no transitions, so there is nothing for `useReducedMotion` to turn off.
 *
 * TWO EMPTY STATES, and only one of them is a state:
 *  - nothing trained at all (an empty plan, or every movement outside the catalog) → renders
 *    NOTHING. "Not trained this week: everything" is not a useful thing to tell someone whose plan
 *    has no exercises in it yet.
 *  - trained, but no gaps → the balanced line. Never a heading over an empty box.
 */
export default function PlanMuscleCoverage({ items, className, compact }: Props) {
  const { t, lang } = useI18n();
  const { primary, secondary, gaps } = planCoverage(items);

  if (primary.length === 0 && secondary.length === 0) return null;

  const names = (muscles: Muscle[]) =>
    muscles.map((m) => t(`muscle.${m}`)).join(lang === "zh-Hant" ? "、" : ", ");

  return (
    <section
      className={`rounded-2xl border border-border-dark bg-surface p-4 ${className ?? ""}`.trim()}
    >
      <h2 className="text-[15.5px] font-bold text-content">{t("plans.muscles.title")}</h2>

      {/* NOT a `sm:` breakpoint: `sm:` asks how wide the WINDOW is, and this card's width comes
          from its container -- with the coach panel open the window is 1280px while the card is
          a fraction of that, which is how the legend ended up a one-character-per-line strip in
          zh-Hant. `compact` is the container signal, passed by the page that owns the layout. */}
      <div
        className={
          compact ? "mt-3" : "mt-3 flex flex-col items-stretch gap-4 sm:flex-row sm:items-center"
        }
      >
        {!compact && (
          // Sized down from the movement page's 208px: this card sits under a whole week of day
          // bands rather than being the point of its own page.
          <div className="mx-auto flex max-w-[380px] flex-1 items-end justify-evenly">
            <figure className="text-center">
              <MuscleMap
                side="front"
                primary={primary}
                secondary={secondary}
                className="h-[164px] w-auto"
              />
              <figcaption className="mt-1.5 text-[11px] font-medium text-muted">
                {t("detail.anterior")}
              </figcaption>
            </figure>
            <figure className="text-center">
              <MuscleMap
                side="back"
                primary={primary}
                secondary={secondary}
                className="h-[164px] w-auto"
              />
              <figcaption className="mt-1.5 text-[11px] font-medium text-muted">
                {t("detail.posterior")}
              </figcaption>
            </figure>
          </div>
        )}

        {/* A stack, the same shape the movement page's legend uses -- one name per line reads at a
            glance where a ragged wrapping flow does not, and a fixed column can never be narrower
            than a label. Compact drops the fixed width and takes the card. */}
        <div className={compact ? "min-w-0 space-y-4" : "min-w-0 shrink-0 space-y-4 sm:w-[236px]"}>
          <MuscleGroupList muscles={primary} tone="primary" />
          <MuscleGroupList muscles={secondary} tone="secondary" />
        </div>
      </div>

      <p className="mt-4 border-t border-border-dark pt-3 text-[12px] leading-relaxed text-faint">
        {gaps.length > 0 ? t("plans.muscles.gaps", { muscles: names(gaps) }) : t("plans.muscles.balanced")}
      </p>
    </section>
  );
}

/** One tone's worth of groups, with the dot treatment the movement detail page's legend uses:
 *  solid violet for the prime movers, pale for the supporting ones. */
function MuscleGroupList({ muscles, tone }: { muscles: Muscle[]; tone: "primary" | "secondary" }) {
  const { t } = useI18n();
  if (muscles.length === 0) return null;
  const dot = tone === "primary" ? "bg-primary" : "bg-primary/30";
  return (
    <div>
      <p className="mb-2 flex items-center gap-2 text-[12px] font-semibold text-content">
        <span className={`h-2.5 w-2.5 rounded-full ${dot}`} />
        {t(tone === "primary" ? "detail.primary" : "detail.secondary")}
      </p>
      <ul className="pl-[18px]">
        {/* `whitespace-nowrap`: a two-word label ("lower back") that wraps leaves its dot alone on
            the line above, which reads as a group with no name. */}
        {muscles.map((m) => (
          <li
            key={m}
            className="mb-1.5 flex items-center gap-2 whitespace-nowrap text-[12.5px] text-muted"
          >
            <span className={`h-2 w-2 shrink-0 rounded-full ${dot}`} />
            {t(`muscle.${m}`)}
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * The one-line version, for a plan CARD: the top few groups, named, with a pale dot each.
 *
 * Deliberately NOT the same shape as the movement pills above it on PlanCard — a second row of
 * identical grey pills turns the card into one undifferentiated blob, and these two rows answer
 * different questions ("which exercises" against "which muscles").
 */
export function MuscleSummaryChips({
  muscles,
  limit = 3,
  className,
}: {
  muscles: Muscle[];
  limit?: number;
  className?: string;
}) {
  const { t } = useI18n();
  if (muscles.length === 0) return null;
  const shown = muscles.slice(0, limit);
  const overflow = muscles.length - shown.length;
  return (
    <p
      className={`flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] font-medium text-faint ${
        className ?? ""
      }`.trim()}
    >
      {/* Named, because on a plan card this line sits directly under the movement chips and two
          rows of bare words is a card that has to be decoded. */}
      <span className="font-semibold uppercase tracking-wider">{t("plans.muscles.trains")}</span>
      {shown.map((m) => (
        <span key={m} className="inline-flex items-center gap-1">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-primary/50" />
          {t(`muscle.${m}`)}
        </span>
      ))}
      {overflow > 0 && <span className="tabular-nums">{t("plans.muscles.more", { n: overflow })}</span>}
    </p>
  );
}
