import type { PlanItem } from "../../api";
import { useI18n } from "../../lib/i18n";
import { movementDetail, type Muscle } from "../../lib/movementDetail";
import { planCoverage } from "../../lib/planMuscles";

interface Props {
  items: PlanItem[];
  className?: string;
  /** Drop the anatomical plate and keep the named groups. For a column too narrow to show the two
   *  bodies at a size where the highlights are readable — under about 320px they are two grey
   *  smudges, which is worse than not drawing them. */
  compact?: boolean;
}

/**
 * What a plan trains, shown the same way a single movement's page shows it: the two bodies with the
 * worked groups lit, and the groups named beside them. The difference is what feeds it — a whole
 * week's items aggregated by `lib/planMuscles.ts` rather than one movement's authored lists — so a
 * user can see their week's shape without opening sixteen movement pages.
 *
 * THE FIGURE IS THE SHIPPED ARTWORK, STACKED — not a drawing of our own. Every catalog movement
 * already has an anatomical plate under `public/movements/muscles-worked/`, the same plate its
 * movement detail page shows, and they are all the SAME base body with different muscles tinted.
 * So the plan's plates layered with `mix-blend-mode: darken` come out as exactly the union of
 * their highlights on one pair of bodies: a week's coverage drawn by the illustrator rather than
 * approximated by us, and primary still distinguishable from secondary (deep violet wins over the
 * pale one, which is the right answer when one movement's supporting group is another's prime
 * mover).
 *
 * WHY `darken` AND NOT `multiply`, which is the usual reflex for stacking artwork: multiply
 * ACCUMULATES. Every plate draws the whole body, so the grey head and the grey untrained muscles
 * are multiplied once per layer — measured on a nine-movement week, the figures came out nearly
 * black. `darken` takes the per-channel minimum, so a grey drawn nine times is still that grey and
 * only a genuinely tinted muscle darkens. Anything that stacks more than two or three plates has
 * to use it.
 *
 * Two consequences worth knowing before editing:
 *  - the blend needs an OPAQUE, light backdrop to behave, hence the explicit `bg-surface` on the
 *    stack, and `isolate` so the blending stops there instead of reaching the card behind it.
 *  - one plate carries anterior AND posterior, so there is ONE image and no "Anterior"/"Posterior"
 *    captions left to write; the legend beside it names the groups, which is also why the layers
 *    carry no alt text of their own.
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
  const plates = planPlates(items);
  // Nothing to layer is not an empty frame: a legend column laid out as if a figure sat beside it
  // is a hole in the card, so the figure's absence collapses the row instead.
  const showFigure = !compact && plates.length > 0;

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
          showFigure
            ? // The group is CAPPED and centred rather than spread across the card. Left to grow,
              // the figure's flex column took ~1600px of a full-width card and centred a 200px
              // image in it, parking the legend against the far edge with a lake of white between
              // the two things the reader is meant to compare.
              "mt-3 mx-auto flex max-w-[600px] flex-col items-stretch gap-6 sm:flex-row sm:items-center"
            : "mt-3"
        }
      >
        {showFigure && (
          <div className="flex shrink-0 justify-center">
            {/* The box is the plates' own ratio (1000x986), sized down from the movement page's
                250px because this card sits under a whole week of day bands rather than being the
                point of its own page. */}
            <div className="relative isolate aspect-[1000/986] h-[232px] bg-surface">
              {plates.map(({ movement, plate }) => (
                <img
                  key={movement}
                  src={plate}
                  alt=""
                  loading="lazy"
                  decoding="async"
                  className="absolute inset-0 h-full w-full object-contain"
                  style={{ mixBlendMode: "darken" }}
                />
              ))}
            </div>
          </div>
        )}

        {/* A stack, the same shape the movement page's legend uses -- one name per line reads at a
            glance where a ragged wrapping flow does not, and a fixed column can never be narrower
            than a label. Compact drops the fixed width and takes the card. */}
        <div
          className={showFigure ? "min-w-0 flex-1 space-y-4" : "min-w-0 space-y-4"}
        >
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

/** The plates to layer: one per DISTINCT movement in the plan, in the order the plan lists them.
 *  Deduplicated because a week that squats three times would otherwise stack the identical image
 *  three times — invisible under multiply, but three downloads and three layers to composite. A
 *  movement outside the catalog, or one whose entry carries no plate, contributes nothing. */
function planPlates(items: PlanItem[]): { movement: string; plate: string }[] {
  const seen = new Set<string>();
  const plates: { movement: string; plate: string }[] = [];
  for (const { movement } of items) {
    if (seen.has(movement)) continue;
    seen.add(movement);
    const plate = movementDetail(movement)?.plate;
    if (plate) plates.push({ movement, plate });
  }
  return plates;
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
      {/* Two columns: a plan lights far more groups than a single movement does, and one name per
          line turned a fifteen-group week into a column taller than the figure beside it. */}
      <ul className="grid grid-cols-2 gap-x-3 pl-[18px]">
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
