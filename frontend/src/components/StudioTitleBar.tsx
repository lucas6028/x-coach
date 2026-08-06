import { CaretDown, Gauge, Lightning, Play } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import { movementLabel, useI18n } from "../lib/i18n";
import type { AnalyzableMovement } from "../lib/movements";
import type { PoseTier } from "../lib/poseTier";

// Model tiers keep MediaPipe's own names in every language — they are product nouns, not copy.
const TIER_NAME: Record<PoseTier, string> = { lite: "Lite", full: "Full", heavy: "Heavy" };
const TIERS: PoseTier[] = ["lite", "full", "heavy"];

interface Props {
  movement: string;
  movements: AnalyzableMovement[];
  onMovementChange: (movement: string) => void;
  tier: PoseTier;
  onTierChange: (tier: PoseTier) => void;
  /** Rendered only once a result is on screen — in the empty state the dropzone right below is
   *  already the call to action, so the button would be a second, weaker copy of it. */
  onNewSession?: () => void;
}

// One of the reference design's header controls: a small white card carrying a caption over the
// current value, with a caret on the right. The real control is a native <select> stretched
// invisibly across the card, so keyboard, mobile pickers and the accessible name all come free —
// the drawn parts are presentation only.
function SelectCard({
  id,
  label,
  value,
  display,
  icon,
  tint,
  options,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  display: string;
  icon: React.ReactNode;
  tint: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <div className="glass-control relative flex min-w-[150px] items-center gap-3 rounded-2xl px-4 py-2.5 transition-colors focus-within:border-[#c9bcff]">
      <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${tint}`}>
        {icon}
      </span>
      <span className="min-w-0 flex-1 text-left leading-tight">
        <label htmlFor={id} className="block text-[10px] font-medium text-[#65719f]">
          {label}
        </label>
        <span className="-mt-0.5 block truncate text-[13px] font-semibold text-[#1e2142]">
          {display}
        </span>
      </span>
      <CaretDown size={13} weight="bold" className="shrink-0 text-[#63709f]" />
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="absolute inset-0 h-full w-full cursor-pointer appearance-none opacity-0"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

// The studio's page header, straight from the reference: breadcrumb, title + subtitle on the
// left, the analysis controls and the primary action on the right.
export default function StudioTitleBar({
  movement,
  movements,
  onMovementChange,
  tier,
  onTierChange,
  onNewSession,
}: Props) {
  const { t } = useI18n();
  const label = movementLabel(t, movement);
  const beta = movements.find((m) => m.name === movement)?.validated === false;

  // Keep a URL-supplied movement the catalog does not know visible rather than silently snapping
  // the control to something the user did not choose.
  const options = movements.map((m) => ({ value: m.name, label: movementLabel(t, m.name) }));
  if (!movements.some((m) => m.name === movement)) options.push({ value: movement, label });

  return (
    // `xc-rise` is the theme's page-header entrance (index.css); it collapses under reduced motion.
    <div className="xc-rise shrink-0">
      {/* `flex-wrap`: the last crumb carries the movement name, which is user-supplied and can be
          long enough to push the row past a phone's width. */}
      <nav className="mb-1 flex flex-wrap items-center gap-x-1.5 text-[12px] font-medium text-[#65719f]">
        <Link to="/" className="transition-colors hover:text-primary">
          {t("studio.crumbHome")}
        </Link>
        <span className="opacity-60">/</span>
        <Link to="/movements" className="transition-colors hover:text-primary">
          {t("studio.crumbWorkout")}
        </Link>
        <span className="opacity-60">/</span>
        <span className="text-[#63709f]">{t("studio.crumbCurrent", { movement: label })}</span>
      </nav>

      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="min-w-0">
          <h1 className="font-display text-[26px] font-bold leading-none tracking-tight text-[#1e2142] lg:text-[32px]">
            {t("studio.title", { movement: label })}
          </h1>
          <p className="mt-1.5 text-[13.5px] font-medium text-[#63709f]">{t("studio.subtitle")}</p>
        </div>

        {/* From `xl` these controls share a line with the page title, and the shell's account
            cluster floats over the top-right corner of that same band (AppLayout) — bottom-
            aligning them ran the movement/precision cards straight under the avatar. The offset
            drops this cluster clear of that 40px strip. Below `xl` the header stacks, so the
            controls are already on their own line and the offset does not apply. */}
        <div className="flex flex-wrap items-center gap-3 xl:mt-[52px]">
          <div className="flex items-center gap-2">
            <SelectCard
              id="movement-select"
              label={t("studio.movement")}
              value={movement}
              display={label}
              tint="bg-[#f3f0ff] text-primary"
              icon={<Lightning size={16} weight="fill" />}
              options={options}
              onChange={onMovementChange}
            />
            {beta && (
              <span
                title={t("movements.betaNote")}
                className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-warning ring-1 ring-warning/40"
              >
                {t("movements.beta")}
              </span>
            )}
          </div>

          {/* The reference's second dropdown ("Device") has no counterpart here — capture source is
              picked in the dropzone itself. The real setting that belongs in this slot is the
              extraction tier, which is persisted and materially changes the verdicts. */}
          <SelectCard
            id="tier-select"
            label={t("tier.label")}
            value={tier}
            display={TIER_NAME[tier]}
            tint="bg-[#f0f0ff] text-[#5b4bff]"
            icon={<Gauge size={16} weight="fill" />}
            options={TIERS.map((x) => ({ value: x, label: TIER_NAME[x] }))}
            onChange={(v) => onTierChange(v as PoseTier)}
          />

          {onNewSession && (
            <button
              onClick={onNewSession}
              // The theme drops gradients on primary buttons in favour of one solid violet with a
              // violet-tinted lift (its `solid` + the containedPrimary shadow); the gradient is
              // kept for small round fills only.
              className="flex items-center gap-2 rounded-2xl bg-[#7a45ff] px-6 py-[13px] text-[13px] font-semibold text-white shadow-[0_10px_24px_rgba(119,62,252,0.20)] transition-all hover:brightness-110 active:scale-[0.98]"
            >
              <Play size={14} weight="fill" />
              {t("studio.newSession")}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
