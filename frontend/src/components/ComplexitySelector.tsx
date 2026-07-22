import { useEffect, useRef, useState } from "react";
import { CaretDown, Gauge } from "@phosphor-icons/react";
import { useI18n } from "../lib/i18n";
import { DEFAULT_ANALYSIS_TIER, type PoseTier } from "../lib/poseTier";

const TIERS: readonly PoseTier[] = ["lite", "full", "heavy"];
// Model tiers keep MediaPipe's own names in every language — they are product nouns, not copy.
const TIER_NAME: Record<PoseTier, string> = { lite: "Lite", full: "Full", heavy: "Heavy" };

// Picks the model tier for the OFFLINE analysis extraction only — the live recording overlay is
// always Lite (LIVE_OVERLAY_TIER), which the panel footer says out loud.
//
// Shaped like Claude Code's effort control rather than a model list: the tiers sit on one
// Faster→Smarter axis with a single marker, and only the CURRENT tier's tradeoff is spelled out.
// A slider is the honest form here — the tiers are ordered by cost/accuracy, not parallel choices.
export default function ComplexitySelector({
  value,
  onChange,
}: {
  value: PoseTier;
  onChange: (tier: PoseTier) => void;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const sliderRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // Arrow keys should land on the axis the moment the panel opens, as they do in Claude Code.
  useEffect(() => {
    if (open) sliderRef.current?.focus();
  }, [open]);

  const index = Math.max(0, TIERS.indexOf(value));
  // Marker/fill position along the axis: first tier sits at 0%, last at 100%.
  const pct = (index / (TIERS.length - 1)) * 100;

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={t("tier.aria", { name: TIER_NAME[value] })}
        className="flex items-center gap-2 rounded-lg border border-border-dark px-3 py-2 text-sm text-muted transition-colors hover:bg-content/5 hover:text-content"
      >
        <Gauge size={16} weight="duotone" className="text-primary" />
        <span className="hidden text-faint sm:inline">{t("tier.label")}</span>
        <span className="font-medium text-content">{TIER_NAME[value]}</span>
        <CaretDown size={12} weight="bold" className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div
          role="dialog"
          aria-label={t("tier.label")}
          className="absolute right-0 z-50 mt-1.5 w-72 overflow-hidden rounded-xl border border-border-dark bg-surface-dark shadow-lg shadow-black/20"
        >
          <div className="px-4 pb-3 pt-3.5">
            {/* Axis ends — the tradeoff the slider travels along. */}
            <div className="flex justify-between text-[11px] text-faint">
              <span>{t("tier.faster")}</span>
              <span>{t("tier.smarter")}</span>
            </div>

            {/* The track. A real range input drives it (keyboard, drag, click-to-seek all come
                free); the visible rail/fill/marker are drawn behind it so they can use the app's
                own tokens, which cross-browser ::-webkit-slider-thumb styling cannot. */}
            <div className="relative mt-2 flex h-4 items-center">
              <div className="h-1 w-full rounded-full bg-content/10" />
              <div className="absolute h-1 rounded-full bg-primary" style={{ width: `${pct}%` }} />
              <div
                className="absolute h-3.5 w-3.5 -translate-x-1/2 rounded-full bg-primary ring-2 ring-surface-dark"
                style={{ left: `${pct}%` }}
              />
              <input
                ref={sliderRef}
                type="range"
                min={0}
                max={TIERS.length - 1}
                step={1}
                value={index}
                onChange={(e) => onChange(TIERS[Number(e.target.value)])}
                aria-label={t("tier.label")}
                aria-valuetext={TIER_NAME[value]}
                className="absolute inset-x-0 h-4 w-full cursor-pointer appearance-none bg-transparent opacity-0"
              />
            </div>

            {/* Tick labels. Pointer-only affordances: the slider above already exposes the value
                to assistive tech, so these stay out of the tab order and off the a11y tree. */}
            <div className="mt-1.5 flex justify-between">
              {TIERS.map((tier) => (
                <button
                  key={tier}
                  type="button"
                  tabIndex={-1}
                  aria-hidden="true"
                  onClick={() => onChange(tier)}
                  className={`text-xs transition-colors ${
                    tier === value ? "font-semibold text-content" : "text-faint hover:text-muted"
                  }`}
                >
                  {TIER_NAME[tier]}
                </button>
              ))}
            </div>

            {/* Only the CURRENT tier's tradeoff — the effort control's sublabel, not a list of
                descriptions to compare. */}
            <p className="mt-2.5 text-xs leading-snug text-faint">
              {t(`tier.${value}.hint`)}
              {value === DEFAULT_ANALYSIS_TIER && ` · ${t("tier.default")}`}
            </p>
          </div>

          <p className="border-t border-border-dark px-4 py-2 text-[11px] leading-snug text-faint">
            {t("tier.note")}
          </p>
        </div>
      )}
    </div>
  );
}
