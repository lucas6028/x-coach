import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { CaretDown, Gauge, X } from "@phosphor-icons/react";
import { useI18n } from "../lib/i18n";
import { DEFAULT_ANALYSIS_TIER, type PoseTier } from "../lib/poseTier";

const TIERS: readonly PoseTier[] = ["lite", "full", "heavy"];
// Model tiers keep MediaPipe's own names in every language — they are product nouns, not copy.
const TIER_NAME: Record<PoseTier, string> = { lite: "Lite", full: "Full", heavy: "Heavy" };

// Picks the model tier for the OFFLINE analysis extraction only — the live recording overlay is
// always Lite (LIVE_OVERLAY_TIER), which the panel footer says out loud.
//
// A reasoning-effort control, not a model list: the tiers sit on one ordered axis (they trade cost
// against accuracy — they are not parallel alternatives), the panel states only where you ARE, and
// the track is the chunky dotted pill with a white knob from the reference design. Accent stays on
// the app's own `primary` token rather than the reference's violet.
export default function ComplexitySelector({
  value,
  onChange,
}: {
  value: PoseTier;
  onChange: (tier: PoseTier) => void;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  // Non-null only while a pointer drag is in flight: the raw fractional position, so the knob
  // tracks the finger continuously instead of teleporting between stops. Released -> nearest stop.
  const [dragPos, setDragPos] = useState<number | null>(null);
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

  // Arrow keys should land on the axis the moment the panel opens.
  useEffect(() => {
    if (open) sliderRef.current?.focus();
  }, [open]);

  const index = Math.max(0, TIERS.indexOf(value));
  const last = TIERS.length - 1;
  // Where the knob is drawn: the live drag position, else the committed tier. Percentages resolve
  // against the knob-inset inner rail, so the knob never hangs off either end of the track.
  const shown = dragPos ?? index;
  const pct = (shown / last) * 100;
  const dragging = dragPos !== null;

  const clampTier = (i: number) => TIERS[Math.min(last, Math.max(0, i))];

  // End of a drag: land on whichever stop is closest to where the finger let go.
  const commit = () => {
    if (dragPos === null) return;
    const nearest = clampTier(Math.round(dragPos));
    setDragPos(null);
    if (nearest !== value) onChange(nearest);
  };

  const step = (delta: number) => {
    setDragPos(null);
    const next = clampTier(index + delta);
    if (next !== value) onChange(next);
  };

  // The input's own step is fine-grained so dragging is smooth, which would make arrow keys crawl
  // a hundredth of a tier at a time — so the keyboard is handled here in whole tiers instead.
  const onSliderKeyDown = (e: ReactKeyboardEvent) => {
    const byKey: Record<string, () => void> = {
      ArrowRight: () => step(1),
      ArrowUp: () => step(1),
      ArrowLeft: () => step(-1),
      ArrowDown: () => step(-1),
      Home: () => step(-last),
      End: () => step(last),
    };
    const handler = byKey[e.key];
    if (!handler) return;
    e.preventDefault();
    handler();
  };

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
          className="absolute right-0 z-50 mt-2 w-80 overflow-hidden rounded-2xl border border-border-dark bg-surface-dark shadow-card"
        >
          <div className="p-4">
            {/* Header: what this is, then a single line saying where you are — the reference's
                "Reasoning effort / Pair pays Medium." pairing. */}
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="font-display text-sm font-semibold text-content">{t("tier.label")}</p>
                <p className="mt-0.5 text-xs leading-snug text-muted">
                  {t(`tier.${value}.hint`)}
                  {value === DEFAULT_ANALYSIS_TIER && ` · ${t("tier.default")}`}
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                aria-label={t("a11y.close")}
                className="-mr-1 -mt-1 shrink-0 rounded-lg p-1 text-faint transition-colors hover:bg-content/5 hover:text-content"
              >
                <X size={16} weight="bold" />
              </button>
            </div>

            {/* The track: a chunky pill with a dot per stop and a white knob riding it. A real
                range input sits invisible on top, so keyboard, drag and click-to-seek all come
                free and the value is exposed as role="slider" — the drawn parts are presentation
                only, which is also the only way to use the app's tokens (::-webkit-slider-thumb
                cannot be styled portably). */}
            <div className="relative mt-4 h-7">
              <div className="absolute inset-x-0 top-1/2 h-2.5 -translate-y-1/2 rounded-full bg-track/50" />
              {/* Knob travel is inset by half a knob so it stays fully on the rail at both ends. */}
              <div className="absolute inset-y-0 left-2.5 right-2.5">
                <div
                  className={`absolute top-1/2 -left-2.5 h-2.5 -translate-y-1/2 rounded-full bg-primary ${
                    dragging ? "" : "transition-[width] duration-150 ease-out"
                  }`}
                  style={{ width: `calc(${pct}% + 0.625rem)` }}
                />
                {TIERS.map((tier, i) => (
                  <span
                    key={tier}
                    className={`absolute top-1/2 h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full ${
                      i <= shown ? "bg-primary-content/60" : "bg-content/30"
                    }`}
                    style={{ left: `${(i / last) * 100}%` }}
                  />
                ))}
                {/* Knob: only slightly wider than the rail, and sharing its centre line, so the two
                    read as one control rather than a ball parked on a bar. */}
                <div
                  className={`absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary bg-white shadow-card ${
                    dragging ? "" : "transition-[left] duration-150 ease-out"
                  }`}
                  style={{ left: `${pct}%` }}
                />
              </div>
              <input
                ref={sliderRef}
                type="range"
                min={0}
                max={last}
                step={0.01}
                value={shown}
                onChange={(e) => setDragPos(Number(e.target.value))}
                onKeyDown={onSliderKeyDown}
                // Every way a drag can end. commit() is idempotent, so overlapping events are fine.
                onPointerUp={commit}
                onPointerCancel={commit}
                onMouseUp={commit}
                onTouchEnd={commit}
                onBlur={commit}
                aria-label={t("tier.label")}
                aria-valuetext={TIER_NAME[clampTier(Math.round(shown))]}
                className="absolute inset-0 h-full w-full cursor-pointer appearance-none rounded-full bg-transparent opacity-0"
              />
            </div>

            {/* Stop labels. The reference has none — its levels are named in the composer chip —
                but ours are product nouns the user has to tell apart, so they stay. Pointer-only:
                the slider already exposes the value, so these keep out of the a11y tree. */}
            <div className="mt-2 flex justify-between">
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
          </div>

          <p className="border-t border-border-dark px-4 py-2.5 text-[11px] leading-snug text-faint">
            {t("tier.note")}
          </p>
        </div>
      )}
    </div>
  );
}
