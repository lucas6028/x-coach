import { useEffect, useRef, useState } from "react";
import { CaretDown, Check, Gauge } from "@phosphor-icons/react";
import { useI18n } from "../lib/i18n";
import { DEFAULT_ANALYSIS_TIER, type PoseTier } from "../lib/poseTier";

const TIERS: readonly PoseTier[] = ["lite", "full", "heavy"];
// Model tiers keep MediaPipe's own names in every language — they are product nouns, not copy.
const TIER_NAME: Record<PoseTier, string> = { lite: "Lite", full: "Full", heavy: "Heavy" };

// Picks the model tier for the OFFLINE analysis extraction only — the live recording overlay is
// always Lite (LIVE_OVERLAY_TIER), which the menu footer says out loud.
//
// Shaped like Claude Code's effort picker: a compact trigger showing the current value, opening a
// menu where each option carries a one-line description of the tradeoff. Dismissal follows the
// AccountMenu pattern (outside click + Escape).
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

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
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
          role="menu"
          className="absolute right-0 z-50 mt-1.5 w-72 overflow-hidden rounded-xl border border-border-dark bg-surface-dark shadow-lg shadow-black/20"
        >
          <div className="p-1">
            {TIERS.map((tier) => {
              const selected = tier === value;
              return (
                <button
                  key={tier}
                  role="menuitemradio"
                  aria-checked={selected}
                  // Name is the tier alone; the tradeoff line is the description. Without this the
                  // accessible name would swallow the hint — and Lite's hint names "Heavy".
                  aria-label={TIER_NAME[tier]}
                  aria-describedby={`tier-hint-${tier}`}
                  onClick={() => {
                    onChange(tier);
                    setOpen(false);
                  }}
                  className="flex w-full items-start gap-2 rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-content/5"
                >
                  <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">
                    {selected && <Check size={14} weight="bold" className="text-primary" />}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2">
                      <span className={`text-sm font-medium ${selected ? "text-content" : "text-muted"}`}>
                        {TIER_NAME[tier]}
                      </span>
                      {tier === DEFAULT_ANALYSIS_TIER && (
                        <span className="rounded-full bg-content/5 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-faint">
                          {t("tier.default")}
                        </span>
                      )}
                    </span>
                    <span id={`tier-hint-${tier}`} className="mt-0.5 block text-xs leading-snug text-faint">
                      {t(`tier.${tier}.hint`)}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
          <p className="border-t border-border-dark px-3 py-2 text-[11px] leading-snug text-faint">
            {t("tier.note")}
          </p>
        </div>
      )}
    </div>
  );
}
