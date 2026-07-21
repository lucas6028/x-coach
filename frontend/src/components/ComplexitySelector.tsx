import type { PoseTier } from "../lib/poseTier";

const TIERS: { tier: PoseTier; label: string; hint: string }[] = [
  { tier: "lite", label: "Lite", hint: "最快，預設" },
  { tier: "full", label: "Full", hint: "較準" },
  { tier: "heavy", label: "Heavy", hint: "最準，較慢" },
];

// Chooses the model tier for the OFFLINE analysis extraction only. The live recording overlay is
// always Lite regardless of this control.
export default function ComplexitySelector({
  value,
  onChange,
}: {
  value: PoseTier;
  onChange: (tier: PoseTier) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-1.5">
      <legend className="text-xs font-semibold uppercase tracking-wider text-faint">分析精度</legend>
      <div role="radiogroup" className="flex gap-2">
        {TIERS.map(({ tier, label, hint }) => (
          <label
            key={tier}
            className={`flex-1 cursor-pointer rounded-lg border px-3 py-2 text-center text-sm transition-colors ${
              value === tier ? "border-primary bg-primary/[0.08] text-content" : "border-border-dark text-muted hover:bg-content/5"
            }`}
          >
            <input
              type="radio"
              name="pose-tier"
              className="sr-only"
              checked={value === tier}
              onChange={() => onChange(tier)}
              aria-label={`${label} — ${hint}`}
            />
            <span className="block font-medium">{label}</span>
            <span className="block text-xs text-faint">{hint}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
