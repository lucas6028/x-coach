import { Fire } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";

// Small "≈ N kcal (estimated)" pill shown on each game's over screen. Deliberately labels the
// number an estimate — a camera can't measure real exertion (see lib/calories.ts).
export default function CalorieBadge({ kcal }: { kcal: number }) {
  const { t } = useI18n();
  return (
    <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-orange-500/25 bg-orange-500/[0.08] px-3.5 py-1.5">
      <Fire size={16} weight="fill" className="text-orange-400" />
      <span className="text-sm font-semibold tabular-nums text-content">
        {t("game.kcal.est", { n: kcal })}
      </span>
      <span className="text-xs text-faint">{t("game.kcal.note")}</span>
    </div>
  );
}
