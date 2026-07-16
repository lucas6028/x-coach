import { useI18n } from "../../lib/i18n";
import { START_LIVES } from "../../lib/ninja/scoring";

interface Props {
  score: number;
  combo: number;
  lives: number;
  // Ticks up on each cut to retrigger the combo pop.
  pop: number;
  bombFlash: boolean;
}

// In-round overlay: score + lives in the corners, a combo pop in the middle, and a bomb banner.
// Purely a function of props.
export default function NinjaHud({ score, combo, lives, pop, bombFlash }: Props) {
  const { t } = useI18n();

  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex flex-col justify-between p-4">
      {/* Top row: score + lives */}
      <div className="flex items-start justify-between gap-3">
        <div className="rounded-xl bg-black/45 px-3 py-2 backdrop-blur">
          <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-300">
            {t("ninja.hud.score")}
          </p>
          <p className="font-mono text-2xl font-bold tabular-nums text-white">
            {score.toLocaleString()}
          </p>
        </div>

        <div className="rounded-xl bg-black/45 px-3 py-2 text-right backdrop-blur">
          <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-300">
            {t("ninja.hud.lives")}
          </p>
          <p className="text-lg leading-tight" aria-label={t("ninja.hud.lives")}>
            {Array.from({ length: START_LIVES }, (_, i) => (i < lives ? "❤️" : "🖤")).join("")}
          </p>
        </div>
      </div>

      {/* Centre: combo pop + bomb banner */}
      <div className="flex flex-1 items-center justify-center">
        {bombFlash ? (
          <p className="font-display text-6xl font-black text-danger drop-shadow-lg">
            {t("ninja.hud.boom")}
          </p>
        ) : (
          combo >= 3 && (
            <p key={pop} className="font-display text-5xl font-black text-secondary drop-shadow-lg animate-[pulse_0.3s_ease-out]">
              {t("ninja.hud.combo", { n: combo })}
            </p>
          )
        )}
      </div>

      {/* Bottom hint */}
      <p className="text-center font-mono text-xs uppercase tracking-[0.2em] text-zinc-300 drop-shadow">
        {t("ninja.hud.slice")}
      </p>
    </div>
  );
}
