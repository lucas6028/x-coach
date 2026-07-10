import { useI18n } from "../../lib/i18n";
import type { GamePose } from "../../lib/game/poses";
import type { Grade } from "../../lib/game/scoring";

interface Props {
  score: number;
  combo: number;
  // Whole seconds left in the round.
  timeLeft: number;
  target: GamePose;
  // 0..1 live match quality against the target, for the hold meter.
  quality: number;
  // 0..1 how far through the required hold the player is.
  holdProgress: number;
  // The grade of the most recently locked pose, briefly flashed. Null clears the flash.
  lastGrade: Grade | null;
}

const GRADE_STYLES: Record<Grade, string> = {
  perfect: "text-secondary",
  great: "text-primary",
  good: "text-content",
  miss: "text-danger",
};

// The in-round overlay: target pose card (top), score/combo/timer (corners), and a
// live hold meter. Purely a function of props so the game loop owns all the state.
export default function GameHud({
  score,
  combo,
  timeLeft,
  target,
  quality,
  holdProgress,
  lastGrade,
}: Props) {
  const { t } = useI18n();
  const low = timeLeft <= 10;

  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex flex-col justify-between p-4">
      {/* Top row: score, target, timer */}
      <div className="flex items-start justify-between gap-3">
        <div className="rounded-xl bg-black/45 px-3 py-2 backdrop-blur">
          <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-300">
            {t("game.hud.score")}
          </p>
          <p className="font-mono text-2xl font-bold tabular-nums text-white">
            {score.toLocaleString()}
          </p>
        </div>

        <div className="flex flex-col items-center rounded-2xl bg-black/45 px-4 py-2 text-center backdrop-blur">
          <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-300">
            {t("game.hud.strike")}
          </p>
          <p className="mt-0.5 flex items-center gap-2 text-lg font-semibold text-white">
            <span aria-hidden className="text-2xl">
              {target.emoji}
            </span>
            {t(target.nameKey)}
          </p>
        </div>

        <div
          className={`rounded-xl bg-black/45 px-3 py-2 text-right backdrop-blur ${
            low ? "text-danger" : "text-white"
          }`}
        >
          <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-300">
            {t("game.hud.time")}
          </p>
          <p className="font-mono text-2xl font-bold tabular-nums">{timeLeft}</p>
        </div>
      </div>

      {/* Bottom: combo + hold meter */}
      <div className="flex flex-col items-center gap-2">
        {lastGrade && lastGrade !== "miss" && (
          <p
            className={`font-display text-3xl font-black uppercase drop-shadow ${GRADE_STYLES[lastGrade]}`}
          >
            {t(`game.grade.${lastGrade}`)}
          </p>
        )}
        {combo >= 2 && (
          <p className="font-mono text-sm font-bold text-secondary drop-shadow">
            {t("game.hud.combo", { n: combo })}
          </p>
        )}
        <div className="h-2.5 w-56 max-w-[70%] overflow-hidden rounded-full bg-black/40">
          <div
            className="h-full rounded-full bg-gradient-to-r from-primary to-secondary transition-[width] duration-100"
            style={{ width: `${Math.round(Math.max(quality, holdProgress) * 100)}%` }}
          />
        </div>
        <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-200 drop-shadow">
          {holdProgress > 0 ? t("game.hud.holding") : t("game.hud.matchPrompt")}
        </p>
      </div>
    </div>
  );
}
