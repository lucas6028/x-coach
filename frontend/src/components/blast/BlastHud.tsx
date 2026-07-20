import { useI18n } from "../../lib/i18n";

interface Props {
  score: number;
  combo: number;
  timeLeft: number;
  // 0..1 charge meter.
  charge: number;
  armed: boolean;
  // Fleeting "+N" / "MISS" flash from the last shot, or null.
  flash: { hits: number; points: number } | null;
}

// In-round overlay: score/time in the corners, a big charge meter + prompt at the
// bottom, and a shot-result flash. Purely a function of props.
export default function BlastHud({ score, combo, timeLeft, charge, armed, flash }: Props) {
  const { t } = useI18n();
  const low = timeLeft <= 8;
  const pct = Math.round(charge * 100);

  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex flex-col justify-between p-4">
      {/* Top row: score + timer */}
      <div className="flex items-start justify-between gap-3">
        <div className="rounded-xl bg-black/45 px-3 py-2 backdrop-blur">
          <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-300">
            {t("blast.hud.score")}
          </p>
          <p className="font-mono text-2xl font-bold tabular-nums text-white">
            {score.toLocaleString()}
          </p>
        </div>

        {combo >= 2 && (
          <div className="self-center rounded-full bg-secondary/20 px-3 py-1 font-mono text-sm font-bold text-secondary backdrop-blur">
            {t("blast.hud.combo", { n: combo })}
          </div>
        )}

        <div
          className={`rounded-xl bg-black/45 px-3 py-2 text-right backdrop-blur ${
            low ? "text-danger" : "text-white"
          }`}
        >
          <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-300">
            {t("blast.hud.time")}
          </p>
          <p className="font-mono text-2xl font-bold tabular-nums">{timeLeft}</p>
        </div>
      </div>

      {/* Shot-result flash */}
      <div className="flex flex-1 items-center justify-center">
        {flash &&
          (flash.hits > 0 ? (
            <p className="font-display text-5xl font-black text-secondary drop-shadow-lg">
              +{flash.points.toLocaleString()}
              {flash.hits > 1 && (
                <span className="ml-2 align-middle text-2xl text-white">
                  ×{flash.hits}
                </span>
              )}
            </p>
          ) : (
            <p className="font-display text-3xl font-black uppercase text-danger/80 drop-shadow">
              {t("blast.hud.whiff")}
            </p>
          ))}
      </div>

      {/* Charge meter + prompt */}
      <div className="flex flex-col items-center gap-2">
        <p className="font-mono text-sm font-bold uppercase tracking-wider text-zinc-100 drop-shadow">
          {armed ? t("blast.hud.fire") : t("blast.hud.charge")}
        </p>
        <div className="h-3 w-72 max-w-[80%] overflow-hidden rounded-full bg-black/40">
          <div
            className={`h-full rounded-full transition-[width] duration-75 ${
              armed
                ? "bg-secondary"
                : "bg-gradient-to-r from-primary to-secondary"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
