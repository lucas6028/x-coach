import { useI18n } from "../../lib/i18n";
import type { Lead } from "../../lib/sixseven/gesture";

interface Props {
  count: number;
  combo: number;
  timeLeft: number;
  lead: Lead;
  // Ticks up on each scored 67 to retrigger the pop animation.
  pop: number;
}

// In-round overlay: the big 67 tally in the centre, timer + combo in the corners, and a live
// 6 / 7 hand indicator at the bottom. Purely a function of props.
export default function SixSevenHud({ count, combo, timeLeft, lead, pop }: Props) {
  const { t } = useI18n();
  const low = timeLeft <= 5;

  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex flex-col justify-between p-4">
      {/* Top row: combo + timer */}
      <div className="flex items-start justify-between gap-3">
        {combo >= 2 ? (
          <div className="rounded-full bg-secondary/20 px-3 py-1 font-mono text-sm font-bold text-secondary backdrop-blur">
            {t("six.hud.combo", { n: combo })}
          </div>
        ) : (
          <span />
        )}
        <div
          className={`rounded-xl bg-black/45 px-3 py-2 text-right backdrop-blur ${
            low ? "text-danger" : "text-white"
          }`}
        >
          <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-300">
            {t("six.hud.time")}
          </p>
          <p className="font-mono text-2xl font-bold tabular-nums">{timeLeft}</p>
        </div>
      </div>

      {/* Centre: the big 67 count */}
      <div className="flex flex-1 flex-col items-center justify-center">
        <p
          key={pop}
          className={`font-display text-7xl font-black tabular-nums text-white drop-shadow-lg ${
            pop ? "animate-[pulse_0.3s_ease-out]" : ""
          }`}
        >
          {count}
        </p>
        <p className="mt-1 font-mono text-xs uppercase tracking-[0.3em] text-zinc-300">
          {t("six.hud.label")}
        </p>
      </div>

      {/* Bottom: live 6 / 7 hand indicator */}
      <div className="flex items-center justify-center gap-4">
        <span
          className={`flex h-12 w-12 items-center justify-center rounded-full font-display text-2xl font-black transition-transform ${
            lead === "left" ? "scale-110 text-[#0b1120]" : "text-white/50"
          }`}
          style={{ background: lead === "left" ? "#22d3ee" : "rgba(255,255,255,0.12)" }}
        >
          6
        </span>
        <span
          className={`flex h-12 w-12 items-center justify-center rounded-full font-display text-2xl font-black transition-transform ${
            lead === "right" ? "scale-110 text-[#0b1120]" : "text-white/50"
          }`}
          style={{ background: lead === "right" ? "#42d159" : "rgba(255,255,255,0.12)" }}
        >
          7
        </span>
      </div>
    </div>
  );
}
