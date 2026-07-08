import { useI18n } from "../../lib/i18n";
import { MATCH_POINTS, type Side } from "../../lib/duel/match";

type SideState = {
  wins: number;
  // 0..1 hold progress toward taking the round.
  hold: number;
  // A body is assigned to this slot this frame.
  present: boolean;
};

interface Props {
  poseEmoji: string;
  poseNameKey: string;
  a: SideState;
  b: SideState;
  // Who just took the round (banner shown during the break), or null mid-round.
  roundFlash: Side | null;
}

const A_COLOR = "#22d3ee";
const B_COLOR = "#f59e0b";

function Pips({ wins, color }: { wins: number; color: string }) {
  return (
    <div className="flex gap-1">
      {Array.from({ length: MATCH_POINTS }, (_, i) => (
        <span
          key={i}
          className="h-2.5 w-2.5 rounded-full"
          style={{
            background: i < wins ? color : "rgba(255,255,255,0.18)",
          }}
        />
      ))}
    </div>
  );
}

function PlayerPanel({
  name,
  color,
  state,
  align,
}: {
  name: string;
  color: string;
  state: SideState;
  align: "left" | "right";
}) {
  return (
    <div
      className={`rounded-xl bg-black/45 px-3 py-2 backdrop-blur ${
        align === "right" ? "text-right" : ""
      }`}
    >
      <div className={`flex items-center gap-2 ${align === "right" ? "flex-row-reverse" : ""}`}>
        <span className="h-3 w-3 rounded-full" style={{ background: color }} />
        <span className="font-mono text-sm font-bold text-white">{name}</span>
        {!state.present && (
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-danger" />
        )}
      </div>
      <div className={`mt-1.5 flex ${align === "right" ? "justify-end" : ""}`}>
        <Pips wins={state.wins} color={color} />
      </div>
    </div>
  );
}

function HoldMeter({ hold, color, align }: { hold: number; color: string; align: "left" | "right" }) {
  return (
    <div className={`h-2.5 w-40 max-w-[42%] overflow-hidden rounded-full bg-black/40 ${align === "right" ? "ml-auto" : ""}`}>
      <div
        className="h-full rounded-full transition-[width] duration-75"
        style={{ width: `${Math.round(Math.min(1, hold) * 100)}%`, background: color }}
      />
    </div>
  );
}

// In-round overlay: both players' score panels + hold meters, the current target pose front and
// centre, and a round-result banner during the break. Purely a function of props.
export default function DuelHud({ poseEmoji, poseNameKey, a, b, roundFlash }: Props) {
  const { t } = useI18n();

  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex flex-col justify-between p-4">
      {/* Top: player A · target pose · player B */}
      <div className="flex items-start justify-between gap-3">
        <PlayerPanel name={t("duel.playerA")} color={A_COLOR} state={a} align="left" />

        <div className="rounded-2xl bg-black/50 px-4 py-2 text-center backdrop-blur">
          <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-300">
            {t("duel.hud.match")}
          </p>
          <p className="text-3xl leading-none" aria-hidden>
            {poseEmoji}
          </p>
          <p className="mt-1 font-display text-sm font-bold text-white">
            {t(`duelPose.${poseNameKey}`)}
          </p>
        </div>

        <PlayerPanel name={t("duel.playerB")} color={B_COLOR} state={b} align="right" />
      </div>

      {/* Round-result banner */}
      <div className="flex flex-1 items-center justify-center">
        {roundFlash && (
          <p
            className="font-display text-4xl font-black drop-shadow-lg"
            style={{ color: roundFlash === "a" ? A_COLOR : B_COLOR }}
          >
            {t("duel.hud.roundWin", {
              p: roundFlash === "a" ? t("duel.playerA") : t("duel.playerB"),
            })}
          </p>
        )}
      </div>

      {/* Bottom: the two hold meters */}
      <div className="flex items-end justify-between gap-3">
        <HoldMeter hold={a.hold} color={A_COLOR} align="left" />
        <span className="pb-0.5 font-display text-sm font-bold uppercase tracking-wider text-zinc-200 drop-shadow">
          {t("duel.hud.hold")}
        </span>
        <HoldMeter hold={b.hold} color={B_COLOR} align="right" />
      </div>
    </div>
  );
}
