import { motion, useReducedMotion } from "motion/react";
import { Camera, Lightning, Trophy } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import { POSES } from "../../lib/game/poses";
import { ROUND_SECONDS } from "../../lib/game/scoring";
import type { ScoreEntry } from "../../lib/game/leaderboard";
import Leaderboard from "./Leaderboard";

interface Props {
  leaderboard: ScoreEntry[];
  onStart: () => void;
  // Present while the camera/model is spinning up so the CTA can show progress.
  starting?: boolean;
  error?: string;
}

// Pre-round screen: the pitch, how it works, the pose gallery, and the local board.
export default function GameStartScreen({ leaderboard, onStart, starting, error }: Props) {
  const { t } = useI18n();
  const reduce = useReducedMotion();

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin">
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="mx-auto flex min-h-full max-w-5xl flex-col gap-8 px-5 py-8 sm:px-6 sm:py-12 lg:flex-row lg:gap-14"
      >
        {/* Left: pitch + start */}
        <div className="lg:flex-1">
          <span className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-primary">
            <Lightning size={13} weight="fill" />
            {t("game.badge")}
          </span>
          <h2 className="mt-4 font-display text-3xl font-bold leading-tight tracking-tight text-content md:text-4xl">
            {t("game.heading")}
          </h2>
          <p className="mt-4 max-w-md leading-relaxed text-muted">{t("game.sub")}</p>

          <ol className="mt-6 space-y-3">
            {["game.how1", "game.how2", "game.how3"].map((k, i) => (
              <li key={k} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                  {i + 1}
                </span>
                <span className="text-sm leading-snug text-muted">{t(k)}</span>
              </li>
            ))}
          </ol>

          <div className="mt-6">
            <button
              onClick={onStart}
              disabled={starting}
              className="flex items-center justify-center gap-2 rounded-2xl bg-primary px-6 py-3.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99] disabled:opacity-60"
            >
              <Camera size={18} weight="duotone" />
              {starting ? t("game.starting") : t("game.startBtn")}
            </button>
            <p className="mt-2 flex items-center gap-1.5 text-xs text-faint">
              <Camera size={13} />
              {t("game.cameraNote", { s: ROUND_SECONDS })}
            </p>
            {error && (
              <p className="mt-3 rounded-xl border border-danger/30 bg-danger/[0.06] p-3 text-sm text-danger">
                {error}
              </p>
            )}
          </div>

          {/* Pose gallery */}
          <div className="mt-8">
            <p className="mb-3 font-mono text-xs uppercase tracking-wider text-faint">
              {t("game.posesTitle")}
            </p>
            <div className="flex flex-wrap gap-2">
              {POSES.map((p) => (
                <span
                  key={p.id}
                  className="inline-flex items-center gap-1.5 rounded-full border border-border-dark bg-content/[0.03] px-3 py-1.5 text-sm text-content"
                >
                  <span aria-hidden>{p.emoji}</span>
                  {t(p.nameKey)}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Right: leaderboard */}
        <div className="lg:w-80 lg:shrink-0">
          <p className="mb-3 flex items-center gap-1.5 font-mono text-xs uppercase tracking-wider text-faint">
            <Trophy size={13} weight="duotone" />
            {t("game.board.title")}
          </p>
          <Leaderboard entries={leaderboard} compact />
        </div>
      </motion.div>
    </div>
  );
}
