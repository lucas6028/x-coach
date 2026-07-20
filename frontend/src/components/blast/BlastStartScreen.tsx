import { motion, useReducedMotion } from "motion/react";
import { Camera, Lightning, Trophy } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import { MEME_EMOJIS } from "../../lib/blast/targets";
import { ROUND_SECONDS } from "../../lib/blast/scoring";
import type { BlastEntry } from "../../lib/blast/leaderboard";
import BlastLeaderboard from "./BlastLeaderboard";

interface Props {
  leaderboard: BlastEntry[];
  onStart: () => void;
  starting?: boolean;
  error?: string;
}

// Pre-round screen: the meme pitch, the charge→fire how-to, the orb deck, and the board.
export default function BlastStartScreen({ leaderboard, onStart, starting, error }: Props) {
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
        <div className="lg:flex-1">
          <span className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-primary">
            <Lightning size={13} weight="fill" />
            {t("blast.badge")}
          </span>
          <h2 className="mt-4 font-display text-3xl font-bold leading-tight tracking-tight text-content md:text-4xl">
            {t("blast.heading")}
          </h2>
          <p className="mt-4 max-w-md leading-relaxed text-muted">{t("blast.sub")}</p>

          <ol className="mt-6 space-y-3">
            {["blast.how1", "blast.how2", "blast.how3"].map((k, i) => (
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
              {starting ? t("blast.starting") : t("blast.startBtn")}
            </button>
            <p className="mt-2 flex items-center gap-1.5 text-xs text-faint">
              <Camera size={13} />
              {t("blast.cameraNote", { s: ROUND_SECONDS })}
            </p>
            {error && (
              <p className="mt-3 rounded-xl border border-danger/30 bg-danger/[0.06] p-3 text-sm text-danger">
                {error}
              </p>
            )}
          </div>

          <div className="mt-8">
            <p className="mb-3 font-mono text-xs uppercase tracking-wider text-faint">
              {t("blast.orbsTitle")}
            </p>
            <div className="flex flex-wrap gap-2 text-2xl">
              {MEME_EMOJIS.map((e) => (
                <span
                  key={e}
                  aria-hidden
                  className="flex h-11 w-11 items-center justify-center rounded-xl border border-border-dark bg-content/[0.03]"
                >
                  {e}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="lg:w-80 lg:shrink-0">
          <p className="mb-3 flex items-center gap-1.5 font-mono text-xs uppercase tracking-wider text-faint">
            <Trophy size={13} weight="duotone" />
            {t("blast.board.title")}
          </p>
          <BlastLeaderboard entries={leaderboard} compact />
        </div>
      </motion.div>
    </div>
  );
}
