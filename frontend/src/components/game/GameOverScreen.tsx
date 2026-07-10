import { useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { ArrowCounterClockwise, Confetti } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import type { ScoreEntry } from "../../lib/game/leaderboard";
import Leaderboard from "./Leaderboard";

export type RoundResult = {
  score: number;
  poses: number;
  bestCombo: number;
};

interface Props {
  result: RoundResult;
  leaderboard: ScoreEntry[];
  // 1-based rank the just-submitted score reached, or null before submission / if it
  // didn't make the board.
  rank: number | null;
  submitted: boolean;
  onSubmit: (name: string) => void;
  onReplay: () => void;
}

const NAME_MAX = 16;

// End-of-round screen: the score, an optional name-entry to claim a board slot, and the
// updated leaderboard with the new entry highlighted.
export default function GameOverScreen({
  result,
  leaderboard,
  rank,
  submitted,
  onSubmit,
  onReplay,
}: Props) {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  const [name, setName] = useState("");

  const handleSubmit = () => {
    const trimmed = name.trim().slice(0, NAME_MAX);
    onSubmit(trimmed || t("game.over.anon"));
  };

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin">
      <motion.div
        initial={reduce ? false : { opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="mx-auto flex min-h-full max-w-md flex-col justify-center gap-6 px-5 py-10"
      >
        <div className="rounded-3xl border border-border-dark bg-surface-dark p-7 text-center">
          <Confetti size={34} weight="duotone" className="mx-auto text-primary" />
          <p className="mt-3 font-mono text-xs uppercase tracking-wider text-faint">
            {t("game.over.title")}
          </p>
          <p className="mt-1 font-display text-5xl font-black tabular-nums text-content">
            {result.score.toLocaleString()}
          </p>
          <div className="mt-4 flex justify-center gap-6 text-sm">
            <div>
              <p className="font-mono text-lg font-bold text-content">{result.poses}</p>
              <p className="text-xs text-muted">{t("game.over.poses")}</p>
            </div>
            <div>
              <p className="font-mono text-lg font-bold text-content">{result.bestCombo}×</p>
              <p className="text-xs text-muted">{t("game.over.combo")}</p>
            </div>
          </div>

          {!submitted ? (
            <div className="mt-6">
              <label className="mb-2 block text-left text-sm font-medium text-content">
                {t("game.over.nameLabel")}
              </label>
              <div className="flex gap-2">
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                  maxLength={NAME_MAX}
                  placeholder={t("game.over.namePlaceholder")}
                  aria-label={t("game.over.nameLabel")}
                  className="min-w-0 flex-1 rounded-xl border border-border-dark bg-content/[0.02] px-3 py-2.5 text-sm text-content outline-none focus:border-primary"
                />
                <button
                  onClick={handleSubmit}
                  className="shrink-0 rounded-xl bg-primary px-4 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99]"
                >
                  {t("game.over.save")}
                </button>
              </div>
            </div>
          ) : (
            <p className="mt-5 text-sm font-medium text-primary">
              {rank && rank > 0
                ? t("game.over.ranked", { rank })
                : t("game.over.notRanked")}
            </p>
          )}
        </div>

        {submitted && (
          <Leaderboard entries={leaderboard} highlightRank={rank ?? undefined} />
        )}

        <button
          onClick={onReplay}
          className="flex items-center justify-center gap-2 rounded-2xl border border-border-dark bg-content/[0.02] px-5 py-3.5 text-sm font-medium text-content transition-colors hover:bg-content/[0.05] active:scale-[0.99]"
        >
          <ArrowCounterClockwise size={18} weight="bold" />
          {t("game.over.replay")}
        </button>
      </motion.div>
    </div>
  );
}
