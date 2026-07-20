import { useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { ArrowCounterClockwise, Confetti } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import type { BlastEntry } from "../../lib/blast/leaderboard";
import BlastLeaderboard from "./BlastLeaderboard";
import CalorieBadge from "../games/CalorieBadge";

export type BlastResult = {
  score: number;
  hits: number;
  bestCombo: number;
  kcal: number;
};

interface Props {
  result: BlastResult;
  leaderboard: BlastEntry[];
  rank: number | null;
  submitted: boolean;
  onSubmit: (name: string) => void;
  onReplay: () => void;
}

const NAME_MAX = 16;

// End-of-round screen: score, optional name entry to claim a board slot, and the board.
export default function BlastOverScreen({
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
    onSubmit(trimmed || t("blast.over.anon"));
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
            {t("blast.over.title")}
          </p>
          <p className="mt-1 font-display text-5xl font-black tabular-nums text-content">
            {result.score.toLocaleString()}
          </p>
          <div className="mt-4 flex justify-center gap-6 text-sm">
            <div>
              <p className="font-mono text-lg font-bold text-content">{result.hits}</p>
              <p className="text-xs text-muted">{t("blast.over.hits")}</p>
            </div>
            <div>
              <p className="font-mono text-lg font-bold text-content">{result.bestCombo}×</p>
              <p className="text-xs text-muted">{t("blast.over.combo")}</p>
            </div>
          </div>

          <CalorieBadge kcal={result.kcal} />

          {!submitted ? (
            <div className="mt-6">
              <label className="mb-2 block text-left text-sm font-medium text-content">
                {t("blast.over.nameLabel")}
              </label>
              <div className="flex gap-2">
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                  maxLength={NAME_MAX}
                  placeholder={t("blast.over.namePlaceholder")}
                  aria-label={t("blast.over.nameLabel")}
                  className="min-w-0 flex-1 rounded-xl border border-border-dark bg-content/[0.02] px-3 py-2.5 text-sm text-content outline-none focus:border-primary"
                />
                <button
                  onClick={handleSubmit}
                  className="shrink-0 rounded-xl bg-primary px-4 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99]"
                >
                  {t("blast.over.save")}
                </button>
              </div>
            </div>
          ) : (
            <p className="mt-5 text-sm font-medium text-primary">
              {rank && rank > 0
                ? t("blast.over.ranked", { rank })
                : t("blast.over.notRanked")}
            </p>
          )}
        </div>

        {submitted && (
          <BlastLeaderboard entries={leaderboard} highlightRank={rank ?? undefined} />
        )}

        <button
          onClick={onReplay}
          className="flex items-center justify-center gap-2 rounded-2xl border border-border-dark bg-content/[0.02] px-5 py-3.5 text-sm font-medium text-content transition-colors hover:bg-content/[0.05] active:scale-[0.99]"
        >
          <ArrowCounterClockwise size={18} weight="bold" />
          {t("blast.over.replay")}
        </button>
      </motion.div>
    </div>
  );
}
