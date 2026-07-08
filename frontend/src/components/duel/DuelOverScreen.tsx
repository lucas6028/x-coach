import { useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { ArrowCounterClockwise, Crown } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import type { Side } from "../../lib/duel/match";
import type { DuelEntry } from "../../lib/duel/leaderboard";
import DuelLeaderboard from "./DuelLeaderboard";

export type DuelResult = {
  winner: Side;
  aWins: number;
  bWins: number;
};

interface Props {
  result: DuelResult;
  results: DuelEntry[];
  submitted: boolean;
  savedTs: number | null;
  onSubmit: (winner: string, loser: string) => void;
  onReplay: () => void;
}

const NAME_MAX = 16;
const A_COLOR = "#22d3ee";
const B_COLOR = "#f59e0b";

// End-of-match screen: who won and by how much, optional name entry to log the duel, and the
// recent-duels board. The winning side's colour frames the card.
export default function DuelOverScreen({
  result,
  results,
  submitted,
  savedTs,
  onSubmit,
  onReplay,
}: Props) {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  const [winnerName, setWinnerName] = useState("");
  const [loserName, setLoserName] = useState("");

  const winnerIsA = result.winner === "a";
  const color = winnerIsA ? A_COLOR : B_COLOR;
  const winnerPoints = winnerIsA ? result.aWins : result.bWins;
  const loserPoints = winnerIsA ? result.bWins : result.aWins;
  const winnerLabel = winnerIsA ? t("duel.playerA") : t("duel.playerB");
  const loserLabel = winnerIsA ? t("duel.playerB") : t("duel.playerA");

  const handleSubmit = () => {
    onSubmit(
      winnerName.trim().slice(0, NAME_MAX) || winnerLabel,
      loserName.trim().slice(0, NAME_MAX) || loserLabel
    );
  };

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin">
      <motion.div
        initial={reduce ? false : { opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="mx-auto flex min-h-full max-w-md flex-col justify-center gap-6 px-5 py-10"
      >
        <div
          className="rounded-3xl border bg-surface-dark p-7 text-center"
          style={{ borderColor: color }}
        >
          <Crown size={34} weight="fill" className="mx-auto" style={{ color }} />
          <p className="mt-3 font-mono text-xs uppercase tracking-wider text-faint">
            {t("duel.over.title")}
          </p>
          <p className="mt-1 font-display text-3xl font-black text-content" style={{ color }}>
            {t("duel.over.wins", { p: winnerLabel })}
          </p>
          <p className="mt-2 font-mono text-4xl font-black tabular-nums text-content">
            <span>{winnerPoints}</span>
            <span className="mx-2 text-muted">–</span>
            <span>{loserPoints}</span>
          </p>

          {!submitted ? (
            <div className="mt-6 space-y-3 text-left">
              <div>
                <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-content">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
                  {t("duel.over.winnerName")}
                </label>
                <input
                  value={winnerName}
                  onChange={(e) => setWinnerName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                  maxLength={NAME_MAX}
                  placeholder={winnerLabel}
                  aria-label={t("duel.over.winnerName")}
                  className="w-full rounded-xl border border-border-dark bg-content/[0.02] px-3 py-2.5 text-sm text-content outline-none focus:border-primary"
                />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-content">
                  {t("duel.over.loserName")}
                </label>
                <input
                  value={loserName}
                  onChange={(e) => setLoserName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                  maxLength={NAME_MAX}
                  placeholder={loserLabel}
                  aria-label={t("duel.over.loserName")}
                  className="w-full rounded-xl border border-border-dark bg-content/[0.02] px-3 py-2.5 text-sm text-content outline-none focus:border-primary"
                />
              </div>
              <button
                onClick={handleSubmit}
                className="w-full rounded-xl bg-primary px-4 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99]"
              >
                {t("duel.over.save")}
              </button>
            </div>
          ) : (
            <p className="mt-5 text-sm font-medium text-primary">{t("duel.over.saved")}</p>
          )}
        </div>

        {submitted && (
          <DuelLeaderboard entries={results} highlightTs={savedTs ?? undefined} />
        )}

        <button
          onClick={onReplay}
          className="flex items-center justify-center gap-2 rounded-2xl border border-border-dark bg-content/[0.02] px-5 py-3.5 text-sm font-medium text-content transition-colors hover:bg-content/[0.05] active:scale-[0.99]"
        >
          <ArrowCounterClockwise size={18} weight="bold" />
          {t("duel.over.rematch")}
        </button>
      </motion.div>
    </div>
  );
}
