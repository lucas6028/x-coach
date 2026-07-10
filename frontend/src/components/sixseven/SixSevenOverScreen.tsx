import { useRef, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { ArrowCounterClockwise, Confetti } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import type { SixSevenEntry } from "../../lib/sixseven/leaderboard";
import SixSevenLeaderboard from "./SixSevenLeaderboard";

export type SixSevenResult = {
  count: number;
  bestCombo: number;
};

interface Props {
  result: SixSevenResult;
  leaderboard: SixSevenEntry[];
  rank: number | null;
  submitted: boolean;
  onSubmit: (name: string) => void;
  onReplay: () => void;
}

const NAME_MAX = 16;

// End-of-round screen: 67 tally, optional name entry to claim a board slot, and the board.
export default function SixSevenOverScreen({
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
  // Enter auto-repeat / double-click can fire several times before the submitted-state
  // re-render unmounts the input; guard so a score is only written once.
  const submittedRef = useRef(false);

  const handleSubmit = () => {
    if (submittedRef.current) return;
    submittedRef.current = true;
    onSubmit(name.trim().slice(0, NAME_MAX) || t("six.over.anon"));
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
            {t("six.over.title")}
          </p>
          <p className="mt-1 font-display text-6xl font-black tabular-nums text-content">
            {result.count}
          </p>
          <p className="mt-1 text-sm text-muted">
            {t("six.over.combo", { n: result.bestCombo })}
          </p>

          {!submitted ? (
            <div className="mt-6">
              <label className="mb-2 block text-left text-sm font-medium text-content">
                {t("six.over.nameLabel")}
              </label>
              <div className="flex gap-2">
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                  maxLength={NAME_MAX}
                  placeholder={t("six.over.namePlaceholder")}
                  aria-label={t("six.over.nameLabel")}
                  className="min-w-0 flex-1 rounded-xl border border-border-dark bg-content/[0.02] px-3 py-2.5 text-sm text-content outline-none focus:border-primary"
                />
                <button
                  onClick={handleSubmit}
                  className="shrink-0 rounded-xl bg-primary px-4 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99]"
                >
                  {t("six.over.save")}
                </button>
              </div>
            </div>
          ) : (
            <p className="mt-5 text-sm font-medium text-primary">
              {rank && rank > 0 ? t("six.over.ranked", { rank }) : t("six.over.notRanked")}
            </p>
          )}
        </div>

        {submitted && (
          <SixSevenLeaderboard entries={leaderboard} highlightRank={rank ?? undefined} />
        )}

        <button
          onClick={onReplay}
          className="flex items-center justify-center gap-2 rounded-2xl border border-border-dark bg-content/[0.02] px-5 py-3.5 text-sm font-medium text-content transition-colors hover:bg-content/[0.05] active:scale-[0.99]"
        >
          <ArrowCounterClockwise size={18} weight="bold" />
          {t("six.over.replay")}
        </button>
      </motion.div>
    </div>
  );
}
