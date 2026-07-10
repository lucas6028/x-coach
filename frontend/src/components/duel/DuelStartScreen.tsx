import { motion, useReducedMotion } from "motion/react";
import { Camera, Sword, Trophy, UsersThree } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import { POSES } from "../../lib/duel/poses";
import { MATCH_POINTS } from "../../lib/duel/match";
import type { DuelEntry } from "../../lib/duel/leaderboard";
import DuelLeaderboard from "./DuelLeaderboard";

interface Props {
  results: DuelEntry[];
  onStart: () => void;
  starting?: boolean;
  error?: string;
}

// Pre-match screen: the versus pitch, the how-to, the pose deck, and the recent-duels board.
export default function DuelStartScreen({ results, onStart, starting, error }: Props) {
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
            <UsersThree size={13} weight="fill" />
            {t("duel.badge")}
          </span>
          <h2 className="mt-4 font-display text-3xl font-bold leading-tight tracking-tight text-content md:text-4xl">
            {t("duel.heading")}
          </h2>
          <p className="mt-4 max-w-md leading-relaxed text-muted">{t("duel.sub")}</p>

          <ol className="mt-6 space-y-3">
            {["duel.how1", "duel.how2", "duel.how3"].map((k, i) => (
              <li key={k} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                  {i + 1}
                </span>
                <span className="text-sm leading-snug text-muted">
                  {t(k, { n: MATCH_POINTS })}
                </span>
              </li>
            ))}
          </ol>

          {/* Player key: cyan A vs amber B — the same colours the overlay uses. */}
          <div className="mt-6 flex items-center gap-3">
            <span className="inline-flex items-center gap-2 rounded-xl border border-border-dark px-3 py-1.5 text-sm">
              <span className="h-3 w-3 rounded-full" style={{ background: "#22d3ee" }} />
              <span className="font-semibold text-content">{t("duel.playerA")}</span>
            </span>
            <Sword size={18} weight="duotone" className="text-faint" />
            <span className="inline-flex items-center gap-2 rounded-xl border border-border-dark px-3 py-1.5 text-sm">
              <span className="h-3 w-3 rounded-full" style={{ background: "#f59e0b" }} />
              <span className="font-semibold text-content">{t("duel.playerB")}</span>
            </span>
          </div>

          <div className="mt-6">
            <button
              onClick={onStart}
              disabled={starting}
              className="flex items-center justify-center gap-2 rounded-2xl bg-primary px-6 py-3.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99] disabled:opacity-60"
            >
              <Camera size={18} weight="duotone" />
              {starting ? t("duel.starting") : t("duel.startBtn")}
            </button>
            <p className="mt-2 flex items-center gap-1.5 text-xs text-faint">
              <UsersThree size={13} />
              {t("duel.cameraNote")}
            </p>
            {error && (
              <p className="mt-3 rounded-xl border border-danger/30 bg-danger/[0.06] p-3 text-sm text-danger">
                {error}
              </p>
            )}
          </div>

          <div className="mt-8">
            <p className="mb-3 font-mono text-xs uppercase tracking-wider text-faint">
              {t("duel.posesTitle")}
            </p>
            <div className="flex flex-wrap gap-2">
              {POSES.map((p) => (
                <span
                  key={p.id}
                  className="flex items-center gap-2 rounded-xl border border-border-dark bg-content/[0.03] px-3 py-1.5 text-sm"
                >
                  <span aria-hidden className="text-lg">
                    {p.emoji}
                  </span>
                  <span className="text-muted">{t(`duelPose.${p.nameKey}`)}</span>
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="lg:w-80 lg:shrink-0">
          <p className="mb-3 flex items-center gap-1.5 font-mono text-xs uppercase tracking-wider text-faint">
            <Trophy size={13} weight="duotone" />
            {t("duel.board.title")}
          </p>
          <DuelLeaderboard entries={results} compact />
        </div>
      </motion.div>
    </div>
  );
}
