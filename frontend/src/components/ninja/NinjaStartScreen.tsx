import { motion, useReducedMotion } from "motion/react";
import { Camera, Knife, Trophy } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import { FRUITS } from "../../lib/ninja/physics";
import { START_LIVES } from "../../lib/ninja/scoring";
import type { NinjaEntry } from "../../lib/ninja/leaderboard";
import NinjaLeaderboard from "./NinjaLeaderboard";

interface Props {
  leaderboard: NinjaEntry[];
  onStart: () => void;
  starting?: boolean;
  error?: string;
}

// Pre-round screen: the pitch, the how-to, the fruit deck, and the board.
export default function NinjaStartScreen({ leaderboard, onStart, starting, error }: Props) {
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
            <Knife size={13} weight="fill" />
            {t("ninja.badge")}
          </span>
          <h2 className="mt-4 font-display text-3xl font-bold leading-tight tracking-tight text-content md:text-4xl">
            {t("ninja.heading")}
          </h2>
          <p className="mt-4 max-w-md leading-relaxed text-muted">{t("ninja.sub")}</p>

          <ol className="mt-6 space-y-3">
            {["ninja.how1", "ninja.how2", "ninja.how3"].map((k, i) => (
              <li key={k} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                  {i + 1}
                </span>
                <span className="text-sm leading-snug text-muted">{t(k, { lives: START_LIVES })}</span>
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
              {starting ? t("ninja.starting") : t("ninja.startBtn")}
            </button>
            <p className="mt-2 flex items-center gap-1.5 text-xs text-faint">
              <Camera size={13} />
              {t("ninja.cameraNote")}
            </p>
            {error && (
              <p className="mt-3 rounded-xl border border-danger/30 bg-danger/[0.06] p-3 text-sm text-danger">
                {error}
              </p>
            )}
          </div>

          <div className="mt-8">
            <p className="mb-3 font-mono text-xs uppercase tracking-wider text-faint">
              {t("ninja.deckTitle")}
            </p>
            <div className="flex flex-wrap gap-2 text-2xl">
              {[...FRUITS, "💣"].map((e) => (
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
            {t("ninja.board.title")}
          </p>
          <NinjaLeaderboard entries={leaderboard} compact />
        </div>
      </motion.div>
    </div>
  );
}
