import { motion, useReducedMotion } from "motion/react";
import { Camera, HandWaving, Trophy } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import { ROUND_SECONDS } from "../../lib/sixseven/counter";
import type { SixSevenEntry } from "../../lib/sixseven/leaderboard";
import SixSevenLeaderboard from "./SixSevenLeaderboard";

interface Props {
  leaderboard: SixSevenEntry[];
  onStart: () => void;
  starting?: boolean;
  error?: string;
}

// Pre-round screen: the meme pitch, the how-to, and the board.
export default function SixSevenStartScreen({ leaderboard, onStart, starting, error }: Props) {
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
            <HandWaving size={13} weight="fill" />
            {t("six.badge")}
          </span>
          <h2 className="mt-4 font-display text-3xl font-bold leading-tight tracking-tight text-content md:text-4xl">
            {t("six.heading")}
          </h2>
          <p className="mt-4 max-w-md leading-relaxed text-muted">{t("six.sub")}</p>

          <ol className="mt-6 space-y-3">
            {["six.how1", "six.how2", "six.how3"].map((k, i) => (
              <li key={k} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                  {i + 1}
                </span>
                <span className="text-sm leading-snug text-muted">{t(k, { s: ROUND_SECONDS })}</span>
              </li>
            ))}
          </ol>

          {/* The 6 / 7 hand key. */}
          <div className="mt-6 flex items-center gap-3">
            <span className="inline-flex items-center gap-2 rounded-xl border border-border-dark px-3 py-1.5 text-sm">
              <span className="flex h-6 w-6 items-center justify-center rounded-full font-bold text-[#0b1120]" style={{ background: "#22d3ee" }}>6</span>
              <span className="text-muted">{t("six.leftHand")}</span>
            </span>
            <span className="inline-flex items-center gap-2 rounded-xl border border-border-dark px-3 py-1.5 text-sm">
              <span className="flex h-6 w-6 items-center justify-center rounded-full font-bold text-[#0b1120]" style={{ background: "#42d159" }}>7</span>
              <span className="text-muted">{t("six.rightHand")}</span>
            </span>
          </div>

          <div className="mt-6">
            <button
              onClick={onStart}
              disabled={starting}
              className="flex items-center justify-center gap-2 rounded-2xl bg-primary px-6 py-3.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99] disabled:opacity-60"
            >
              <Camera size={18} weight="duotone" />
              {starting ? t("six.starting") : t("six.startBtn")}
            </button>
            <p className="mt-2 flex items-center gap-1.5 text-xs text-faint">
              <Camera size={13} />
              {t("six.cameraNote", { s: ROUND_SECONDS })}
            </p>
            {error && (
              <p className="mt-3 rounded-xl border border-danger/30 bg-danger/[0.06] p-3 text-sm text-danger">
                {error}
              </p>
            )}
          </div>
        </div>

        <div className="lg:w-80 lg:shrink-0">
          <p className="mb-3 flex items-center gap-1.5 font-mono text-xs uppercase tracking-wider text-faint">
            <Trophy size={13} weight="duotone" />
            {t("six.board.title")}
          </p>
          <SixSevenLeaderboard entries={leaderboard} compact />
        </div>
      </motion.div>
    </div>
  );
}
