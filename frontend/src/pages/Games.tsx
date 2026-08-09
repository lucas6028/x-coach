import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "motion/react";
import {
  ArrowRight,
  Fire,
  GameController,
  HandWaving,
  Knife,
  MaskHappy,
  Trophy,
  Warning,
  type Icon,
} from "@phosphor-icons/react";
import AppLayout from "../components/AppLayout";
import { useI18n } from "../lib/i18n";
import { useLiffContext } from "../lib/liffContext";
import { loadCalories } from "../lib/calorieStore";
import { bestScore as bestNinjaScore } from "../lib/ninja/leaderboard";
import { loadLeaderboard as loadSixBoard } from "../lib/sixseven/leaderboard";
import { bestScore as bestWebScore } from "../lib/webslinger/leaderboard";
import type { GameId } from "../lib/calories";

interface GameCard {
  id: GameId;
  to: string;
  emoji: string;
  gradient: string;
  icon: Icon;
  titleKey: string;
  descKey: string;
  tagKey: string;
  best: number;
  bestLabelKey: string;
}

// The games hub — an App-Store / Steam-style catalog of the pose mini-games. Cards link into the
// existing game routes; this page deliberately does NOT import the game pages themselves, so the
// ~800 kB MediaPipe bundle only loads once a player actually opens a game.
export default function Games() {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  const { isInClient } = useLiffContext();

  const totals = loadCalories();
  const bestSix = loadSixBoard()[0]?.count ?? 0;

  const games: GameCard[] = [
    {
      id: "ninja",
      to: "/ninja",
      emoji: "🍉",
      gradient: "from-rose-500/25 via-orange-400/15 to-amber-300/10",
      icon: Knife,
      titleKey: "ninja.title",
      descKey: "games.ninja.desc",
      tagKey: "ninja.badge",
      best: bestNinjaScore(),
      bestLabelKey: "games.stat.bestScore",
    },
    {
      id: "sixseven",
      to: "/67",
      emoji: "🙌",
      gradient: "from-violet-500/25 via-fuchsia-400/15 to-sky-300/10",
      icon: HandWaving,
      titleKey: "six.title",
      descKey: "games.six.desc",
      tagKey: "six.badge",
      best: bestSix,
      bestLabelKey: "games.stat.bestCount",
    },
    {
      id: "webslinger",
      to: "/web-slinger",
      emoji: "🕸️",
      gradient: "from-rose-700/30 via-slate-700/20 to-sky-400/10",
      icon: MaskHappy,
      titleKey: "web.title",
      descKey: "games.web.desc",
      tagKey: "web.badge",
      best: bestWebScore(),
      bestLabelKey: "games.stat.bestScore",
    },
  ];

  return (
    <AppLayout>
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="mx-auto max-w-5xl px-5 py-8 sm:px-6 sm:py-12"
        >
          <span className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-primary">
            <GameController size={13} weight="fill" />
            {t("games.badge")}
          </span>
          <h2 className="mt-4 font-display text-3xl font-bold leading-tight tracking-tight text-content md:text-4xl">
            {t("games.heading")}
          </h2>
          <p className="mt-3 max-w-xl leading-relaxed text-muted">{t("games.sub")}</p>

          {/* Inside LINE, getUserMedia can hang on iOS (see lib/camera). The games already
              recover from that after the fact; this warns before the player commits. */}
          {isInClient && (
            <div className="mt-5 flex items-start gap-3 rounded-2xl border border-amber-500/25 bg-amber-500/[0.07] p-4">
              <Warning size={20} weight="fill" className="mt-0.5 shrink-0 text-amber-400" />
              <p className="text-sm leading-relaxed text-muted">{t("games.liffCameraHint")}</p>
            </div>
          )}

          {/* Lifetime calorie total across every game. */}
          <div className="mt-7 flex items-center gap-4 rounded-2xl border border-orange-500/20 bg-gradient-to-br from-orange-500/[0.08] to-amber-400/[0.04] p-5">
            <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-orange-500/15 text-orange-400">
              <Fire size={26} weight="fill" />
            </span>
            <div className="min-w-0">
              <p className="font-mono text-[11px] uppercase tracking-wider text-faint">
                {t("games.totalTitle")}
              </p>
              {totals.sessions > 0 ? (
                <>
                  <p className="font-display text-3xl font-black tabular-nums text-content">
                    <span>{totals.total.toLocaleString()}</span>{" "}
                    <span className="text-lg font-bold text-muted">{t("games.kcalUnit")}</span>
                  </p>
                  <p className="text-xs text-muted">
                    {totals.sessions === 1
                      ? t("games.totalSubOne")
                      : t("games.totalSub", { n: totals.sessions })}
                  </p>
                </>
              ) : (
                <p className="mt-1 text-sm text-muted">{t("games.totalEmpty")}</p>
              )}
            </div>
          </div>

          {/* Catalog grid. */}
          <div className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-2">
            {games.map((game) => {
              const GameIcon = game.icon;
              return (
                <Link
                  key={game.id}
                  to={game.to}
                  className="group flex flex-col overflow-hidden rounded-2xl border border-border-dark bg-surface-dark transition-colors hover:border-primary/40"
                >
                  <div
                    className={`relative flex h-36 items-center justify-center bg-gradient-to-br ${game.gradient}`}
                  >
                    <span aria-hidden className="text-6xl drop-shadow-lg">
                      {game.emoji}
                    </span>
                    <span className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-black/30 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-white/90 backdrop-blur-sm">
                      <GameIcon size={12} weight="fill" />
                      {t(game.tagKey)}
                    </span>
                  </div>

                  <div className="flex flex-1 flex-col p-5">
                    <h3 className="font-display text-xl font-bold text-content">
                      {t(game.titleKey)}
                    </h3>
                    <p className="mt-1.5 flex-1 text-sm leading-relaxed text-muted">
                      {t(game.descKey)}
                    </p>

                    <div className="mt-4 flex items-center justify-between gap-3">
                      <div className="flex items-center gap-4 text-xs">
                        <span className="flex items-center gap-1.5 text-muted">
                          <Trophy size={14} weight="duotone" className="text-primary" />
                          <span className="tabular-nums font-semibold text-content">
                            {game.best.toLocaleString()}
                          </span>
                          <span className="text-faint">{t(game.bestLabelKey)}</span>
                        </span>
                        <span className="flex items-center gap-1.5 text-muted">
                          <Fire size={14} weight="fill" className="text-orange-400" />
                          <span className="tabular-nums font-semibold text-content">
                            {totals.byGame[game.id].toLocaleString()}
                          </span>
                          <span className="text-faint">{t("games.kcalUnit")}</span>
                        </span>
                      </div>
                      <span className="flex items-center gap-1.5 rounded-xl bg-primary px-3.5 py-2 text-sm font-semibold text-primary-content transition-transform group-hover:translate-x-0.5">
                        {t("games.play")}
                        <ArrowRight size={16} weight="bold" />
                      </span>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </motion.div>
      </div>
    </AppLayout>
  );
}
