import { useEffect, useState } from "react";
import { CaretRight, Lock, VideoCamera } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { useNavigate } from "react-router-dom";
import AppLayout from "../components/AppLayout";
import { api } from "../api";
import { MOVEMENT_GROUPS, type AnalyzableMovement } from "../lib/movements";
import { useI18n, movementLabel } from "../lib/i18n";

// The movement menu: every movement the coach knows, laid out at once and grouped by body region.
// Picking one hands off to the analysis studio. Only the movements the pipeline can actually
// analyse are actionable (see GET /api/movements); the rest are listed but inert, so the menu
// stays complete without promising an analysis we would run with the wrong rules.
export default function Movements() {
  const { t } = useI18n();
  const navigate = useNavigate();
  const reduce = useReducedMotion();

  // Which movements are analyzable comes from the server, derived from the detector registry, so
  // this page needs no edit when a detector is registered. On failure fall back to Squat-only:
  // the studio still works, and the alternative -- offering every movement -- would run the wrong
  // rules over the user's video.
  const [live, setLive] = useState<AnalyzableMovement[]>([{ name: "Squat", validated: true }]);
  useEffect(() => {
    let cancelled = false;
    api
      .getMovements()
      .then((ms) => {
        if (!cancelled && ms.length) setLive(ms);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  // The flat position of each group's first card, so the reveal cascades down the whole page in
  // reading order rather than restarting at every section.
  const groupOffsets: number[] = [];
  MOVEMENT_GROUPS.reduce((acc, group) => {
    groupOffsets.push(acc);
    return acc + group.items.length;
  }, 0);

  return (
    <AppLayout>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-5xl px-4 py-8 lg:px-6 lg:py-12">
          <h1 className="font-display text-2xl font-bold text-content">{t("movements.title")}</h1>
          <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted">{t("movements.subtitle")}</p>

          <div className="mt-9 flex flex-col gap-9">
            {MOVEMENT_GROUPS.map((group, gi) => (
              <section key={group.key}>
                <div className="flex items-center gap-3">
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-faint">{t(group.key)}</h2>
                  <span className="h-px flex-1 bg-border-dark" />
                </div>

                <ul className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {group.items.map((movement, i) => {
                    const entry = live.find((m) => m.name === movement);
                    const name = movementLabel(t, movement);
                    return (
                      <motion.li
                        key={movement}
                        initial={reduce ? false : { opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{
                          duration: 0.35,
                          delay: (groupOffsets[gi] + i) * 0.025,
                          ease: [0.16, 1, 0.3, 1],
                        }}
                      >
                        {entry ? (
                          <button
                            onClick={() => navigate(`/app?movement=${encodeURIComponent(movement)}`)}
                            className="flex w-full items-center gap-3 rounded-xl border border-primary/25 bg-primary/[0.07] px-4 py-3.5 text-left transition-colors hover:bg-primary/[0.12] active:translate-y-px"
                          >
                            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                              <VideoCamera size={18} weight="duotone" />
                            </span>
                            <span className="min-w-0 flex-1">
                              <span className="block truncate font-medium text-content">
                                {name}
                                {!entry.validated && (
                                  <span
                                    title={t("movements.betaNote")}
                                    className="ml-2 rounded px-1.5 py-0.5 align-middle text-[10px] font-semibold uppercase tracking-wide text-warning ring-1 ring-warning/40"
                                  >
                                    {t("movements.beta")}
                                  </span>
                                )}
                              </span>
                              <span className="block text-xs text-primary">{t("movements.analyze")}</span>
                            </span>
                            <CaretRight size={14} weight="bold" className="shrink-0 text-primary" />
                          </button>
                        ) : (
                          // Not a disabled <button>: there is no action behind it, so it is listed as
                          // plain content instead of a control that pretends to be one.
                          <div className="flex w-full items-center gap-3 rounded-xl border border-border-dark bg-surface px-4 py-3.5">
                            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-content/5 text-faint">
                              <Lock size={18} weight="duotone" />
                            </span>
                            <span className="min-w-0 flex-1">
                              <span className="block truncate font-medium text-muted">{name}</span>
                              <span className="block text-xs text-faint">{t("movements.soon")}</span>
                            </span>
                          </div>
                        )}
                      </motion.li>
                    );
                  })}
                </ul>
              </section>
            ))}
          </div>
        </main>
      </div>
    </AppLayout>
  );
}
