import { useEffect, useMemo, useState } from "react";
import { MagnifyingGlass, UploadSimple } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { Link, useNavigate } from "react-router-dom";
import AppLayout from "../components/AppLayout";
import MovementCard from "../components/movements/MovementCard";
import { api } from "../api";
import { MOVEMENT_GROUPS, type AnalyzableMovement } from "../lib/movements";
import { useI18n, movementLabel } from "../lib/i18n";

// The movement menu, rebuilt against the exercise_library_muse-spark reference: a search-and-
// filter row over a grid of preview cards.
//
// THE ONE PLACE THIS DEPARTS FROM THE REFERENCE'S STRUCTURE: the reference filters a single flat
// grid with its category pills. Here the pills narrow which SECTIONS are shown, and the body-region
// sections stay — that grouping is what someone picking a movement to train actually reasons in,
// and it is the arrangement this page already had. So the pills are a coarse jump, not the only
// way to tell a squat from a row.
//
// Also dropped: the mock's "All categories" dropdown, which duplicates the pill row beside it.
//
// Picking a movement hands off to the analysis studio. Only the movements the pipeline can
// actually analyse are actionable (see GET /api/movements); the rest are listed but inert, so the
// menu stays complete without promising an analysis we would run with the wrong rules.
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

  const [search, setSearch] = useState("");
  const [region, setRegion] = useState<string>("all");

  // Matching is on the LABEL the reader can see, not on the canonical English key: a zh reader
  // typing 深蹲 is searching the only name this page ever showed them. Sections that end up empty
  // are dropped entirely rather than left as a heading over nothing.
  const groups = useMemo(() => {
    const q = search.trim().toLowerCase();
    return MOVEMENT_GROUPS.filter((g) => region === "all" || g.key === region)
      .map((g) => ({
        key: g.key,
        items: g.items.filter((m) => !q || movementLabel(t, m).toLowerCase().includes(q)),
      }))
      .filter((g) => g.items.length > 0);
  }, [search, region, t]);

  // The flat position of each group's first card, so the reveal cascades down the whole page in
  // reading order rather than restarting at every section.
  const groupOffsets: number[] = [];
  groups.reduce((acc, group) => {
    groupOffsets.push(acc);
    return acc + group.items.length;
  }, 0);

  const regions = [
    { key: "all", label: t("movements.filterAll") },
    ...MOVEMENT_GROUPS.map((g) => ({ key: g.key as string, label: t(g.key) })),
  ];

  return (
    <AppLayout>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-6xl px-4 py-8 lg:px-6 lg:py-12">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h1 className="font-display text-2xl font-bold text-content">
                {t("movements.title")}
              </h1>
              <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted">
                {t("movements.subtitle")}
              </p>
            </div>
            <Link
              to="/app"
              className="inline-flex shrink-0 items-center gap-2 rounded-full bg-primary px-5 py-2 text-[13px] font-semibold text-primary-content shadow-accent transition-colors hover:bg-primary/90 active:scale-[0.99]"
            >
              <UploadSimple size={15} weight="bold" />
              {t("movements.uploadCta")}
            </Link>
          </div>

          {/* Search + region pills. The pills scroll rather than wrap on a narrow screen: five
              short pills wrapping to two lines would push the first card below the fold on a
              phone, and unlike History's three dropdown filters there is nothing here worth
              folding behind a funnel — a pill row IS the compact form. */}
          <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="relative sm:w-[260px]">
              <MagnifyingGlass
                size={15}
                className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-faint"
              />
              <input
                type="search"
                aria-label={t("movements.searchPlaceholder")}
                placeholder={t("movements.searchPlaceholder")}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="glass-control h-[44px] w-full rounded-full pl-10 pr-4 text-[13px] font-medium text-content outline-none transition-colors placeholder:font-normal placeholder:text-faint hover:border-primary/40 focus:border-primary/40 [&::-webkit-search-cancel-button]:appearance-none"
              />
            </div>

            {/* Not a <ul>: this is a row of controls, and the page's lists are its four sections.
                `min-w-0` is what lets the row actually scroll rather than size to its content —
                a flex item's default `min-width: auto` floors it at its intrinsic width. */}
            <div className="scrollbar-none flex min-w-0 gap-2 overflow-x-auto sm:flex-wrap">
              {regions.map((r) => {
                const active = r.key === region;
                return (
                  <button
                    key={r.key}
                    type="button"
                    aria-pressed={active}
                    onClick={() => setRegion(r.key)}
                    className={`h-[38px] shrink-0 rounded-full px-4 text-[13px] font-medium transition-colors ${
                      active
                        ? "bg-primary text-primary-content shadow-accent"
                        : "glass-control text-muted hover:text-content"
                    }`}
                  >
                    {r.label}
                  </button>
                );
              })}
            </div>
          </div>

          {groups.length === 0 ? (
            <div className="mt-8 rounded-2xl border border-border-dark bg-surface px-6 py-12 text-center text-sm text-muted">
              {t("movements.noMatch")}
            </div>
          ) : (
            <div className="mt-8 flex flex-col gap-9">
              {groups.map((group, gi) => (
                <section key={group.key}>
                  <div className="flex items-center gap-3">
                    <h2 className="text-xs font-semibold uppercase tracking-wider text-faint">
                      {t(group.key)}
                    </h2>
                    <span className="h-px flex-1 bg-border-dark" />
                  </div>

                  <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 lg:gap-4">
                    {group.items.map((movement, i) => {
                      const entry = live.find((m) => m.name === movement);
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
                          <MovementCard
                            movement={movement}
                            live={entry ? { validated: entry.validated } : undefined}
                            onPick={() =>
                              navigate(`/app?movement=${encodeURIComponent(movement)}`)
                            }
                          />
                        </motion.li>
                      );
                    })}
                  </ul>
                </section>
              ))}
            </div>
          )}
        </main>
      </div>
    </AppLayout>
  );
}
