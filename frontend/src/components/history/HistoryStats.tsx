import { useMemo } from "react";
import { Barbell, ChartLineUp, ClockCounterClockwise } from "@phosphor-icons/react";
import type { HistoryItem } from "../../api";
import { movementLabel, useI18n } from "../../lib/i18n";

interface Props {
  /** The rows actually loaded — one page, so possibly fewer than `total`. */
  items: HistoryItem[];
  /** The caller's all-time count, straight from the API. */
  total: number;
}

// One summary tile. Deliberately a <div>, not a <section> with a heading: the day separators below
// are the page's only h2s, and the history tests count them.
function Tile({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-border-dark bg-surface-dark p-4 shadow-card">
      <p className="text-[11px] tracking-wide text-faint">{label}</p>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

// A 7-day sparkline of how many analyses landed each day. Decorative — it says "you trained on
// four of the last seven days", not a precise series — so it carries no axis and no tooltip.
function Spark({ counts }: { counts: number[] }) {
  const max = Math.max(1, ...counts);
  const step = counts.length > 1 ? 64 / (counts.length - 1) : 0;
  const pts = counts.map((c, i) => [2 + i * step, 30 - (c / max) * 24] as const);
  const line = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ");
  return (
    // `primary` is a fixed hex in the Tailwind config, not a CSS variable, so the strokes take it
    // from `currentColor` rather than a `rgb(var(--…))` that would resolve to nothing.
    <svg viewBox="0 0 68 34" className="h-9 w-[68px] shrink-0 text-primary" aria-hidden="true">
      <path d={`${line} L 66 34 L 2 34 Z`} fill="currentColor" opacity="0.1" />
      <path
        d={line}
        fill="none"
        stroke="currentColor"
        opacity="0.55"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// A ring showing the clean-rep share. `strokeDasharray` is expressed against the circumference so
// the arc is the percentage, not an eyeballed constant.
function Ring({ pct }: { pct: number }) {
  const r = 18;
  const c = 2 * Math.PI * r;
  return (
    <svg
      viewBox="0 0 52 52"
      className="h-[52px] w-[52px] shrink-0 -rotate-90 text-primary"
      aria-hidden="true"
    >
      <circle cx="26" cy="26" r={r} fill="none" stroke="rgb(var(--c-border))" strokeWidth="6" />
      <circle
        cx="26"
        cy="26"
        r={r}
        fill="none"
        stroke="currentColor"
        strokeWidth="6"
        strokeLinecap="round"
        strokeDasharray={`${(pct / 100) * c} ${c}`}
      />
    </svg>
  );
}

// The reference design's summary strip: four tiles above the record grid.
//
// Only "total" is an all-time number — the API hands it back with the page. Everything else is
// computed from the rows on this page (50 by default), so when there are more the strip says which
// window it is describing rather than passing a partial figure off as a lifetime one.
export default function HistoryStats({ items, total }: Props) {
  const { t, lang } = useI18n();

  const { cleanPct, top, latest, spark } = useMemo(() => {
    const clean = items.filter((it) => it.fault_count === 0).length;

    // Most-trained movement, by the label the cards show — so "Squat" counts rows that predate the
    // movement column together with rows that name it, exactly as the grid below groups them.
    const tally = new Map<string, number>();
    for (const it of items) {
      const name = movementLabel(t, it.movement ?? "Squat");
      tally.set(name, (tally.get(name) ?? 0) + 1);
    }
    let best: { name: string; count: number } | null = null;
    for (const [name, count] of tally) {
      if (!best || count > best.count) best = { name, count };
    }

    // Counts for the last seven days, oldest first. Local midnights, so "today" matches the day
    // separator the grid draws.
    const midnight = new Date();
    midnight.setHours(0, 0, 0, 0);
    const days = new Array(7).fill(0);
    for (const it of items) {
      const d = new Date(it.created_at);
      if (Number.isNaN(d.getTime())) continue;
      const ago = Math.floor((midnight.getTime() - d.setHours(0, 0, 0, 0)) / 86_400_000);
      if (ago >= 0 && ago < 7) days[6 - ago] += 1;
    }

    return {
      cleanPct: items.length ? Math.round((clean / items.length) * 100) : null,
      top: best,
      latest: items[0] ?? null,
      spark: days,
    };
  }, [items, t]);

  const latestAt = latest && new Date(latest.created_at);
  const latestLabel =
    latestAt && !Number.isNaN(latestAt.getTime())
      ? latestAt.toLocaleString(lang, { dateStyle: "short", timeStyle: "short" })
      : (latest?.created_at ?? t("history.statNone"));

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
        <Tile label={t("history.statTotal")}>
          <div className="flex items-end justify-between gap-2">
            <span className="text-[22px] font-bold leading-none text-content">{total}</span>
            <Spark counts={spark} />
          </div>
        </Tile>

        <Tile label={t("history.statCleanRate")}>
          <div className="flex items-center justify-between gap-2">
            <span className="text-[22px] font-bold leading-none text-content">
              {cleanPct === null ? t("history.statNone") : `${cleanPct}%`}
            </span>
            <Ring pct={cleanPct ?? 0} />
          </div>
        </Tile>

        <Tile label={t("history.statTopMovement")}>
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
              <Barbell size={18} weight="duotone" />
            </span>
            <div className="min-w-0">
              <p className="truncate text-[13px] font-semibold leading-none text-content">
                {top ? top.name : t("history.statNone")}
              </p>
              {top && (
                <p className="mt-1 text-[11px] text-faint">
                  {t("history.statTimes", { count: top.count })}
                </p>
              )}
            </div>
          </div>
        </Tile>

        <Tile label={t("history.statLatest")}>
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
              {latest ? (
                <ClockCounterClockwise size={18} weight="duotone" />
              ) : (
                <ChartLineUp size={18} weight="duotone" />
              )}
            </span>
            <div className="min-w-0">
              <p className="truncate text-[13px] font-semibold leading-none text-content">
                {latest ? latestLabel : t("history.statNone")}
              </p>
              {latest && (
                <p className="mt-1 truncate text-[11px] text-faint">
                  {movementLabel(t, latest.movement ?? "Squat")}
                </p>
              )}
            </div>
          </div>
        </Tile>
      </div>

      {/* Said once, under the strip, rather than repeated on each derived tile. */}
      {items.length < total && (
        <p className="mt-2 text-[11px] text-faint">
          {t("history.statsScope", { loaded: items.length, total })}
        </p>
      )}
    </div>
  );
}
