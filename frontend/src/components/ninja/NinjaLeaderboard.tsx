import { Trophy } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import type { NinjaEntry } from "../../lib/ninja/leaderboard";

interface Props {
  entries: NinjaEntry[];
  highlightRank?: number;
  compact?: boolean;
}

const MEDALS = ["🥇", "🥈", "🥉"];

// Local top-10 board. Presentational — the caller loads/persists via lib/ninja.
export default function NinjaLeaderboard({ entries, highlightRank, compact }: Props) {
  const { t } = useI18n();

  if (entries.length === 0) {
    return (
      <div className="rounded-2xl border border-border-dark bg-surface-dark p-6 text-center">
        <Trophy size={28} weight="duotone" className="mx-auto text-faint" />
        <p className="mt-2 text-sm text-muted">{t("ninja.board.empty")}</p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-border-dark bg-surface-dark">
      {!compact && (
        <div className="flex items-center gap-2 border-b border-border-dark px-4 py-3">
          <Trophy size={18} weight="duotone" className="text-primary" />
          <span className="font-display text-sm font-semibold text-content">
            {t("ninja.board.title")}
          </span>
        </div>
      )}
      <ol className="divide-y divide-border-dark">
        {entries.map((e, i) => {
          const rank = i + 1;
          const highlighted = rank === highlightRank;
          return (
            <li
              key={`${e.ts}-${i}`}
              className={`flex items-center gap-3 px-4 py-2.5 ${highlighted ? "bg-primary/10" : ""}`}
            >
              <span className="w-6 shrink-0 text-center text-sm font-semibold text-muted">
                {MEDALS[i] ?? rank}
              </span>
              <span className="min-w-0 flex-1 truncate text-sm font-medium text-content">
                {e.name}
                {highlighted && (
                  <span className="ml-2 rounded-full bg-primary/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary">
                    {t("ninja.board.you")}
                  </span>
                )}
              </span>
              <span className="shrink-0 font-mono text-sm font-bold tabular-nums text-content">
                {e.score.toLocaleString()}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
