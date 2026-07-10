import { Trophy } from "@phosphor-icons/react";
import { useI18n } from "../../lib/i18n";
import type { DuelEntry } from "../../lib/duel/leaderboard";

interface Props {
  entries: DuelEntry[];
  // ts of a just-saved duel to highlight.
  highlightTs?: number;
  compact?: boolean;
}

// Local log of the ten most recent duels. Presentational — the caller loads/persists via
// lib/duel/leaderboard.
export default function DuelLeaderboard({ entries, highlightTs, compact }: Props) {
  const { t } = useI18n();

  if (entries.length === 0) {
    return (
      <div className="rounded-2xl border border-border-dark bg-surface-dark p-6 text-center">
        <Trophy size={28} weight="duotone" className="mx-auto text-faint" />
        <p className="mt-2 text-sm text-muted">{t("duel.board.empty")}</p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-border-dark bg-surface-dark">
      {!compact && (
        <div className="flex items-center gap-2 border-b border-border-dark px-4 py-3">
          <Trophy size={18} weight="duotone" className="text-primary" />
          <span className="font-display text-sm font-semibold text-content">
            {t("duel.board.title")}
          </span>
        </div>
      )}
      <ol className="divide-y divide-border-dark">
        {entries.map((e, i) => {
          const highlighted = e.ts === highlightTs;
          return (
            <li
              key={`${e.ts}-${i}`}
              className={`flex items-center gap-3 px-4 py-2.5 ${
                highlighted ? "bg-primary/10" : ""
              }`}
            >
              <span aria-hidden className="w-5 shrink-0 text-center text-sm">
                🏆
              </span>
              <span className="min-w-0 flex-1 truncate text-sm text-content">
                <span className="font-semibold">{e.winner}</span>
                <span className="text-muted"> {t("duel.board.beat")} </span>
                <span className="text-muted">{e.loser}</span>
              </span>
              <span className="shrink-0 font-mono text-sm font-bold tabular-nums text-content">
                {e.winnerPoints}–{e.loserPoints}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
