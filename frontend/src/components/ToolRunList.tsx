import { useState } from "react";
import { useI18n } from "../lib/i18n";
import type { ToolRun } from "../api";
import { LumenLoader } from "./LumenLoader";

// The tools we have i18n labels for. A name outside this list falls back to the generic label —
// `t()` returns the key itself on a miss (i18n.tsx), so an unguarded lookup would render a raw key
// like "chat.tool.something_else" into the tray.
const TOOL_LABEL_KEYS = ["get_analysis", "kg_query", "rag_search"] as const;

// A committed `ToolRun` plus the one live field the renderer cares about. `pending` is absent on a
// message restored from history — a stored record is finished by definition.
type DisplayToolRun = ToolRun & { pending?: boolean };

/** The tool calls behind one answer, in call order, above the answer they produced.
 *
 * Used twice: for a committed assistant message (from `message.tools`) and for the turn currently
 * streaming (from live state). Same markup both times, so nothing shifts when the turn commits.
 */
export function ToolRunList({ runs }: { runs: DisplayToolRun[] }) {
  if (!runs.length) return null;
  return (
    <div className="flex flex-col gap-2">
      {runs.map((run, i) => (
        <ToolRunRow key={i} run={run} />
      ))}
    </div>
  );
}

/** One tool call: the lookup line, plus a collapsed count of what it cited.
 *
 * A child component rather than inline markup because the expanded state is per-row and a hook
 * cannot live inside a `.map`. That state is deliberately ephemeral — a row collapses when the turn
 * commits, since the live list is replaced by the committed message's own. Preserving it would mean
 * hoisting row state into CoachTray and keying it across both render paths, which is not worth it
 * for a state the user usually enters after reading.
 *
 * A source's `kind` decides the heading, not the tool that produced it. `kg_query`'s entries come
 * back with `kind: "concept"` because knowledge-graph nodes carry no citation anywhere in the
 * graph; counting them under the same word as retrieved documents would tell the user a concept is
 * a source. Keying off `kind` means a future tool that also returns concept-kind sources gets the
 * safe wording automatically.
 */
function ToolRunRow({ run }: { run: DisplayToolRun }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const known = TOOL_LABEL_KEYS.includes(run.name as (typeof TOOL_LABEL_KEYS)[number]);
  const sources = Array.isArray(run.sources) ? run.sources : [];
  const isConcept = sources.some((s) => s.kind === "concept");
  return (
    <div className="text-xs text-muted" aria-busy={!!run.pending}>
      {/* The label, the query, and the pending marker share ONE element: the marker is an element
          child, which testing-library's text matcher ignores, so the line still matches by text and
          still parents the source block below it. */}
      <div className="flex items-center gap-2">
        {known ? t(`chat.tool.${run.name}`) : t("chat.tool.generic")}
        {run.query ? `${t("chat.tool.sep")}${run.query}` : ""}
        {run.pending && (
          // aria-hidden because LumenLoader's dots carry role="status": today exactly one exists at
          // a time (CoachTray's, gated on toolRuns.length === 0), but a three-tool turn would mount
          // three simultaneous live regions all announcing the same string. `aria-busy` on the row
          // states the same thing once, in the right place.
          <span aria-hidden="true" className="inline-flex">
            <LumenLoader variant="dots" />
          </span>
        )}
      </div>
      {sources.length > 0 && (
        <div className="mt-1 flex flex-col gap-0.5 pl-3">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="self-start text-left text-faint transition-colors hover:text-muted"
          >
            <span aria-hidden="true" className="mr-1 inline-block">
              {open ? "⌄" : "›"}
            </span>
            {t(isConcept ? "chat.tool.conceptsN" : "chat.tool.sourcesN", { n: sources.length })}
          </button>
          {open &&
            sources.map((s, j) => (
              <div key={j} className="text-faint">
                {s.label}
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
