import { useI18n } from "../lib/i18n";
import type { ToolRun } from "../api";

// The tools we have i18n labels for. A name outside this list falls back to the generic label —
// `t()` returns the key itself on a miss (i18n.tsx), so an unguarded lookup would render a raw key
// like "chat.tool.something_else" into the tray.
const TOOL_LABEL_KEYS = ["get_analysis", "kg_query", "rag_search"] as const;

/** The tool calls behind one answer, in call order, above the answer they produced.
 *
 * Used twice: for a committed assistant message (from `message.tools`) and for the turn currently
 * streaming (from live state). Same markup both times, so nothing shifts when the turn commits.
 *
 * `kg_query`'s entries are headed differently from the other tools' on purpose. Its "sources" are
 * knowledge-graph concepts, which carry no citation anywhere in the graph; showing them under the
 * same heading as a retrieved document would tell the user a concept is a source.
 */
export function ToolRunList({ runs }: { runs: ToolRun[] }) {
  const { t } = useI18n();
  if (!runs.length) return null;
  return (
    <div className="flex flex-col gap-2">
      {runs.map((run, i) => {
        const known = TOOL_LABEL_KEYS.includes(run.name as (typeof TOOL_LABEL_KEYS)[number]);
        const heading = run.name === "kg_query" ? t("chat.tool.concepts") : t("chat.tool.sources");
        return (
          <div key={i} className="text-xs text-muted">
            <div>
              {known ? t(`chat.tool.${run.name}` as never) : t("chat.tool.generic")}
              {run.query ? `${t("chat.tool.sep")}${run.query}` : ""}
            </div>
            {Array.isArray(run.sources) && run.sources.length > 0 && (
              <div className="mt-1 pl-3 flex flex-col gap-0.5">
                <div className="text-faint">{heading}</div>
                {run.sources.map((s, j) => (
                  <div key={j} className="text-faint">
                    {s.label}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
