import { useMemo } from "react";
import { motion, useReducedMotion } from "motion/react";
import type { Analysis, Detection, Retrieval, RagResult } from "../api";
import { fmtTime } from "../lib/format";
import { useI18n, faultLabel, phaseLabel, severityText } from "../lib/i18n";

interface Props {
  analysis: Analysis;
  currentTime: number;
  onSeek: (t: number) => void;
}

interface SummaryNode {
  node_id: string;
  label: string;
}

function summaryCategory(retrieval: Retrieval | undefined, key: string): string[] {
  if (!retrieval) return [];
  const results = (retrieval.context.results as Array<Record<string, unknown>>) || [];
  const out: string[] = [];
  for (const seed of results) {
    const summary = (seed.summary as Record<string, SummaryNode[]>) || {};
    for (const n of summary[key] || []) out.push(n.node_id);
  }
  return Array.from(new Set(out)).slice(0, 4);
}

function ragSnippet(retrieval: Retrieval | undefined): string | null {
  if (!retrieval || retrieval.retrieval_mode !== "rag") return null;
  const results = (retrieval.context.results as RagResult[]) || [];
  return results.length ? results[0].text.slice(0, 200) + "…" : null;
}

function keyEvidence(d: Detection): string {
  const ev = d.evidence || {};
  const entries = Object.entries(ev).filter(([, v]) => typeof v === "number");
  if (!entries.length) return "";
  const [k, v] = entries[0];
  return `${k.replace(/_/g, " ")} ${typeof v === "number" ? v.toFixed(2) : v}`;
}

// Severity drives a small graded signal (real semantic state, not decoration):
// high = danger red, moderate = amber, mild = muted neutral.
type SevLevel = "high" | "moderate" | "mild";
function sevLevel(sev: number): SevLevel {
  if (sev >= 0.75) return "high";
  if (sev >= 0.4) return "moderate";
  return "mild";
}
const SEV_DOT: Record<SevLevel, string> = {
  high: "bg-danger",
  moderate: "bg-amber-500",
  mild: "bg-faint",
};
const SEV_CHIP: Record<SevLevel, string> = {
  high: "bg-danger/10 text-danger",
  moderate: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
  mild: "bg-content/5 text-muted",
};

function FaultCard({
  d,
  retrieval,
  active,
  last,
  onSeek,
}: {
  d: Detection;
  retrieval: Retrieval | undefined;
  active: boolean;
  last: boolean;
  onSeek: (t: number) => void;
}) {
  const { t } = useI18n();
  const causes = summaryCategory(retrieval, "causes");
  const risks = summaryCategory(retrieval, "risks");
  const corrections = summaryCategory(retrieval, "corrections");
  const snippet = ragSnippet(retrieval);
  const level = sevLevel(d.severity);
  const evidence = keyEvidence(d);

  return (
    <div className="flex gap-3">
      {/* temporal rail: faults are ordered by when they happen in the rep */}
      <div className="flex flex-col items-center pt-1.5">
        <span className="relative flex h-2.5 w-2.5">
          {active && (
            <span className={`absolute inline-flex h-full w-full rounded-full ${SEV_DOT[level]} animate-ping opacity-60`} />
          )}
          <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${SEV_DOT[level]}`} />
        </span>
        {!last && <span className="mt-1 w-px flex-1 bg-border-dark" />}
      </div>

      <button
        onClick={() => onSeek(d.start_time)}
        className={`mb-1 w-full overflow-hidden rounded-xl border text-left transition-colors ${
          active
            ? "border-primary/50 bg-primary/[0.04] ring-1 ring-primary/30"
            : "border-border-dark bg-surface hover:bg-content/[0.03]"
        }`}
      >
        {/* header: time, fault, severity */}
        <div className="flex items-start justify-between gap-3 px-4 pt-3.5">
          <div className="min-w-0">
            <span className="font-mono text-[11px] text-faint">{fmtTime(d.start_time)}</span>
            <p className="font-display font-semibold leading-tight text-content">
              {faultLabel(t, d.fault_name)}
            </p>
            <p className="mt-0.5 text-xs text-muted">
              {t("feedback.phaseTag", { phase: phaseLabel(t, d.phase) })}
            </p>
          </div>
          <span
            className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium ${SEV_CHIP[level]}`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${SEV_DOT[level]}`} />
            {severityText(t, d.severity)}
          </span>
        </div>

        {evidence && (
          <div className="mt-2.5 px-4">
            <span className="inline-block rounded-md bg-content/5 px-2 py-1 font-mono text-[11px] text-muted">
              {evidence}
            </span>
          </div>
        )}

        {/* reasoning: cause and risk retrieved from the knowledge graph */}
        {(causes.length > 0 || risks.length > 0) && (
          <dl className="mt-3 space-y-2 border-t border-border-dark/70 px-4 pt-3">
            {causes.length > 0 && (
              <div className="flex gap-2 text-xs leading-relaxed">
                <dt className="shrink-0 text-muted">{t("feedback.likelyCause")}</dt>
                <dd className="font-medium text-primary">{causes.join(", ")}</dd>
              </div>
            )}
            {risks.length > 0 && (
              <div className="flex gap-2 text-xs leading-relaxed">
                <dt className="shrink-0 text-muted">{t("feedback.injuryRisk")}</dt>
                <dd className="font-medium text-danger">{risks.join(", ")}</dd>
              </div>
            )}
          </dl>
        )}

        {/* the actionable cue carries the most visual weight */}
        {corrections.length > 0 && (
          <div className="mx-4 mb-4 mt-3 flex items-start gap-2.5 rounded-lg bg-secondary/10 p-3">
            <span className="material-symbols-outlined mt-px text-base text-secondary">arrow_forward</span>
            <div className="min-w-0">
              <span className="text-[10px] font-bold uppercase tracking-wider text-secondary">
                {t("feedback.cue")}
              </span>
              <p className="text-sm font-medium text-content">{corrections.join(" · ")}</p>
            </div>
          </div>
        )}

        {snippet && !corrections.length && (
          <p className="mx-4 mb-4 mt-3 border-l-2 border-border-dark pl-3 text-[11px] italic leading-relaxed text-muted">
            {snippet}
          </p>
        )}

        {!causes.length && !risks.length && !corrections.length && !snippet && (
          <div className="px-4 pb-4 pt-2" />
        )}
      </button>
    </div>
  );
}

export default function ReasoningLog({ analysis, currentTime, onSeek }: Props) {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  const retrievalByFault = useMemo(() => {
    const m = new Map<string, Retrieval>();
    for (const r of analysis.retrievals) if (!m.has(r.fault_id)) m.set(r.fault_id, r);
    return m;
  }, [analysis]);

  const detections = analysis.detections;

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-background">
      <div className="flex items-center justify-between gap-2 border-b border-border-dark bg-surface-dark px-4 py-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-content">
          <span className="material-symbols-outlined text-base text-primary">psychology</span>
          {t("feedback.title")}
        </h2>
        <span className="rounded-full border border-primary/20 bg-primary/10 px-2.5 py-1 font-mono text-[10px] text-primary">
          {t("feedback.badge")}
        </span>
      </div>

      <div className="scrollbar-thin flex-1 space-y-4 overflow-y-auto p-4">
        {detections.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
            <span className="flex h-14 w-14 items-center justify-center rounded-full bg-secondary/10">
              <span className="material-symbols-outlined text-3xl text-secondary">check_circle</span>
            </span>
            <p className="max-w-[15rem] text-sm text-muted">{t("feedback.noFaults")}</p>
          </div>
        ) : (
          detections.map((d, i) => (
            <motion.div
              key={i}
              initial={reduce ? false : { opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: i * 0.05, ease: [0.16, 1, 0.3, 1] }}
            >
              <FaultCard
                d={d}
                retrieval={retrievalByFault.get(d.fault_id)}
                active={currentTime >= d.start_time && currentTime <= d.end_time}
                last={i === detections.length - 1}
                onSeek={onSeek}
              />
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
}
