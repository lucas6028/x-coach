import { ArrowRight } from "@phosphor-icons/react";
import type { Detection, Retrieval } from "../api";
import { fmtTime } from "../lib/format";
import { keyEvidence, ragSnippet, summaryCategory } from "../lib/retrieval";
import { useI18n, faultLabel, phaseLabel, severityText } from "../lib/i18n";

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

// One detected fault rendered as the coach's grounded analysis card: timecode + fault + severity,
// the measured evidence, the KG-retrieved likely cause / injury risk, and the corrective cue
// (highest visual weight). Clicking seeks the video; `active` marks the fault the playhead is in.
// Lives on a temporal rail so a sequence of faults reads as one connected thought.
export default function FaultCard({
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
            <ArrowRight size={16} weight="bold" className="mt-px shrink-0 text-secondary" />
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
