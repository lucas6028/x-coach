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
  high: "bg-[#ff6b6b]",
  moderate: "bg-[#e0a33a]",
  mild: "bg-[#b8bcd3]",
};
const SEV_CHIP: Record<SevLevel, string> = {
  high: "bg-[#fff5f5] text-[#e05252] border border-[#ffe0e0]",
  moderate: "bg-[#fff6d9] text-[#b8922e] border border-[#ffe9a8]",
  mild: "bg-[#f5f6fb] text-[#59648f] border border-[#f0f1f8]",
};

type Rung = { kind: "cause" | "risk" | "fix"; label: string; text: string };

// One detected fault rendered as the coach's grounded analysis card, read as a causal ladder:
// timecode + fault + severity up top, the measured evidence, then a connected chain of the
// KG-retrieved likely cause, the injury risk it leads to, and the corrective cue (the terminal
// rung, highest visual weight). Clicking seeks the video; `active` marks the fault the playhead
// is currently inside (carried by the primary ring).
export default function FaultCard({
  d,
  retrieval,
  active,
  onSeek,
}: {
  d: Detection;
  retrieval: Retrieval | undefined;
  active: boolean;
  onSeek: (t: number) => void;
}) {
  const { t } = useI18n();
  const causes = summaryCategory(retrieval, "causes");
  const risks = summaryCategory(retrieval, "risks");
  const corrections = summaryCategory(retrieval, "corrections");
  const snippet = ragSnippet(retrieval);
  const level = sevLevel(d.severity);
  const evidence = keyEvidence(d);

  // Cause leads to risk leads to fix. Only the rungs the KG actually supplied are shown; the fix
  // is always the terminal rung so the chain resolves on the action.
  const rungs: Rung[] = [];
  if (causes.length) rungs.push({ kind: "cause", label: t("feedback.cause"), text: causes.join(", ") });
  if (risks.length) rungs.push({ kind: "risk", label: t("feedback.risk"), text: risks.join(", ") });
  if (corrections.length) rungs.push({ kind: "fix", label: t("feedback.cue"), text: corrections.join(" · ") });

  return (
    <button
      onClick={() => onSeek(d.start_time)}
      // `glass-face` (fill + shadow only, no border) so the active state's violet ring and border
      // survive — the panel class sets `border`, which would overwrite the one signal that says
      // which fault the playhead is currently inside.
      className={`glass-face block w-full rounded-[18px] border p-4 text-left transition-colors ${
        active
          ? "border-primary/60 ring-2 ring-primary/25"
          : "border-white/80 hover:border-[#ddd8f5]"
      }`}
    >
      {/* header: time + phase, fault, severity */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <span className="font-mono text-[11px] text-faint">
            {fmtTime(d.start_time)} · {phaseLabel(t, d.phase)}
          </span>
          <p className="font-display font-semibold leading-tight text-content">
            {faultLabel(t, d.fault_name)}
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
        <span className="mt-2 inline-block rounded-md bg-content/5 px-2 py-1 font-mono text-[11px] text-muted">
          {evidence}
        </span>
      )}

      {/* the causal ladder: cause -> risk -> fix, connected as one chain */}
      {rungs.length > 0 && (
        <ol className="mt-3.5">
          {rungs.map((r, i) => {
            const notLast = i < rungs.length - 1;
            if (r.kind === "fix") {
              return (
                <li key={r.kind} className="relative flex gap-3">
                  <span className="relative mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-secondary/15">
                    <ArrowRight size={10} weight="bold" className="text-secondary" />
                  </span>
                  <span className="leading-snug">
                    <span className="block text-[10px] font-semibold uppercase tracking-wide text-secondary">
                      {r.label}
                    </span>
                    <span className="text-sm font-semibold text-content">{r.text}</span>
                  </span>
                </li>
              );
            }
            const dot = r.kind === "cause" ? "bg-primary/70" : "bg-danger/80";
            const textCls = r.kind === "cause" ? "text-content" : "font-medium text-danger";
            return (
              <li key={r.kind} className={`relative flex gap-3 ${notLast ? "pb-3" : ""}`}>
                {notLast && <span className="absolute left-[3px] top-3 h-full w-px bg-border-dark" />}
                <span className={`relative mt-1 h-2 w-2 shrink-0 rounded-full ${dot}`} />
                <span className="text-xs leading-relaxed">
                  <span className="block text-[10px] font-semibold uppercase tracking-wide text-faint">
                    {r.label}
                  </span>
                  <span className={textCls}>{r.text}</span>
                </span>
              </li>
            );
          })}
        </ol>
      )}

      {snippet && !corrections.length && (
        <p className="mt-3 border-l-2 border-border-dark pl-3 text-[11px] italic leading-relaxed text-muted">
          {snippet}
        </p>
      )}
    </button>
  );
}
