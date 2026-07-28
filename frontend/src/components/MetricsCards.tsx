import { motion, useReducedMotion } from "motion/react";
import type { Analysis } from "../api";
import { useI18n, viewLabel } from "../lib/i18n";
import { wasMeasured } from "../lib/quality";

interface Props {
  analysis: Analysis;
  // Portrait clips are narrow, so the panel folds to 2x2 to stay clear of the
  // top-left status badge; landscape clips get a single horizontal row.
  portrait?: boolean;
}

// One cell of the over-video metrics HUD: label over a prominent mono value,
// with an optional supporting figure beneath.
function Stat({
  label,
  value,
  sub,
  tone = "default",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "default" | "danger" | "good";
}) {
  const valueColor =
    tone === "danger" ? "text-danger" : tone === "good" ? "text-secondary" : "text-white";
  return (
    <div className="bg-black/55 px-2.5 py-2">
      <div className="whitespace-nowrap text-[9px] font-semibold uppercase leading-none tracking-wider text-white/55">
        {label}
      </div>
      <div className={`mt-1.5 whitespace-nowrap font-mono text-[13px] font-bold leading-none tabular-nums ${valueColor}`}>
        {value}
      </div>
      {sub && (
        <div className="mt-1 whitespace-nowrap text-[9px] leading-none tabular-nums text-white/55">
          {sub}
        </div>
      )}
    </div>
  );
}

// Biomechanics metrics as a frosted-glass telemetry strip overlaid on the video
// (see VideoPanel). The 1px gaps over a translucent-white backing render hairline
// dividers cleanly in both the 1x4 row and the 2x2 fold. White-on-dark is
// intentional: the panel always sits over the black video, so it reads
// identically in both themes — matching the sibling status badge and controls.
export default function MetricsCards({ analysis, portrait = false }: Props) {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  const q = analysis.quality;
  const faults = analysis.detections;
  const topSeverity = faults.reduce((m, d) => Math.max(m, d.severity), 0);
  const visibility = (q.lower_body_visibility_mean ?? 0) * 100;
  const validRatio = (q.valid_frame_ratio ?? 0) * 100;
  const extracted = q.extracted_frames ?? 0;
  const hasFaults = faults.length > 0;
  // The detector returns an empty `detections` list for BOTH "no faults found" and "the clip was
  // never measurable", so `faults.length` alone printed a green "Faults 0 / clean rep" card
  // directly beside "Valid Frames 0%". `wasMeasured` is the SHARED criterion — CoachTray's
  // clean-rep banner reads the same helper, and the two co-render on one screen, so they must not
  // be able to disagree. See src/lib/quality.ts for the full rationale and the stated gap.
  const measured = wasMeasured(q);

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className={`grid grid-cols-[repeat(2,auto)] gap-px overflow-hidden rounded-xl bg-white/15 shadow-lg shadow-black/40 ring-1 ring-white/15 backdrop-blur-md ${
        portrait ? "" : "sm:grid-cols-[repeat(4,auto)]"
      }`}
    >
      <Stat
        label={t("metric.cameraView")}
        value={viewLabel(t, analysis.view.view_type)}
        sub={t("metric.conf", { v: (analysis.view.view_confidence ?? 0).toFixed(2) })}
      />
      <Stat
        label={t("metric.faults")}
        value={String(faults.length)}
        sub={
          hasFaults
            ? t("metric.peakSeverity", { v: topSeverity.toFixed(2) })
            : measured
              ? t("metric.cleanRep")
              : t("metric.notMeasured")
        }
        tone={hasFaults ? "danger" : measured ? "good" : "default"}
      />
      <Stat label={t("metric.lowerBodyVis")} value={`${visibility.toFixed(0)}%`} />
      {/* Denominator: the frames that were EXTRACTED, when the payload says. Under RS-SP2 only the
          scored reps carry landmarks, so the whole-clip ratio legitimately falls to ~30% and would
          read as bad tracking — a pipeline decision presented as a measurement problem. Analyses
          predating SP2 (and CLI output) have no extracted_frames and keep the old denominator. */}
      <Stat
        label={t("metric.validFrames")}
        value={`${(extracted > 0 ? (q.valid_frames ?? 0) / extracted * 100 : validRatio).toFixed(0)}%`}
        sub={extracted > 0
          ? t("metric.framesRatioExtracted", { valid: q.valid_frames ?? 0, extracted })
          : t("metric.framesRatio", { valid: q.valid_frames ?? 0, total: q.total_frames ?? 0 })}
      />
    </motion.div>
  );
}
