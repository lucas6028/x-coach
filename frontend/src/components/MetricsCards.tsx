import type { Analysis } from "../api";
import { useI18n, viewLabel } from "../lib/i18n";

interface Props {
  analysis: Analysis;
}

function Card({
  label,
  value,
  sub,
  danger,
}: {
  label: string;
  value: string;
  sub?: string;
  danger?: boolean;
}) {
  return (
    <div
      className={`p-3 bg-surface-dark rounded border flex flex-col gap-1 ${
        danger ? "border-danger/40" : "border-border-dark"
      }`}
    >
      <span className="text-[10px] text-muted uppercase tracking-wider font-semibold">
        {label}
      </span>
      <span className={`text-xl font-bold font-mono ${danger ? "text-danger" : "text-content"}`}>
        {value}
      </span>
      {sub && <p className="text-[10px] text-muted">{sub}</p>}
    </div>
  );
}

export default function MetricsCards({ analysis }: Props) {
  const { t } = useI18n();
  const q = analysis.quality;
  const faults = analysis.detections;
  const topSeverity = faults.reduce((m, d) => Math.max(m, d.severity), 0);
  const visibility = (q.lower_body_visibility_mean ?? 0) * 100;
  const validRatio = (q.valid_frame_ratio ?? 0) * 100;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <Card
        label={t("metric.cameraView")}
        value={viewLabel(t, analysis.view.view_type)}
        sub={t("metric.conf", { v: (analysis.view.view_confidence ?? 0).toFixed(2) })}
      />
      <Card
        label={t("metric.faults")}
        value={String(faults.length)}
        sub={
          faults.length
            ? t("metric.peakSeverity", { v: topSeverity.toFixed(2) })
            : t("metric.cleanRep")
        }
        danger={faults.length > 0}
      />
      <Card
        label={t("metric.lowerBodyVis")}
        value={`${visibility.toFixed(0)}%`}
        sub={t("metric.landmarkConf")}
      />
      <Card
        label={t("metric.validFrames")}
        value={`${validRatio.toFixed(0)}%`}
        sub={t("metric.framesRatio", { valid: q.valid_frames ?? 0, total: q.total_frames ?? 0 })}
      />
    </div>
  );
}
