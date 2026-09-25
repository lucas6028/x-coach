import { ChartBar } from "@phosphor-icons/react";
import type { Analysis } from "../../api";
import { faultLabel, useI18n, viewLabel, type TFunc } from "../../lib/i18n";
import { validFrameStat } from "../../lib/quality";
import StudioCard from "./StudioCard";

interface Cell {
  label: string;
  value: string;
  sub: string;
  /** 0–1; drives the bar width. */
  fill: number;
  tone: "bad" | "good";
}

// The reference's "Key Metrics" panel shows three biomechanics figures with their optimal band.
// The real equivalent is the detector's own primary evidence: each detection names the metric it
// measured (`primary_label`), the value it read and the threshold it breached. Those cells come
// first; any remaining slots are filled with clip-quality figures, so the panel is always three
// rows long and every number in it was actually measured.
function cellsFor(analysis: Analysis, t: TFunc): Cell[] {
  const cells: Cell[] = [];

  for (const d of analysis.detections) {
    if (cells.length === 3) break;
    const ev = d.evidence || {};
    const label = ev.primary_label;
    const value = ev.primary_value;
    if (typeof label !== "string" || typeof value !== "number") continue;
    const threshold = ev.primary_threshold;
    cells.push({
      label,
      value: value.toFixed(2),
      sub:
        typeof threshold === "number"
          ? t("studio.metricLimit", {
              op: value < threshold ? "≥" : "≤",
              v: String(threshold),
            })
          : faultLabel(t, d.fault_name),
      // Distance to the breached threshold, so a marginal miss reads differently from a big one.
      fill: typeof threshold === "number" && threshold !== 0 ? Math.min(1, value / threshold) : 0.5,
      tone: "bad",
    });
  }

  const q = analysis.quality;
  const frames = validFrameStat(q);
  const quality: Cell[] = [
    // Denominator: the frames that were EXTRACTED, when the payload says so — see
    // `validFrameStat`. Value, bar and tone all read the SAME ratio; feeding the bar
    // `valid_frame_ratio` while the number came from the extracted denominator would print "96%"
    // over a 29%-wide red bar.
    {
      label: t("metric.validFrames"),
      value: `${(frames.ratio * 100).toFixed(0)}%`,
      sub: frames.extracted > 0
        ? t("metric.framesRatioExtracted", { valid: frames.valid, extracted: frames.extracted })
        : t("metric.framesRatio", { valid: frames.valid, total: q.total_frames ?? 0 }),
      fill: frames.ratio,
      tone: frames.ratio >= 0.8 ? "good" : "bad",
    },
    {
      label: t("metric.lowerBodyVis"),
      value: `${((q.lower_body_visibility_mean ?? 0) * 100).toFixed(0)}%`,
      sub: t("metric.landmarkConf"),
      fill: q.lower_body_visibility_mean ?? 0,
      tone: (q.lower_body_visibility_mean ?? 0) >= 0.6 ? "good" : "bad",
    },
    {
      label: t("metric.cameraView"),
      value: viewLabel(t, analysis.view.view_type),
      sub: t("metric.conf", { v: (analysis.view.view_confidence ?? 0).toFixed(2) }),
      fill: analysis.view.view_confidence ?? 0,
      tone: (analysis.view.view_confidence ?? 0) >= 0.6 ? "good" : "bad",
    },
  ];

  for (const c of quality) {
    if (cells.length === 3) break;
    cells.push(c);
  }
  return cells;
}

export default function KeyMetricsCard({ analysis }: { analysis: Analysis }) {
  const { t } = useI18n();
  const cells = cellsFor(analysis, t);

  return (
    <StudioCard icon={<ChartBar size={14} weight="bold" />} title={t("studio.keyMetrics")} index={1}>
      <ul className="flex flex-col gap-2">
        {cells.map((m, i) => (
          <li key={i} className="glass-control rounded-xl px-3 py-2.5">
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <div className="min-w-0">
                <div className="truncate text-[10px] font-semibold uppercase leading-tight tracking-wide text-[#63709f]">
                  {m.label}
                </div>
                <div className="truncate text-[10px] font-medium leading-tight text-[#63709f]">{m.sub}</div>
              </div>
              <div className="shrink-0 text-[15px] font-extrabold text-[#1e2142]">{m.value}</div>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-[#f0f1f8]">
              <div
                className={`h-full rounded-full ${m.tone === "bad" ? "bg-[#ff6b6b]" : "bg-[#22c55e]"}`}
                style={{ width: `${Math.round(Math.min(1, Math.max(0, m.fill)) * 100)}%` }}
              />
            </div>
          </li>
        ))}
      </ul>
    </StudioCard>
  );
}
