import type { Analysis } from "../api";
import { fmtTime } from "../lib/format";
import { useI18n, faultLabel } from "../lib/i18n";

interface Props {
  analysis: Analysis;
  duration: number;
  currentTime: number;
  onSeek: (t: number) => void;
}

// Error/neutral segment bar + playhead. Errors are red (opacity scaled by severity).
export default function Timeline({ analysis, duration, currentTime, onSeek }: Props) {
  const { t } = useI18n();
  const dur = duration || analysis.metadata.total_frames / (analysis.metadata.fps || 30) || 1;
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / dur) * 100))}%`;

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    onSeek(ratio * dur);
  };

  return (
    <div className="px-1 select-none">
      <div
        className="relative h-6 w-full flex items-center cursor-pointer group"
        onClick={handleClick}
      >
        {/* track */}
        <div className="absolute inset-x-0 h-2 bg-track/50 rounded-full ring-1 ring-inset ring-white/5" />
        {/* progress */}
        <div
          className="absolute h-2 rounded-full bg-gradient-to-r from-primary to-cyan-400 shadow-[0_0_8px_theme(colors.primary)]"
          style={{ width: pct(currentTime) }}
        />
        {/* Spans that carry no pose data because RS-SP2 never extracted them. NEUTRAL, never a
            warning colour: they are not a problem, they are unexamined — and an empty timeline
            must not read as "these reps were fine". */}
        {(analysis.reps?.segments ?? [])
          .filter((s) => !s.analyzed)
          .map((s) => (
            <div
              key={`un-${s.index}`}
              data-testid="unanalyzed-span"
              title={t("timeline.unanalyzed")}
              className="absolute h-2 rounded-full bg-track opacity-70
                         [background-image:repeating-linear-gradient(45deg,transparent,transparent_3px,rgba(255,255,255,0.12)_3px,rgba(255,255,255,0.12)_6px)]"
              style={{
                left: pct(s.start_time),
                width: `${Math.max(1.5, ((s.end_time - s.start_time) / dur) * 100)}%`,
              }}
            />
          ))}
        {/* fault segments */}
        {analysis.detections.map((d, i) => (
          <div
            key={i}
            title={`${faultLabel(t, d.fault_name)} (${fmtTime(d.start_time)}–${fmtTime(d.end_time)})`}
            onClick={(e) => {
              e.stopPropagation();
              onSeek(d.start_time);
            }}
            className="absolute h-2 bg-danger rounded-full z-10 hover:h-3 transition-all"
            style={{
              left: pct(d.start_time),
              width: `${Math.max(1.5, ((d.end_time - d.start_time) / dur) * 100)}%`,
              opacity: 0.5 + 0.5 * Math.min(1, d.severity),
              boxShadow: "0 0 8px rgba(239,68,68,0.8)",
            }}
          />
        ))}
        {/* playhead */}
        <div
          className="absolute z-20 -translate-x-1/2 w-3.5 h-3.5 rounded-full bg-white ring-2 ring-primary shadow-[0_0_8px_rgba(255,255,255,0.7)] transition-transform group-hover:scale-110"
          style={{ left: pct(currentTime) }}
        />
      </div>
      <div className="flex items-center gap-4 mt-1.5 text-[10px] text-muted">
        <span className="font-mono tabular-nums text-content">
          {fmtTime(currentTime)} <span className="text-faint">/</span> {fmtTime(dur)}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-danger inline-block shadow-[0_0_5px_rgba(239,68,68,0.7)]" />
          {t("timeline.fault")}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-track inline-block" />
          {t("timeline.neutral")}
        </span>
        {analysis.reps && (
          <span className="text-muted">
            {analysis.reps.fallback
              ? t("timeline.wholeClip")
              : t("timeline.repsSummary", {
                  detected: analysis.reps.detected,
                  // Separator is translated data (see `timeline.repsListSeparator`), not a
                  // hardcoded character here — a full-width "、" reads wrong inside English text.
                  list: analysis.reps.analyzed.join(t("timeline.repsListSeparator")),
                })}
          </span>
        )}
      </div>
    </div>
  );
}
