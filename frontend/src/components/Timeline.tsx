import type { Analysis } from "../api";
import { fmtTime } from "../lib/format";
import { useI18n, faultLabel } from "../lib/i18n";

interface Props {
  analysis: Analysis;
  duration: number;
  currentTime: number;
  onSeek: (t: number) => void;
}

// The scrub bar inside the video card's control pill (reference design): a violet progress fill
// on a translucent track, the detected faults marked in red along it, and a white knob. Clicking
// the track seeks; clicking a fault marker jumps to that fault.
//
// The legend that used to sit under a standalone timeline strip is gone with the strip itself —
// the card now carries ONE scrub bar, and a legend inside a 6px control pill would not fit. The
// markers keep their per-fault `title`, which is what actually named them.
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
    <div className="flex min-w-0 flex-1 select-none items-center gap-3">
      <span className="shrink-0 text-[11px] font-medium tabular-nums text-white/90">
        {fmtTime(currentTime)} <span className="text-white/45">/</span> {fmtTime(dur)}
      </span>
      <div
        className="group relative mx-1 h-4 flex-1 cursor-pointer"
        onClick={handleClick}
        title={t("timeline.neutral")}
      >
        <div className="absolute inset-x-0 top-1/2 h-[6px] -translate-y-1/2 overflow-hidden rounded-full bg-white/30" />
        <div
          className="absolute top-1/2 h-[6px] -translate-y-1/2 rounded-full bg-[#8b7bff]"
          style={{ width: pct(currentTime) }}
        />
        {analysis.detections.map((d, i) => (
          <div
            key={i}
            title={`${t("timeline.fault")}: ${faultLabel(t, d.fault_name)} (${fmtTime(d.start_time)}–${fmtTime(d.end_time)})`}
            onClick={(e) => {
              e.stopPropagation();
              onSeek(d.start_time);
            }}
            className="absolute top-1/2 z-10 h-[6px] -translate-y-1/2 rounded-full bg-[#ff6b6b] transition-all hover:h-[10px]"
            style={{
              left: pct(d.start_time),
              width: `${Math.max(1.5, ((d.end_time - d.start_time) / dur) * 100)}%`,
              opacity: 0.6 + 0.4 * Math.min(1, d.severity),
            }}
          />
        ))}
        <div
          className="absolute top-1/2 z-20 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-[#8b7bff] bg-white shadow transition-transform group-hover:scale-110"
          style={{ left: pct(currentTime) }}
        />
      </div>
    </div>
  );
}
