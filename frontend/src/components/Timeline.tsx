import type { Analysis } from "../api";
import { fmtTime } from "../lib/format";

interface Props {
  analysis: Analysis;
  duration: number;
  currentTime: number;
  onSeek: (t: number) => void;
}

// Error/neutral segment bar + playhead. Errors are red (opacity scaled by severity).
export default function Timeline({ analysis, duration, currentTime, onSeek }: Props) {
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
        <div className="absolute inset-x-0 h-2 bg-gray-700/50 rounded-full ring-1 ring-inset ring-white/5" />
        {/* progress */}
        <div
          className="absolute h-2 rounded-full bg-gradient-to-r from-primary to-cyan-400 shadow-[0_0_8px_theme(colors.primary)]"
          style={{ width: pct(currentTime) }}
        />
        {/* fault segments */}
        {analysis.detections.map((d, i) => (
          <div
            key={i}
            title={`${d.fault_name} (${fmtTime(d.start_time)}–${fmtTime(d.end_time)})`}
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
      <div className="flex items-center gap-4 mt-1.5 text-[10px] text-gray-400">
        <span className="font-mono tabular-nums text-gray-300">
          {fmtTime(currentTime)} <span className="text-gray-600">/</span> {fmtTime(dur)}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-danger inline-block shadow-[0_0_5px_rgba(239,68,68,0.7)]" />
          Fault
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-gray-600 inline-block" />
          Neutral
        </span>
      </div>
    </div>
  );
}
