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
    <div className="px-1">
      <div
        className="relative h-6 w-full flex items-center cursor-pointer group"
        onClick={handleClick}
      >
        <div className="absolute inset-x-0 h-1.5 bg-gray-700 rounded-full overflow-hidden" />
        {/* progress */}
        <div
          className="absolute h-1.5 bg-primary/70 rounded-full"
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
            className="absolute h-1.5 bg-danger rounded-sm z-10 hover:h-2.5 transition-all"
            style={{
              left: pct(d.start_time),
              width: `${Math.max(1.5, ((d.end_time - d.start_time) / dur) * 100)}%`,
              opacity: 0.45 + 0.55 * Math.min(1, d.severity),
              boxShadow: "0 0 6px #ef4444",
            }}
          />
        ))}
        {/* playhead */}
        <div
          className="absolute h-4 w-0.5 bg-white rounded-full z-20"
          style={{ left: pct(currentTime) }}
        />
      </div>
      <div className="flex items-center gap-4 mt-1 text-[10px] text-gray-500">
        <span className="font-mono">{fmtTime(currentTime)} / {fmtTime(dur)}</span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm bg-danger inline-block" /> Fault
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm bg-gray-600 inline-block" /> Neutral
        </span>
      </div>
    </div>
  );
}
