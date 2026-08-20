import { useRef } from "react";
import { CheckCircle, CornersOut, Pause, Play, Warning } from "@phosphor-icons/react";
import type { Analysis } from "../../api";
import { fmtTime } from "../../lib/format";
import { faultLabel, useI18n } from "../../lib/i18n";
import { keyEvidence } from "../../lib/retrieval";
import { useVideoPlayback } from "../../lib/useVideoPlayback";
import SkeletonOverlay from "../SkeletonOverlay";
import FaultChips from "./FaultChips";

interface Props {
  analysis: Analysis;
  videoRef: React.RefObject<HTMLVideoElement>;
  videoSrc: string | null;
  onTimeUpdate: (t: number) => void;
  onActiveFault: (faultId: string | null) => void;
  onSeek: (t: number) => void;
}

/**
 * The phone stage from the mock: the clip with its skeleton, a status pill and expand control in
 * the top corners, a floating metrics card down the right, labelled fault callouts pinned to the
 * body, and a control bar overlaid along the bottom.
 *
 * The mock's metrics card lists Reps / Tempo / Knee Angle. Tempo has no counterpart anywhere in
 * this pipeline and is dropped rather than faked; the third row is the detector's own primary
 * evidence, which is what "Knee Angle 82°" actually is.
 */
export default function MobileVideoCard({
  analysis,
  videoRef,
  videoSrc,
  onTimeUpdate,
  onActiveFault,
  onSeek,
}: Props) {
  const { t } = useI18n();
  const wrapRef = useRef<HTMLDivElement>(null);
  const { playing, time, duration, togglePlay, toggleFullscreen } = useVideoPlayback(
    videoRef,
    analysis.video_id,
    onTimeUpdate,
    wrapRef
  );

  const faultCount = analysis.detections.length;
  const dur = duration || analysis.metadata.total_frames / (analysis.metadata.fps || 30) || 1;
  const pct = (v: number) => `${Math.min(100, Math.max(0, (v / dur) * 100))}%`;

  // Rep count comes from the per-rep detector when the analysis carries it. There is no "target"
  // anywhere in this product, so the mock's "8 / 12 · Target 12" is a plain count here.
  const reps = analysis.detections.find((d) => typeof d.rep_count === "number")?.rep_count;
  const primary = analysis.detections.find((d) => keyEvidence(d));

  const scrub = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(((e.clientX - r.left) / r.width) * dur);
  };

  return (
    <div
      ref={wrapRef}
      className="relative overflow-hidden rounded-[22px] bg-black shadow-[0_10px_30px_rgba(105,112,175,0.18)]"
    >
      <div className="relative" style={{ aspectRatio: "1.32 / 1" }}>
        <video
          ref={videoRef}
          {...(videoSrc ? { src: videoSrc } : {})}
          className="absolute inset-0 h-full w-full object-contain"
          playsInline
          onClick={togglePlay}
        />
        <SkeletonOverlay analysis={analysis} videoRef={videoRef} onActiveFault={onActiveFault} />
        <FaultChips analysis={analysis} videoRef={videoRef} time={time} />

        {/* Status pill. The mock says "Live Analysis"; this pipeline is offline, so the pill
            carries the verdict it actually has. */}
        <div className="glass-over-video absolute left-3 top-3 z-20 flex items-center gap-2 rounded-full px-3 py-1.5">
          {faultCount > 0 ? (
            <Warning size={12} weight="fill" className="text-[#ff5a5a]" />
          ) : (
            <CheckCircle size={12} weight="fill" className="text-[#22c55e]" />
          )}
          <span className="text-[11px] font-semibold text-[#1e2142]">
            {faultCount > 0
              ? faultCount === 1
                ? t("video.faultOne")
                : t("video.faultMany", { count: faultCount })
              : t("video.noFaults")}
          </span>
        </div>

        <button
          onClick={toggleFullscreen}
          aria-label={t("a11y.fullscreen")}
          className="absolute right-3 top-3 z-20 flex h-8 w-8 items-center justify-center rounded-full bg-black/40 text-white transition-colors active:bg-black/60"
        >
          <CornersOut size={15} />
        </button>

        {/* Floating metrics card. Only rows backed by real data render, so a clip with neither a
            rep count nor primary evidence drops the card rather than showing empty rows. */}
        {(reps !== undefined || primary) && (
          <div className="glass-over-video absolute right-3 top-14 z-20 w-[132px] divide-y divide-[#ebeaf6] rounded-[14px] px-3">
            {reps !== undefined && (
              <div className="py-2.5">
                <div className="text-[10px] font-medium text-[#65719f]">{t("mobile.reps")}</div>
                <div className="text-[15px] font-extrabold text-[#1e2142]">{reps}</div>
              </div>
            )}
            {primary && (
              <div className="py-2.5">
                <div className="truncate text-[10px] font-medium text-[#65719f]">
                  {String(primary.evidence.primary_label)}
                </div>
                <div className="text-[15px] font-extrabold text-[#1e2142]">
                  {Number(primary.evidence.primary_value).toFixed(2)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Control bar. The mock's 1x speed pill has no playback-rate control behind it here and
            is left out; the fault markers from the desktop scrubber are kept, since on a phone
            they are the only way to reach a fault in the clip. */}
        <div className="absolute inset-x-0 bottom-0 z-20 flex items-center gap-2.5 bg-gradient-to-t from-black/75 to-transparent px-3 pb-3 pt-8">
          <button
            onClick={togglePlay}
            aria-label={playing ? t("a11y.pause") : t("a11y.play")}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[#2b2b45]/85 text-white"
          >
            {playing ? <Pause size={16} weight="fill" /> : <Play size={16} weight="fill" />}
          </button>
          <span className="shrink-0 text-[11px] font-medium tabular-nums text-white/90">
            {fmtTime(time)}
          </span>
          <div className="group relative h-4 flex-1 cursor-pointer" onClick={scrub}>
            <div className="absolute inset-x-0 top-1/2 h-[5px] -translate-y-1/2 rounded-full bg-white/30" />
            <div
              className="absolute top-1/2 h-[5px] -translate-y-1/2 rounded-full bg-[#8b7bff]"
              style={{ width: pct(time) }}
            />
            {/* Spans RS-SP2 never extracted, hatched exactly as the desktop scrubber hatches them
                (Timeline.tsx carries the full rationale). The phone shows its own scrub bar rather
                than mounting Timeline, so without this the mobile viewer would get a verdict
                computed on 3 of 5 reps with nothing on screen saying so. */}
            {(analysis.reps?.segments ?? [])
              .filter((seg) => !seg.analyzed)
              .map((seg) => (
                <div
                  key={`un-${seg.index}`}
                  data-testid="unanalyzed-span"
                  title={t("timeline.unanalyzed")}
                  className="absolute top-1/2 h-[5px] -translate-y-1/2 rounded-full bg-white/25"
                  style={{
                    left: pct(seg.start_time),
                    width: `${Math.max(2, ((seg.end_time - seg.start_time) / dur) * 100)}%`,
                    backgroundImage:
                      "repeating-linear-gradient(45deg, transparent, transparent 3px, " +
                      "var(--c-hatch) 3px, var(--c-hatch) 6px)",
                  }}
                />
              ))}
            {analysis.detections.map((d, i) => (
              <div
                key={i}
                title={faultLabel(t, d.fault_name)}
                onClick={(e) => {
                  e.stopPropagation();
                  onSeek(d.start_time);
                }}
                className="absolute top-1/2 z-10 h-[5px] -translate-y-1/2 rounded-full bg-[#ff6b6b]"
                style={{
                  left: pct(d.start_time),
                  width: `${Math.max(2, ((d.end_time - d.start_time) / dur) * 100)}%`,
                }}
              />
            ))}
            <div
              className="absolute top-1/2 z-20 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow"
              style={{ left: pct(time) }}
            />
          </div>
          <span className="shrink-0 text-[11px] font-medium tabular-nums text-white/60">
            {fmtTime(dur)}
          </span>
        </div>
        {/* How much of the clip the verdict covers. Its own line under the control bar, because
            the phone bar has no room left beside the scrubber — but it is not optional: without
            it a three-rep verdict on a five-rep clip reads as a verdict on all five. */}
        {analysis.reps && (
          <div className="absolute inset-x-0 bottom-0 z-20 px-3 pb-1 text-center text-[10px] font-medium tabular-nums text-white/70">
            {analysis.reps.fallback
              ? t("timeline.wholeClip")
              : t("timeline.repsSummary", {
                  detected: analysis.reps.detected,
                  list: analysis.reps.analyzed.join(t("timeline.repsListSeparator")),
                })}
          </div>
        )}
      </div>
    </div>
  );
}
