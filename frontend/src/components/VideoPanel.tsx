import { useEffect, useRef, useState } from "react";
import { api, type Analysis } from "../api";
import { CheckCircle, CornersOut, Pause, Play, Warning } from "@phosphor-icons/react";
import { fmtTime } from "../lib/format";
import { useI18n } from "../lib/i18n";
import MetricsCards from "./MetricsCards";
import SkeletonOverlay from "./SkeletonOverlay";
import Timeline from "./Timeline";

interface Props {
  analysis: Analysis;
  videoRef: React.RefObject<HTMLVideoElement>;
  onTimeUpdate: (t: number) => void;
  onActiveFault: (faultId: string | null) => void;
  onSeek: (t: number) => void;
}

export default function VideoPanel({
  analysis,
  videoRef,
  onTimeUpdate,
  onActiveFault,
  onSeek,
}: Props) {
  const { t } = useI18n();
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const wrapRef = useRef<HTMLDivElement>(null);

  const { width, height } = analysis.metadata;
  const aspect = width && height ? width / height : 9 / 16;
  const portrait = aspect < 1;

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onTime = () => {
      setTime(v.currentTime);
      onTimeUpdate(v.currentTime);
    };
    const onMeta = () => setDuration(v.duration || 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    v.addEventListener("timeupdate", onTime);
    v.addEventListener("loadedmetadata", onMeta);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    return () => {
      v.removeEventListener("timeupdate", onTime);
      v.removeEventListener("loadedmetadata", onMeta);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
    };
  }, [videoRef, onTimeUpdate, analysis.video_id]);

  // `timeupdate` only fires ~4x/sec, so drive the playhead from rAF while playing
  // for a smooth, frame-rate timeline (timeupdate still covers paused/seek updates).
  useEffect(() => {
    if (!playing) return;
    const v = videoRef.current;
    if (!v) return;
    let raf = 0;
    const tick = () => {
      setTime(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, videoRef]);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => undefined);
    else v.pause();
  };

  const toggleFullscreen = () => {
    const el = wrapRef.current;
    if (!el) return;
    if (document.fullscreenElement) document.exitFullscreen().catch(() => undefined);
    else el.requestFullscreen?.().catch(() => undefined);
  };

  const faultCount = analysis.detections.length;

  return (
    <div className="flex flex-col gap-3 lg:h-full lg:min-h-0">
      {/* On desktop the video grows to fill the column (no dead space below it):
          portrait fills height, landscape fills width — each keeps its aspect so
          neither letterboxes. Mobile keeps a fixed, scrollable height. */}
      <div className="flex items-center justify-center lg:flex-1 lg:min-h-0">
        <div
          ref={wrapRef}
          className={`group relative h-[58vh] max-w-full bg-black rounded-xl ring-1 ring-border-dark shadow-2xl shadow-black/60 overflow-hidden ${
            portrait ? "lg:h-full lg:w-auto" : "lg:h-auto lg:w-full lg:max-h-full"
          }`}
          style={{ aspectRatio: String(aspect) }}
        >
          <video
            ref={videoRef}
            src={api.videoFileUrl(analysis.video_id)}
            className="absolute inset-0 w-full h-full object-contain"
            playsInline
            onClick={togglePlay}
          />
          <SkeletonOverlay analysis={analysis} videoRef={videoRef} onActiveFault={onActiveFault} />

          {/* status badge. The metrics panel sits beside it: it folds to icon-only
              whenever the panel is alongside on a narrow row (mobile, or any
              portrait clip), and shows its full label only on a wide landscape
              row. The panel's faults cell carries the status when the label hides. */}
          <div
            className={`absolute top-3 left-3 z-20 flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium text-white backdrop-blur-md shadow-lg ${
              portrait ? "gap-0" : "gap-0 sm:gap-1.5"
            } ${faultCount > 0 ? "bg-danger/85" : "bg-secondary/85"}`}
          >
            {faultCount > 0 ? (
              <Warning size={14} weight="fill" />
            ) : (
              <CheckCircle size={14} weight="fill" />
            )}
            <span className={portrait ? "hidden" : "hidden sm:inline"}>
              {faultCount > 0
                ? faultCount === 1
                  ? t("video.faultOne")
                  : t("video.faultMany", { count: faultCount })
                : t("video.noFaults")}
            </span>
          </div>

          {/* biomechanics metrics HUD (top-right). pointer-events-none keeps the
              corner click-through to play/pause; the panel itself is read-only. */}
          <div className="absolute top-3 right-3 z-20 pointer-events-none">
            <MetricsCards analysis={analysis} portrait={portrait} />
          </div>

          {/* center play / pause overlay */}
          <button
            onClick={togglePlay}
            aria-label={playing ? t("a11y.pause") : t("a11y.play")}
            className={`absolute inset-0 z-10 flex items-center justify-center transition-opacity duration-200 ${
              playing ? "opacity-0 group-hover:opacity-100" : "opacity-100"
            }`}
          >
            <span className="flex items-center justify-center w-16 h-16 rounded-full bg-black/45 backdrop-blur-sm ring-1 ring-white/25 text-white transition-transform hover:scale-110">
              {playing ? <Pause size={34} weight="fill" /> : <Play size={34} weight="fill" />}
            </span>
          </button>

          {/* control bar */}
          <div
            className={`absolute bottom-0 inset-x-0 z-20 px-3 pb-3 pt-10 bg-gradient-to-t from-black/90 via-black/40 to-transparent transition-opacity duration-200 ${
              playing ? "opacity-0 group-hover:opacity-100" : "opacity-100"
            }`}
          >
            <div className="flex items-center gap-3">
              <button
                onClick={togglePlay}
                aria-label={playing ? t("a11y.pause") : t("a11y.play")}
                className="text-white hover:text-primary transition-colors"
              >
                {playing ? <Pause size={22} weight="fill" /> : <Play size={22} weight="fill" />}
              </button>
              <span className="font-mono text-[11px] text-white/80 tabular-nums">
                {fmtTime(time)} <span className="text-white/40">/</span> {fmtTime(duration || 0)}
              </span>
              <div className="flex-1" />
              <button
                onClick={toggleFullscreen}
                aria-label={t("a11y.fullscreen")}
                className="text-white/80 hover:text-primary transition-colors"
              >
                <CornersOut size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>

      <Timeline analysis={analysis} duration={duration} currentTime={time} onSeek={onSeek} />
    </div>
  );
}
