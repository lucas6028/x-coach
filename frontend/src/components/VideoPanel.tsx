import { useEffect, useRef, useState } from "react";
import { api, type Analysis } from "../api";
import { fmtTime } from "../lib/format";
import { useI18n } from "../lib/i18n";
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
    <div className="flex flex-col gap-3">
      <div className="flex justify-center">
        <div
          ref={wrapRef}
          className="group relative h-[58vh] max-w-full bg-black rounded-xl ring-1 ring-border-dark shadow-2xl shadow-black/60 overflow-hidden"
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

          {/* status badge */}
          <div
            className={`absolute top-3 left-3 z-20 flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium text-white backdrop-blur-md shadow-lg ${
              faultCount > 0 ? "bg-danger/85" : "bg-secondary/85"
            }`}
          >
            <span className="material-symbols-outlined text-sm leading-none">
              {faultCount > 0 ? "warning" : "check_circle"}
            </span>
            {faultCount > 0
              ? faultCount === 1
                ? t("video.faultOne")
                : t("video.faultMany", { count: faultCount })
              : t("video.noFaults")}
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
              <span className="material-symbols-outlined text-4xl leading-none">
                {playing ? "pause" : "play_arrow"}
              </span>
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
                <span className="material-symbols-outlined">
                  {playing ? "pause" : "play_arrow"}
                </span>
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
                <span className="material-symbols-outlined text-xl">fullscreen</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <Timeline analysis={analysis} duration={duration} currentTime={time} onSeek={onSeek} />
    </div>
  );
}
