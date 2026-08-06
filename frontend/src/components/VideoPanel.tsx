import { useEffect, useRef, useState } from "react";
import { api, type Analysis } from "../api";
import { CheckCircle, CornersOut, Pause, Play, Warning } from "@phosphor-icons/react";
import { useI18n } from "../lib/i18n";
import SkeletonOverlay from "./SkeletonOverlay";
import Timeline from "./Timeline";
import DetectedErrorsCard from "./studio/DetectedErrorsCard";
import FormScoreCard from "./studio/FormScoreCard";

interface Props {
  analysis: Analysis;
  videoRef: React.RefObject<HTMLVideoElement>;
  onTimeUpdate: (t: number) => void;
  onActiveFault: (faultId: string | null) => void;
  onSeek: (t: number) => void;
  /** Which fault the playhead is inside, so the overlaid error list can mark it. */
  activeFaultId?: string | null;
}

/**
 * Where this analysis's video actually lives.
 *
 * Three sources, resolved in order of what is already known:
 *  - a library demo clip is a public file the backend streams directly;
 *  - a fresh upload's presigned URL rides along on the analyze response;
 *  - a history replay has neither, because storing a presigned URL in the row would mean
 *    replaying an expired one — so it re-signs through the ownership-checked endpoint.
 *
 * `null` while resolving and after a failure: the panel renders the analysis without playback
 * rather than blocking the page on storage.
 */
function useVideoSrc(analysis: Analysis): string | null {
  const { source, video_id: videoId, video_url: videoUrl } = analysis;
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    if (source === "library") {
      setSrc(api.videoFileUrl(videoId));
      return;
    }
    if (videoUrl) {
      setSrc(videoUrl);
      return;
    }
    let cancelled = false;
    setSrc(null);
    api
      .uploadMedia(videoId)
      .then((media) => {
        if (!cancelled) setSrc(media.video_url);
      })
      .catch(() => {
        if (!cancelled) setSrc(null);
      });
    return () => {
      cancelled = true;
    };
  }, [source, videoId, videoUrl]);

  return src;
}

// The reference design's video card: a rounded frame holding the clip and its skeleton overlay, a
// status badge and expand control in the top corners, the floating analysis cards down the right
// edge, and a frosted control pill along the bottom.
export default function VideoPanel({
  analysis,
  videoRef,
  onTimeUpdate,
  onActiveFault,
  onSeek,
  activeFaultId = null,
}: Props) {
  const { t } = useI18n();
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const wrapRef = useRef<HTMLDivElement>(null);

  const videoSrc = useVideoSrc(analysis);

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
    // `shrink-0`: the card is a flex item in a bounded, scrolling column, so without it the
    // aspect-ratio stage gets squeezed flat on a short viewport instead of scrolling.
    <div
      ref={wrapRef}
      className="group relative shrink-0 overflow-hidden rounded-[24px] border border-[#e6e8f0] bg-[#d6dbe3] shadow-[0_4px_24px_rgba(0,0,0,0.06)]"
    >
      {/* One fixed stage for every clip: the reference's 1.72:1 landscape frame, regardless of
          what the upload's own aspect is. A portrait phone clip pillarboxes inside it rather than
          growing the card, so the page does not reflow between a phone recording and a landscape
          one, and the floating cards keep the same position. The <video> is object-contain and
          SkeletonOverlay maps the landmarks onto the RENDERED video rect (not the canvas box), so
          the skeleton still tracks the body inside the letterbox. */}
      <div className="relative overflow-hidden bg-black" style={{ aspectRatio: "1.72 / 1" }}>
        <video
          ref={videoRef}
          // Omitted entirely while unresolved: an empty `src` makes the browser re-request the
          // page URL as media and log a decode error.
          {...(videoSrc ? { src: videoSrc } : {})}
          className="absolute inset-0 h-full w-full object-contain"
          playsInline
          onClick={togglePlay}
        />
        <SkeletonOverlay analysis={analysis} videoRef={videoRef} onActiveFault={onActiveFault} />

        {/* Top-left status pill. Everything overlaid on the clip uses the theme's glass vocabulary
            at high opacity and WITHOUT backdrop-blur: blurring over a surface that is decoding
            video and repainting a skeleton canvas every frame is exactly the frame-drop the
            theme's own blur discipline exists to avoid (see index.css). */}
        <div className="glass-over-video absolute left-4 top-4 z-20 flex items-center gap-2 rounded-full px-3 py-1.5">
          {faultCount > 0 ? (
            <Warning size={13} weight="fill" className="text-[#ff5a5a]" />
          ) : (
            <CheckCircle size={13} weight="fill" className="text-[#22c55e]" />
          )}
          <span className="text-[12px] font-semibold text-[#1e2142]">
            {faultCount > 0
              ? faultCount === 1
                ? t("video.faultOne")
                : t("video.faultMany", { count: faultCount })
              : t("video.noFaults")}
          </span>
        </div>

        {/* Top-right expand. */}
        <button
          onClick={toggleFullscreen}
          aria-label={t("a11y.fullscreen")}
          className="absolute right-4 top-4 z-20 flex h-9 w-9 items-center justify-center rounded-full border border-white/20 bg-[#3a3d4d]/85 text-white shadow-[0_10px_24px_rgba(20,24,60,0.28)] transition-colors hover:bg-[#2a2d3d]/90"
        >
          <CornersOut size={16} />
        </button>

        {/* The floating analysis stack down the right edge. Bounded between the status pill and
            the control bar and allowed to scroll inside that band: the reference mock has a tall
            stage, but a real clip's card can be short enough to clip the second card off the
            bottom. Hidden on the narrowest screens, where it would cover the athlete — the same
            information lives in the coach column there. */}
        <div className="scrollbar-none absolute bottom-[60px] right-3 top-[60px] z-20 hidden w-[172px] flex-col gap-3 overflow-y-auto sm:flex sm:right-4 lg:w-[180px]">
          <DetectedErrorsCard
            detections={analysis.detections}
            onSeek={onSeek}
            activeFaultId={activeFaultId}
          />
          <FormScoreCard analysis={analysis} />
        </div>

        {/* Centre play / pause. */}
        <button
          onClick={togglePlay}
          aria-label={playing ? t("a11y.pause") : t("a11y.play")}
          className={`absolute inset-0 z-10 flex items-center justify-center transition-opacity duration-200 ${
            playing ? "opacity-0 group-hover:opacity-100" : "opacity-100"
          }`}
        >
          <span className="flex h-16 w-16 items-center justify-center rounded-full bg-black/50 text-white ring-1 ring-white/25 transition-transform hover:scale-110">
            {playing ? <Pause size={32} weight="fill" /> : <Play size={32} weight="fill" />}
          </span>
        </button>

        {/* Bottom control pill. */}
        {/* Dark glass: translucent enough that the clip reads through it, which is legible here
            because everything on this bar is white-on-dark — the inverse of the light cards on
            the right, which have to stay opaque. Still no blur; this is the hot path. */}
        <div className="absolute inset-x-3 bottom-3 z-20 flex items-center gap-3 rounded-full border border-white/15 bg-[#373a4a]/85 px-3 py-2 shadow-[inset_0_1px_0_rgba(255,255,255,0.14),0_10px_28px_rgba(0,0,0,0.32)]">
          <button
            onClick={togglePlay}
            aria-label={playing ? t("a11y.pause") : t("a11y.play")}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white/15 text-white transition-colors hover:bg-white/25"
          >
            {playing ? <Pause size={14} weight="fill" /> : <Play size={14} weight="fill" />}
          </button>
          <Timeline analysis={analysis} duration={duration} currentTime={time} onSeek={onSeek} />
        </div>
      </div>
    </div>
  );
}
