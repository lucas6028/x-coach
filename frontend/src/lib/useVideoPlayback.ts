import { useEffect, useState, type RefObject } from "react";

/**
 * Playback state for the analysis clip: play/pause, the playhead, the duration, and fullscreen.
 *
 * Extracted from VideoPanel so the desktop card and the phone card share one implementation. Both
 * render the same <video> with the same overlays; forking this would mean two rAF loops to keep in
 * step and two chances to drop the `timeupdate` teardown.
 */
export function useVideoPlayback(
  videoRef: RefObject<HTMLVideoElement>,
  /** Remounting key — the listeners rebind when the clip changes. */
  videoId: string,
  onTimeUpdate: (t: number) => void,
  /** The element to take fullscreen (the card, so the overlays go with it). */
  wrapRef: RefObject<HTMLElement>
) {
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onTime = () => {
      setTime(v.currentTime);
      onTimeUpdate(v.currentTime);
    };
    // A browser-recorded clip is a live-muxed WebM with no Duration element, so the element
    // reports `Infinity` (see lib/mediaDuration.ts). Infinity is truthy, so passing it on would
    // slip past the `duration || metadata` fallback the scrub bars use and freeze the playhead at
    // `t / Infinity` = 0%. Report 0 for any length that is not a real one and let them fall back.
    const onMeta = () =>
      setDuration(Number.isFinite(v.duration) && v.duration > 0 ? v.duration : 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    v.addEventListener("timeupdate", onTime);
    v.addEventListener("loadedmetadata", onMeta);
    // Chrome learns a live-muxed clip's true length once playback has reached the end; take it
    // when it arrives so the readout stops being the metadata's frames/fps estimate.
    v.addEventListener("durationchange", onMeta);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    return () => {
      v.removeEventListener("timeupdate", onTime);
      v.removeEventListener("loadedmetadata", onMeta);
      v.removeEventListener("durationchange", onMeta);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
    };
  }, [videoRef, onTimeUpdate, videoId]);

  // `timeupdate` only fires ~4x/sec, so drive the playhead from rAF while playing for a smooth,
  // frame-rate timeline (timeupdate still covers paused/seek updates). The fault chips ride this
  // same update rather than starting a loop of their own.
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

  return { playing, time, duration, togglePlay, toggleFullscreen };
}
