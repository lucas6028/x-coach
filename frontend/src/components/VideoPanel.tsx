import { useEffect, useRef, useState } from "react";
import { api, type Analysis } from "../api";
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

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => undefined);
    else v.pause();
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-center">
      <div
        ref={wrapRef}
        className="relative h-[58vh] max-w-full bg-black rounded-lg border border-border-dark overflow-hidden"
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

        <div className="absolute bottom-0 inset-x-0 p-3 bg-gradient-to-t from-black/90 to-transparent z-20">
          <div className="flex items-center gap-3 mb-1">
            <button onClick={togglePlay} className="text-white hover:text-primary">
              <span className="material-symbols-outlined">
                {playing ? "pause" : "play_arrow"}
              </span>
            </button>
            <span className="text-white/80 font-mono text-[11px]">
              {analysis.detections.length} fault
              {analysis.detections.length === 1 ? "" : "s"} detected
            </span>
          </div>
        </div>
      </div>
      </div>

      <Timeline analysis={analysis} duration={duration} currentTime={time} onSeek={onSeek} />
    </div>
  );
}
