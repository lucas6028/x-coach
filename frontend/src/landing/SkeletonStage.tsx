import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { Play, Pause } from "@phosphor-icons/react";
import { POSE_CONNECTIONS } from "../lib/pose";

// Compact pose payload produced by scripts/landing/extract_demo_skeletons.py.
type PoseData = {
  fps: number;
  width: number;
  height: number;
  frames: ([number, number, number][] | null)[];
};

interface Props {
  clipId: string;
  src: string;
  poster: string;
  name: string;
  analyzingLabel: string;
  playLabel: string;
  pauseLabel: string;
  onHoverChange: (hovered: boolean) => void;
}

const VIS = 0.4;
const REVEAL = { duration: 2.2, delay: 0.7, ease: [0.16, 1, 0.3, 1] as const };

function Corner({ pos }: { pos: "tl" | "tr" | "bl" | "br" }) {
  const map = {
    tl: "left-3 top-3 border-l-2 border-t-2",
    tr: "right-3 top-3 border-r-2 border-t-2",
    bl: "left-3 bottom-3 border-l-2 border-b-2",
    br: "right-3 bottom-3 border-r-2 border-b-2",
  } as const;
  return <span aria-hidden className={`absolute h-5 w-5 border-[#3ee07a]/50 ${map[pos]}`} />;
}

// Real MediaPipe skeleton drawn on a canvas synced to the <video>, revealed left-to-right
// by a wipe so the clip transitions from original footage to the analyzed overlay.
export default function SkeletonStage({
  clipId,
  src,
  poster,
  name,
  analyzingLabel,
  playLabel,
  pauseLabel,
  onHoverChange,
}: Props) {
  const reduce = useReducedMotion();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [pose, setPose] = useState<PoseData | null>(null);
  const [playing, setPlaying] = useState(!reduce);

  // Load the per-clip landmark track.
  useEffect(() => {
    let on = true;
    setPose(null);
    fetch(`/demo/${clipId}.pose.json`)
      .then((r) => r.json())
      .then((d: PoseData) => on && setPose(d))
      .catch(() => undefined);
    return () => {
      on = false;
    };
  }, [clipId]);

  // Draw the skeleton each frame, mapped onto the letterboxed (object-contain) video rect.
  useEffect(() => {
    if (!pose) return;
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const fps = pose.fps || 25;
    const frames = pose.frames;
    let raf = 0;

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.round(rect.width * dpr);
      const h = Math.round(rect.height * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const idx = Math.min(frames.length - 1, Math.max(0, Math.round(video.currentTime * fps)));
      const lm = frames[idx];
      if (!lm) return;

      const vw = video.videoWidth || pose.width || 1;
      const vh = video.videoHeight || pose.height || 1;
      const videoAspect = vw / vh;
      const boxAspect = canvas.width / canvas.height;
      let rW: number, rH: number, oX: number, oY: number;
      if (boxAspect > videoAspect) {
        rH = canvas.height;
        rW = rH * videoAspect;
        oX = (canvas.width - rW) / 2;
        oY = 0;
      } else {
        rW = canvas.width;
        rH = rW / videoAspect;
        oX = 0;
        oY = (canvas.height - rH) / 2;
      }
      const px = (x: number) => oX + x * rW;
      const py = (y: number) => oY + y * rH;

      ctx.lineCap = "round";
      for (const [a, b] of POSE_CONNECTIONS) {
        const pa = lm[a];
        const pb = lm[b];
        if (!pa || !pb || pa[2] < VIS || pb[2] < VIS) continue;
        ctx.beginPath();
        ctx.moveTo(px(pa[0]), py(pa[1]));
        ctx.lineTo(px(pb[0]), py(pb[1]));
        ctx.strokeStyle = "#3ee07a";
        ctx.lineWidth = 3 * dpr;
        ctx.shadowColor = "#16b8a8";
        ctx.shadowBlur = 8 * dpr;
        ctx.stroke();
      }
      ctx.shadowBlur = 0;

      for (let i = 0; i < lm.length; i++) {
        const p = lm[i];
        if (!p || p[2] < VIS) continue;
        ctx.beginPath();
        ctx.arc(px(p[0]), py(p[1]), 3 * dpr, 0, Math.PI * 2);
        ctx.fillStyle = "#eafff0";
        ctx.fill();
      }
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [pose]);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => undefined);
    else v.pause();
  };

  return (
    <div
      onMouseEnter={() => onHoverChange(true)}
      onMouseLeave={() => onHoverChange(false)}
      className="group relative aspect-square w-full overflow-hidden rounded-2xl border border-white/10 bg-[#070b0a] shadow-[0_30px_80px_-30px_rgba(0,0,0,0.8)]"
    >
      {/* base: original footage */}
      <video
        ref={videoRef}
        src={src}
        poster={poster}
        autoPlay={!reduce}
        muted
        loop
        playsInline
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onClick={togglePlay}
        className="absolute inset-0 h-full w-full object-contain"
      />

      {/* analyzed layer, revealed by the wipe */}
      <motion.div
        className="absolute inset-0"
        initial={reduce ? false : { clipPath: "inset(0 100% 0 0)" }}
        animate={{ clipPath: "inset(0 0% 0 0)" }}
        transition={REVEAL}
        style={reduce ? { clipPath: "inset(0 0% 0 0)" } : undefined}
      >
        <div className="absolute inset-0 bg-[#06140c]/25" />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-50"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.05) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.05) 1px,transparent 1px)",
            backgroundSize: "44px 44px",
          }}
        />
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      </motion.div>

      {/* wipe divider */}
      {!reduce && (
        <motion.div
          aria-hidden
          className="absolute inset-y-0 z-10 w-px bg-gradient-to-b from-[#5ffb6f] via-[#16b8a8] to-[#5ffb6f] shadow-[0_0_18px_2px_rgba(62,224,122,0.55)]"
          initial={{ left: "0%", opacity: 0 }}
          animate={{ left: "100%", opacity: [0, 1, 1, 0] }}
          transition={{ ...REVEAL, opacity: { ...REVEAL, times: [0, 0.06, 0.88, 1] } }}
        />
      )}

      <Corner pos="tl" />
      <Corner pos="tr" />
      <Corner pos="bl" />
      <Corner pos="br" />

      {/* status chip */}
      <div className="pointer-events-none absolute left-4 top-4 z-20 flex items-center gap-2 rounded-full border border-white/10 bg-black/40 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-zinc-200 backdrop-blur">
        <motion.span
          className="h-1.5 w-1.5 rounded-full bg-[#3ee07a]"
          animate={reduce ? undefined : { opacity: [1, 0.3, 1] }}
          transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
        />
        {analyzingLabel}
      </div>

      {/* active exercise name */}
      <div className="pointer-events-none absolute bottom-4 left-4 z-20">
        <p className="font-display text-xl font-semibold text-zinc-50 drop-shadow">{name}</p>
        <span className="mt-1 block h-0.5 w-10 rounded-full bg-gradient-to-r from-[#5ffb6f] to-[#16b8a8]" />
      </div>

      {/* play / pause */}
      <button
        onClick={togglePlay}
        aria-label={playing ? pauseLabel : playLabel}
        className="absolute bottom-4 right-4 z-20 flex h-10 w-10 items-center justify-center rounded-full border border-white/15 bg-black/45 text-zinc-100 backdrop-blur transition-transform hover:scale-105 active:scale-95"
      >
        {playing ? <Pause weight="fill" size={16} /> : <Play weight="fill" size={16} />}
      </button>
    </div>
  );
}
