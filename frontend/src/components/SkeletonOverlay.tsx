import { useEffect, useRef } from "react";
import type { Analysis } from "../api";
import { POSE_CONNECTIONS, FAULT_LANDMARKS, edgeIsFaulty } from "../lib/pose";
import { containRect } from "../lib/videoRect";

interface Props {
  analysis: Analysis;
  videoRef: React.RefObject<HTMLVideoElement>;
  onActiveFault: (faultId: string | null) => void;
}

const PRIMARY = "#0f758a";
const DANGER = "#ef4444";
const VIS_THRESHOLD = 0.4;

// Draws the 33-landmark skeleton on a canvas, synced to the <video> via requestAnimationFrame.
// Limbs implicated by a fault active at the current frame are drawn in red.
export default function SkeletonOverlay({ analysis, videoRef, onActiveFault }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastFaultRef = useRef<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const fps = analysis.pose.fps || 30;
    const frames = analysis.pose.frames;
    let raf = 0;

    const draw = () => {
      raf = requestAnimationFrame(draw);

      // Match canvas backing store to its displayed size (handles resize + DPR).
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
      const frame = frames[idx];

      // Determine which faults are active at this frame and the implicated landmark set.
      const activeLandmarks = new Set<number>();
      let activeFault: string | null = null;
      for (const d of analysis.detections) {
        if (idx >= d.start_frame && idx <= d.end_frame) {
          if (!activeFault) activeFault = d.fault_id;
          for (const lm of FAULT_LANDMARKS[d.fault_id] || []) activeLandmarks.add(lm);
        }
      }
      if (activeFault !== lastFaultRef.current) {
        lastFaultRef.current = activeFault;
        onActiveFault(activeFault);
      }

      if (!frame || !frame.lm) return;
      const lm = frame.lm;

      // The <video> uses object-contain, so it is letterboxed inside the canvas box.
      // Map the normalized [0,1] landmarks onto the *rendered* video rectangle, not the
      // full canvas, otherwise the skeleton drifts off the body.
      const vw = video.videoWidth || analysis.metadata.width || analysis.pose.width || 1;
      const vh = video.videoHeight || analysis.metadata.height || analysis.pose.height || 1;
      // Shared with the DOM-positioned fault chips (lib/videoRect) — they annotate the same
      // joints this canvas highlights, so the two must agree on where the video actually is.
      const { offsetX: oX, offsetY: oY, width: rW, height: rH } = containRect(
        canvas.width,
        canvas.height,
        vw / vh
      );
      const px = (x: number) => oX + x * rW;
      const py = (y: number) => oY + y * rH;

      // Edges
      for (const [a, b] of POSE_CONNECTIONS) {
        const pa = lm[a];
        const pb = lm[b];
        if (!pa || !pb || pa[2] < VIS_THRESHOLD || pb[2] < VIS_THRESHOLD) continue;
        const faulty = edgeIsFaulty(a, b, activeLandmarks);
        ctx.beginPath();
        ctx.moveTo(px(pa[0]), py(pa[1]));
        ctx.lineTo(px(pb[0]), py(pb[1]));
        ctx.strokeStyle = faulty ? DANGER : PRIMARY;
        ctx.lineWidth = (faulty ? 4 : 2.5) * dpr;
        ctx.lineCap = "round";
        if (faulty) {
          ctx.shadowColor = DANGER;
          ctx.shadowBlur = 8 * dpr;
        } else {
          ctx.shadowBlur = 0;
        }
        ctx.stroke();
      }
      ctx.shadowBlur = 0;

      // Joints
      for (let i = 0; i < lm.length; i++) {
        const p = lm[i];
        if (!p || p[2] < VIS_THRESHOLD) continue;
        const faulty = activeLandmarks.has(i);
        ctx.beginPath();
        ctx.arc(px(p[0]), py(p[1]), (faulty ? 4 : 3) * dpr, 0, Math.PI * 2);
        ctx.fillStyle = faulty ? DANGER : "#3bc9db";
        ctx.fill();
      }
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [analysis, videoRef, onActiveFault]);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none z-10" />;
}
