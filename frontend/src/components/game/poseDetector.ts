// The MediaPipe boundary for the Pose Match game: create a live PoseLandmarker and
// draw its output onto a canvas. Isolated here (and excluded from coverage) because it
// depends on WASM + WebGL + a real camera, none of which exist under jsdom. Everything
// the game *reasons* about lives in ../../lib/game/* and is unit-tested there.
import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { POSE_CONNECTIONS } from "../../lib/pose";

const VERSION = "0.10.35";
// WASM runtime bundled with the pinned tasks-vision release.
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
// Lite model keeps first-load small — plenty accurate for coarse full-body poses.
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

export type { NormalizedLandmark };

// Build a VIDEO-mode single-pose landmarker. Rejects if the model / WASM can't load.
export async function createPoseLandmarker(): Promise<PoseLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

// Draw the skeleton, mirrored to match the selfie-view video, colouring the whole body
// green while `matching` (the player is nailing the current target) and neutral otherwise.
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
  matching: boolean
): void {
  const stroke = matching ? "#3ee07a" : "#e5e7eb";
  const glow = matching ? "#16b8a8" : "#0f758a";
  // Mirror horizontally so moving right on screen moves the on-screen skeleton right.
  const px = (x: number) => (1 - x) * width;
  const py = (y: number) => y * height;

  ctx.lineCap = "round";
  ctx.strokeStyle = stroke;
  ctx.shadowColor = glow;
  ctx.shadowBlur = 10;
  ctx.lineWidth = 5;
  for (const [a, b] of POSE_CONNECTIONS) {
    const pa = landmarks[a];
    const pb = landmarks[b];
    if (!pa || !pb) continue;
    if ((pa.visibility ?? 1) < 0.4 || (pb.visibility ?? 1) < 0.4) continue;
    ctx.beginPath();
    ctx.moveTo(px(pa.x), py(pa.y));
    ctx.lineTo(px(pb.x), py(pb.y));
    ctx.stroke();
  }
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#f8fafc";
  for (const p of landmarks) {
    if (!p || (p.visibility ?? 1) < 0.4) continue;
    ctx.beginPath();
    ctx.arc(px(p.x), py(p.y), 4, 0, Math.PI * 2);
    ctx.fill();
  }
}
