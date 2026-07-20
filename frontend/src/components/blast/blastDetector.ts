// The MediaPipe + canvas boundary for Meme Blaster. Isolated here (and excluded from
// coverage) because it needs WASM + WebGL + a real camera, none of which exist under
// jsdom. All gameplay reasoning lives in ../../lib/blast/* and is unit-tested there.
import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { LM } from "../../lib/pose";
import type { Target } from "../../lib/blast/targets";

const VERSION = "0.10.35";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

export type { NormalizedLandmark };

export async function createPoseLandmarker(): Promise<PoseLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

// Everything the render pass needs about the current frame.
export type Scene = {
  landmarks: NormalizedLandmark[] | null;
  targets: Target[];
  charge: number; // 0..1
  armed: boolean;
  // Active beam (screen-space aim height + remaining life 0..1), or null.
  beam: { y: number; life: number } | null;
};

// Mirror horizontally so the selfie view feels natural.
const mx = (x: number, w: number) => (1 - x) * w;

export function drawScene(
  ctx: CanvasRenderingContext2D,
  scene: Scene,
  width: number,
  height: number
): void {
  ctx.clearRect(0, 0, width, height);

  // Targets — meme orbs drawn as large emoji (screen space, not mirrored).
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `${Math.round(height * 0.09)}px serif`;
  for (const t of scene.targets) {
    ctx.globalAlpha = 1;
    ctx.fillText(t.emoji, t.x * width, t.y * height);
  }
  ctx.globalAlpha = 1;

  // Active beam — a bright horizontal energy bar sweeping across at the aim height.
  if (scene.beam) {
    const by = scene.beam.y * height;
    const h = height * 0.14;
    const grad = ctx.createLinearGradient(0, by - h / 2, width, by - h / 2);
    grad.addColorStop(0, "rgba(62,224,122,0)");
    grad.addColorStop(0.5, `rgba(94,251,111,${0.55 * scene.beam.life})`);
    grad.addColorStop(1, `rgba(22,184,168,${0.2 * scene.beam.life})`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, by - h / 2, width, h);
    ctx.strokeStyle = `rgba(234,255,240,${scene.beam.life})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, by);
    ctx.lineTo(width, by);
    ctx.stroke();
  }

  // Hands — wrist markers + a charging orb between them (mirrored to match the video).
  const lm = scene.landmarks;
  if (lm) {
    const lw = lm[LM.LEFT_WRIST];
    const rw = lm[LM.RIGHT_WRIST];
    if (lw && rw) {
      const lx = mx(lw.x, width);
      const ly = lw.y * height;
      const rx = mx(rw.x, width);
      const ry = rw.y * height;
      const cx = (lx + rx) / 2;
      const cy = (ly + ry) / 2;

      if (scene.charge > 0) {
        const r = height * (0.03 + 0.09 * scene.charge);
        const orb = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        const core = scene.armed ? "#eafff0" : "#5ffb6f";
        orb.addColorStop(0, core);
        orb.addColorStop(1, "rgba(22,184,168,0)");
        ctx.fillStyle = orb;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = scene.armed ? "#5ffb6f" : "#e5e7eb";
      for (const [x, y] of [
        [lx, ly],
        [rx, ry],
      ]) {
        ctx.beginPath();
        ctx.arc(x, y, height * 0.018, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}
