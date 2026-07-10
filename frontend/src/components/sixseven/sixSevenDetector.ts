// The MediaPipe + canvas boundary for the 67 game. Isolated here (and excluded from coverage)
// because it needs WASM + WebGL + a real camera, none of which exist under jsdom. All the game
// reasoning lives in ../../lib/sixseven/* and is unit-tested there.
import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { LM } from "../../lib/pose";
import { visible, type Lead } from "../../lib/sixseven/gesture";

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

export type Scene = {
  landmarks: NormalizedLandmark[] | null;
  // Which hand is up this frame.
  lead: Lead;
  // Fleeting pop when a 67 just scored (0..1 life), or null.
  pop: number | null;
};

// "6" rides the left hand, "7" the right — the two halves of the chant.
const LEFT_COLOR = "#22d3ee";
const RIGHT_COLOR = "#42d159";

// Mirror horizontally so the selfie view feels natural.
const mx = (x: number, w: number) => (1 - x) * w;

function drawHand(
  ctx: CanvasRenderingContext2D,
  p: NormalizedLandmark,
  width: number,
  height: number,
  label: string,
  color: string,
  active: boolean,
  pop: number | null
) {
  const x = mx(p.x, width);
  const y = p.y * height;
  const r = height * (active ? 0.05 : 0.032);

  if (active) {
    const glow = ctx.createRadialGradient(x, y, 0, x, y, r * (pop != null ? 1.8 + pop : 1.6));
    glow.addColorStop(0, color);
    glow.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, r * (pop != null ? 1.8 + pop : 1.6), 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = active ? "#0b1120" : color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = active ? color : "#0b1120";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `bold ${Math.round(r * 1.3)}px sans-serif`;
  ctx.fillText(label, x, y);
}

export function drawScene(
  ctx: CanvasRenderingContext2D,
  scene: Scene,
  width: number,
  height: number
): void {
  ctx.clearRect(0, 0, width, height);
  const lm = scene.landmarks;
  if (!lm) return;

  const lw = lm[LM.LEFT_WRIST];
  const rw = lm[LM.RIGHT_WRIST];
  if (visible(lw)) {
    drawHand(ctx, lw!, width, height, "6", LEFT_COLOR, scene.lead === "left", scene.lead === "left" ? scene.pop : null);
  }
  if (visible(rw)) {
    drawHand(ctx, rw!, width, height, "7", RIGHT_COLOR, scene.lead === "right", scene.lead === "right" ? scene.pop : null);
  }
}
