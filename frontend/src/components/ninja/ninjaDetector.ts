// The MediaPipe + canvas boundary for Fruit Ninja. Isolated here (and excluded from coverage)
// because it needs WASM + WebGL + a real camera, none of which exist under jsdom. All the game
// reasoning lives in ../../lib/ninja/* and is unit-tested there.
import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import type { Entity } from "../../lib/ninja/physics";

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

export type Point = { x: number; y: number };

export type Scene = {
  entities: Entity[];
  // Recent wrist positions per hand (oldest→newest), for the blade trail.
  trails: Point[][];
  // Fleeting bomb flash (0..1 life) when a bomb just went off, or null.
  bombFlash: number | null;
};

// Mirror horizontally so the selfie view feels natural.
const mx = (x: number, w: number) => (1 - x) * w;

const BLADE_COLORS = ["#22d3ee", "#f59e0b"];

export function drawScene(
  ctx: CanvasRenderingContext2D,
  scene: Scene,
  width: number,
  height: number
): void {
  ctx.clearRect(0, 0, width, height);

  // Fruits + bombs as big emoji (screen space, not mirrored — emoji aren't handed).
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const e of scene.entities) {
    ctx.font = `${Math.round(e.radius * 2 * height)}px serif`;
    ctx.fillText(e.emoji, e.x * width, e.y * height);
  }

  // Blade trails: a fading, thickening stroke through each hand's recent positions.
  scene.trails.forEach((trail, i) => {
    if (trail.length < 2) return;
    const color = BLADE_COLORS[i % BLADE_COLORS.length];
    for (let k = 1; k < trail.length; k += 1) {
      const a = trail[k - 1];
      const b = trail[k];
      const life = k / trail.length;
      ctx.strokeStyle = color;
      ctx.globalAlpha = life * 0.85;
      ctx.lineWidth = 2 + life * 10;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(mx(a.x, width), a.y * height);
      ctx.lineTo(mx(b.x, width), b.y * height);
      ctx.stroke();
    }
    // Bright blade tip.
    const tip = trail[trail.length - 1];
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#f8fafc";
    ctx.beginPath();
    ctx.arc(mx(tip.x, width), tip.y * height, height * 0.012, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;

  // Bomb blast: a red full-screen flash that fades out.
  if (scene.bombFlash != null) {
    ctx.fillStyle = `rgba(239,68,68,${0.5 * scene.bombFlash})`;
    ctx.fillRect(0, 0, width, height);
  }
}
