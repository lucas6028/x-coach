// The MediaPipe + canvas boundary for Fruit Ninja. Isolated here (and excluded from coverage)
// because it needs WASM + WebGL + a real camera, none of which exist under jsdom. All the game
// reasoning lives in ../../lib/ninja/* and is unit-tested there.
import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import type { Entity } from "../../lib/ninja/physics";
import type { Piece } from "../../lib/ninja/pieces";

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
  // Flying halves of just-sliced fruit.
  pieces: Piece[];
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

  // Fruits + bombs as big emoji. Mirror the *position* (like the selfie video and the blade
  // trail) so a fruit lines up with the hand the player sees; the glyph itself isn't flipped, so
  // it stays readable.
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const e of scene.entities) {
    ctx.font = `${Math.round(e.radius * 2 * height)}px serif`;
    ctx.fillText(e.emoji, mx(e.x, width), e.y * height);
  }

  // Sliced-fruit halves: draw the emoji clipped to one side in the piece's spinning, fading frame,
  // nudged outward along the cut so the two halves read as pulling apart.
  for (const p of scene.pieces) {
    const size = p.radius * 2 * height;
    ctx.save();
    ctx.translate(mx(p.x, width), p.y * height);
    ctx.rotate(p.rot);
    ctx.globalAlpha = Math.max(0, Math.min(1, p.life));
    ctx.beginPath();
    const gap = size * 0.06 * (1 - p.life); // seam opens as the halves drift apart
    if (p.half === "left") ctx.rect(-size, -size, size - gap, size * 2);
    else ctx.rect(gap, -size, size, size * 2);
    ctx.clip();
    ctx.font = `${Math.round(size)}px serif`;
    ctx.fillText(p.emoji, 0, 0);
    ctx.restore();
  }
  ctx.globalAlpha = 1;

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
