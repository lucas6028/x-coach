// The MediaPipe + canvas boundary for Pose Duel. Isolated here (and excluded from coverage)
// because it needs WASM + WebGL + a real camera, none of which exist under jsdom. All the game
// reasoning lives in ../../lib/duel/* and is unit-tested there. The one thing that matters for
// multiplayer: numPoses is 2, so a frame can carry two bodies.
import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { POSE_CONNECTIONS } from "../../lib/pose";
import { MIN_VISIBILITY } from "../../lib/duel/angles";
import type { Landmarks, Players } from "../../lib/duel/assign";

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
    numPoses: 2,
  });
}

// Per-player render state: colour, hold progress (0..1), and whether they're matching now.
export type SideVisual = { hold: number; matched: boolean };

export type Scene = {
  players: Players;
  a: SideVisual;
  b: SideVisual;
};

// Cyan for player A, amber for player B — chosen over left/right labels so the selfie mirror
// never confuses who's who.
const COLOR_A = { core: "#22d3ee", dim: "rgba(34,211,238,0.45)" };
const COLOR_B = { core: "#f59e0b", dim: "rgba(245,158,11,0.45)" };

// Mirror horizontally so the selfie view feels natural.
const mx = (x: number, w: number) => (1 - x) * w;

function vis(p: NormalizedLandmark | null | undefined): boolean {
  return !!p && (p.visibility ?? 1) >= MIN_VISIBILITY;
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  lm: Landmarks,
  width: number,
  height: number,
  color: { core: string; dim: string },
  matched: boolean
) {
  const marks = lm as (NormalizedLandmark | null | undefined)[];
  ctx.lineWidth = matched ? 6 : 4;
  ctx.strokeStyle = matched ? color.core : color.dim;
  for (const [i, j] of POSE_CONNECTIONS) {
    const a = marks[i];
    const b = marks[j];
    if (!vis(a) || !vis(b)) continue;
    ctx.beginPath();
    ctx.moveTo(mx(a!.x, width), a!.y * height);
    ctx.lineTo(mx(b!.x, width), b!.y * height);
    ctx.stroke();
  }
  ctx.fillStyle = color.core;
  for (const p of marks) {
    if (!vis(p)) continue;
    ctx.beginPath();
    ctx.arc(mx(p!.x, width), p!.y * height, matched ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// A ring above a player's head that fills as they hold the pose.
function drawHoldRing(
  ctx: CanvasRenderingContext2D,
  lm: Landmarks,
  width: number,
  height: number,
  color: { core: string; dim: string },
  hold: number
) {
  const marks = lm as (NormalizedLandmark | null | undefined)[];
  const ls = marks[11];
  const rs = marks[12];
  if (!vis(ls) || !vis(rs)) return;
  const cx = mx((ls!.x + rs!.x) / 2, width);
  const shoulderY = ((ls!.y + rs!.y) / 2) * height;
  const cy = shoulderY - height * 0.14;
  const r = height * 0.045;

  ctx.lineWidth = 6;
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  if (hold > 0) {
    ctx.strokeStyle = color.core;
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, hold));
    ctx.stroke();
  }
}

export function drawScene(
  ctx: CanvasRenderingContext2D,
  scene: Scene,
  width: number,
  height: number
): void {
  ctx.clearRect(0, 0, width, height);
  const { a: la, b: lb } = scene.players;
  if (la) {
    drawSkeleton(ctx, la, width, height, COLOR_A, scene.a.matched);
    drawHoldRing(ctx, la, width, height, COLOR_A, scene.a.hold);
  }
  if (lb) {
    drawSkeleton(ctx, lb, width, height, COLOR_B, scene.b.matched);
    drawHoldRing(ctx, lb, width, height, COLOR_B, scene.b.hold);
  }
}
