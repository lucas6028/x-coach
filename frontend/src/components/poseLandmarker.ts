// Shared MediaPipe PoseLandmarker loader for the pose-driven mini-games (Fruit Ninja, 67).
// Isolated here (and excluded from coverage) because it needs WASM + WebGL, none of which exist
// under jsdom.
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import { MODEL_URL, type PoseTier } from "../lib/poseTier";

const VERSION = "0.10.35";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;

export async function createPoseLandmarker(tier: PoseTier = "lite"): Promise<PoseLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL[tier], delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}
