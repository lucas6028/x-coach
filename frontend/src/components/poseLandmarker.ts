// Shared MediaPipe PoseLandmarker loader for the pose-driven mini-games (Fruit Ninja, 67).
// Isolated here (and excluded from coverage) because it needs WASM + WebGL, none of which exist
// under jsdom.
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

const VERSION = "0.10.35";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

export async function createPoseLandmarker(): Promise<PoseLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}
