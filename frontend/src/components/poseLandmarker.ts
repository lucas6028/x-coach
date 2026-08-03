// Shared MediaPipe PoseLandmarker loader for the pose-driven mini-games (Fruit Ninja, 67).
// Isolated here (and excluded from coverage) because it needs WASM + WebGL, none of which exist
// under jsdom.
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import { MODEL_URL, type PoseTier } from "../lib/poseTier";

const VERSION = "0.10.35";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
// The task instance is intentionally per-session (callers must close it), while the Vision
// fileset is immutable. Caching it avoids repeat WASM fetch/compile work after route changes.
let filesetPromise: ReturnType<typeof FilesetResolver.forVisionTasks> | undefined;

function getVisionFileset() {
  if (!filesetPromise) {
    filesetPromise = FilesetResolver.forVisionTasks(WASM_BASE).catch((error) => {
      // A transient CDN/WebView failure must remain retryable; do not permanently cache a reject.
      filesetPromise = undefined;
      throw error;
    });
  }
  return filesetPromise;
}

export async function createPoseLandmarker(tier: PoseTier = "lite"): Promise<PoseLandmarker> {
  const fileset = await getVisionFileset();
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL[tier], delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}
