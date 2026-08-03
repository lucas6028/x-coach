// MediaPipe's synchronous video API blocks whichever thread owns the task. This worker owns the
// long-running upload analysis task, leaving the UI thread free for video decode and progress UI.
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import { MODEL_URL, type PoseTier } from "../lib/poseTier";

const VERSION = "0.10.35";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;

type WorkerScope = {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: WorkerResponse): void;
};

type WorkerRequest =
  | { type: "init"; tier: PoseTier }
  | { type: "infer"; id: number; bitmap: ImageBitmap; timestamp: number }
  | { type: "close" };

type Landmark = { x: number; y: number; z: number; visibility?: number };
type WorkerResponse =
  | { type: "ready" }
  | { type: "result"; id: number; landmarks: Landmark[] | null; worldLandmarks: Landmark[] | null }
  | { type: "error"; id?: number; message: string };

const scope = globalThis as unknown as WorkerScope;
let landmarker: PoseLandmarker | null = null;

function copyLandmarks(points?: readonly Landmark[]): Landmark[] | null {
  return points?.map(({ x, y, z, visibility }) => ({ x, y, z, visibility })) ?? null;
}

scope.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
      landmarker = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL[data.tier], delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
      });
      scope.postMessage({ type: "ready" });
      return;
    }
    if (data.type === "close") {
      landmarker?.close();
      landmarker = null;
      return;
    }
    if (!landmarker) throw new Error("Pose worker received a frame before initialization.");
    try {
      const result = landmarker.detectForVideo(data.bitmap, data.timestamp);
      scope.postMessage({
        type: "result",
        id: data.id,
        landmarks: copyLandmarks(result.landmarks?.[0]),
        worldLandmarks: copyLandmarks(result.worldLandmarks?.[0]),
      });
    } finally {
      data.bitmap.close();
    }
  } catch (error) {
    scope.postMessage({
      type: "error",
      id: data.type === "infer" ? data.id : undefined,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
