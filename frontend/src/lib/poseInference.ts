import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import type { PoseTier } from "./poseTier";

export interface InferenceLandmark { x: number; y: number; z: number; visibility?: number }
export interface PoseInferenceResult {
  landmarks: InferenceLandmark[] | null;
  worldLandmarks: InferenceLandmark[] | null;
}
export interface PoseInferenceRunner {
  detect(video: HTMLVideoElement, timestamp: number): Promise<PoseInferenceResult>;
  close(): void;
}

type WorkerResponse =
  | { type: "ready" }
  | { type: "result"; id: number; landmarks: InferenceLandmark[] | null; worldLandmarks: InferenceLandmark[] | null }
  | { type: "error"; id?: number; message: string };
type PendingRequest = {
  resolve: (value: PoseInferenceResult) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

const WORKER_INIT_TIMEOUT_MS = 15_000;
const WORKER_FRAME_TIMEOUT_MS = 15_000;

/** An infrastructure failure that makes a worker unsuitable for the remaining analysis. */
class WorkerUnavailableError extends Error {}

function fromMainThread(landmarker: PoseLandmarker): PoseInferenceRunner {
  return {
    async detect(video, timestamp) {
      const result = landmarker.detectForVideo(video, timestamp);
      return { landmarks: result.landmarks?.[0] ?? null, worldLandmarks: result.worldLandmarks?.[0] ?? null };
    },
    close: () => landmarker.close(),
  };
}

async function createMainThreadRunner(tier: PoseTier): Promise<PoseInferenceRunner> {
  const { createPoseLandmarker } = await import("../components/poseLandmarker");
  return fromMainThread(await createPoseLandmarker(tier));
}

async function createWorkerRunner(tier: PoseTier): Promise<PoseInferenceRunner> {
  if (typeof Worker === "undefined" || typeof OffscreenCanvas === "undefined" || typeof createImageBitmap !== "function") {
    throw new WorkerUnavailableError("Worker image inference is unavailable.");
  }
  const worker = new Worker(new URL("../workers/poseInference.worker.ts", import.meta.url), { type: "module" });
  const pending = new Map<number, PendingRequest>();
  let nextId = 1;
  let closed = false;

  const rejectPending = (reason: Error) => {
    pending.forEach(({ reject, timer }) => { clearTimeout(timer); reject(reason); });
    pending.clear();
  };
  const stopWorker = (reason: Error) => {
    if (closed) return;
    closed = true;
    rejectPending(reason);
    worker.terminate();
  };

  await new Promise<void>((resolve, reject) => {
    const initTimer = setTimeout(() => {
      stopWorker(new WorkerUnavailableError("Pose worker initialization timed out."));
      reject(new WorkerUnavailableError("Pose worker initialization timed out."));
    }, WORKER_INIT_TIMEOUT_MS);
    const fail = (message: string) => {
      clearTimeout(initTimer);
      const error = new WorkerUnavailableError(message);
      stopWorker(error);
      reject(error);
    };
    worker.onerror = () => fail("Pose worker failed to initialize.");
    worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === "ready") {
        clearTimeout(initTimer);
        resolve();
      } else if (data.type === "error") {
        fail(data.message);
      }
    };
    worker.postMessage({ type: "init", tier });
  });

  worker.onerror = () => stopWorker(new WorkerUnavailableError("Pose worker stopped unexpectedly."));
  worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    if (data.type === "result") {
      const request = pending.get(data.id);
      pending.delete(data.id);
      if (request) {
        clearTimeout(request.timer);
        request.resolve({ landmarks: data.landmarks, worldLandmarks: data.worldLandmarks });
      }
    } else if (data.type === "error") {
      if (data.id === undefined) {
        stopWorker(new WorkerUnavailableError(data.message));
        return;
      }
      const request = pending.get(data.id);
      pending.delete(data.id);
      if (request) {
        clearTimeout(request.timer);
        // A task error can be frame-specific. Preserve it rather than permanently demoting a
        // worker because a transient snapshot/inference failure happened once.
        request.reject(new Error(data.message));
      }
    }
  };

  return {
    async detect(video, timestamp) {
      if (closed) throw new WorkerUnavailableError("Pose worker is closed.");
      // Snapshot errors are not evidence that the worker itself is incompatible.
      const bitmap = await createImageBitmap(video);
      if (closed) {
        bitmap.close();
        throw new WorkerUnavailableError("Pose worker is closed.");
      }
      const id = nextId++;
      return new Promise<PoseInferenceResult>((resolve, reject) => {
        const timer = setTimeout(() => {
          stopWorker(new WorkerUnavailableError("Pose worker inference timed out."));
        }, WORKER_FRAME_TIMEOUT_MS);
        pending.set(id, { resolve, reject, timer });
        try {
          worker.postMessage({ type: "infer", id, bitmap, timestamp }, [bitmap]);
        } catch (error) {
          const request = pending.get(id);
          pending.delete(id);
          if (request) clearTimeout(request.timer);
          bitmap.close();
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      });
    },
    close() {
      if (closed) return;
      closed = true;
      rejectPending(new Error("Pose worker is closed."));
      try { worker.postMessage({ type: "close" }); } catch { /* terminate below is sufficient */ }
      worker.terminate();
    },
  };
}

/** Prefer off-main-thread inference, falling back only when the worker infrastructure is unusable. */
export async function createPoseInferenceRunner(tier: PoseTier): Promise<PoseInferenceRunner> {
  let workerRunner: PoseInferenceRunner;
  try {
    workerRunner = await createWorkerRunner(tier);
  } catch {
    return createMainThreadRunner(tier);
  }
  let fallback: PoseInferenceRunner | null = null;
  let closed = false;
  return {
    async detect(video, timestamp) {
      if (closed) throw new Error("Pose inference runner is closed.");
      if (fallback) return fallback.detect(video, timestamp);
      try {
        return await workerRunner.detect(video, timestamp);
      } catch (error) {
        if (!(error instanceof WorkerUnavailableError)) throw error;
        workerRunner.close();
        fallback = await createMainThreadRunner(tier);
        return fallback.detect(video, timestamp);
      }
    },
    close() {
      if (closed) return;
      closed = true;
      workerRunner.close();
      fallback?.close();
    },
  };
}
