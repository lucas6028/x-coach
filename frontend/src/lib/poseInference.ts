import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import type { PoseTier } from "./poseTier";

export interface InferenceLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

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

function fromMainThread(landmarker: PoseLandmarker): PoseInferenceRunner {
  return {
    async detect(video, timestamp) {
      const result = landmarker.detectForVideo(video, timestamp);
      return {
        landmarks: result.landmarks?.[0] ?? null,
        worldLandmarks: result.worldLandmarks?.[0] ?? null,
      };
    },
    close: () => landmarker.close(),
  };
}

async function createMainThreadRunner(tier: PoseTier): Promise<PoseInferenceRunner> {
  // Keep the compatibility path out of the normal worker download. It is only needed by WebViews
  // that cannot provide the required worker image/GPU primitives.
  const { createPoseLandmarker } = await import("../components/poseLandmarker");
  return fromMainThread(await createPoseLandmarker(tier));
}

async function createWorkerRunner(tier: PoseTier): Promise<PoseInferenceRunner> {
  if (
    typeof Worker === "undefined" ||
    typeof OffscreenCanvas === "undefined" ||
    typeof createImageBitmap !== "function"
  ) {
    throw new Error("Worker image inference is unavailable.");
  }

  const worker = new Worker(new URL("../workers/poseInference.worker.ts", import.meta.url), {
    type: "module",
  });
  const pending = new Map<number, { resolve: (value: PoseInferenceResult) => void; reject: (reason: Error) => void }>();
  let nextId = 1;
  let closed = false;
  let initTimer: ReturnType<typeof setTimeout> | undefined;

  const rejectPending = (reason: Error) => {
    pending.forEach(({ reject }) => reject(reason));
    pending.clear();
  };

  await new Promise<void>((resolve, reject) => {
    const fail = (reason: Error) => {
      clearTimeout(initTimer);
      worker.terminate();
      reject(reason);
    };
    initTimer = setTimeout(() => fail(new Error("Pose worker initialization timed out.")), 15_000);
    worker.onerror = () => fail(new Error("Pose worker failed to initialize."));
    worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === "ready") {
        clearTimeout(initTimer);
        resolve();
      } else if (data.type === "error") {
        fail(new Error(data.message));
      }
    };
    worker.postMessage({ type: "init", tier });
  });

  worker.onerror = () => rejectPending(new Error("Pose worker stopped unexpectedly."));
  worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    if (data.type === "result") {
      const request = pending.get(data.id);
      pending.delete(data.id);
      request?.resolve({ landmarks: data.landmarks, worldLandmarks: data.worldLandmarks });
    } else if (data.type === "error") {
      if (data.id === undefined) {
        rejectPending(new Error(data.message));
      } else {
        const request = pending.get(data.id);
        pending.delete(data.id);
        request?.reject(new Error(data.message));
      }
    }
  };

  return {
    async detect(video, timestamp) {
      if (closed) throw new Error("Pose worker is closed.");
      const bitmap = await createImageBitmap(video);
      const id = nextId++;
      return new Promise<PoseInferenceResult>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        try {
          worker.postMessage({ type: "infer", id, bitmap, timestamp }, [bitmap]);
        } catch (error) {
          pending.delete(id);
          bitmap.close();
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      });
    },
    close() {
      if (closed) return;
      closed = true;
      rejectPending(new Error("Pose worker is closed."));
      try {
        worker.postMessage({ type: "close" });
      } catch {
        // Worker termination below is sufficient when it has already stopped.
      }
      worker.terminate();
    },
  };
}

/** Prefer an off-main-thread worker for a long analysis; keep the established GPU path as fallback. */
export async function createPoseInferenceRunner(tier: PoseTier): Promise<PoseInferenceRunner> {
  try {
    const workerRunner = await createWorkerRunner(tier);
    let fallback: PoseInferenceRunner | null = null;
    let closed = false;
    return {
      async detect(video, timestamp) {
        if (closed) throw new Error("Pose inference runner is closed.");
        if (fallback) return fallback.detect(video, timestamp);
        try {
          return await workerRunner.detect(video, timestamp);
        } catch {
          // Some WebViews advertise worker primitives but cannot run MediaPipe/WebGL in that
          // worker. Recover the current frame with the existing main-thread GPU delegate.
          try {
            workerRunner.close();
          } catch {
            // The worker may already be terminated; the compatibility path still must run.
          }
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
  } catch {
    return createMainThreadRunner(tier);
  }
}
