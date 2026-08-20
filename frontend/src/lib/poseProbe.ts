// Full live-pose chain probe for the LIFF device check: camera → decoded <video> frame →
// MediaPipe PoseLandmarker (WASM + WebGL GPU delegate, fetched from CDN) → sustained
// detectForVideo loop with an FPS measurement. Camera working alone does NOT mean the
// games are playable inside the LINE in-app WebView — model download, shader-compile
// warmup (which can freeze the main thread for tens of seconds on mobile), and per-frame
// inference speed each fail independently there, so the probe reports per-stage results.
//
// Impure MediaPipe/WebGL/camera glue — excluded from coverage like the other detector
// boundaries (see vite.config.ts + codecov.yml); it cannot run under jsdom.

import { CameraError, getCameraStream } from "./camera";
import { waitForVideoFrame } from "./videoFrame";

export interface PoseProbeResult {
  ok: boolean;
  /** The stage reached: where it failed, or "done" after a full measured run. */
  stage: "camera" | "video" | "model" | "warmup" | "detect" | "done";
  message: string;
  /** WASM + model download/compile time (CDN fetch included). */
  modelLoadMs?: number;
  /** First inference incl. GPU shader compile — the "freeze" risk on mobile WebViews. */
  warmupMs?: number;
  /** Sustained inference throughput over the measurement window. */
  avgFps?: number;
  /** Whether any frame actually yielded pose landmarks (camera + model really agree). */
  landmarksSeen?: boolean;
}

const MEASURE_MS = 5_000;

/**
 * Run the exact pipeline the mini-games use (same createPoseLandmarker, same constraints)
 * against `video`, measure it, and tear everything down. Never rejects — every failure
 * mode is folded into the result so the diagnostics page can just render it.
 */
export async function probeLivePose(
  video: HTMLVideoElement,
  { measureMs = MEASURE_MS }: { measureMs?: number } = {}
): Promise<PoseProbeResult> {
  let stream: MediaStream | null = null;
  let landmarker: { detectForVideo(v: HTMLVideoElement, t: number): unknown; close(): void } | null =
    null;
  const teardown = () => {
    stream?.getTracks().forEach((track) => track.stop());
    try {
      landmarker?.close();
    } catch {
      /* a landmarker that failed mid-init may throw on close — nothing left to free */
    }
    video.srcObject = null;
  };

  try {
    // Stage 1: camera (same constraints as the games).
    try {
      stream = await getCameraStream({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      });
    } catch (err) {
      const reason = err instanceof CameraError ? err.reason : "error";
      return { ok: false, stage: "camera", message: `getUserMedia failed: ${reason}` };
    }

    // Stage 2: a decoded frame the model can read.
    video.srcObject = stream;
    try {
      await video.play();
    } catch {
      /* iOS may reject play() pre-gesture; loadeddata below is the real gate */
    }
    await waitForVideoFrame(video);
    if (video.readyState < 2) {
      return { ok: false, stage: "video", message: "Camera stream never produced a frame." };
    }

    // Stage 3: WASM + model from the CDN (jsdelivr + storage.googleapis.com).
    const modelStart = performance.now();
    try {
      const { createPoseLandmarker } = await import("../components/poseLandmarker");
      landmarker = await createPoseLandmarker();
    } catch (err) {
      return {
        ok: false,
        stage: "model",
        message: `Model load failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
    const modelLoadMs = Math.round(performance.now() - modelStart);

    // Stage 4: first inference — GPU delegate shader compile happens here.
    const warmupStart = performance.now();
    let landmarksSeen = false;
    const sawLandmarks = (result: unknown): boolean =>
      Array.isArray((result as { landmarks?: unknown[] })?.landmarks) &&
      ((result as { landmarks: unknown[] }).landmarks.length ?? 0) > 0;
    try {
      landmarksSeen = sawLandmarks(landmarker.detectForVideo(video, performance.now()));
    } catch (err) {
      return {
        ok: false,
        stage: "warmup",
        message: `First inference failed: ${err instanceof Error ? err.message : String(err)}`,
        modelLoadMs,
      };
    }
    const warmupMs = Math.round(performance.now() - warmupStart);

    // Stage 5: sustained loop — the number that decides "playable or not".
    let frames = 0;
    const loopStart = performance.now();
    try {
      while (performance.now() - loopStart < measureMs) {
        const result = landmarker.detectForVideo(video, performance.now());
        landmarksSeen = landmarksSeen || sawLandmarks(result);
        frames += 1;
        // Yield to the event loop like a rAF-driven game does, so the measurement reflects
        // a realistic frame cadence instead of a busy-loop burn.
        await new Promise<void>((r) => requestAnimationFrame(() => r()));
      }
    } catch (err) {
      return {
        ok: false,
        stage: "detect",
        message: `Inference loop failed: ${err instanceof Error ? err.message : String(err)}`,
        modelLoadMs,
        warmupMs,
      };
    }
    const elapsed = performance.now() - loopStart;
    const avgFps = elapsed > 0 ? Math.round((frames / elapsed) * 1000 * 10) / 10 : 0;

    return {
      ok: true,
      stage: "done",
      message: `${frames} frames in ${Math.round(elapsed)}ms.`,
      modelLoadMs,
      warmupMs,
      avgFps,
      landmarksSeen,
    };
  } finally {
    teardown();
  }
}
