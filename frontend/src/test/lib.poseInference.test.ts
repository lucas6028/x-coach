import { afterEach, describe, expect, it, vi } from "vitest";

const mockLandmarker = vi.hoisted(() => ({ detectForVideo: vi.fn(), close: vi.fn() }));
const mockCreateLandmarker = vi.hoisted(() => vi.fn());
vi.mock("../components/poseLandmarker", () => ({ createPoseLandmarker: mockCreateLandmarker }));

import { createPoseInferenceRunner } from "../lib/poseInference";

const poseResult = {
  landmarks: [{ x: 0.1, y: 0.2, z: 0.3, visibility: 0.9 }],
  worldLandmarks: [{ x: 1, y: 2, z: 3, visibility: 0.8 }],
};

class FakeWorker {
  static mode: "result" | "error" | "hang" = "result";
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  terminate = vi.fn();
  postMessage(message: { type: string; id?: number }) {
    if (message.type === "init") queueMicrotask(() => this.onmessage?.({ data: { type: "ready" } } as MessageEvent));
    if (message.type === "infer" && FakeWorker.mode === "result") {
      queueMicrotask(() => this.onmessage?.({ data: { type: "result", id: message.id, ...poseResult } } as MessageEvent));
    }
    if (message.type === "infer" && FakeWorker.mode === "error") this.onerror?.(new Event("error"));
  }
}

function enableWorker(mode: typeof FakeWorker.mode = "result") {
  FakeWorker.mode = mode;
  vi.stubGlobal("Worker", FakeWorker);
  vi.stubGlobal("OffscreenCanvas", class {});
  vi.stubGlobal("createImageBitmap", vi.fn().mockResolvedValue({ close: vi.fn() }));
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("pose inference runner", () => {
  it("falls back to the established GPU task when worker image inference is unavailable", async () => {
    vi.stubGlobal("Worker", undefined);
    vi.stubGlobal("OffscreenCanvas", undefined);
    mockLandmarker.detectForVideo.mockReturnValue({ landmarks: [poseResult.landmarks], worldLandmarks: [poseResult.worldLandmarks] });
    mockCreateLandmarker.mockResolvedValue(mockLandmarker);

    const runner = await createPoseInferenceRunner("lite");
    await expect(runner.detect({} as HTMLVideoElement, 42)).resolves.toEqual(poseResult);
    runner.close();
    expect(mockCreateLandmarker).toHaveBeenCalledWith("lite");
  });

  it("uses the worker result when its image and GPU path are healthy", async () => {
    enableWorker();
    const runner = await createPoseInferenceRunner("lite");
    await expect(runner.detect({} as HTMLVideoElement, 42)).resolves.toEqual(poseResult);
    expect(mockCreateLandmarker).not.toHaveBeenCalled();
    runner.close();
  });

  it("falls back when the worker dies with a request pending", async () => {
    enableWorker("error");
    mockLandmarker.detectForVideo.mockReturnValue({ landmarks: [poseResult.landmarks], worldLandmarks: [poseResult.worldLandmarks] });
    mockCreateLandmarker.mockResolvedValue(mockLandmarker);
    const runner = await createPoseInferenceRunner("lite");
    await expect(runner.detect({} as HTMLVideoElement, 42)).resolves.toEqual(poseResult);
    expect(mockCreateLandmarker).toHaveBeenCalledOnce();
    runner.close();
  });

  it("times out a hung worker request and falls back instead of leaving it pending", async () => {
    vi.useFakeTimers();
    enableWorker("hang");
    mockLandmarker.detectForVideo.mockReturnValue({ landmarks: [poseResult.landmarks], worldLandmarks: [poseResult.worldLandmarks] });
    mockCreateLandmarker.mockResolvedValue(mockLandmarker);
    const runner = await createPoseInferenceRunner("lite");
    const result = runner.detect({} as HTMLVideoElement, 42);
    await vi.advanceTimersByTimeAsync(15_000);
    await expect(result).resolves.toEqual(poseResult);
    runner.close();
  });

  it("does not demote a healthy worker when creating an image snapshot fails", async () => {
    enableWorker();
    vi.stubGlobal("createImageBitmap", vi.fn().mockRejectedValue(new Error("snapshot failed")));
    const runner = await createPoseInferenceRunner("lite");
    await expect(runner.detect({} as HTMLVideoElement, 42)).rejects.toThrow("snapshot failed");
    expect(mockCreateLandmarker).not.toHaveBeenCalled();
    runner.close();
  });
});
