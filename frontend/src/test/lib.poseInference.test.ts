import { afterEach, describe, expect, it, vi } from "vitest";

const mockLandmarker = vi.hoisted(() => ({
  detectForVideo: vi.fn(),
  close: vi.fn(),
}));
const mockCreateLandmarker = vi.hoisted(() => vi.fn());

vi.mock("../components/poseLandmarker", () => ({
  createPoseLandmarker: mockCreateLandmarker,
}));

import { createPoseInferenceRunner } from "../lib/poseInference";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("pose inference runner", () => {
  it("falls back to the established GPU task when worker image inference is unavailable", async () => {
    vi.stubGlobal("Worker", undefined);
    vi.stubGlobal("OffscreenCanvas", undefined);
    mockLandmarker.detectForVideo.mockReturnValue({
      landmarks: [[{ x: 0.1, y: 0.2, z: 0.3, visibility: 0.9 }]],
      worldLandmarks: [[{ x: 1, y: 2, z: 3, visibility: 0.8 }]],
    });
    mockCreateLandmarker.mockResolvedValue(mockLandmarker);

    const runner = await createPoseInferenceRunner("lite");
    await expect(runner.detect({} as HTMLVideoElement, 42)).resolves.toEqual({
      landmarks: [{ x: 0.1, y: 0.2, z: 0.3, visibility: 0.9 }],
      worldLandmarks: [{ x: 1, y: 2, z: 3, visibility: 0.8 }],
    });
    runner.close();

    expect(mockCreateLandmarker).toHaveBeenCalledWith("lite");
    expect(mockLandmarker.detectForVideo).toHaveBeenCalledWith(expect.anything(), 42);
    expect(mockLandmarker.close).toHaveBeenCalledOnce();
  });
});
