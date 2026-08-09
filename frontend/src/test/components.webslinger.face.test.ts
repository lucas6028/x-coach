import { describe, expect, it } from "vitest";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { extractFaceLandmarks } from "../components/webslinger/webSlingerDetector";

function faceFrame(): NormalizedLandmark[] {
  const landmarks = Array.from(
    { length: 9 },
    () => ({ x: 0.5, y: 0.5, z: 0, visibility: 1 }) as NormalizedLandmark
  );
  landmarks[0] = { x: 0.5, y: 0.44, z: 0, visibility: 1 };
  landmarks[2] = { x: 0.46, y: 0.4, z: 0, visibility: 1 };
  landmarks[5] = { x: 0.54, y: 0.4, z: 0, visibility: 1 };
  landmarks[7] = { x: 0.4, y: 0.44, z: 0, visibility: 1 };
  landmarks[8] = { x: 0.6, y: 0.44, z: 0, visibility: 1 };
  return landmarks;
}

describe("web-slinger face tracking", () => {
  it("extracts the five stable anchors used by the mask", () => {
    expect(extractFaceLandmarks(faceFrame())).toEqual({
      nose: { x: 0.5, y: 0.44 },
      leftEye: { x: 0.46, y: 0.4 },
      rightEye: { x: 0.54, y: 0.4 },
      leftEar: { x: 0.4, y: 0.44 },
      rightEar: { x: 0.6, y: 0.44 },
    });
  });

  it("hides the mask when a required face anchor is unreliable", () => {
    const landmarks = faceFrame();
    landmarks[7] = { ...landmarks[7], visibility: 0.1 };
    expect(extractFaceLandmarks(landmarks)).toBeNull();
    expect(extractFaceLandmarks(null)).toBeNull();
  });
});
