// Client-side pose extraction: decode a recorded/uploaded clip frame-by-frame, run MediaPipe,
// and emit pose JSON byte-compatible with src/pose/process_videos.py so the backend detector is
// untouched. The pure serializer (landmarksToFrame) is unit-tested; the video/WASM glue below
// is excluded from jsdom coverage like the other browser-only detector boundaries.
import { resolveDuration } from "./mediaDuration";
import type { PoseTier } from "./poseTier";
import { createPoseInferenceRunner } from "./poseInference";

const LANDMARK_COUNT = 33;
export const MAX_POSE_ANALYSIS_DURATION_SECONDS = 90;

export function assertPoseAnalysisDuration(duration: number): void {
  if (duration > MAX_POSE_ANALYSIS_DURATION_SECONDS) {
    throw new Error(`Videos longer than ${MAX_POSE_ANALYSIS_DURATION_SECONDS} seconds cannot be analyzed.`);
  }
}

// Worker structured-clone results from some WebViews can omit optional fields. Never serialize a
// partial landmark: JSON.stringify would drop that key and the server would reject the payload.
interface MpLandmark { x?: number; y?: number; z?: number; visibility?: number }
export interface PoseJsonLandmark { x: number; y: number; z: number; visibility: number }
export interface PoseJsonFrame {
  frame_index: number;
  landmarks: PoseJsonLandmark[] | null;
  world_landmarks: PoseJsonLandmark[] | null;
}
export interface PoseJson {
  metadata: { fps: number; width: number; height: number; total_frames: number };
  frames: PoseJsonFrame[];
}

const isFiniteCoordinate = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const toPts = (lms?: MpLandmark[]): PoseJsonLandmark[] | null => {
  if (!lms || lms.length < LANDMARK_COUNT) return null;
  if (!lms.every((l) =>
    isFiniteCoordinate(l.x) &&
    isFiniteCoordinate(l.y) &&
    isFiniteCoordinate(l.z) &&
    (l.visibility === undefined || isFiniteCoordinate(l.visibility))
  )) return null;
  return lms.map((l) => ({ x: l.x!, y: l.y!, z: l.z!, visibility: l.visibility ?? 0 }));
};

export function landmarksToFrame(
  frameIndex: number,
  landmarks?: MpLandmark[],
  worldLandmarks?: MpLandmark[]
): PoseJsonFrame {
  return { frame_index: frameIndex, landmarks: toPts(landmarks), world_landmarks: toPts(worldLandmarks) };
}

/* c8 ignore start -- video decode and MediaPipe require browser media/WebGL primitives. */
export async function extractPoseFromBlob(
  blob: Blob,
  tier: PoseTier,
  onProgress?: (p: number) => void
): Promise<PoseJson> {
  const url = URL.createObjectURL(blob);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  const metadataReady = new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Could not decode the video."));
  });
  metadataReady.catch(() => undefined);
  video.src = url;
  const landmarker = await createPoseInferenceRunner(tier);
  const frames: PoseJsonFrame[] = [];
  try {
    await metadataReady;
    const fps = 30;
    const duration = await resolveDuration(video);
    assertPoseAnalysisDuration(duration);
    let i = 0;
    for (let t = 0; t < duration; t += 1 / fps) {
      video.currentTime = t;
      await new Promise<void>((resolve) => { video.onseeked = () => resolve(); });
      const result = await landmarker.detect(video, Math.round(t * 1000));
      frames.push(landmarksToFrame(i, result.landmarks ?? undefined, result.worldLandmarks ?? undefined));
      i += 1;
      onProgress?.(Math.min(1, t / duration));
    }
    onProgress?.(1);
    return {
      metadata: { fps, width: video.videoWidth, height: video.videoHeight, total_frames: frames.length },
      frames,
    };
  } finally {
    landmarker.close();
    URL.revokeObjectURL(url);
  }
}
/* c8 ignore stop */
