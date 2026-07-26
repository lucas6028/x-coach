// Client-side pose extraction: decode a recorded/uploaded clip frame-by-frame, run MediaPipe,
// and emit pose JSON byte-compatible with src/pose/process_videos.py so the backend detector is
// untouched. The pure serializer (landmarksToFrame) is unit-tested; the <video>/rVFC/WASM glue
// in extractPoseFromBlob is impure and coverage-excluded like the other detector boundaries.
import { createPoseLandmarker } from "../components/poseLandmarker";
import type { PoseTier } from "./poseTier";

const LANDMARK_COUNT = 33;

interface MpLandmark { x: number; y: number; z: number; visibility?: number }
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

const toPts = (lms?: MpLandmark[]): PoseJsonLandmark[] | null =>
  lms && lms.length >= LANDMARK_COUNT
    ? lms.map((l) => ({ x: l.x, y: l.y, z: l.z, visibility: l.visibility ?? 0 }))
    : null;

export function landmarksToFrame(
  frameIndex: number,
  landmarks?: MpLandmark[],
  worldLandmarks?: MpLandmark[]
): PoseJsonFrame {
  return { frame_index: frameIndex, landmarks: toPts(landmarks), world_landmarks: toPts(worldLandmarks) };
}

/* c8 ignore start — <video>/requestVideoFrameCallback/WASM glue, unrunnable under jsdom */
export async function extractPoseFromBlob(
  blob: Blob,
  tier: PoseTier,
  onProgress?: (p: number) => void
): Promise<PoseJson> {
  const url = URL.createObjectURL(blob);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.src = url;
  const landmarker = await createPoseLandmarker(tier);
  const frames: PoseJsonFrame[] = [];
  try {
    await new Promise<void>((res, rej) => {
      video.onloadedmetadata = () => res();
      video.onerror = () => rej(new Error("Could not decode the video."));
    });
    const fps = 30;
    const duration = video.duration || 0;
    let i = 0;
    // Seek-and-detect: step through the clip at a fixed cadence so frame_index is deterministic
    // and aligned to the stored video (rVFC live-rate would drift on drops).
    for (let t = 0; t < duration; t += 1 / fps) {
      video.currentTime = t;
      await new Promise<void>((r) => { video.onseeked = () => r(); });
      const result = landmarker.detectForVideo(video, Math.round(t * 1000));
      frames.push(landmarksToFrame(i, result.landmarks?.[0], result.worldLandmarks?.[0]));
      i += 1;
      onProgress?.(duration ? Math.min(1, t / duration) : 1);
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
