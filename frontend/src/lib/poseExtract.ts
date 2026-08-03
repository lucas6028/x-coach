// Client-side pose extraction: decode a recorded/uploaded clip frame-by-frame, run MediaPipe,
// and emit pose JSON byte-compatible with src/pose/process_videos.py so the backend detector is
// untouched. The pure serializer (landmarksToFrame) is unit-tested; the <video>/rVFC/WASM glue
// in extractPoseFromBlob is impure and coverage-excluded like the other detector boundaries.
import type { PoseTier } from "./poseTier";
import { createPoseInferenceRunner } from "./poseInference";

const LANDMARK_COUNT = 33;

// The task API normally supplies every coordinate, but worker structured-clone results from
// some WebViews can omit optional fields. Never serialize a partial landmark: JSON.stringify
// would drop that key and the server correctly rejects the whole pose payload as malformed.
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

  // Treat a damaged MediaPipe result as a no-pose frame. This is safe for the detector and
  // guarantees every non-null frame conforms to the API's x/y/z/visibility contract.
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

// WHY THIS EXISTS. A MediaRecorder muxes its WebM as a LIVE stream: the Segment has unknown size
// and the Info element carries only TimecodeScale — no Duration — with no Cues index. Verified by
// parsing the clips this path actually produced (data/runtime/uploads/*.webm). The browser
// therefore cannot report a length, and `video.duration` comes back NaN (observed) or Infinity for
// a RECORDED clip, while an UPLOADED file reports a real number.
//
// `extractPoseFromBlob` bounds its sampling loop by that value, so every live recording extracted
// ZERO frames and the app told the user "no frame in this clip could be measured" — a
// verdict-shaped message for what was really a container quirk.
//
// The remedy is the standard one: seek far past any plausible end. The browser clamps the seek to
// the true end of the media and fires `durationchange` carrying the recovered duration.
const SEEK_PROBE = 1e101;

/** The slice of HTMLVideoElement the duration probe touches. Narrow on purpose: jsdom has no
 *  decoder, so a full <video> cannot be exercised in tests, and this protocol is precisely where
 *  the live-record bug lived — it needs to be testable. */
export interface DurationProbe {
  duration: number;
  currentTime: number;
  addEventListener(type: string, listener: () => void): void;
  removeEventListener(type: string, listener: () => void): void;
}

/**
 * Resolve a usable clip length, recovering it from the media itself when the container omits one.
 *
 * Rejects rather than returning 0 when the length never arrives: a 0 would flow into the sampling
 * loop as "no frames", and the app renders an empty frame list as a *form verdict* ("nothing could
 * be measured") rather than a failure. Reporting a decode problem as a coaching result is the exact
 * failure this codebase treats as unacceptable, so an honest error wins.
 */
export function resolveDuration(video: DurationProbe, timeoutMs = 5000): Promise<number> {
  // A well-formed upload already knows its length; probing it would be a pointless seek on the one
  // path that has no bug.
  if (Number.isFinite(video.duration) && video.duration > 0) return Promise.resolve(video.duration);

  return new Promise<number>((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener("durationchange", check);
      video.removeEventListener("seeked", check);
    };
    function check() {
      // `durationchange` can fire while the length is still unknown — keep waiting until it is real.
      if (settled || !Number.isFinite(video.duration) || video.duration <= 0) return;
      settled = true;
      cleanup();
      video.currentTime = 0; // rewind: the caller samples forward from the start
      resolve(video.duration);
    }
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      video.currentTime = 0;
      reject(new Error("Could not read the clip's length."));
    }, timeoutMs);
    // Attached BEFORE the seek: the browser may answer synchronously, and a listener registered
    // afterwards would miss the only event it will ever get.
    video.addEventListener("durationchange", check);
    // Secondary signal only. A recorded clip has no Cues index, so `seeked` firing is not something
    // this may depend on.
    video.addEventListener("seeked", check);
    video.currentTime = SEEK_PROBE;
  });
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
  // Handlers attached BEFORE `src` is assigned, and before the model download below. `loadedmetadata`
  // fires once and is not replayed: registering it after an intervening await races the blob's own
  // load, and losing that race wedges the extraction forever.
  const metadataReady = new Promise<void>((res, rej) => {
    video.onloadedmetadata = () => res();
    video.onerror = () => rej(new Error("Could not decode the video."));
  });
  // Keep an early decode failure from surfacing as an unhandled rejection while the model loads;
  // `await metadataReady` below still sees it.
  metadataReady.catch(() => undefined);
  video.src = url;
  const landmarker = await createPoseInferenceRunner(tier);
  const frames: PoseJsonFrame[] = [];
  try {
    await metadataReady;
    const fps = 30;
    // NOT `video.duration || 0` — a live-recorded clip reports no length and that silently sampled
    // nothing. See resolveDuration.
    const duration = await resolveDuration(video);
    let i = 0;
    // Seek-and-detect: step through the clip at a fixed cadence so frame_index is deterministic
    // and aligned to the stored video (rVFC live-rate would drift on drops).
    for (let t = 0; t < duration; t += 1 / fps) {
      video.currentTime = t;
      await new Promise<void>((r) => { video.onseeked = () => r(); });
      const result = await landmarker.detect(video, Math.round(t * 1000));
      frames.push(landmarksToFrame(i, result.landmarks ?? undefined, result.worldLandmarks ?? undefined));
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
