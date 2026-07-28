// Client-side pose extraction: decode a recorded/uploaded clip frame-by-frame, run MediaPipe,
// and emit pose JSON byte-compatible with src/pose/process_videos.py so the backend detector is
// untouched. The pure serializer (landmarksToFrame) is unit-tested; the <video>/rVFC/WASM glue
// in extractPoseFromBlob is impure and coverage-excluded like the other detector boundaries.
import { createPoseLandmarker } from "../components/poseLandmarker";
import type { PoseTier } from "./poseTier";
import { LIVE_OVERLAY_TIER } from "./poseTier";
import { TS_REP_SIGNALS, centeredMedian, type SignalLandmark } from "./repSignal";
import { segmentReps, selectReps } from "./repSegmentation";
import {
  COARSE_SMOOTH_WINDOW, COARSE_STRIDE, CANONICAL_FPS, coarseBand, frameIndexAt, mergeSpans,
  refineWindow, spanForRep, spanFrameIndices, type FrameSpan, type Refinement,
} from "./repSpans";

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

export type RepsFallback =
  | "no_reps_detected" | "only_partial_reps" | "segmentation_disabled" | null;

export interface RepSegment {
  index: number;
  start_frame: number;
  end_frame: number;
  partial: boolean;
  analyzed: boolean;
  refined: Refinement;
}

export interface RepsPlan {
  max_reps: number;
  fallback: RepsFallback;
  segments: RepSegment[];
}

/**
 * Decide which frames to extract densely (spec §2.1, §4.1).
 *
 * EVERY fallback returns the WHOLE clip as one span. Sending a sparse frame list with no windows
 * would leave the backend to segment data that does not exist, and reporting a segmentation
 * failure as "no faults found" is the failure mode this codebase refuses (see resolveDuration's
 * comment below for the same rule applied to decoding). Not saving time is the correct trade.
 */
export function planReps(
  coarseSignal: number[], maxReps: number, lastFrameIndex: number, movement: string
): { plan: RepsPlan; spans: FrameSpan[] } {
  const wholeClip: FrameSpan[] = [{ start: 0, end: lastFrameIndex }];
  const fallbackPlan = (fallback: RepsFallback, segments: RepSegment[] = []) =>
    ({ plan: { max_reps: maxReps, fallback, segments }, spans: wholeClip });

  if (!(movement in TS_REP_SIGNALS)) return fallbackPlan("segmentation_disabled");

  const smoothed = centeredMedian(coarseSignal, COARSE_SMOOTH_WINDOW);
  const reps = segmentReps(smoothed, { fps: CANONICAL_FPS / COARSE_STRIDE });
  if (reps.length === 0) return fallbackPlan("no_reps_detected");
  if (reps.every((rep) => rep.partial)) {
    // A tightly-trimmed single-rep clip looks like this; analysing it whole is correct for it.
    return fallbackPlan("only_partial_reps");
  }

  const analyzed = new Set(selectReps(reps, maxReps).map((rep) => rep.index));
  const segments: RepSegment[] = reps.map((rep) => ({
    index: rep.index,
    start_frame: rep.start * COARSE_STRIDE,
    end_frame: Math.min(lastFrameIndex, rep.end * COARSE_STRIDE),
    partial: rep.partial,
    analyzed: analyzed.has(rep.index),
    refined: false, // upgraded by refineSegments once the dense signal exists
  }));

  const spans = mergeSpans(
    reps.filter((rep) => analyzed.has(rep.index))
        .map((rep) => spanForRep(smoothed, rep, lastFrameIndex))
  );
  return { plan: { max_reps: maxReps, fallback: null, segments }, spans };
}

/**
 * Replace each analyzed segment's coarse boundary with the one the dense signal gives (§2.1.1).
 *
 * `coarseSignal` is here for its BAND, not its samples: a span holds about one repetition's worth
 * of samples, and percentiles taken over that narrow a slice shift the hysteresis band and move
 * the boundary with it. Measured on 46 clips — per-span percentiles refine 92.9% of reps exactly
 * (p95 15.3 frames, max 46); the same spans given the coarse pass's whole-clip range refine 98.6%
 * exactly (p95 0, max 1). The coarse pass covers the whole clip, which is why that range exists to
 * be handed over at all.
 */
export function refineSegments(
  plan: RepsPlan, spans: FrameSpan[], denseSignal: (number | undefined)[],
  lastFrameIndex: number, coarseSignal: number[]
): RepsPlan {
  if (plan.fallback !== null) return plan;
  // Squat's defaults; a movement with rep knobs would pass its own, and coarseBand orients
  // internally so the band lands in the same space segmentReps compares against.
  const band = coarseBand(coarseSignal);
  const segments = plan.segments.map((segment) => {
    if (!segment.analyzed) return segment;
    const coarse = { start: segment.start_frame, end: segment.end_frame };
    const span = spans.find((s) => s.start <= coarse.start && coarse.end <= s.end)
      ?? spans.find((s) => s.start <= coarse.end && coarse.start <= s.end);
    if (!span) return segment;
    const { start, end, refined } =
      refineWindow(denseSignal, span, coarse, CANONICAL_FPS, lastFrameIndex, band);
    return { ...segment, start_frame: start, end_frame: end, refined };
  });
  return { ...plan, segments };
}

/* c8 ignore start — <video>/requestVideoFrameCallback/WASM glue, unrunnable under jsdom */

/** Seek to each frame_index in turn and run the landmarker. Shared by both passes. */
async function sampleFrames(
  video: HTMLVideoElement,
  landmarker: { detectForVideo(v: HTMLVideoElement, t: number): {
    landmarks?: MpLandmark[][]; worldLandmarks?: MpLandmark[][] } },
  frameIndices: number[],
  onProgress?: (p: number) => void
): Promise<PoseJsonFrame[]> {
  const out: PoseJsonFrame[] = [];
  for (let n = 0; n < frameIndices.length; n += 1) {
    const index = frameIndices[n];
    const t = index / CANONICAL_FPS;
    video.currentTime = t;
    await new Promise<void>((r) => { video.onseeked = () => r(); });
    const result = landmarker.detectForVideo(video, Math.round(t * 1000));
    out.push(landmarksToFrame(index, result.landmarks?.[0], result.worldLandmarks?.[0]));
    onProgress?.((n + 1) / frameIndices.length);
  }
  return out;
}

/**
 * Two-pass extraction: find the reps cheaply, then measure only the selected ones (spec §2.1).
 *
 * Two passes are FORCED, not chosen: selectReps takes first/middle/last, so the total rep count
 * must be known before selecting, and no single streaming pass can know it. "Just take the first
 * three" is the failure SP1 rejected by name — fatigue breakdown shows up in the LAST rep.
 *
 * The returned pose JSON is FULL LENGTH with `landmarks: null` outside the extracted spans, which
 * keeps RepWindow positions equal to frame_index and frame_metrics one row per frame (spec §2.2).
 */
export async function extractPoseWithReps(
  blob: Blob,
  tier: PoseTier,
  movement: string,
  maxReps: number,
  onProgress?: (p: number) => void
): Promise<{ pose: PoseJson; reps: RepsPlan }> {
  const url = URL.createObjectURL(blob);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  const metadataReady = new Promise<void>((res, rej) => {
    video.onloadedmetadata = () => res();
    video.onerror = () => rej(new Error("Could not decode the video."));
  });
  metadataReady.catch(() => undefined);
  video.src = url;

  const coarseLandmarker = await createPoseLandmarker(LIVE_OVERLAY_TIER);
  try {
    await metadataReady;
    const duration = await resolveDuration(video);
    const lastFrameIndex = Math.max(0, frameIndexAt(duration) - 1);

    // Pass 1 — coarse. Lite, every COARSE_STRIDE-th frame, only to locate repetitions.
    const coarseIndices: number[] = [];
    for (let i = 0; i <= lastFrameIndex; i += COARSE_STRIDE) coarseIndices.push(i);
    const coarseFrames = await sampleFrames(video, coarseLandmarker, coarseIndices,
      (p) => onProgress?.(p * 0.3));
    const signal = TS_REP_SIGNALS[movement];
    const coarseSignal = coarseFrames.map((f) =>
      signal ? signal(f.landmarks as SignalLandmark[] | null) : NaN);
    const { plan, spans } = planReps(coarseSignal, maxReps, lastFrameIndex, movement);

    // Pass 2 — dense, at the user's tier, over the padded spans only.
    const denseIndices = spanFrameIndices(spans);
    const denseLandmarker = tier === LIVE_OVERLAY_TIER
      ? coarseLandmarker
      : await createPoseLandmarker(tier);
    let denseFrames: PoseJsonFrame[];
    try {
      denseFrames = await sampleFrames(video, denseLandmarker, denseIndices,
        (p) => onProgress?.(0.3 + p * 0.7));
    } finally {
      if (denseLandmarker !== coarseLandmarker) denseLandmarker.close();
    }

    // Full-length frame list: extracted frames in place, `null` landmarks everywhere else.
    const byIndex = new Map(denseFrames.map((f) => [f.frame_index, f]));
    const frames: PoseJsonFrame[] = [];
    for (let i = 0; i <= lastFrameIndex; i += 1) {
      frames.push(byIndex.get(i) ?? landmarksToFrame(i, undefined, undefined));
    }

    const denseSignal: (number | undefined)[] = new Array(lastFrameIndex + 1).fill(undefined);
    if (signal) {
      for (const frame of denseFrames) {
        denseSignal[frame.frame_index] = signal(frame.landmarks as SignalLandmark[] | null);
      }
    }
    onProgress?.(1);
    return {
      pose: {
        metadata: {
          fps: CANONICAL_FPS, width: video.videoWidth, height: video.videoHeight,
          total_frames: frames.length,
        },
        frames,
      },
      reps: refineSegments(plan, spans, denseSignal, lastFrameIndex, coarseSignal),
    };
  } finally {
    coarseLandmarker.close();
    URL.revokeObjectURL(url);
  }
}

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
  const landmarker = await createPoseLandmarker(tier);
  const frames: PoseJsonFrame[] = [];
  try {
    await metadataReady;
    const fps = CANONICAL_FPS;
    // NOT `video.duration || 0` — a live-recorded clip reports no length and that silently sampled
    // nothing. See resolveDuration.
    const duration = await resolveDuration(video);
    // Seek-and-detect: step through the clip at a fixed cadence so frame_index is deterministic
    // and aligned to the stored video (rVFC live-rate would drift on drops). The index comes from
    // the TIMESTAMP, not a counter — the coarse pass steps differently and must agree. See
    // repSpans.frameIndexAt.
    for (let t = 0; t < duration; t += 1 / fps) {
      video.currentTime = t;
      await new Promise<void>((r) => { video.onseeked = () => r(); });
      const result = landmarker.detectForVideo(video, Math.round(t * 1000));
      frames.push(landmarksToFrame(frameIndexAt(t), result.landmarks?.[0], result.worldLandmarks?.[0]));
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
