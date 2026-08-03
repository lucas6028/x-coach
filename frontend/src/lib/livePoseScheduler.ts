// A display can tick at 60/120 Hz while its camera produces 30 fps. Running synchronous pose
// inference on every render tick wastes GPU time and creates main-thread contention. This helper
// admits only fresh frames at the requested maximum cadence.

export interface LivePoseSchedule {
  lastVideoTime: number;
  lastInferenceAt: number;
}

export const LIVE_POSE_FPS = 30;

export function createLivePoseSchedule(): LivePoseSchedule {
  return { lastVideoTime: -1, lastInferenceAt: -Infinity };
}

export function shouldRunLivePoseInference(
  schedule: LivePoseSchedule,
  videoTime: number,
  now: number,
  targetFps = LIVE_POSE_FPS
): boolean {
  // `currentTime` can legitimately jump backwards when a media element restarts. Only an
  // already-inferred frame is stale; a frame rejected by the cadence budget remains eligible.
  if (!Number.isFinite(videoTime) || videoTime === schedule.lastVideoTime) return false;
  if (now - schedule.lastInferenceAt < 1000 / Math.max(1, targetFps)) return false;
  schedule.lastVideoTime = videoTime;
  schedule.lastInferenceAt = now;
  return true;
}
