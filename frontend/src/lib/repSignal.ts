// The 1-D signal a movement's repetitions are found in, computed in the browser (RS-SP2 spec §2.6).
//
// PORTED, NOT REINVENTED. Every function here mirrors a specific Python one, because the backend
// trusts the rep windows this signal produces and would never see a disagreement (spec §2.3, §2.7):
//   avgKneeAngle   <- pose_rule_detector.py:143-144,167 + geometry.py:60-73 (angle_degrees, 3-D)
//   visibility gate<- geometry.py:42-49 (visible_point), threshold 0.50
//   centeredMedian <- geometry.py:108-119
// The shared fixture pins signal->windows; it does NOT pin that both sides compute the same signal.
// This file is the only thing that does, so change it only alongside its Python twin.

const LANDMARK_COUNT = 33;
/** geometry.py:7 — a landmark at or above this is trusted; below it the point does not exist. */
export const VISIBILITY_THRESHOLD = 0.5;

const LEFT_HIP = 23;
const RIGHT_HIP = 24;
const LEFT_KNEE = 25;
const RIGHT_KNEE = 26;
const LEFT_ANKLE = 27;
const RIGHT_ANKLE = 28;

export interface SignalLandmark { x: number; y: number; z: number; visibility: number }

type Point = [number, number, number];

/**
 * geometry.py:42-49. Returns null for an absent, non-finite, or insufficiently visible point.
 *
 * Only the first `dims` coordinates are checked. The validity gate (pose_rule_detector.py:134)
 * uses dims=2 to allow frames where z is missing; angle_degrees uses dims=3 for full 3-D
 * calculation. This separation is critical: a frame may be valid for segmentation but have
 * one ankle's z as NaN, and that frame's metric must come from the other side only.
 */
function visiblePoint(lms: SignalLandmark[], index: number, dims: number = 3): Point | null {
  const lm = lms[index];
  if (!lm) return null;
  const { x, y, z, visibility } = lm;
  // Only check the first `dims` coordinates plus visibility.
  const coords = [x, y, z].slice(0, dims);
  if (![...coords, visibility].every(Number.isFinite)) return null;
  if (visibility < VISIBILITY_THRESHOLD) return null;
  return [x, y, z];
}

/** geometry.py:60-73. The angle at `b` in degrees, in 3-D. NaN when any point is unusable. */
function angleDegrees(lms: SignalLandmark[], a: number, b: number, c: number): number {
  const pa = visiblePoint(lms, a);
  const pb = visiblePoint(lms, b);
  const pc = visiblePoint(lms, c);
  if (!pa || !pb || !pc) return NaN;
  const ba: Point = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
  const bc: Point = [pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]];
  const norm = (v: Point) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const denominator = norm(ba) * norm(bc);
  if (denominator <= 1e-8) return NaN;
  const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
  const cosine = Math.min(1, Math.max(-1, dot / denominator));
  return (Math.acos(cosine) * 180) / Math.PI;
}

/** geometry.py:101-105. Mean of the finite entries; NaN when there are none. */
function meanFinite(values: number[]): number {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return NaN;
  return finite.reduce((sum, v) => sum + v, 0) / finite.length;
}

/**
 * geometry.py:108-119. Median over a centred window, NON-FINITE ENTRIES SKIPPED.
 *
 * Skipping rather than propagating is what lets a padded span be smoothed at all: RS-SP2 leaves
 * holes in the frame sequence, and a NaN-propagating median would spread each hole by the window
 * radius. The window simply shrinks instead.
 */
export function centeredMedian(values: number[], window: number): number[] {
  if (values.length === 0) return [];
  const radius = Math.max(0, Math.floor(window / 2));
  return values.map((_, index) => {
    const start = Math.max(0, index - radius);
    const end = Math.min(values.length, index + radius + 1);
    const finite = values.slice(start, end).filter(Number.isFinite).sort((a, b) => a - b);
    if (finite.length === 0) return NaN;
    const mid = Math.floor(finite.length / 2);
    return finite.length % 2 === 1 ? finite[mid] : (finite[mid - 1] + finite[mid]) / 2;
  });
}

/**
 * The squat rep signal: the mean of the two knee angles.
 *
 * Mirrors raw_frame_metrics' validity rule (pose_rule_detector.py:133-141) — hips, knees and
 * ankles must all be visible or the frame has no metrics at all — and then its
 * `mean_finite([left_knee_angle, right_knee_angle])`.
 */
export function avgKneeAngle(landmarks: SignalLandmark[] | null | undefined): number {
  if (!landmarks || landmarks.length < LANDMARK_COUNT) return NaN;
  const required = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE];
  // Validity gate uses dims=2 (pose_rule_detector.py:134) to allow NaN z.
  if (required.some((index) => visiblePoint(landmarks, index, 2) === null)) return NaN;
  return meanFinite([
    angleDegrees(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    angleDegrees(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
  ]);
}

/**
 * Which movements can be segmented in the browser, keyed by the registry's canonical name.
 *
 * A movement ABSENT here is not broken — it takes the whole-clip fallback (spec §4.1,
 * `segmentation_disabled`) and behaves exactly as it does today. That is why SP2 can ship with
 * only Squat without blocking any other movement.
 */
export const TS_REP_SIGNALS: Record<string, (lm: SignalLandmark[] | null | undefined) => number> = {
  Squat: avgKneeAngle,
};
