// The 1-D signal a movement's repetitions are found in, computed in the browser (RS-SP2 spec §2.6).
//
// PORTED, NOT REINVENTED. Every function here mirrors a specific Python one, because the backend
// trusts the rep windows this signal produces and would never see a disagreement (spec §2.3, §2.7):
//   avgKneeAngle           <- pose_rule_detector.py:143-144,167
//   pushUpMinElbowAngle    <- pushup.py:473,498 (pushup_compute_raw)
//   lungeMinKneeAngle      <- lunge.py:241,263 (lunge_compute_raw)
//   rowMinElbowAngle       <- row.py:167,191 (row_compute_raw)
//   bicepCurlAvgElbowAngle <- bicep_curl.py:110,138 (bicep_curl_compute_raw)
//   sitUpHipAngle          <- situp.py:129,162 (situp_compute_raw)
//   angleDegrees           <- geometry.py:60-73 (angle_degrees, 3-D)
//   visibility gate        <- geometry.py:42-49 (visible_point), threshold 0.50
//   centeredMedian         <- geometry.py:108-119
// The shared fixture pins signal->windows; it does NOT pin that both sides compute the same signal.
// tests/fixtures/rep_signal_golden.json (lib.repSignal.golden.test.ts) does, so change a function
// here only alongside its Python twin, and regenerate that file when you do.

const LANDMARK_COUNT = 33;
/** geometry.py:7 — a landmark at or above this is trusted; below it the point does not exist. */
export const VISIBILITY_THRESHOLD = 0.5;

const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_ELBOW = 13;
const RIGHT_ELBOW = 14;
const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;
const LEFT_KNEE = 25;
const RIGHT_KNEE = 26;
const LEFT_ANKLE = 27;
const RIGHT_ANKLE = 28;
const LEFT_FOOT_INDEX = 31;
const RIGHT_FOOT_INDEX = 32;

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

/** `min(finite)` as the movement modules write it; NaN when neither side was measurable. */
function minFinite(values: number[]): number {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return NaN;
  return Math.min(...finite);
}

/** One side's angle, as `[a, b, c]` with the angle measured at `b`. */
type Joint = [number, number, number];

const KNEES: Joint[] = [[LEFT_HIP, LEFT_KNEE, LEFT_ANKLE], [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]];
const ELBOWS: Joint[] = [
  [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST],
];
const HIPS: Joint[] = [[LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE], [RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE]];

/**
 * The shape every movement's compute_raw shares: if ANY `required` landmark fails the gate the
 * frame is `valid=False` and carries no metrics at all, otherwise the two sides' angles are
 * reduced to one value. The gate is per movement and is not just the joints the angle reads (Push-up
 * requires the ankles, Lunge the toes), so a port that gated on the angle's own landmarks alone
 * would measure frames Python refuses.
 *
 * The gate uses dims=2 (`visible_point(points, index, dims=2)` in every module) so a NaN z passes
 * it, while `angleDegrees` uses dims=3 — such a frame is valid but that side's angle is NaN, and
 * the value comes from the other side only.
 */
function gatedAngles(
  landmarks: SignalLandmark[] | null | undefined,
  required: number[],
  joints: Joint[],
  reduce: (values: number[]) => number
): number {
  if (!landmarks || landmarks.length < LANDMARK_COUNT) return NaN;
  if (required.some((index) => visiblePoint(landmarks, index, 2) === null)) return NaN;
  return reduce(joints.map(([a, b, c]) => angleDegrees(landmarks, a, b, c)));
}

/**
 * The squat rep signal: the mean of the two knee angles.
 *
 * Mirrors raw_frame_metrics' validity rule (pose_rule_detector.py:133-141) — hips, knees and
 * ankles must all be visible or the frame has no metrics at all — and then its
 * `mean_finite([left_knee_angle, right_knee_angle])`.
 */
export function avgKneeAngle(landmarks: SignalLandmark[] | null | undefined): number {
  return gatedAngles(
    landmarks, [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE], KNEES, meanFinite
  );
}

/** Push-up `min_elbow_angle`: the more-flexed arm drives depth. Ankles are required — no plank line
 *  without them — so this is NOT interchangeable with `rowMinElbowAngle`. */
export function pushUpMinElbowAngle(landmarks: SignalLandmark[] | null | undefined): number {
  return gatedAngles(landmarks, [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
  ], ELBOWS, minFinite);
}

/** Lunge `min_knee_angle`: the more-flexed knee. Toes are required for the knee-past-toes metric,
 *  so a frame without them has no rep signal either. */
export function lungeMinKneeAngle(landmarks: SignalLandmark[] | null | undefined): number {
  return gatedAngles(landmarks, [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
  ], KNEES, minFinite);
}

/** Row `min_elbow_angle`. Push-up's formula under a different gate: shoulders to hips, no legs. */
export function rowMinElbowAngle(landmarks: SignalLandmark[] | null | undefined): number {
  return gatedAngles(landmarks, [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
  ], ELBOWS, minFinite);
}

/** Bicep Curl `avg_elbow_angle` — the MEAN, not the min: the arms move in phase. Row's gate. */
export function bicepCurlAvgElbowAngle(landmarks: SignalLandmark[] | null | undefined): number {
  return gatedAngles(landmarks, [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
  ], ELBOWS, meanFinite);
}

/** Sit-up `hip_angle_deg`: the mean shoulder-hip-knee angle, same side throughout. */
export function sitUpHipAngle(landmarks: SignalLandmark[] | null | undefined): number {
  return gatedAngles(landmarks, [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
  ], HIPS, meanFinite);
}

/**
 * Which movements can be segmented in the browser, keyed by the registry's canonical name.
 *
 * A movement ABSENT here is not broken — it takes the whole-clip fallback (spec §4.1,
 * `segmentation_disabled`) and behaves exactly as it does today.
 *
 * EVERY ENTRY MUST USE THE DEFAULT REP KNOBS (polarity "min", no rectify, rep_start "extended",
 * the default min_rep_seconds), because `planReps`/`refineSegments` pass none, and `valleyPosition`
 * anchors spans on an argmin. lib.repSignal.golden.test.ts enforces this against the Python
 * detectors, so a movement like Jumping Jacks (polarity "max") fails there instead of shipping wrong
 * windows; wiring one up means plumbing its knobs through those functions first.
 *
 * The coarse-pass constants (COARSE_STRIDE, REP_PADDING_FRAMES) were measured on squat clips only.
 * The other movements inherit them unmeasured; a rep the padding is too narrow for surfaces as
 * `refined: "clipped"`, which the backend logs.
 */
export const TS_REP_SIGNALS: Record<string, (lm: SignalLandmark[] | null | undefined) => number> = {
  Squat: avgKneeAngle,
  "Push-up": pushUpMinElbowAngle,
  Lunge: lungeMinKneeAngle,
  Row: rowMinElbowAngle,
  "Bicep Curl": bicepCurlAvgElbowAngle,
  "Sit-up": sitUpHipAngle,
};
