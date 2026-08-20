// Split a movement clip into repetitions from a single 1-D metric series.
//
// A LINE-BY-LINE PORT of src/pose/rep_segmentation.py, which is the authority: read its module
// docstring for why there is deliberately NO noise-vs-range gate (four attempts at one each
// false-rejected ordinary training signal), and why _climbBackward's strict `>` is load-bearing.
// tests/fixtures/rep_segmentation_cases.json pins both implementations to the same outputs.
//
// TWO PORTING TRAPS, both measured, both covered by tests:
//   - numpy's percentile interpolates linearly (np.percentile([1,2,3,4], 5) === 1.15). Picking the
//     k-th element instead shifts the hysteresis band and moves every boundary.
//   - Python's int(round(x)) is banker's rounding. Math.round is not, and the two disagree on
//     6- and 10-rep clips, which would make the languages analyse different repetitions.

/** Robust bounds for the signal's dynamic range, so one bad frame cannot define it. */
export const PERCENTILE_LOW = 5;
export const PERCENTILE_HIGH = 95;
/** Hysteresis band, as fractions of the dynamic range from the effort-peak end. */
export const ENTER_FRACTION = 0.35;
export const EXIT_FRACTION = 0.65;
/** Floor on repetition duration — the ONLY thing separating a real excursion from a blip. */
export const DEFAULT_MIN_REP_SECONDS = 0.4;

const POLARITIES = ["min", "max"] as const;
const REP_STARTS = ["extended", "flexed"] as const;
export type Polarity = (typeof POLARITIES)[number];
export type RepStart = (typeof REP_STARTS)[number];

export interface RepWindow {
  /** 1-based: it is what a user is told ("your 3rd rep"). */
  index: number;
  /** Inclusive POSITION IN THE PASSED SEQUENCE, not a frame_index. */
  start: number;
  end: number;
  partial: boolean;
}

export interface SegmentOptions {
  fps: number;
  polarity?: Polarity;
  rectify?: boolean;
  repStart?: RepStart;
  minRepSeconds?: number;
  /** In ORIENTED space -- i.e. already run through `oriented(signal, polarity, rectify)` below,
   *  not the caller's raw signal. `coarseBand` in repSpans.ts orients internally, using the same
   *  `oriented()` exported from this module, so a `{ polarity, rectify }` matching the ones
   *  passed here is what keeps the two in the same space -- there is no raw-space variant to
   *  misuse. */
  band?: { low: number; high: number };
}

/** numpy.percentile's default linear interpolation, over values that are already sorted. */
export function percentile(sorted: number[], p: number): number {
  if (sorted.length === 1) return sorted[0];
  const position = ((sorted.length - 1) * p) / 100;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

/** Python's round(): ties go to the even integer, unlike Math.round which always goes up. */
function roundHalfToEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Normalise any movement's signal to the convention "the effort peak is a LOW value". */
export function oriented(signal: number[], polarity: Polarity, rectify: boolean): number[] {
  return signal.map((value) => {
    // A bipolar signal (torso twist: centre -> A -> centre -> B) has two excursions in opposite
    // directions. Rectifying makes each swing its own excursion from zero.
    const v = rectify ? Math.abs(value) : value;
    return polarity === "max" ? -v : v;
  });
}

/** Maximal inclusive runs at/below `threshold`, skipping non-finite samples. */
function runsAtOrBelow(values: number[], threshold: number): [number, number][] {
  const runs: [number, number][] = [];
  let start: number | null = null;
  values.forEach((value, index) => {
    if (!Number.isFinite(value)) return; // an unmeasurable frame neither opens nor closes a run
    if (value <= threshold && start === null) start = index;
    else if (value > threshold && start !== null) {
      runs.push([start, index - 1]);
      start = null;
    }
  });
  if (start !== null) runs.push([start, values.length - 1]);
  return runs;
}

function lastAtOrAbove(values: number[], threshold: number, before: number): number | null {
  for (let i = before - 1; i >= 0; i -= 1) {
    if (Number.isFinite(values[i]) && values[i] >= threshold) return i;
  }
  return null;
}

function firstAtOrAbove(values: number[], threshold: number, after: number): number | null {
  for (let i = after + 1; i < values.length; i += 1) {
    if (Number.isFinite(values[i]) && values[i] >= threshold) return i;
  }
  return null;
}

/**
 * Walk back from an exit crossing to the top of the excursion, STOPPING AT A PLATEAU.
 *
 * The strict `>` is what makes a window's length equal the excursion's length rather than the
 * clip's, which in turn is what lets `finalize`'s min-frames filter reject a blip on duration
 * alone. See the Python docstring for the full argument.
 */
function climbBackward(values: number[], index: number): number {
  let i = index;
  while (i > 0 && Number.isFinite(values[i - 1]) && values[i - 1] > values[i]) i -= 1;
  return i;
}

function climbForward(values: number[], index: number): number {
  let i = index;
  const last = values.length - 1;
  while (i < last && Number.isFinite(values[i + 1]) && values[i + 1] > values[i]) i += 1;
  return i;
}

/** The full extent of the excursion a deep run belongs to: top, through the bottom, to top. */
function excursionBounds(
  values: number[], deepStart: number, deepEnd: number, exit: number
): [number, number, boolean] {
  const before = lastAtOrAbove(values, exit, deepStart);
  const after = firstAtOrAbove(values, exit, deepEnd);
  const start = before === null ? 0 : climbBackward(values, before);
  const end = after === null ? values.length - 1 : climbForward(values, after);
  return [start, end, before === null || after === null];
}

/** Boundaries at the EXTENDED end: a rep runs standing -> bottom -> standing. */
function windowsFromPlateaus(
  values: number[], deepRuns: [number, number][], exit: number, minFrames: number
): RepWindow[] {
  return finalize(deepRuns.map(([s, e]) => excursionBounds(values, s, e, exit)), minFrames);
}

/**
 * Boundaries at the FLEXED end: a rep runs floor -> lockout -> floor (deadlift).
 *
 * Filters each deep run on the duration of the excursion it belongs to, because a valley-to-valley
 * window's length is the rep PERIOD, not any one excursion — without this the flexed path has no
 * anomaly rejection at all. See the Python docstring for the boundary-run trade this accepts.
 */
function windowsFromValleys(
  values: number[], deepRuns: [number, number][], exit: number, minFrames: number
): RepWindow[] {
  const realRuns = deepRuns.filter(([s, e]) => {
    const [start, end] = excursionBounds(values, s, e, exit);
    return end - start + 1 >= minFrames;
  });
  if (realRuns.length === 0) return [];

  const valleys = realRuns.map(([s, e]) => {
    let best = s;
    let bestValue = Infinity;
    for (let i = s; i <= e; i += 1) {
      if (Number.isFinite(values[i]) && values[i] < bestValue) { bestValue = values[i]; best = i; }
    }
    return best;
  });

  const spans: [number, number, boolean][] = [];
  if (valleys[0] > 0) spans.push([0, valleys[0] - 1, true]);
  for (let i = 0; i + 1 < valleys.length; i += 1) spans.push([valleys[i], valleys[i + 1] - 1, false]);
  if (valleys[valleys.length - 1] < values.length - 1) {
    spans.push([valleys[valleys.length - 1], values.length - 1, true]);
  }
  return finalize(spans, minFrames);
}

/** De-duplicate, resolve shared boundaries, drop too-short spans, and number the rest. */
function finalize(spans: [number, number, boolean][], minFrames: number): RepWindow[] {
  const sorted = [...spans].sort((a, b) =>
    a[0] - b[0] || a[1] - b[1] || Number(a[2]) - Number(b[2]));

  const unique: [number, number, boolean][] = [];
  const seen = new Set<string>();
  for (const span of sorted) {
    const key = `${span[0]}:${span[1]}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(span);
  }

  const windows: RepWindow[] = [];
  unique.forEach(([start, rawEnd, partial], position) => {
    // Adjacent reps meet at a single frame — the peak between them belongs to the rep that STARTS
    // there — so the earlier window gives it up, or one frame is phased and scored twice.
    const end = position + 1 < unique.length ? Math.min(rawEnd, unique[position + 1][0] - 1) : rawEnd;
    if (end - start + 1 < minFrames) return;
    windows.push({ index: windows.length + 1, start, end, partial });
  });
  return windows;
}

/**
 * Segment `signal` into repetitions.
 *
 * Returns `[]` — never a guess — when the signal carries no repetition structure. The caller is
 * required to fall back to whole-clip analysis in that case, NOT to report no faults.
 */
export function segmentReps(signal: number[], options: SegmentOptions): RepWindow[] {
  const polarity = options.polarity ?? "min";
  const repStart = options.repStart ?? "extended";
  const minRepSeconds = options.minRepSeconds ?? DEFAULT_MIN_REP_SECONDS;
  if (!POLARITIES.includes(polarity)) throw new Error(`polarity must be min or max, got ${polarity}`);
  if (!REP_STARTS.includes(repStart)) throw new Error(`repStart must be extended or flexed, got ${repStart}`);

  const values = oriented(signal, polarity, options.rectify ?? false);
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  const minFrames = Math.max(3, roundHalfToEven(minRepSeconds * Math.max(options.fps, 1)));
  if (finite.length < 2 * minFrames) return [];

  // The band may be supplied by the caller. RS-SP2 refines a rep's boundary inside a padded span,
  // and a span's OWN percentiles are computed over one repetition's worth of samples -- narrow
  // enough that the hysteresis band shifts and the boundary moves with it. Measured on 46 real
  // clips: per-span percentiles refine 92.9% of reps exactly (p95 15.3 frames, max 46), while the
  // same spans given the whole clip's range refine 98.6% exactly (p95 0, max 1). The caller has a
  // whole-clip range available because the coarse pass covers the whole clip. See the SP2 spec
  // §2.1.1, and `coarseBand` in repSpans.ts.
  const low = options.band ? options.band.low : percentile(finite, PERCENTILE_LOW);
  const high = options.band ? options.band.high : percentile(finite, PERCENTILE_HIGH);
  const span = high - low;
  if (span <= 0) return [];

  const enter = low + ENTER_FRACTION * span;
  const exit = low + EXIT_FRACTION * span;
  const deepRuns = runsAtOrBelow(values, enter);
  if (deepRuns.length === 0) return [];

  return repStart === "flexed"
    ? windowsFromValleys(values, deepRuns, exit, minFrames)
    : windowsFromPlateaus(values, deepRuns, exit, minFrames);
}

/**
 * Choose which repetitions to actually analyze: first / middle / last.
 *
 * Not "the first N": the first rep carries warm-up errors, the middle one steady state, the last
 * one fatigue breakdown, and sampling only the middle systematically hides the fault a lifter most
 * needs told. Partial reps are skipped when complete ones exist, but kept when they are all there
 * is — analyzing a truncated rep beats analyzing nothing.
 */
export function selectReps(reps: RepWindow[], maxReps: number | null): RepWindow[] {
  const complete = reps.filter((rep) => !rep.partial);
  const candidates = complete.length > 0 ? complete : [...reps];
  if (candidates.length === 0) return [];
  if (!maxReps || maxReps <= 0 || candidates.length <= maxReps) return candidates;

  // np.linspace(start, stop, num) with num === 1 returns [start], not a division by (num - 1);
  // that division only makes sense for spacing out 2+ points. Guarding it here mirrors that
  // numpy behaviour directly instead of letting `(last * i) / (maxReps - 1)` divide by zero at
  // i === 0, which would produce NaN and silently smuggle `candidates[NaN] === undefined` into
  // the result. Python's `select_reps(reps, max_reps=1)` returns `[candidates[0]]`.
  if (maxReps === 1) return [candidates[0]];

  const last = candidates.length - 1;
  const positions = new Set<number>();
  for (let i = 0; i < maxReps; i += 1) {
    positions.add(roundHalfToEven((last * i) / (maxReps - 1)));
  }
  return [...positions].sort((a, b) => a - b).map((position) => candidates[position]);
}
