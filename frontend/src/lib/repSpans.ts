// Frame bookkeeping shared by the coarse and dense extraction passes (RS-SP2 spec §2.5).
//
// WHY A FUNCTION AND NOT A COUNTER. poseExtract used to number frames with an incrementing
// counter, which equals round(t * 30) only when the sampling step happens to be 1/30. The coarse
// pass steps by 3/30, so a counter would number its samples 0,1,2,… while the video's frames are
// 0,3,6,… — every rep window derived from it would land in the wrong index space, silently. Both
// passes now derive the index from the TIMESTAMP, so they share one coordinate system.

import { centeredMedian } from "./repSignal";
import {
  PERCENTILE_HIGH,
  PERCENTILE_LOW,
  oriented,
  percentile as percentileOf,
  segmentReps,
  type Polarity,
  type RepWindow,
} from "./repSegmentation";

/** The grid every frame_index is expressed on, matching poseExtract's fixed sampling cadence. */
export const CANONICAL_FPS = 30;

/** The frame_index of the sample at `t` seconds. */
export function frameIndexAt(t: number): number {
  return Math.round(t * CANONICAL_FPS);
}

/** Coarse pass samples every Nth frame of the canonical grid, so sample k IS frame k*N. */
export const COARSE_STRIDE = 3;
/** Smoothing windows. Aligned by TIME, not frame count: SP1's 5 frames at 30fps is 0.17s, and
 *  reusing 5 at the coarse 10fps would smooth over 0.5s and flatten a squat's bottom. */
export const DENSE_SMOOTH_WINDOW = 5;
export const COARSE_SMOOTH_WINDOW = 3;

/**
 * How far past the coarse window's own half-width a span must reach (spec §2.8).
 *
 * MEASURED, not chosen: across 46 real squat pose JSONs (70 reps), a span of
 * `valley +/- (coarseHalf + 24)` contained the dense-derived window for 98.6% of reps; 8 frames
 * covered 95.7% and 32 covered all 70. The remaining 1.4% surface as refined:"clipped" rather than
 * silently losing part of a rep.
 *
 * WHY THE SPAN IS ANCHORED ON THE VALLEY. The same measurement put the coarse-vs-dense BOUNDARY
 * error at p95 15 frames and max 45, and sweeping the stride from 2 to 6 barely moved it — the
 * error comes from the hysteresis band's percentiles shifting with the sample distribution, not
 * from resolution, so a denser coarse pass cannot fix it. The VALLEY, an argmin rather than a
 * threshold crossing, landed within 5 frames every time. Anchoring there is what turns the padding
 * constant from "absorb a 36-frame tail" into "absorb a 7-frame one".
 */
export const REP_PADDING_FRAMES = 24;

export interface FrameSpan { start: number; end: number }

/** The position of the deepest sample in `window`, skipping non-finite samples. */
export function valleyPosition(signal: number[], window: RepWindow): number {
  let best = window.start;
  let bestValue = Infinity;
  for (let i = window.start; i <= window.end; i += 1) {
    if (Number.isFinite(signal[i]) && signal[i] < bestValue) { bestValue = signal[i]; best = i; }
  }
  return best;
}

/**
 * The whole-clip dynamic range, taken from the pass that covers the whole clip (spec §2.1.1).
 *
 * Orients internally with `oriented()` -- the SAME function `segmentReps` uses to build the
 * values it compares `band` against -- before taking percentiles, so the returned band is
 * GUARANTEED to be in the same space `SegmentOptions.band` expects. There is no raw-space call
 * to make by mistake: `polarity`/`rectify` here must simply match whatever `segmentReps({
 * polarity, rectify, band })` call the result is handed to. Defaults match `SegmentOptions`'s own
 * (`polarity: "min"`, `rectify: false`), so every current Squat call site is unaffected.
 *
 * For Squat, `oriented()` is the identity, so the 46-clip measurement `refineWindow` cites cannot
 * distinguish an oriented call from an unoriented one -- see the coarseBand test with
 * `polarity: "max"` for the case that does.
 */
export function coarseBand(
  coarseSignal: number[], polarity: Polarity = "min", rectify = false
): { low: number; high: number } | null {
  const values = oriented(coarseSignal, polarity, rectify);
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return null;
  return { low: percentileOf(finite, PERCENTILE_LOW), high: percentileOf(finite, PERCENTILE_HIGH) };
}

/** The frames to extract densely for one coarse-detected rep, in canonical frame_index space. */
export function spanForRep(
  coarseSignal: number[], rep: RepWindow, lastFrameIndex: number
): FrameSpan {
  const valleyFrame = valleyPosition(coarseSignal, rep) * COARSE_STRIDE;
  // floor, not ceil: the 98.6% coverage figure REP_PADDING_FRAMES is set from was measured with
  // Python's `(end - start + 1) * STRIDE // 2`, and tests/test_coarse_segmentation_corpus.py
  // re-measures it the same way. A one-frame difference is immaterial next to a 24-frame pad, but
  // the constant's justification is only reproducible if both sides compute the span identically.
  const coarseHalf = Math.floor(((rep.end - rep.start + 1) * COARSE_STRIDE) / 2);
  const half = coarseHalf + REP_PADDING_FRAMES;
  return {
    start: Math.max(0, valleyFrame - half),
    end: Math.min(lastFrameIndex, valleyFrame + half),
  };
}

/** Union of overlapping or touching spans, sorted — so no frame is ever extracted twice. */
export function mergeSpans(spans: FrameSpan[]): FrameSpan[] {
  const sorted = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: FrameSpan[] = [];
  for (const span of sorted) {
    const last = merged[merged.length - 1];
    if (last && span.start <= last.end + 1) last.end = Math.max(last.end, span.end);
    else merged.push({ ...span });
  }
  return merged;
}

/** Every frame_index covered by `spans`, ascending. */
export function spanFrameIndices(spans: FrameSpan[]): number[] {
  const indices: number[] = [];
  for (const span of spans) {
    for (let i = span.start; i <= span.end; i += 1) indices.push(i);
  }
  return indices;
}

export type Refinement = true | false | "clipped";

/**
 * Re-derive a rep's boundary from the DENSE signal inside its extracted span (spec §2.1.1).
 *
 * The coarse boundary can be tens of frames off (see REP_PADDING_FRAMES), and `assign_phases`
 * takes a window's first 15% as setup — so a start that lands late puts "setup" mid-descent,
 * which is exactly the bug SP1 exists to fix, arriving by a new route. Padding cannot correct
 * that; only measuring the boundary on data that exists at full rate can.
 *
 * MEASURED, on the same 46 clips REP_PADDING_FRAMES came from: re-deriving the hysteresis band
 * from one span's OWN percentiles -- a span holds only about one repetition's worth of samples --
 * refines just 92.9% of reps to the whole-clip dense boundary EXACTLY (p95 15.3 frames, max 46).
 * Handing the same spans the WHOLE CLIP's range instead (the `band` parameter below, sourced from
 * `coarseBand` on the coarse pass's signal -- it already covers the whole clip) raises that to
 * 98.6% exact, p95 0, max 1. The 46-frame tail is not a rounding error: it lands "setup" in the
 * middle of a descent, which is the bug this whole rep-segmentation line of work exists to fix,
 * arriving by a new route. Pass `band`; do not let this re-derive its own.
 *
 * `denseSignal` is indexed by frame_index, with `undefined` wherever nothing was extracted.
 * `lastFrameIndex` is the clip's own end, and it is not optional bookkeeping: a window touching a
 * span edge that IS the clip edge has not been clipped -- nothing exists beyond it to extract.
 * Conflating the two reported 43% of real reps as clipped instead of the true 1.4%.
 */
export function refineWindow(
  denseSignal: (number | undefined)[], span: FrameSpan, coarse: FrameSpan,
  fps: number, lastFrameIndex: number, band: { low: number; high: number } | null
): { start: number; end: number; refined: Refinement } {
  const slice: number[] = [];
  for (let i = span.start; i <= span.end; i += 1) {
    const value = denseSignal[i];
    slice.push(value === undefined ? NaN : value);
  }
  const windows = segmentReps(centeredMedian(slice, DENSE_SMOOTH_WINDOW), { fps, band: band ?? undefined });
  if (windows.length === 0) return { start: coarse.start, end: coarse.end, refined: false };

  // The coarse window says WHICH rep this span is about; pick the refined window that overlaps it
  // most, so a neighbouring rep caught by the padding cannot steal the boundary.
  const best = windows.reduce((a, b) => (overlap(b, span, coarse) > overlap(a, span, coarse) ? b : a));
  const start = span.start + best.start;
  const end = span.start + best.end;
  const clipped =
    (start === span.start && span.start > 0) ||
    (end === span.end && span.end < lastFrameIndex);
  return { start, end, refined: clipped ? "clipped" : true };
}

function overlap(window: RepWindow, span: FrameSpan, coarse: FrameSpan): number {
  const start = span.start + window.start;
  const end = span.start + window.end;
  return Math.max(0, Math.min(end, coarse.end) - Math.max(start, coarse.start) + 1);
}
