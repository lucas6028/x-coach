import { describe, it, expect } from "vitest";
import {
  CANONICAL_FPS,
  COARSE_STRIDE,
  REP_PADDING_FRAMES,
  frameIndexAt,
  mergeSpans,
  refineWindow,
  spanForRep,
  spanFrameIndices,
  valleyPosition,
} from "../lib/repSpans";
import { segmentReps } from "../lib/repSegmentation";

describe("frameIndexAt", () => {
  it("puts every sample on the canonical 30fps grid", () => {
    expect(CANONICAL_FPS).toBe(30);
    expect(frameIndexAt(0)).toBe(0);
    expect(frameIndexAt(1 / 30)).toBe(1);
    expect(frameIndexAt(1)).toBe(30);
  });

  it("gives a coarse pass the SAME indices a dense pass would give", () => {
    // The bug this pins: an incrementing counter makes coarse sample k index k, not 3k.
    const dense = Array.from({ length: 10 }, (_, i) => frameIndexAt(i / 30));
    const coarse = [0, 3, 6, 9].map((i) => frameIndexAt(i / 30));
    expect(coarse).toEqual([0, 3, 6, 9]);
    expect(coarse.every((c) => dense.includes(c))).toBe(true);
  });
});

// A cosine rep: `count` excursions from 170 degrees down to 60 and back, `period` samples each.
function repSignal(count: number, period: number): number[] {
  return Array.from({ length: count * period }, (_, i) =>
    115 + 55 * Math.cos((2 * Math.PI * (i % period)) / period));
}

describe("valleyPosition", () => {
  it("finds the deepest sample inside the window", () => {
    const signal = [10, 8, 3, 8, 10];
    expect(valleyPosition(signal, { index: 1, start: 0, end: 4, partial: false })).toBe(2);
  });

  it("ignores non-finite samples rather than returning one", () => {
    const signal = [10, NaN, 3, NaN, 10];
    expect(valleyPosition(signal, { index: 1, start: 0, end: 4, partial: false })).toBe(2);
  });
});

describe("spanForRep", () => {
  it("anchors on the valley and spans the coarse half-width plus the padding", () => {
    // 30 coarse samples, valley at 15 => frame 45 on the canonical grid; half-width 15 samples
    // => 45 frames; span = 45 +/- (45 + 24).
    const coarse = repSignal(1, 30);
    const [rep] = segmentReps(coarse, { fps: 10 });
    const span = spanForRep(coarse, rep, 89);
    const valleyFrame = valleyPosition(coarse, rep) * COARSE_STRIDE;
    expect(span.start).toBeLessThan(valleyFrame);
    expect(span.end).toBeGreaterThan(valleyFrame);
    // Unclamped this would be valleyFrame +/- 69 = [-24, 114], symmetric around 45. But this clip
    // is only 90 frames (0..89) and 45 is not its exact midpoint (44.5 is), so once the pad
    // saturates BOTH edges — as it does here — the returned span is the whole clip, not a
    // perfectly centred window. Assert that directly instead of an equality the discretisation
    // makes unsatisfiable.
    expect(span).toEqual({ start: 0, end: 89 });
  });

  it("computes the exact frame span from the valley, coarseHalf, and padding when nothing clamps", () => {
    // A synthetic 10-sample coarse window [40, 49] with its lone minimum at coarse position 45,
    // chosen so the padded span lands comfortably inside [0, 300] and neither edge clamps — this
    // pins the arithmetic itself, not clamping behaviour.
    //   valleyFrame = 45 * COARSE_STRIDE(3)                 = 135
    //   coarseHalf  = floor((49 - 40 + 1) * COARSE_STRIDE / 2) = floor(10 * 3 / 2) = 15
    //   half        = coarseHalf(15) + REP_PADDING_FRAMES(24)  = 39
    //   span        = [135 - 39, 135 + 39]                     = [96, 174]
    const coarse = new Array(50).fill(100);
    coarse[45] = 10;
    const rep = { index: 1, start: 40, end: 49, partial: false };
    const span = spanForRep(coarse, rep, 300);
    expect(span).toEqual({ start: 96, end: 174 });
  });

  it("clamps only the edge that actually exceeds the clip", () => {
    // Same rep as above (unclamped span [96, 174]), but the clip now ends at frame 150 — short
    // enough that the end clamps (174 > 150) while the start (96) does not. A stub that returns
    // {0, lastFrameIndex} unconditionally would produce {0, 150} here, not {96, 150}.
    const coarse = new Array(50).fill(100);
    coarse[45] = 10;
    const rep = { index: 1, start: 40, end: 49, partial: false };
    const span = spanForRep(coarse, rep, 150);
    expect(span).toEqual({ start: 96, end: 150 });
  });

  it("pads by the measured constant", () => {
    expect(REP_PADDING_FRAMES).toBe(24);
    expect(COARSE_STRIDE).toBe(3);
  });
});

describe("mergeSpans", () => {
  it("merges overlapping spans so no frame is extracted twice", () => {
    expect(mergeSpans([{ start: 0, end: 50 }, { start: 40, end: 90 }])).toEqual([{ start: 0, end: 90 }]);
  });

  it("merges spans that only touch", () => {
    expect(mergeSpans([{ start: 0, end: 10 }, { start: 11, end: 20 }])).toEqual([{ start: 0, end: 20 }]);
  });

  it("keeps disjoint spans apart and sorts them", () => {
    expect(mergeSpans([{ start: 60, end: 80 }, { start: 0, end: 10 }]))
      .toEqual([{ start: 0, end: 10 }, { start: 60, end: 80 }]);
  });

  it("returns [] for no spans", () => {
    expect(mergeSpans([])).toEqual([]);
  });
});

describe("spanFrameIndices", () => {
  it("enumerates every frame in every span, once, in order", () => {
    expect(spanFrameIndices([{ start: 0, end: 2 }, { start: 5, end: 6 }])).toEqual([0, 1, 2, 5, 6]);
  });
});

describe("refineWindow", () => {
  // A dense 90-frame rep sitting inside a 200-frame clip, spanned from frame 0 to 119.
  const LAST = 199;
  const dense: (number | undefined)[] = new Array(LAST + 1).fill(undefined);
  repSignal(1, 90).forEach((v, i) => { dense[i + 15] = v; });

  it("recovers the dense boundary from a coarse one that is 10 frames late", () => {
    const out = refineWindow(dense, { start: 0, end: 119 }, { start: 25, end: 100 }, 30, LAST);
    expect(out.refined).toBe(true);
    expect(Math.abs(out.start - 15)).toBeLessThanOrEqual(2);
  });

  it("reports 'clipped' when the span cut the rep off mid-clip", () => {
    // The span ends at 89 while the clip runs to 199, so there WAS more to extract and the
    // padding was too small — the one case that must stay visible.
    const out = refineWindow(dense, { start: 15, end: 89 }, { start: 20, end: 85 }, 30, LAST);
    expect(out.refined).toBe("clipped");
  });

  it("does NOT report 'clipped' when the span edge is the clip's own edge", () => {
    // Nothing exists beyond the clip, so touching that edge is not a padding failure. Measured:
    // conflating the two reported 43% of real reps as clipped instead of the true 1.4%.
    const flush: (number | undefined)[] = new Array(90).fill(undefined);
    repSignal(1, 90).forEach((v, i) => { flush[i] = v; });
    const out = refineWindow(flush, { start: 0, end: 89 }, { start: 0, end: 89 }, 30, 89);
    expect(out.refined).toBe(true);
  });

  it("picks the window overlapping the coarse one when padding caught a neighbour", () => {
    // mergeSpans can fuse two adjacent reps into one span, so the span legitimately holds two
    // windows and the overlap tiebreak decides — it is load-bearing, not a safety net.
    const two: (number | undefined)[] = new Array(LAST + 1).fill(undefined);
    repSignal(2, 90).forEach((v, i) => { two[i] = v; });
    const out = refineWindow(two, { start: 0, end: 179 }, { start: 95, end: 175 }, 30, LAST);
    expect(out.start).toBeGreaterThanOrEqual(80);
  });

  it("falls back to the coarse boundary when the span holds no window", () => {
    const flat: (number | undefined)[] = new Array(LAST + 1).fill(5);
    const out = refineWindow(flat, { start: 0, end: 119 }, { start: 20, end: 100 }, 30, LAST);
    expect(out).toEqual({ start: 20, end: 100, refined: false });
  });
});
