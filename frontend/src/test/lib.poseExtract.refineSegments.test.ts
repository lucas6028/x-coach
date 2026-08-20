// `refineSegments`' neighbour clamp (RS-SP2 whole-branch review, fix 5), isolated from
// `refineWindow`'s own segmentation math by mocking it — the clamp must hold regardless of WHY a
// refined boundary overshot, so a deterministic stand-in is more precise here than trying to coax
// a real signal into overshooting.
import { describe, it, expect, vi } from "vitest";

vi.mock("../lib/repSpans", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/repSpans")>();
  return { ...actual, refineWindow: vi.fn() };
});

import { refineSegments, type RepsPlan } from "../lib/poseExtract";
import { refineWindow } from "../lib/repSpans";

const mockRefineWindow = vi.mocked(refineWindow);

describe("refineSegments — clamping refined boundaries against neighbours", () => {
  it("clamps an analyzed segment's end when refinement pushed it into the next segment's start", () => {
    // Simulate a refinement that (for whatever reason — a noisy span, a neighbour's excursion
    // bleeding into the padding) extends 40 frames past its own coarse end, straight into the next
    // segment's coarse start. Nothing upstream of the clamp being tested would catch this: each
    // segment is refined independently, from its own span, with no view of its neighbour.
    mockRefineWindow.mockImplementation((_dense, _span, coarse) => (
      { start: coarse.start, end: coarse.end + 40, refined: true }
    ));

    const plan: RepsPlan = {
      max_reps: 2,
      fallback: null,
      segments: [
        { index: 1, start_frame: 0, end_frame: 50, partial: false, analyzed: true, refined: false },
        { index: 2, start_frame: 60, end_frame: 110, partial: false, analyzed: true, refined: false },
      ],
    };
    const spans = [{ start: 0, end: 200 }];
    const denseSignal = new Array(201).fill(1);
    const coarseSignal = new Array(70).fill(1);

    const result = refineSegments(plan, spans, denseSignal, 200, coarseSignal);

    // Without the clamp this would be [0, 90] and [60, 150] — end_frame 90 overlaps start_frame
    // 60, which `_validate_reps`' overlap check (backend/app/routers/analyze.py) would 400 on.
    expect(result.segments[0].end_frame).toBeLessThan(result.segments[1].start_frame);
    // The EARLIER segment gives up the shared ground (mirrors `finalize` in repSegmentation.ts);
    // the later, still-refined segment keeps the boundary refinement actually computed for it.
    expect(result.segments[1].start_frame).toBe(60);
    expect(result.segments[1].end_frame).toBe(150);
  });

  it("chains a clamp through three overlapping segments in one left-to-right pass", () => {
    mockRefineWindow.mockImplementation((_dense, _span, coarse) => (
      { start: coarse.start, end: coarse.end + 40, refined: true }
    ));

    const plan: RepsPlan = {
      max_reps: 3,
      fallback: null,
      segments: [
        { index: 1, start_frame: 0, end_frame: 50, partial: false, analyzed: true, refined: false },
        { index: 2, start_frame: 60, end_frame: 110, partial: false, analyzed: true, refined: false },
        { index: 3, start_frame: 120, end_frame: 170, partial: false, analyzed: true, refined: false },
      ],
    };
    const spans = [{ start: 0, end: 300 }];
    const denseSignal = new Array(301).fill(1);
    const coarseSignal = new Array(100).fill(1);

    const result = refineSegments(plan, spans, denseSignal, 300, coarseSignal);
    const segs = result.segments;
    for (let i = 0; i + 1 < segs.length; i += 1) {
      expect(segs[i].end_frame).toBeLessThan(segs[i + 1].start_frame);
    }
  });

  it("leaves already non-overlapping refined segments untouched", () => {
    mockRefineWindow.mockImplementation((_dense, _span, coarse) => (
      { start: coarse.start, end: coarse.end, refined: true }
    ));

    const plan: RepsPlan = {
      max_reps: 2,
      fallback: null,
      segments: [
        { index: 1, start_frame: 0, end_frame: 50, partial: false, analyzed: true, refined: false },
        { index: 2, start_frame: 60, end_frame: 110, partial: false, analyzed: true, refined: false },
      ],
    };
    const spans = [{ start: 0, end: 200 }];
    const denseSignal = new Array(201).fill(1);
    const coarseSignal = new Array(70).fill(1);

    const result = refineSegments(plan, spans, denseSignal, 200, coarseSignal);
    expect(result.segments[0]).toMatchObject({ start_frame: 0, end_frame: 50 });
    expect(result.segments[1]).toMatchObject({ start_frame: 60, end_frame: 110 });
  });
});
