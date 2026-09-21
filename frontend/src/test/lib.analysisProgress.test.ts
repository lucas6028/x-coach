import { describe, it, expect, vi, afterEach } from "vitest";
import {
  createProgressTracker,
  windowedRemainingSec,
  PHASE_SHARE,
  type AnalysisProgress,
} from "../lib/analysisProgress";

afterEach(() => vi.useRealTimers());

// A tracker on a hand-driven clock, collecting everything it emits.
function harness() {
  let clock = 0;
  const seen: AnalysisProgress[] = [];
  const tracker = createProgressTracker((p) => seen.push(p), () => clock);
  return {
    tracker,
    seen,
    last: () => seen[seen.length - 1],
    advance: (ms: number) => { clock += ms; },
  };
}

describe("windowedRemainingSec", () => {
  it("has no estimate from fewer than two samples", () => {
    expect(windowedRemainingSec([])).toBeNull();
    expect(windowedRemainingSec([{ t: 0, p: 0.5 }])).toBeNull();
  });

  it("has no estimate until the samples span a full second", () => {
    expect(windowedRemainingSec([{ t: 0, p: 0 }, { t: 900, p: 0.5 }])).toBeNull();
  });

  it("has no estimate while progress is stalled", () => {
    expect(windowedRemainingSec([{ t: 0, p: 0.3 }, { t: 2000, p: 0.3 }])).toBeNull();
  });

  it("extrapolates the window's rate over what is left", () => {
    // 20% in 2s -> 10%/s -> 60% left takes 6s.
    expect(windowedRemainingSec([{ t: 0, p: 0.2 }, { t: 2000, p: 0.4 }])).toBeCloseTo(6);
  });
});

describe("createProgressTracker", () => {
  it("maps extraction onto its share of the overall bar", () => {
    const h = harness();
    h.tracker.extract(0.5);
    expect(h.last().phase).toBe("extract");
    expect(h.last().fraction).toBeCloseTo(PHASE_SHARE.extract * 0.5);
  });

  it("reports no seconds before there is a measured rate", () => {
    const h = harness();
    h.tracker.extract(0.01);
    expect(h.last().remainingSec).toBeNull();
  });

  it("estimates seconds from the extraction rate plus the phases still to come", () => {
    const h = harness();
    h.tracker.extract(0);
    h.advance(2000);
    h.tracker.extract(0.2);
    // 10%/s -> 8s of extraction left, plus the 5s assumed for upload + server.
    expect(h.last().remainingSec).toBe(13);
  });

  it("throttles per-frame callbacks", () => {
    const h = harness();
    h.tracker.extract(0.1);
    h.advance(10);
    h.tracker.extract(0.11);
    h.advance(10);
    h.tracker.extract(0.12);
    expect(h.seen).toHaveLength(1);
    h.advance(200);
    h.tracker.extract(0.13);
    expect(h.seen).toHaveLength(2);
  });

  it("follows the recent rate, not the average since the start", () => {
    const h = harness();
    // Fast coarse pass: 30% in 3s.
    for (let i = 0; i <= 3; i += 1) { h.tracker.extract(i * 0.1); h.advance(1000); }
    // Slow dense pass: 1%/s for 20s. An all-run average would still be quoting ~2.5%/s.
    for (let i = 1; i <= 20; i += 1) { h.tracker.extract(0.3 + i * 0.01); h.advance(1000); }
    // 50% left at 1%/s is 50s (+5s tail); the run-average answer would be about 25s.
    expect(h.last().remainingSec).toBeGreaterThan(45);
  });

  it("never moves the bar backwards", () => {
    const h = harness();
    h.tracker.extract(0.8);
    h.advance(300);
    h.tracker.extract(0.4);
    expect(h.last().fraction).toBeCloseTo(PHASE_SHARE.extract * 0.8);
  });

  it("emits the switch to the upload phase even inside the throttle window", () => {
    const h = harness();
    h.tracker.extract(1);
    h.advance(1);
    h.tracker.upload(0);
    expect(h.last().phase).toBe("upload");
    expect(h.last().fraction).toBeCloseTo(PHASE_SHARE.extract);
  });

  it("ignores late extraction callbacks once uploading", () => {
    const h = harness();
    h.tracker.upload(0.5);
    const count = h.seen.length;
    h.advance(500);
    h.tracker.extract(0.9);
    expect(h.seen).toHaveLength(count);
  });

  it("fills the server phase asymptotically, with no seconds, and stops on stop()", () => {
    vi.useFakeTimers();
    const h = harness();
    h.tracker.upload(1);
    expect(h.last().phase).toBe("server");
    expect(h.last().remainingSec).toBeNull();
    const atStart = h.last().fraction;

    h.advance(4000);
    vi.advanceTimersByTime(250);
    const later = h.last().fraction;
    expect(later).toBeGreaterThan(atStart);

    // An hour in, the bar is still short of full: only the real response completes it.
    h.advance(3_600_000);
    vi.advanceTimersByTime(250);
    expect(h.last().fraction).toBeLessThanOrEqual(1);
    expect(h.last().remainingSec).toBeNull();

    h.tracker.stop();
    const count = h.seen.length;
    vi.advanceTimersByTime(5000);
    expect(h.seen).toHaveLength(count);
    h.tracker.stop(); // idempotent
  });

  it("does not restart the server phase on a repeated upload-complete event", () => {
    vi.useFakeTimers();
    const h = harness();
    h.tracker.upload(1);
    h.tracker.upload(1);
    const count = h.seen.length;
    vi.advanceTimersByTime(250);
    // One timer, one tick.
    expect(h.seen).toHaveLength(count + 1);
    h.tracker.stop();
  });
});
