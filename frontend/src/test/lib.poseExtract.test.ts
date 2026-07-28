import { afterEach, describe, expect, it, vi } from "vitest";
import { landmarksToFrame, planReps, resolveDuration } from "../lib/poseExtract";
import { COARSE_STRIDE } from "../lib/repSpans";

const lm = (n: number) => Array.from({ length: n }, (_, i) => ({ x: i / 100, y: i / 50, z: 0.1, visibility: 0.9 }));

describe("landmarksToFrame", () => {
  it("serializes 33 landmarks + world landmarks into the shared schema", () => {
    const world = Array.from({ length: 33 }, (_, i) => ({ x: i + 1, y: i + 1, z: 0.2, visibility: 0.5 }));
    const frame = landmarksToFrame(7, lm(33), world);
    expect(frame.frame_index).toBe(7);
    expect(frame.landmarks).toHaveLength(33);
    expect(frame.landmarks![0]).toEqual({ x: 0, y: 0, z: 0.1, visibility: 0.9 });
    expect(frame.world_landmarks).toHaveLength(33);
    expect(frame.world_landmarks![0]).toEqual({ x: 1, y: 1, z: 0.2, visibility: 0.5 });
  });

  it("defaults a landmark's missing visibility to 0", () => {
    const noVis = Array.from({ length: 33 }, () => ({ x: 0.1, y: 0.2, z: 0.3 }));
    const frame = landmarksToFrame(0, noVis, noVis);
    expect(frame.landmarks![0]).toEqual({ x: 0.1, y: 0.2, z: 0.3, visibility: 0 });
  });

  it("emits null landmarks when the frame has no full 33-point pose", () => {
    expect(landmarksToFrame(1, undefined, undefined).landmarks).toBeNull();
    expect(landmarksToFrame(2, lm(20), lm(20)).landmarks).toBeNull(); // detector needs >=33
  });
});

// A stand-in for the slice of <video> `resolveDuration` touches. jsdom has no decoder, so this is
// the only way to get the duration-recovery protocol under test at all — and the protocol is
// exactly where the live-record bug lived.
//
// `currentTime` is a real setter that records every write and can fire events synchronously, so a
// regression that seeks BEFORE attaching its listeners misses the event and times out rather than
// silently passing.
class FakeVideo {
  duration: number;
  seeks: number[] = [];
  private listeners = new Map<string, Set<() => void>>();
  /** Fired synchronously on the next `currentTime` write, mimicking the browser's response to the
   *  probe seek: clamp to the true end of the media, then announce the recovered duration. */
  onSeek: ((self: FakeVideo) => void) | null = null;

  constructor(duration: number) {
    this.duration = duration;
  }

  private _currentTime = 0;
  get currentTime() {
    return this._currentTime;
  }
  set currentTime(t: number) {
    this._currentTime = t;
    this.seeks.push(t);
    const hook = this.onSeek;
    if (hook) {
      this.onSeek = null; // one-shot: the reset-to-0 write must not re-trigger it
      hook(this);
    }
  }

  addEventListener(type: string, cb: () => void) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(cb);
  }

  removeEventListener(type: string, cb: () => void) {
    this.listeners.get(type)?.delete(cb);
  }

  emit(type: string) {
    for (const cb of [...(this.listeners.get(type) ?? [])]) cb();
  }

  get listenerCount() {
    return [...this.listeners.values()].reduce((n, s) => n + s.size, 0);
  }
}

describe("resolveDuration", () => {
  afterEach(() => vi.useRealTimers());

  it("trusts a container that already reports a duration, without seeking", async () => {
    // The upload path. It works today and must stay untouched — probing a well-formed file would
    // be a pointless seek on the one path with no bug.
    const video = new FakeVideo(12.5);
    await expect(resolveDuration(video)).resolves.toBe(12.5);
    expect(video.seeks).toEqual([]);
  });

  // THE LIVE-RECORD BUG. Verified against the clips this path actually produced
  // (data/runtime/uploads/*.webm): MediaRecorder writes a Segment of unknown size and an Info
  // element carrying only TimecodeScale — no Duration — so the browser cannot report one. The old
  // code did `video.duration || 0`, looped `t < 0`, and emitted zero frames.
  it.each([
    ["NaN", NaN],
    ["Infinity", Infinity],
    ["zero", 0],
  ])("recovers the real duration when the container reports %s", async (_label, reported) => {
    const video = new FakeVideo(reported);
    video.onSeek = (v) => {
      v.duration = 8.25; // the browser clamps the probe seek and learns the true end
      v.emit("durationchange");
    };
    await expect(resolveDuration(video)).resolves.toBe(8.25);
    expect(video.seeks[0]).toBeGreaterThan(1e6); // sought far past any plausible end
  });

  it("resolves from durationchange alone, with no seeked event", async () => {
    // A MediaRecorder clip has no Cues index, so `seeked` firing is not something we may assume.
    // If this implementation depended on it, the user's phone would hang instead of analysing.
    const video = new FakeVideo(NaN);
    video.onSeek = (v) => {
      v.duration = 4;
      v.emit("durationchange");
    };
    await expect(resolveDuration(video)).resolves.toBe(4);
  });

  it("keeps waiting through a durationchange that is still not finite", async () => {
    const video = new FakeVideo(NaN);
    video.onSeek = (v) => v.emit("durationchange"); // still NaN — not an answer
    const pending = resolveDuration(video);
    video.duration = 6.5;
    video.emit("durationchange");
    await expect(pending).resolves.toBe(6.5);
  });

  it("rewinds to the start so the caller's sampling loop begins at t=0", async () => {
    const video = new FakeVideo(NaN);
    video.onSeek = (v) => {
      v.duration = 3;
      v.emit("durationchange");
    };
    await resolveDuration(video);
    expect(video.currentTime).toBe(0);
    expect(video.seeks[video.seeks.length - 1]).toBe(0); // `.at()` is past this project's TS lib target
  });

  it("throws instead of hanging when the duration never arrives", async () => {
    // Deliberately NOT a silent 0. Returning 0 would reproduce the original bug: an empty frame
    // list that the app renders as "no frame could be measured" — a verdict-shaped answer for what
    // is really a decode failure. An honest error is the lesser harm.
    vi.useFakeTimers();
    const video = new FakeVideo(NaN);
    const pending = resolveDuration(video, 5000);
    const assertion = expect(pending).rejects.toThrow(/length/i);
    await vi.advanceTimersByTimeAsync(5000);
    await assertion;
  });

  it("detaches its listeners however it settles", async () => {
    const ok = new FakeVideo(NaN);
    ok.onSeek = (v) => {
      v.duration = 2;
      v.emit("durationchange");
    };
    await resolveDuration(ok);
    expect(ok.listenerCount).toBe(0);

    vi.useFakeTimers();
    const timedOut = new FakeVideo(NaN);
    const pending = resolveDuration(timedOut, 1000);
    const assertion = expect(pending).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(1000);
    await assertion;
    expect(timedOut.listenerCount).toBe(0);
  });
});

function coarseRepSignal(count: number, period: number): number[] {
  return Array.from({ length: count * period }, (_, i) =>
    115 + 55 * Math.cos((2 * Math.PI * (i % period)) / period));
}

describe("planReps", () => {
  const LAST = 5 * 30 * COARSE_STRIDE - 1; // five 30-sample coarse reps on the canonical grid

  it("marks the first / middle / last of five reps as analyzed", () => {
    const { plan } = planReps(coarseRepSignal(5, 30), 3, LAST, "Squat");
    expect(plan.fallback).toBeNull();
    expect(plan.segments).toHaveLength(5);
    expect(plan.segments.filter((s) => s.analyzed).map((s) => s.index)).toEqual([1, 3, 5]);
  });

  it("returns spans only for the analyzed reps", () => {
    const { spans } = planReps(coarseRepSignal(5, 30), 3, LAST, "Squat");
    expect(spans.length).toBeGreaterThan(0);
    expect(spans.length).toBeLessThanOrEqual(3);
  });

  it("reports frame_index, not coarse positions", () => {
    const { plan } = planReps(coarseRepSignal(5, 30), 3, LAST, "Squat");
    // Rep 2 of 5 cannot start before frame 30 if each rep is 90 canonical frames long.
    expect(plan.segments[1].start_frame).toBeGreaterThanOrEqual(COARSE_STRIDE * 20);
  });

  it("falls back to the whole clip when nothing segments", () => {
    const { plan, spans } = planReps(new Array(150).fill(5), 3, LAST, "Squat");
    expect(plan.fallback).toBe("no_reps_detected");
    expect(plan.segments).toEqual([]);
    expect(spans).toEqual([{ start: 0, end: LAST }]);
  });

  it("falls back for a movement with no browser-side signal", () => {
    const { plan, spans } = planReps(coarseRepSignal(5, 30), 3, LAST, "Deadlift");
    expect(plan.fallback).toBe("segmentation_disabled");
    expect(spans).toEqual([{ start: 0, end: LAST }]);
  });

  it("falls back when every rep is partial", () => {
    // A clip that STARTS at the bottom and only rises: the single window has no crossing to climb
    // from on its left, so it is partial. Verified against Python — segment_reps returns exactly
    // one window with partial=True — so this asserts unconditionally.
    const rising = Array.from({ length: 30 }, (_, i) =>
      115 - 55 * Math.cos((2 * Math.PI * i) / 60));
    const { plan, spans } = planReps(rising, 3, 89, "Squat");
    expect(plan.fallback).toBe("only_partial_reps");
    expect(plan.segments).toEqual([]);
    expect(spans).toEqual([{ start: 0, end: 89 }]);
  });

  it("NEVER returns an empty span list — a fallback still extracts everything", () => {
    for (const signal of [new Array(150).fill(5), coarseRepSignal(5, 30)]) {
      expect(planReps(signal, 3, LAST, "Squat").spans.length).toBeGreaterThan(0);
    }
  });
});
