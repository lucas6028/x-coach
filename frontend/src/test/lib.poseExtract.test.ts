import { afterEach, describe, expect, it, vi } from "vitest";
import { landmarksToFrame, resolveDuration } from "../lib/poseExtract";

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
