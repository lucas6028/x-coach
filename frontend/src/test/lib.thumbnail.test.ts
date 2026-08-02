import { afterEach, describe, expect, it, vi } from "vitest";
import { THUMBNAIL_MAX_EDGE, thumbnailSize, thumbnailTime, withTimeout } from "../lib/thumbnail";

describe("thumbnailSize", () => {
  it("leaves a small frame alone", () => {
    expect(thumbnailSize(320, 240)).toEqual({ width: 320, height: 240 });
  });

  it("scales a landscape frame down by its longest edge", () => {
    expect(thumbnailSize(1920, 1080)).toEqual({ width: 480, height: 270 });
  });

  it("scales a portrait frame down by its longest edge", () => {
    expect(thumbnailSize(1080, 1920)).toEqual({ width: 270, height: 480 });
  });

  it("never returns a zero dimension for an extreme aspect ratio", () => {
    const { width, height } = thumbnailSize(4000, 1);
    expect(width).toBe(THUMBNAIL_MAX_EDGE);
    expect(height).toBeGreaterThanOrEqual(1);
  });
});

describe("thumbnailTime", () => {
  it("picks a frame a quarter of the way in", () => {
    expect(thumbnailTime(8)).toBe(2);
  });

  it("falls back to the first frame when the length is unusable", () => {
    // A MediaRecorder clip whose duration never resolved. 0 is a real frame; NaN is not a time.
    expect(thumbnailTime(Number.NaN)).toBe(0);
    expect(thumbnailTime(0)).toBe(0);
    expect(thumbnailTime(Infinity)).toBe(0);
  });
});

describe("withTimeout", () => {
  afterEach(() => vi.useRealTimers());

  it("resolves with the promise's value when it settles first", async () => {
    await expect(withTimeout(Promise.resolve("frame"), 5000)).resolves.toBe("frame");
  });

  it("rejects with the promise's own error when it rejects first", async () => {
    await expect(withTimeout(Promise.reject(new Error("decode failed")), 5000)).rejects.toThrow(
      "decode failed"
    );
  });

  it("rejects with a timeout error when the promise never settles in time", async () => {
    vi.useFakeTimers();
    const never = new Promise<void>(() => undefined);
    const result = withTimeout(never, 5000);
    const assertion = expect(result).rejects.toThrow("timed out");
    await vi.advanceTimersByTimeAsync(5000);
    await assertion;
  });
});
