import { afterEach, describe, expect, it, vi } from "vitest";
import { THUMBNAIL_MAX_EDGE, isBlankFrame, thumbnailSize, thumbnailTime, withTimeout } from "../lib/thumbnail";

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

describe("isBlankFrame", () => {
  const rgba = (...pixels: [number, number, number, number][]) => new Uint8ClampedArray(pixels.flat());

  it("flags the solid black frame iPhone Safari recordings stored", () => {
    expect(isBlankFrame(rgba([0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255]))).toBe(true);
  });

  it("flags the transparent canvas a draw that did nothing leaves behind", () => {
    expect(isBlankFrame(new Uint8ClampedArray(4 * 16))).toBe(true);
  });

  it("flags any single flat colour, not only black", () => {
    expect(isBlankFrame(rgba([128, 128, 128, 255], [129, 127, 128, 255]))).toBe(true);
  });

  // A dim scene is still a real frame: sensor noise spreads its values, and "dark" alone must not
  // throw away a thumbnail recorded in a badly lit gym.
  it("keeps a dark frame that has any real variation", () => {
    expect(isBlankFrame(rgba([3, 4, 2, 255], [9, 6, 5, 255], [1, 2, 3, 255]))).toBe(false);
  });

  it("keeps a frame whose variation is in one channel only", () => {
    expect(isBlankFrame(rgba([0, 0, 0, 255], [0, 0, 40, 255]))).toBe(false);
  });

  it("treats an empty buffer as blank", () => {
    expect(isBlankFrame(new Uint8ClampedArray(0))).toBe(true);
  });
});
