import { describe, expect, it } from "vitest";
import { THUMBNAIL_MAX_EDGE, thumbnailSize, thumbnailTime } from "../lib/thumbnail";

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
