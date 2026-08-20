import { describe, it, expect } from "vitest";
import { containRect } from "../lib/videoRect";

// The letterbox arithmetic the skeleton canvas and the phone's fault chips BOTH map through. It is
// pure and exact, so it is pinned by arithmetic here rather than inferred from a rendered overlay:
// under jsdom the canvas never rasterises, so a drawing test could not tell a correct mapping from
// a broken one.
describe("containRect", () => {
  it("pillarboxes a clip narrower than its box: full height, bars left and right", () => {
    // 2:1 box, 1:1 clip → 100 wide of content centred in 200.
    expect(containRect(200, 100, 1)).toEqual({
      offsetX: 50,
      offsetY: 0,
      width: 100,
      height: 100,
    });
  });

  it("letterboxes a clip wider than its box: full width, bars top and bottom", () => {
    // 1:1 box, 2:1 clip → 50 tall of content centred in 100.
    expect(containRect(100, 100, 2)).toEqual({
      offsetX: 0,
      offsetY: 25,
      width: 100,
      height: 50,
    });
  });

  it("fills the box exactly when the aspects match, leaving no dead space", () => {
    expect(containRect(160, 90, 16 / 9)).toEqual({
      offsetX: 0,
      offsetY: 0,
      width: 160,
      height: 90,
    });
  });

  // The portrait-phone-clip case the extraction was written for: a 9:16 clip in a landscape stage
  // is pillarboxed hard, and mapping landmarks onto the full box would throw the skeleton most of
  // the way off the body.
  it("pillarboxes a portrait clip in a landscape stage", () => {
    const r = containRect(320, 180, 9 / 16);
    expect(r.height).toBe(180);
    expect(r.width).toBeCloseTo(101.25);
    expect(r.offsetX).toBeCloseTo((320 - 101.25) / 2);
    expect(r.offsetY).toBe(0);
  });

  // Before `loadedmetadata` the caller has no intrinsic size, so videoWidth/videoHeight are 0 and
  // the aspect arrives as 0 or NaN. Falling back to the box keeps the overlay drawing over the whole
  // card for a frame instead of collapsing it into a zero-width sliver in the corner.
  it.each([
    ["zero", 0],
    ["negative", -2],
    ["NaN", NaN],
    ["Infinity", Infinity],
  ])("falls back to the whole box for a %s aspect", (_name, aspect) => {
    expect(containRect(300, 200, aspect)).toEqual({
      offsetX: 0,
      offsetY: 0,
      width: 300,
      height: 200,
    });
  });
});
