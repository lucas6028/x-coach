import { describe, it, expect } from "vitest";
import {
  angleIndexFromDrag,
  keyFrameLabelParts,
  sideViewIndex,
  tileBackground,
} from "../lib/turntable";
import type { NlfKeyFrame } from "../api";

describe("angleIndexFromDrag", () => {
  it("steps positively when dragging right", () => {
    expect(angleIndexFromDrag(0, 14, 14, 24)).toBe(1);
    expect(angleIndexFromDrag(0, 28, 14, 24)).toBe(2);
  });

  it("steps negatively when dragging left", () => {
    expect(angleIndexFromDrag(5, -14, 14, 24)).toBe(4);
  });

  it("wraps forward past the last tile", () => {
    expect(angleIndexFromDrag(23, 14, 14, 24)).toBe(0);
    expect(angleIndexFromDrag(23, 28, 14, 24)).toBe(1);
  });

  it("wraps backward past the first tile", () => {
    expect(angleIndexFromDrag(0, -14, 14, 24)).toBe(23);
    expect(angleIndexFromDrag(1, -28, 14, 24)).toBe(23);
  });

  it("rounds a partial drag to the nearest step", () => {
    // 20px at 14px/step rounds to 1 step (20/14 ≈ 1.43 -> 1).
    expect(angleIndexFromDrag(0, 20, 14, 24)).toBe(1);
    // 6px rounds down to 0 steps.
    expect(angleIndexFromDrag(0, 6, 14, 24)).toBe(0);
  });

  it("returns 0 for a degenerate (non-positive) angle count", () => {
    expect(angleIndexFromDrag(0, 100, 14, 0)).toBe(0);
  });
});

describe("sideViewIndex", () => {
  it("is a quarter turn from the camera's own view", () => {
    expect(sideViewIndex(24)).toBe(6);
  });
});

describe("tileBackground", () => {
  it("scales the strip so each tile fills the display box, offsetting to the selected tile", () => {
    expect(tileBackground(0, 24, 288)).toEqual({
      backgroundSize: "6912px 288px",
      backgroundPosition: "-0px 0",
    });
    expect(tileBackground(6, 24, 288)).toEqual({
      backgroundSize: "6912px 288px",
      backgroundPosition: "-1728px 0",
    });
  });
});

function repBottom(over: Partial<NlfKeyFrame> = {}): NlfKeyFrame {
  return {
    t_s: 1.2,
    kind: "rep_bottom",
    fault_ids: [],
    rep_index: 2,
    strip_url: "https://example.com/strip1.webp",
    ...over,
  };
}

function faultPeak(over: Partial<NlfKeyFrame> = {}): NlfKeyFrame {
  return {
    t_s: 3.4,
    kind: "fault_peak",
    fault_ids: ["knees_inward"],
    rep_index: null,
    strip_url: "https://example.com/strip2.webp",
    ...over,
  };
}

describe("keyFrameLabelParts", () => {
  it("carries the rep index for a rep-bottom key frame", () => {
    expect(keyFrameLabelParts(repBottom({ rep_index: 3 }))).toEqual({
      kind: "rep_bottom",
      repIndex: 3,
    });
  });

  it("carries a null rep index unchanged", () => {
    expect(keyFrameLabelParts(repBottom({ rep_index: null }))).toEqual({
      kind: "rep_bottom",
      repIndex: null,
    });
  });

  it("carries the fault ids for a fault-peak key frame", () => {
    expect(keyFrameLabelParts(faultPeak({ fault_ids: ["heel_rise", "knees_inward"] }))).toEqual({
      kind: "fault_peak",
      faultIds: ["heel_rise", "knees_inward"],
    });
  });
});
