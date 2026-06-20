import { describe, it, expect } from "vitest";
import { fmtTime, titleCase } from "../lib/format";

describe("fmtTime", () => {
  it("formats zero as 0:00", () => {
    expect(fmtTime(0)).toBe("0:00");
  });

  it("formats sub-minute seconds", () => {
    expect(fmtTime(5)).toBe("0:05");
    expect(fmtTime(59)).toBe("0:59");
  });

  it("formats exactly one minute", () => {
    expect(fmtTime(60)).toBe("1:00");
  });

  it("formats minutes and seconds", () => {
    expect(fmtTime(90)).toBe("1:30");
    expect(fmtTime(125)).toBe("2:05");
  });

  it("floors fractional seconds", () => {
    expect(fmtTime(1.9)).toBe("0:01");
    expect(fmtTime(59.99)).toBe("0:59");
  });

  it("clamps negative values to 0:00", () => {
    expect(fmtTime(-5)).toBe("0:00");
  });

  it("clamps Infinity to 0:00", () => {
    expect(fmtTime(Infinity)).toBe("0:00");
  });

  it("clamps NaN to 0:00", () => {
    expect(fmtTime(NaN)).toBe("0:00");
  });

  it("pads seconds to two digits", () => {
    expect(fmtTime(61)).toBe("1:01");
  });
});

describe("titleCase", () => {
  it("capitalises single words", () => {
    expect(titleCase("hello")).toBe("Hello");
  });

  it("replaces underscores with spaces and capitalises each word", () => {
    expect(titleCase("knees_inward")).toBe("Knees Inward");
  });

  it("handles multiple underscores", () => {
    expect(titleCase("excessive_forward_lean")).toBe("Excessive Forward Lean");
  });

  it("leaves already-cased words alone except for first letter", () => {
    expect(titleCase("fOO_bAR")).toBe("FOO BAR");
  });

  it("returns empty string unchanged", () => {
    expect(titleCase("")).toBe("");
  });
});
