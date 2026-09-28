import { describe, it, expect } from "vitest";
import { fmtStorage, fmtTime, titleCase } from "../lib/format";

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

describe("fmtStorage", () => {
  const MB = 1024 * 1024;

  it("uses binary megabytes, so a 500 MiB quota reads 500 MB like the upload error", () => {
    expect(fmtStorage(500 * MB)).toBe("500 MB");
  });

  it("keeps one decimal below 10 and drops it from 10 up", () => {
    expect(fmtStorage(4.25 * MB)).toBe("4.3 MB");
    expect(fmtStorage(37.4 * MB)).toBe("37 MB");
  });

  it("formats zero as 0 MB", () => {
    expect(fmtStorage(0)).toBe("0 MB");
  });

  it("switches to GB at 1024 MB", () => {
    expect(fmtStorage(1023 * MB)).toBe("1023 MB");
    expect(fmtStorage(1024 * MB)).toBe("1 GB");
    expect(fmtStorage(1536 * MB)).toBe("1.5 GB");
    expect(fmtStorage(100 * 1024 * MB)).toBe("100 GB");
  });

  it("clamps negative and non-finite values to 0 MB", () => {
    expect(fmtStorage(-1)).toBe("0 MB");
    expect(fmtStorage(NaN)).toBe("0 MB");
    expect(fmtStorage(Infinity)).toBe("0 MB");
  });
});
