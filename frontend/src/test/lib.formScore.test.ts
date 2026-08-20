import { describe, it, expect } from "vitest";
import { bandFor, formScore, scoreFromDetections } from "../lib/formScore";
import { mockAnalysis, mockCleanAnalysis, mockDetection, mockUnmeasuredAnalysis } from "./fixtures";

const withSeverity = (...severities: number[]) =>
  severities.map((severity) => ({ ...mockDetection, severity }));

describe("scoreFromDetections — the published rule", () => {
  it("gives a clean rep 100", () => {
    expect(scoreFromDetections([])).toEqual({ value: 100, band: "excellent" });
  });

  it("deducts 25 points per unit of severity", () => {
    expect(scoreFromDetections(withSeverity(0.5)).value).toBe(88); // 100 − 12.5, rounded
    expect(scoreFromDetections(withSeverity(1)).value).toBe(75);
    expect(scoreFromDetections(withSeverity(0.8, 0.8)).value).toBe(60);
  });

  it("floors at 20 rather than reaching zero — zero would read as a failed analysis", () => {
    expect(scoreFromDetections(withSeverity(1, 1, 1, 1, 1, 1)).value).toBe(20);
  });

  // A detector that ever emitted an out-of-range severity would otherwise push the score above
  // 100 or below the floor, i.e. print a number the rule does not define.
  it("clamps a severity outside 0–1 before applying it", () => {
    expect(scoreFromDetections(withSeverity(-3)).value).toBe(100);
    expect(scoreFromDetections(withSeverity(4)).value).toBe(75);
  });
});

describe("bandFor", () => {
  it.each([
    [100, "excellent"],
    [90, "excellent"],
    [89, "good"],
    [75, "good"],
    [74, "fair"],
    [55, "fair"],
    [54, "poor"],
    [20, "poor"],
  ])("maps %i to %s", (value, band) => {
    expect(bandFor(value)).toBe(band);
  });
});

describe("formScore — the unmeasured guard", () => {
  it("scores a measured clip", () => {
    // mockAnalysis carries one severity-0.8 fault: 100 − 20 = 80.
    expect(formScore(mockAnalysis)).toEqual({ value: 80, band: "good" });
  });

  it("scores a measured clean rep 100", () => {
    expect(formScore(mockCleanAnalysis)?.value).toBe(100);
  });

  // The whole point of the guard: an unmeasured clip has an empty detection list for the SAME
  // reason a flawless one does, so scoring it would print "100 — Excellent" over a clip nothing
  // was measured on.
  it("returns null when nothing in the clip was measurable", () => {
    expect(formScore(mockUnmeasuredAnalysis)).toBeNull();
  });
});
