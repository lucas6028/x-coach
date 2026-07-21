import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_ANALYSIS_TIER, MODEL_URL, loadAnalysisTier, saveAnalysisTier } from "../lib/poseTier";

afterEach(() => localStorage.clear());

describe("poseTier", () => {
  it("maps every tier to its distinct .task model", () => {
    expect(MODEL_URL.lite).toContain("pose_landmarker_lite");
    expect(MODEL_URL.full).toContain("pose_landmarker_full");
    expect(MODEL_URL.heavy).toContain("pose_landmarker_heavy");
  });

  it("defaults to the validated tier when storage is empty", () => {
    expect(loadAnalysisTier()).toBe(DEFAULT_ANALYSIS_TIER);
  });

  it("round-trips a saved tier and ignores garbage", () => {
    saveAnalysisTier("heavy");
    expect(loadAnalysisTier()).toBe("heavy");
    localStorage.setItem("xcoach.poseTier", "bogus");
    expect(loadAnalysisTier()).toBe(DEFAULT_ANALYSIS_TIER);
  });
});
