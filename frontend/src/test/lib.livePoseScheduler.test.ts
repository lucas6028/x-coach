import {
  createLivePoseSchedule,
  shouldRunLivePoseInference,
} from "../lib/livePoseScheduler";
import { describe, expect, it } from "vitest";

describe("live pose scheduler", () => {
  it("caps fresh 60fps camera frames to a 30fps inference budget", () => {
    const schedule = createLivePoseSchedule();

    expect(shouldRunLivePoseInference(schedule, 0, 0, 30)).toBe(true);
    expect(shouldRunLivePoseInference(schedule, 1 / 60, 16, 30)).toBe(false);
    expect(shouldRunLivePoseInference(schedule, 2 / 60, 34, 30)).toBe(true);
  });

  it("re-offers a fresh frame that arrived just before the cadence budget opened", () => {
    const schedule = createLivePoseSchedule();

    expect(shouldRunLivePoseInference(schedule, 0, 0, 30)).toBe(true);
    expect(shouldRunLivePoseInference(schedule, 1 / 30, 33, 30)).toBe(false);
    expect(shouldRunLivePoseInference(schedule, 1 / 30, 34, 30)).toBe(true);
  });

  it("does not infer the same paused frame twice", () => {
    const schedule = createLivePoseSchedule();

    expect(shouldRunLivePoseInference(schedule, 1, 0)).toBe(true);
    expect(shouldRunLivePoseInference(schedule, 1, 100)).toBe(false);
  });

  it("accepts a media restart but rejects repeated and invalid timestamps", () => {
    const schedule = createLivePoseSchedule();

    expect(shouldRunLivePoseInference(schedule, 2, 0)).toBe(true);
    expect(shouldRunLivePoseInference(schedule, 1, 40)).toBe(true);
    expect(shouldRunLivePoseInference(schedule, Number.NaN, 80)).toBe(false);
  });
});
