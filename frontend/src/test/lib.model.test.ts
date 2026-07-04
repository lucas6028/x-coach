import { describe, it, expect, beforeEach } from "vitest";
import {
  CHAT_MODELS,
  DEFAULT_CHAT_MODEL,
  getStoredModel,
  setStoredModel,
} from "../lib/model";

describe("lib/model", () => {
  beforeEach(() => localStorage.clear());

  it("defaults to deepseek-v4-flash when unset", () => {
    expect(getStoredModel()).toBe(DEFAULT_CHAT_MODEL);
    expect(DEFAULT_CHAT_MODEL).toBe("deepseek/deepseek-v4-flash");
  });

  it("round-trips a valid selection", () => {
    setStoredModel("minimax/minimax-m3");
    expect(getStoredModel()).toBe("minimax/minimax-m3");
  });

  it("falls back to the default for an unrecognized stored value", () => {
    localStorage.setItem("chat_model", "evil/expensive-model");
    expect(getStoredModel()).toBe(DEFAULT_CHAT_MODEL);
  });

  it("offers exactly the four allow-listed models, each with a label", () => {
    expect(CHAT_MODELS.map((m) => m.id)).toEqual([
      "deepseek/deepseek-v4-flash",
      "xiaomi/mimo-v2.5",
      "minimax/minimax-m3",
      "tencent/hy3-preview",
    ]);
    CHAT_MODELS.forEach((m) => expect(m.label.length).toBeGreaterThan(0));
  });
});
