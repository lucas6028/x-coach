import { describe, it, expect, beforeEach } from "vitest";
import { getStoredModel, setStoredModel } from "../lib/model";

describe("lib/model", () => {
  beforeEach(() => localStorage.clear());

  it("returns '' (follow server default) when unset", () => {
    expect(getStoredModel()).toBe("");
  });

  it("round-trips a pinned selection", () => {
    setStoredModel("minimax/minimax-m3");
    expect(getStoredModel()).toBe("minimax/minimax-m3");
  });

  it("clears back to the server default with an empty id", () => {
    setStoredModel("minimax/minimax-m3");
    setStoredModel("");
    expect(getStoredModel()).toBe("");
  });
});
