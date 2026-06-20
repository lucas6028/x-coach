import { describe, it, expect, beforeEach, vi } from "vitest";
import { getStoredTheme } from "../lib/theme";

describe("getStoredTheme", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("returns 'system' when nothing is stored", () => {
    expect(getStoredTheme()).toBe("system");
  });

  it("returns 'light' when stored", () => {
    localStorage.setItem("theme", "light");
    expect(getStoredTheme()).toBe("light");
  });

  it("returns 'dark' when stored", () => {
    localStorage.setItem("theme", "dark");
    expect(getStoredTheme()).toBe("dark");
  });

  it("returns 'system' when stored", () => {
    localStorage.setItem("theme", "system");
    expect(getStoredTheme()).toBe("system");
  });

  it("falls back to 'system' for an invalid stored value", () => {
    localStorage.setItem("theme", "midnight");
    expect(getStoredTheme()).toBe("system");
  });
});
