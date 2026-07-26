import { describe, it, expect, vi, beforeEach } from "vitest";

// The SDK module is mocked once with a getter so each test can swap the fake in place —
// no vi.resetModules() needed (which would break coverage attribution for lib/liff).
const { sdkState } = vi.hoisted(() => ({
  sdkState: { sdk: null as unknown },
}));

vi.mock("@line/liff", () => ({
  get default() {
    return sdkState.sdk;
  },
}));

import {
  initLiff,
  isInLiffClient,
  isLiffConfigured,
  getLiffIdToken,
  _resetLiffForTests,
} from "../lib/liff";

function fakeSdk(overrides: Record<string, unknown> = {}) {
  return {
    init: vi.fn().mockResolvedValue(undefined),
    isInClient: vi.fn().mockReturnValue(true),
    isLoggedIn: vi.fn().mockReturnValue(true),
    getIDToken: vi.fn().mockReturnValue("id-token"),
    ...overrides,
  };
}

beforeEach(() => {
  vi.unstubAllEnvs();
  // Stub VITE_LIFF_ID to empty EXPLICITLY rather than relying on it being absent. unstubAllEnvs
  // restores the *ambient* environment, so on a developer machine whose .env supplies a real LIFF
  // id the "unconfigured" case below was silently configured and the assertion inverted -- the
  // degradation path stopped being tested exactly where someone had LINE set up locally. The
  // configured describe re-stubs this to a real id in its own beforeEach, which runs after.
  vi.stubEnv("VITE_LIFF_ID", "");
  _resetLiffForTests();
  sdkState.sdk = fakeSdk();
});

describe("lib/liff (unconfigured)", () => {
  it("degrades to null/false without VITE_LIFF_ID", async () => {
    expect(isLiffConfigured()).toBe(false);
    expect(await initLiff()).toBeNull();
    expect(await isInLiffClient()).toBe(false);
    expect(await getLiffIdToken()).toBeNull();
  });
});

describe("lib/liff (configured)", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_LIFF_ID", "1234567890-test");
  });

  it("initialises once with the configured id and caches the SDK", async () => {
    const sdk = fakeSdk();
    sdkState.sdk = sdk;
    expect(isLiffConfigured()).toBe(true);
    const first = await initLiff();
    const second = await initLiff();
    expect(first).toBe(sdk);
    expect(second).toBe(sdk);
    expect(sdk.init).toHaveBeenCalledTimes(1);
    expect(sdk.init).toHaveBeenCalledWith({ liffId: "1234567890-test" });
  });

  it("degrades to null when init rejects", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    sdkState.sdk = fakeSdk({ init: vi.fn().mockRejectedValue(new Error("bad liff id")) });
    expect(await initLiff()).toBeNull();
    expect(await isInLiffClient()).toBe(false);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("reports in-client state and the ID token", async () => {
    expect(await isInLiffClient()).toBe(true);
    expect(await getLiffIdToken()).toBe("id-token");
  });

  it("still returns the ID token in an external browser once logged in", async () => {
    sdkState.sdk = fakeSdk({
      isInClient: vi.fn().mockReturnValue(false),
      isLoggedIn: vi.fn().mockReturnValue(true),
    });
    expect(await isInLiffClient()).toBe(false);
    expect(await getLiffIdToken()).toBe("id-token");
  });

  it("returns no ID token when not logged in", async () => {
    sdkState.sdk = fakeSdk({ isLoggedIn: vi.fn().mockReturnValue(false) });
    expect(await getLiffIdToken()).toBeNull();
  });

  it("returns null when getIDToken throws", async () => {
    sdkState.sdk = fakeSdk({
      getIDToken: vi.fn(() => {
        throw new Error("no openid scope");
      }),
    });
    expect(await getLiffIdToken()).toBeNull();
  });

  it("_resetLiffForTests forces a re-init", async () => {
    const sdk = fakeSdk();
    sdkState.sdk = sdk;
    await initLiff();
    _resetLiffForTests();
    await initLiff();
    expect(sdk.init).toHaveBeenCalledTimes(2);
  });
});
