import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";

// The provider is the unit under test; lib/liff is its (already-tested) dependency.
const { liffState } = vi.hoisted(() => ({
  // `hang: true` makes initLiff() return a promise that never settles (never-resolves
  // resilience test); `reject: true` makes it reject (defense-in-depth, since the real
  // initLiff() never actually rejects — see lib/liff.ts).
  liffState: { configured: true, sdk: null as unknown, hang: false, reject: false },
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => liffState.configured,
  initLiff: () => {
    if (liffState.hang) return new Promise(() => {});
    if (liffState.reject) return Promise.reject(new Error("boom"));
    return Promise.resolve(liffState.sdk);
  },
}));

import { LiffProvider, useLiffContext } from "../lib/liffContext";

function Probe() {
  const { ready, isInClient } = useLiffContext();
  return <span>{`ready=${ready} inClient=${isInClient}`}</span>;
}

const renderProbe = () =>
  render(
    <LiffProvider>
      <Probe />
    </LiffProvider>
  );

// jsdom's navigator.userAgent is read-only; redefine it per test.
function stubUserAgent(ua: string) {
  Object.defineProperty(window.navigator, "userAgent", { value: ua, configurable: true });
}

const REAL_UA = window.navigator.userAgent;

beforeEach(() => {
  liffState.configured = true;
  liffState.sdk = { isInClient: () => true };
  liffState.hang = false;
  liffState.reject = false;
  stubUserAgent(REAL_UA);
  window.history.replaceState({}, "", "/app");
});

afterEach(() => {
  stubUserAgent(REAL_UA);
  vi.useRealTimers();
});

describe("LiffProvider — optimistic guess before init resolves", () => {
  it("guesses in-client from the LINE user agent", () => {
    stubUserAgent("Mozilla/5.0 (iPhone) AppleWebKit/605.1.15 Line/14.2.0");
    renderProbe();
    // First paint, before the init promise resolves: the guess is already applied.
    expect(screen.getByText("ready=false inClient=true")).toBeInTheDocument();
  });

  it("guesses in-client from a liff.state query param", () => {
    window.history.replaceState({}, "", "/app?liff.state=%2Fhistory");
    renderProbe();
    expect(screen.getByText("ready=false inClient=true")).toBeInTheDocument();
  });

  it("guesses in-client from a liff-referrer query param", () => {
    window.history.replaceState({}, "", "/app?liff-referrer=https%3A%2F%2Fliff.line.me");
    renderProbe();
    expect(screen.getByText("ready=false inClient=true")).toBeInTheDocument();
  });

  it("guesses web for a plain browser", () => {
    renderProbe();
    expect(screen.getByText("ready=false inClient=false")).toBeInTheDocument();
  });
});

describe("LiffProvider — correction once the SDK answers", () => {
  it("corrects an optimistic in-client guess that was wrong", async () => {
    stubUserAgent("Mozilla/5.0 (iPhone) Line/14.2.0");
    liffState.sdk = { isInClient: () => false };
    renderProbe();
    await waitFor(() =>
      expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument()
    );
  });

  it("promotes a web guess to in-client when the SDK says so", async () => {
    liffState.sdk = { isInClient: () => true };
    renderProbe();
    await waitFor(() =>
      expect(screen.getByText("ready=true inClient=true")).toBeInTheDocument()
    );
  });

  it("reads a failed init (null) as not in-client", async () => {
    liffState.sdk = null;
    renderProbe();
    await waitFor(() =>
      expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument()
    );
  });
});

// Fix: neither of these may strand `ready` at false forever — on the app's entry route
// (Landing/Login) that now means a permanent spinner, not just a stale guess.
describe("LiffProvider — settles ready even when the SDK-side callback misbehaves", () => {
  it("degrades to plain web when liff.isInClient() throws", async () => {
    stubUserAgent("Mozilla/5.0 (iPhone) Line/14.2.0"); // optimistic guess: in-client
    liffState.sdk = {
      isInClient: () => {
        throw new Error("boom");
      },
    };
    renderProbe();
    expect(screen.getByText("ready=false inClient=true")).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument()
    );
  });

  it("degrades to plain web when initLiff() itself rejects", async () => {
    stubUserAgent("Mozilla/5.0 (iPhone) Line/14.2.0"); // optimistic guess: in-client
    liffState.reject = true;
    renderProbe();
    expect(screen.getByText("ready=false inClient=true")).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument()
    );
  });

  it("degrades to plain web via the timeout when initLiff() never settles", () => {
    vi.useFakeTimers();
    stubUserAgent("Mozilla/5.0 (iPhone) Line/14.2.0"); // optimistic guess: in-client
    liffState.hang = true;
    renderProbe();
    expect(screen.getByText("ready=false inClient=true")).toBeInTheDocument();
    // Still pending well before the timeout — proves the settle came from the race, not
    // from some other timer firing early.
    act(() => {
      vi.advanceTimersByTime(5_999);
    });
    expect(screen.getByText("ready=false inClient=true")).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument();
  });
});

describe("LiffProvider — unconfigured", () => {
  it("is immediately ready and never in-client without VITE_LIFF_ID", () => {
    liffState.configured = false;
    stubUserAgent("Mozilla/5.0 (iPhone) Line/14.2.0");
    renderProbe();
    // No LIFF id: the LINE user agent is irrelevant, and there is nothing to wait for.
    expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument();
  });
});

describe("useLiffContext — outside a provider", () => {
  it("defaults to the web shell", () => {
    render(<Probe />);
    expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument();
  });
});
