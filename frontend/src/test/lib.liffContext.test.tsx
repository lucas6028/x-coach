import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

// The provider is the unit under test; lib/liff is its (already-tested) dependency.
const { liffState } = vi.hoisted(() => ({
  liffState: { configured: true, sdk: null as unknown },
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => liffState.configured,
  initLiff: () => Promise.resolve(liffState.sdk),
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
  stubUserAgent(REAL_UA);
  window.history.replaceState({}, "", "/app");
});

afterEach(() => {
  stubUserAgent(REAL_UA);
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
