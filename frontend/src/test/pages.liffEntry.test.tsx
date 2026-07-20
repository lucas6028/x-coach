import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";

const { liffState } = vi.hoisted(() => ({
  liffState: { configured: true, sdk: null as unknown },
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => liffState.configured,
  initLiff: () => Promise.resolve(liffState.sdk),
}));

import Landing from "../landing/Landing";
import Login from "../pages/Login";
import { LiffProvider } from "../lib/liffContext";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";

// Reflects the router's current search string so a redirect target that drops (or keeps) it is
// observable — real window.location is untouched by MemoryRouter, so this reads the router's own
// location, exactly like the fix under test does via useLocation() in Landing/Login.
function StudioStandIn() {
  const location = useLocation();
  return <p>{`studio${location.search}`}</p>;
}

// Render the real route table for the two entry routes plus a studio stand-in, so a redirect
// is observable as "the studio marker is on screen".
const renderAt = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <LiffProvider>
        <AuthProvider>
          <I18nProvider>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/login" element={<Login />} />
              <Route path="/app" element={<StudioStandIn />} />
            </Routes>
          </I18nProvider>
        </AuthProvider>
      </LiffProvider>
    </MemoryRouter>
  );

// jsdom's navigator.userAgent is read-only; redefine it per test (mirrors lib.liffContext.test.tsx).
function stubUserAgent(ua: string) {
  Object.defineProperty(window.navigator, "userAgent", { value: ua, configurable: true });
}
const REAL_UA = window.navigator.userAgent;

beforeEach(() => {
  liffState.configured = true;
  // `isLoggedIn` is required: lib/auth's LINE auto-login effect calls
  // `liff.isLoggedIn()` outside its try block (auth.tsx:213). A fake SDK missing it throws
  // an unhandled rejection inside that effect and pollutes the test output.
  liffState.sdk = { isInClient: () => true, isLoggedIn: () => false };
  stubUserAgent(REAL_UA);
});

afterEach(() => {
  stubUserAgent(REAL_UA);
});

describe("entry points inside the LINE app", () => {
  it("redirects the landing page to the studio", async () => {
    renderAt("/");
    await waitFor(() => expect(screen.getByText(/^studio/)).toBeInTheDocument());
  });

  it("redirects the login page to the studio", async () => {
    renderAt("/login");
    await waitFor(() => expect(screen.getByText(/^studio/)).toBeInTheDocument());
  });

  it("preserves the query string (liff.state) when redirecting the landing page", async () => {
    renderAt("/?liff.state=%2Fhistory");
    await waitFor(() =>
      expect(screen.getByText("studio?liff.state=%2Fhistory")).toBeInTheDocument()
    );
  });

  it("preserves the query string (liff.state) when redirecting the login page", async () => {
    renderAt("/login?liff.state=%2Fhistory");
    await waitFor(() =>
      expect(screen.getByText("studio?liff.state=%2Fhistory")).toBeInTheDocument()
    );
  });
});

describe("entry points on the web (regression guard)", () => {
  beforeEach(() => {
    liffState.sdk = { isInClient: () => false, isLoggedIn: () => false };
  });

  it("keeps the landing page", async () => {
    renderAt("/");
    await waitFor(() => expect(screen.queryByText(/^studio/)).not.toBeInTheDocument());
    // Precise match on the landing page's known CTA copy (see landing.test.tsx), not a
    // permissive regex that would also match the redirect target's own "studio" text.
    expect(screen.getAllByRole("link", { name: /Open the demo/i }).length).toBeGreaterThan(0);
  });

  it("keeps the login form", async () => {
    renderAt("/login");
    await waitFor(() => expect(screen.queryByText(/^studio/)).not.toBeInTheDocument());
    // Prove the form actually rendered, not just that "studio" is absent (which would also
    // be true of a blank page).
    expect(screen.getByRole("heading", { name: /sign in/i })).toBeInTheDocument();
  });
});

// Fix 2: the optimistic in-client guess (LINE user agent / liff.state query params — see
// lib/liffContext) is not a confirmed LIFF context. Opening the plain site URL inside LINE's
// in-app browser (e.g. a shared link in a chat) matches the same user-agent signal but
// liff.isInClient() answers false once the SDK actually confirms. The redirect off Landing/Login
// is irreversible (no route back), so it must never fire on the unconfirmed guess.
describe("entry points — guess says in-client, SDK corrects to web", () => {
  beforeEach(() => {
    // LINE's in-app browser UA, so the synchronous first-paint guess says "in-client" — but the
    // (mocked) SDK will say otherwise once it resolves.
    stubUserAgent("Mozilla/5.0 (iPhone) AppleWebKit/605.1.15 Line/14.2.0");
    liffState.sdk = { isInClient: () => false, isLoggedIn: () => false };
  });

  it("Landing: shows a neutral loading state instead of the marketing page or a redirect, then settles on the marketing page", async () => {
    renderAt("/");
    // Pending window: neither the redirect target nor the marketing page's own content is up yet.
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.queryByText(/^studio/)).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Open the demo/i })).not.toBeInTheDocument();
    // Fix: this fires on every LINE cold start before any video exists, so the wait must not
    // narrate the analysis pipeline (LumenLoader "scan" copy) — that describes work that isn't
    // happening yet. A neutral "Loading…" replaces it.
    expect(screen.getByText("Loading…")).toBeInTheDocument();
    expect(
      screen.queryByText(/Reading pose|Checking mechanics|Lighting the why/i)
    ).not.toBeInTheDocument();
    // Once the SDK corrects the guess: the real marketing page, never the studio.
    await waitFor(() =>
      expect(screen.getAllByRole("link", { name: /Open the demo/i }).length).toBeGreaterThan(0)
    );
    expect(screen.queryByText(/^studio/)).not.toBeInTheDocument();
  });

  it("Login: shows a neutral loading state instead of the sign-in form or a redirect, then settles on the sign-in form", async () => {
    renderAt("/login");
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.queryByText(/^studio/)).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /sign in/i })).not.toBeInTheDocument();
    // Same fix as Landing: neutral copy, not the analysis-pipeline narration.
    expect(screen.getByText("Loading…")).toBeInTheDocument();
    expect(
      screen.queryByText(/Reading pose|Checking mechanics|Lighting the why/i)
    ).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("heading", { name: /sign in/i })).toBeInTheDocument());
    expect(screen.queryByText(/^studio/)).not.toBeInTheDocument();
  });
});
