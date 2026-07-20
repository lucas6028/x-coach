import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

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
              <Route path="/app" element={<p>studio</p>} />
            </Routes>
          </I18nProvider>
        </AuthProvider>
      </LiffProvider>
    </MemoryRouter>
  );

beforeEach(() => {
  liffState.configured = true;
  // `isLoggedIn` is required: lib/auth's LINE auto-login effect calls
  // `liff.isLoggedIn()` outside its try block (auth.tsx:213). A fake SDK missing it throws
  // an unhandled rejection inside that effect and pollutes the test output.
  liffState.sdk = { isInClient: () => true, isLoggedIn: () => false };
});

describe("entry points inside the LINE app", () => {
  it("redirects the landing page to the studio", async () => {
    renderAt("/");
    await waitFor(() => expect(screen.getByText("studio")).toBeInTheDocument());
  });

  it("redirects the login page to the studio", async () => {
    renderAt("/login");
    await waitFor(() => expect(screen.getByText("studio")).toBeInTheDocument());
  });
});

describe("entry points on the web (regression guard)", () => {
  beforeEach(() => {
    liffState.sdk = { isInClient: () => false, isLoggedIn: () => false };
  });

  it("keeps the landing page", async () => {
    renderAt("/");
    await waitFor(() => expect(screen.queryByText("studio")).not.toBeInTheDocument());
    // Precise match on the landing page's known CTA copy (see landing.test.tsx), not a
    // permissive regex that would also match the redirect target's own "studio" text.
    expect(screen.getAllByRole("link", { name: /Open the demo/i }).length).toBeGreaterThan(0);
  });

  it("keeps the login form", async () => {
    renderAt("/login");
    await waitFor(() => expect(screen.queryByText("studio")).not.toBeInTheDocument());
    // Prove the form actually rendered, not just that "studio" is absent (which would also
    // be true of a blank page).
    expect(screen.getByRole("heading", { name: /sign in/i })).toBeInTheDocument();
  });
});
