import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
import RequireAuth from "../components/RequireAuth";

const mockUseAuth = vi.mocked(useAuth);

function renderGuard() {
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={["/history"]}>
        <Routes>
          <Route
            path="/history"
            element={
              <RequireAuth>
                <div>secret content</div>
              </RequireAuth>
            }
          />
          <Route path="/login" element={<div>login page</div>} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

beforeEach(() => mockUseAuth.mockReset());

describe("RequireAuth", () => {
  it("shows a placeholder while the session is loading", () => {
    mockUseAuth.mockReturnValue({ user: null, loading: true } as ReturnType<typeof useAuth>);
    renderGuard();
    expect(screen.getByText("Checking your session…")).toBeInTheDocument();
    expect(screen.queryByText("secret content")).not.toBeInTheDocument();
  });

  it("redirects to /login when there is no user", () => {
    mockUseAuth.mockReturnValue({ user: null, loading: false } as ReturnType<typeof useAuth>);
    renderGuard();
    expect(screen.getByText("login page")).toBeInTheDocument();
    expect(screen.queryByText("secret content")).not.toBeInTheDocument();
  });

  it("renders children when authenticated", () => {
    mockUseAuth.mockReturnValue({
      user: { email: "a@x.com" },
      loading: false,
    } as ReturnType<typeof useAuth>);
    renderGuard();
    expect(screen.getByText("secret content")).toBeInTheDocument();
  });

  // Fix 4: inside LINE, lib/auth's silent token exchange runs with `loading` already false but
  // `lineAuthenticating` still true. Without this, tapping a guarded tab in that window bounced
  // to /login — which itself redirects straight back to /app in-client, so the one screen that
  // could explain the wait was unreachable.
  it("shows the loading placeholder (not a redirect) while the LINE token exchange is in flight", () => {
    mockUseAuth.mockReturnValue({
      user: null,
      loading: false,
      lineAuthenticating: true,
    } as ReturnType<typeof useAuth>);
    renderGuard();
    expect(screen.getByText("Checking your session…")).toBeInTheDocument();
    expect(screen.queryByText("login page")).not.toBeInTheDocument();
    expect(screen.queryByText("secret content")).not.toBeInTheDocument();
  });

  it("redirects to /login once the LINE exchange finishes with no session", () => {
    mockUseAuth.mockReturnValue({
      user: null,
      loading: false,
      lineAuthenticating: false,
    } as ReturnType<typeof useAuth>);
    renderGuard();
    expect(screen.getByText("login page")).toBeInTheDocument();
  });
});
