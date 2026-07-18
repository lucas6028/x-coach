import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

// Header picks the top-right affordance from useAuth's { user, lineAuthenticating }:
// account menu (signed in) · "signing in with LINE" indicator (silent auto-login running) ·
// sign-in link (logged out and idle). Mock useAuth so all three are reachable — the real
// AuthProvider can't be coaxed into the mid-exchange state without a full LIFF setup.
const { mockUseAuth } = vi.hoisted(() => ({ mockUseAuth: vi.fn() }));
vi.mock("../lib/auth", () => ({ useAuth: mockUseAuth }));

import Header from "../components/Header";

function renderHeader() {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <Header analysis={null} loading={false} />
      </I18nProvider>
    </MemoryRouter>
  );
}

describe("Header — auth affordance", () => {
  beforeEach(() => mockUseAuth.mockReset());

  it("shows the sign-in link when logged out and idle", () => {
    mockUseAuth.mockReturnValue({ user: null, lineAuthenticating: false });
    renderHeader();
    expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute("href", "/login");
    expect(screen.queryByText(/signing in with line/i)).not.toBeInTheDocument();
  });

  it("shows a 'signing in with LINE' indicator while the auto-login exchange runs", () => {
    mockUseAuth.mockReturnValue({ user: null, lineAuthenticating: true });
    renderHeader();
    expect(screen.getByText(/signing in with line/i)).toBeInTheDocument();
    // The log-in link must NOT be shown — that's the misleading state we're replacing.
    expect(screen.queryByRole("link", { name: /sign in/i })).not.toBeInTheDocument();
  });

  it("shows the account menu once signed in, indicator gone", () => {
    mockUseAuth.mockReturnValue({
      user: { email: "ada@example.com", user_metadata: {} },
      lineAuthenticating: false,
      signOut: vi.fn(),
    });
    renderHeader();
    expect(screen.getByRole("button", { name: /account menu/i })).toBeInTheDocument();
    expect(screen.queryByText(/signing in with line/i)).not.toBeInTheDocument();
  });
});
