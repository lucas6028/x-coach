import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

// The rail's footer picks its affordance from useAuth's { user, lineAuthenticating }:
// account menu (signed in) · "signing in with LINE" indicator (silent auto-login running) ·
// sign-in link (logged out and idle). Mock useAuth so all three are reachable — the real
// AuthProvider can't be coaxed into the mid-exchange state without a full LIFF setup.
// (This cluster used to sit in the content card's top row; see components.Header.test.tsx for
// what that row carries now.)
const { mockUseAuth } = vi.hoisted(() => ({ mockUseAuth: vi.fn() }));
vi.mock("../lib/auth", () => ({ useAuth: mockUseAuth }));

import Sidebar from "../components/Sidebar";

function renderRail(open = true) {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <Sidebar
          open={open}
          width={open ? 236 : 76}
          animate
          onNewAnalysis={vi.fn()}
        />
      </I18nProvider>
    </MemoryRouter>
  );
}

const signedIn = {
  user: { email: "ada@example.com", user_metadata: {} },
  lineAuthenticating: false,
  signOut: vi.fn(),
};

describe("Sidebar — account footer", () => {
  beforeEach(() => mockUseAuth.mockReset());

  it("shows the sign-in link when logged out and idle", () => {
    mockUseAuth.mockReturnValue({ user: null, lineAuthenticating: false });
    renderRail();
    expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute("href", "/login");
    expect(screen.queryByText(/signing in with line/i)).not.toBeInTheDocument();
  });

  it("shows a 'signing in with LINE' indicator while the auto-login exchange runs", () => {
    mockUseAuth.mockReturnValue({ user: null, lineAuthenticating: true });
    renderRail();
    expect(screen.getByText(/signing in with line/i)).toBeInTheDocument();
    // The log-in link must NOT be shown — that's the misleading state we're replacing.
    expect(screen.queryByRole("link", { name: /sign in/i })).not.toBeInTheDocument();
  });

  it("shows the account menu once signed in, indicator gone", () => {
    mockUseAuth.mockReturnValue(signedIn);
    renderRail();
    expect(screen.getByRole("button", { name: /account menu/i })).toBeInTheDocument();
    expect(screen.queryByText(/signing in with line/i)).not.toBeInTheDocument();
  });

  // The 76px rail has room for the avatar but not the name — but the control must still BE there.
  // Dropping the account cluster on collapse would leave no way to sign out without expanding.
  it("keeps the account control when collapsed, without the name", () => {
    mockUseAuth.mockReturnValue(signedIn);
    renderRail(false);
    expect(screen.getByRole("button", { name: /account menu/i })).toBeInTheDocument();
    expect(screen.queryByText("ada@example.com")).not.toBeInTheDocument();
  });

  it("shows the display name beside the avatar when expanded", () => {
    mockUseAuth.mockReturnValue(signedIn);
    renderRail();
    expect(screen.getByText("ada@example.com")).toBeInTheDocument();
  });
});
