import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
import AdminLogin from "../pages/admin/AdminLogin";
import RequireAuth from "../components/RequireAuth";

const mockUseAuth = vi.mocked(useAuth);

function makeAuth(overrides: Partial<ReturnType<typeof useAuth>> = {}) {
  return {
    user: null,
    session: null,
    loading: false,
    configured: true,
    signInWithPassword: vi.fn().mockResolvedValue(undefined),
    signUpWithPassword: vi.fn().mockResolvedValue({ needsConfirmation: false }),
    signInWithGoogle: vi.fn().mockResolvedValue(undefined),
    signOut: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  } as ReturnType<typeof useAuth>;
}

// Render the admin login at /admin/login with a stub /admin destination so a successful
// sign-in (or an already-authenticated session) resolves to a visible marker.
function renderAdminLogin(initial = "/admin/login") {
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={[initial]}>
        <Routes>
          <Route path="/admin/login" element={<AdminLogin />} />
          <Route path="/admin" element={<div>admin console</div>} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

beforeEach(() => mockUseAuth.mockReset());

describe("AdminLogin", () => {
  it("renders email + password with NO Google/OAuth button", () => {
    mockUseAuth.mockReturnValue(makeAuth());
    renderAdminLogin();
    expect(screen.getByRole("heading", { name: "Admin console" })).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Google/i })).not.toBeInTheDocument();
  });

  it("signs in with the entered credentials and navigates to /admin", async () => {
    const auth = makeAuth();
    mockUseAuth.mockReturnValue(auth);
    renderAdminLogin();
    await userEvent.type(screen.getByLabelText("Email"), "root@x.com");
    await userEvent.type(screen.getByLabelText("Password"), "secret1");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(screen.getByText("admin console")).toBeInTheDocument());
    expect(auth.signInWithPassword).toHaveBeenCalledWith("root@x.com", "secret1");
  });

  it("shows an error when sign-in fails", async () => {
    const auth = makeAuth({
      signInWithPassword: vi.fn().mockRejectedValue(new Error("nope")),
    });
    mockUseAuth.mockReturnValue(auth);
    renderAdminLogin();
    await userEvent.type(screen.getByLabelText("Email"), "root@x.com");
    await userEvent.type(screen.getByLabelText("Password"), "secret1");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() =>
      expect(screen.getByText(/Couldn't sign you in/i)).toBeInTheDocument()
    );
  });

  it("warns and disables submit when auth is not configured", () => {
    mockUseAuth.mockReturnValue(makeAuth({ configured: false }));
    renderAdminLogin();
    expect(screen.getByText(/isn't set up/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign in" })).toBeDisabled();
  });

  it("redirects an already-authenticated session to /admin", () => {
    mockUseAuth.mockReturnValue(makeAuth({ user: { email: "root@x.com" } as never }));
    renderAdminLogin();
    expect(screen.getByText("admin console")).toBeInTheDocument();
  });
});

describe("RequireAuth redirectTo", () => {
  it("sends a logged-out visitor of the admin tree to /admin/login (not /login)", () => {
    mockUseAuth.mockReturnValue(makeAuth({ user: null }));
    render(
      <I18nProvider>
        <MemoryRouter initialEntries={["/admin"]}>
          <Routes>
            <Route
              path="/admin"
              element={
                <RequireAuth redirectTo="/admin/login">
                  <div>admin console</div>
                </RequireAuth>
              }
            />
            <Route path="/admin/login" element={<div>admin sign in</div>} />
            <Route path="/login" element={<div>user sign in</div>} />
          </Routes>
        </MemoryRouter>
      </I18nProvider>
    );
    expect(screen.getByText("admin sign in")).toBeInTheDocument();
    expect(screen.queryByText("user sign in")).not.toBeInTheDocument();
  });
});
