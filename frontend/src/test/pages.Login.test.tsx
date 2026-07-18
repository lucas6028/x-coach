import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
import Login from "../pages/Login";

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
    signInWithLine: vi.fn().mockResolvedValue(undefined),
    signOut: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  } as ReturnType<typeof useAuth>;
}

function renderLogin() {
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={["/login"]}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/app" element={<div>app studio</div>} />
          <Route path="/" element={<div>home page</div>} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

beforeEach(() => mockUseAuth.mockReset());

describe("Login", () => {
  it("renders the sign-in form by default", () => {
    mockUseAuth.mockReturnValue(makeAuth());
    renderLogin();
    expect(screen.getByRole("heading", { name: "Sign in" })).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Continue with Google/i })).toBeInTheDocument();
  });

  it("toggles to the sign-up form", async () => {
    mockUseAuth.mockReturnValue(makeAuth());
    renderLogin();
    await userEvent.click(screen.getByRole("button", { name: "Create an account" }));
    expect(screen.getByRole("heading", { name: "Create your account" })).toBeInTheDocument();
  });

  it("signs in and navigates to the studio", async () => {
    const auth = makeAuth();
    mockUseAuth.mockReturnValue(auth);
    renderLogin();
    await userEvent.type(screen.getByLabelText("Email"), "ada@x.com");
    await userEvent.type(screen.getByLabelText("Password"), "secret1");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(screen.getByText("app studio")).toBeInTheDocument());
    expect(auth.signInWithPassword).toHaveBeenCalledWith("ada@x.com", "secret1");
  });

  it("shows an error when sign-in fails", async () => {
    const auth = makeAuth({
      signInWithPassword: vi.fn().mockRejectedValue(new Error("Invalid login credentials")),
    });
    mockUseAuth.mockReturnValue(auth);
    renderLogin();
    await userEvent.type(screen.getByLabelText("Email"), "ada@x.com");
    await userEvent.type(screen.getByLabelText("Password"), "secret1");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() =>
      expect(screen.getByText("Invalid login credentials")).toBeInTheDocument()
    );
  });

  it("shows the confirm-email notice on sign-up", async () => {
    const auth = makeAuth({
      signUpWithPassword: vi.fn().mockResolvedValue({ needsConfirmation: true }),
    });
    mockUseAuth.mockReturnValue(auth);
    renderLogin();
    await userEvent.click(screen.getByRole("button", { name: "Create an account" }));
    await userEvent.type(screen.getByLabelText("Email"), "ada@x.com");
    await userEvent.type(screen.getByLabelText("Password"), "secret1");
    await userEvent.click(screen.getByRole("button", { name: "Create account" }));
    await waitFor(() =>
      expect(screen.getByText(/Check your inbox/i)).toBeInTheDocument()
    );
  });

  it("starts Google sign-in", async () => {
    const auth = makeAuth();
    mockUseAuth.mockReturnValue(auth);
    renderLogin();
    await userEvent.click(screen.getByRole("button", { name: /Continue with Google/i }));
    expect(auth.signInWithGoogle).toHaveBeenCalled();
  });

  it("starts LINE sign-in", async () => {
    const auth = makeAuth();
    mockUseAuth.mockReturnValue(auth);
    renderLogin();
    await userEvent.click(screen.getByRole("button", { name: /Continue with LINE/i }));
    expect(auth.signInWithLine).toHaveBeenCalled();
  });

  it("shows an error when LINE sign-in fails", async () => {
    const auth = makeAuth({
      signInWithLine: vi.fn().mockRejectedValue(new Error("LINE login is not configured.")),
    });
    mockUseAuth.mockReturnValue(auth);
    renderLogin();
    await userEvent.click(screen.getByRole("button", { name: /Continue with LINE/i }));
    await waitFor(() =>
      expect(screen.getByText("LINE login is not configured.")).toBeInTheDocument()
    );
  });

  it("warns and disables submit when auth is not configured", () => {
    mockUseAuth.mockReturnValue(makeAuth({ configured: false }));
    renderLogin();
    expect(screen.getByText(/isn't set up/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign in" })).toBeDisabled();
  });

  it("redirects an already-authenticated user", () => {
    mockUseAuth.mockReturnValue(makeAuth({ user: { email: "ada@x.com" } as never }));
    renderLogin();
    expect(screen.getByText("app studio")).toBeInTheDocument();
  });
});
