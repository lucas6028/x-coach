import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

const { mockUseAuth, signOut } = vi.hoisted(() => ({
  mockUseAuth: vi.fn(),
  signOut: vi.fn(),
}));

vi.mock("../lib/auth", () => ({ useAuth: mockUseAuth }));

import AccountMenu from "../components/AccountMenu";

function withUser(user: unknown) {
  mockUseAuth.mockReturnValue({ user, signOut });
}

function renderMenu() {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <AccountMenu />
      </I18nProvider>
    </MemoryRouter>
  );
}

describe("AccountMenu", () => {
  beforeEach(() => {
    signOut.mockReset();
    mockUseAuth.mockReset();
  });

  it("renders nothing when signed out", () => {
    withUser(null);
    const { container } = renderMenu();
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the profile image when an avatar URL is present", () => {
    withUser({
      email: "lifter@example.com",
      user_metadata: { avatar_url: "https://example.com/me.png", full_name: "Test Lifter" },
    });
    const { container } = renderMenu();
    const img = container.querySelector("img");
    expect(img).toHaveAttribute("src", "https://example.com/me.png");
  });

  it("falls back to an initial when there is no avatar", () => {
    withUser({ email: "ada@example.com", user_metadata: {} });
    const { container } = renderMenu();
    expect(container.querySelector("img")).not.toBeInTheDocument();
    expect(screen.getByText("A")).toBeInTheDocument();
  });

  it("does not show the email in the dropdown", async () => {
    const user = userEvent.setup();
    withUser({ email: "ada@example.com", user_metadata: {} });
    renderMenu();
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    expect(screen.queryByText("ada@example.com")).not.toBeInTheDocument();
  });

  it("links to settings and signs out from the menu", async () => {
    const user = userEvent.setup();
    withUser({ email: "ada@example.com", user_metadata: {} });
    renderMenu();

    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /account menu/i }));

    expect(screen.getByRole("menu")).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /settings/i })).toHaveAttribute("href", "/settings");

    await user.click(screen.getByRole("menuitem", { name: /sign out/i }));
    expect(signOut).toHaveBeenCalledOnce();
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });
});
