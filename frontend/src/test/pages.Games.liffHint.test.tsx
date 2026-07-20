import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

const { liffState } = vi.hoisted(() => ({
  liffState: { configured: true, sdk: null as unknown },
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => liffState.configured,
  initLiff: () => Promise.resolve(liffState.sdk),
}));

import Games from "../pages/Games";
import { LiffProvider } from "../lib/liffContext";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";

const HINT = /camera may not work inside LINE/i;

const renderGames = () =>
  render(
    <MemoryRouter initialEntries={["/games"]}>
      <LiffProvider>
        <AuthProvider>
          <I18nProvider>
            <Games />
          </I18nProvider>
        </AuthProvider>
      </LiffProvider>
    </MemoryRouter>
  );

beforeEach(() => {
  liffState.configured = true;
  // isLoggedIn is required alongside isInClient: auth.tsx's auto-login effect calls
  // liff.isLoggedIn() outside its try block, so a fake SDK missing it throws an
  // unhandled rejection and pollutes test output (known issue, tasks 3/4).
  liffState.sdk = { isInClient: () => false, isLoggedIn: () => false };
});

describe("Games — camera hint", () => {
  it("warns about the LINE in-app camera when running in-client", async () => {
    liffState.sdk = { isInClient: () => true, isLoggedIn: () => false };
    renderGames();
    await waitFor(() => expect(screen.getByText(HINT)).toBeInTheDocument());
  });

  it("shows no hint on the web", async () => {
    renderGames();
    await waitFor(() => expect(screen.getByText("Pose Arcade")).toBeInTheDocument());
    expect(screen.queryByText(HINT)).not.toBeInTheDocument();
  });
});
