import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// Drive the branch through the real provider, with lib/liff (the SDK edge) faked.
const { liffState } = vi.hoisted(() => ({
  liffState: { configured: true, sdk: null as unknown },
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => liffState.configured,
  initLiff: () => Promise.resolve(liffState.sdk),
}));

import AppLayout from "../components/AppLayout";
import { LiffProvider } from "../lib/liffContext";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";

const renderLayout = () =>
  render(
    <MemoryRouter initialEntries={["/app"]}>
      <LiffProvider>
        <AuthProvider>
          <I18nProvider>
            <AppLayout title="My records">
              <p>page body</p>
            </AppLayout>
          </I18nProvider>
        </AuthProvider>
      </LiffProvider>
    </MemoryRouter>
  );

beforeEach(() => {
  liffState.configured = true;
  liffState.sdk = { isInClient: () => false, isLoggedIn: () => false };
});

describe("AppLayout — inside the LINE app", () => {
  beforeEach(() => {
    liffState.sdk = { isInClient: () => true, isLoggedIn: () => false };
  });

  it("renders the tab bar and drops the sidebar", async () => {
    renderLayout();
    await waitFor(() => expect(screen.getByRole("navigation")).toBeInTheDocument());
    // The web sidebar's signature entries are gone.
    expect(screen.queryByText("Prototype v0.1")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /New analysis/i })).not.toBeInTheDocument();
    // The tab bar's four destinations are present.
    expect(screen.getByRole("link", { name: /Analyse/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Settings/i })).toBeInTheDocument();
  });

  it("drops the web navbar's brand lockup", async () => {
    renderLayout();
    await waitFor(() => expect(screen.getByRole("navigation")).toBeInTheDocument());
    expect(screen.queryByLabelText("X-Coach")).not.toBeInTheDocument();
  });

  it("still renders the page body", async () => {
    renderLayout();
    await waitFor(() => expect(screen.getByText("page body")).toBeInTheDocument());
  });
});

describe("AppLayout — on the web (regression guard)", () => {
  it("keeps the existing navbar + sidebar shell", async () => {
    renderLayout();
    // AppLayout renders Sidebar twice (desktop rail hidden on mobile, mobile drawer hidden on desktop),
    // so both "Prototype v0.1" and "New analysis" button appear twice in the DOM.
    await waitFor(() => expect(screen.getAllByText("Prototype v0.1")).toHaveLength(2));
    expect(screen.getByLabelText("X-Coach")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /New analysis/i })).toHaveLength(2);
    expect(screen.getByText("page body")).toBeInTheDocument();
  });
});
