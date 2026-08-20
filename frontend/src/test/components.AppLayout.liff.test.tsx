import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";

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
            <AppLayout>
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
    // The web sidebar itself (its version of "New analysis" — a full-width, labelled button) is
    // gone. The shell header carries its own icon-only "New analysis" action instead (see the
    // "header actions" describe block below), so this only checks the sidebar's own labelled
    // rendering is absent, not the accessible name.
    expect(screen.queryByText("New analysis")).not.toBeInTheDocument();
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

  it("threads the studio's new-analysis action into the shell header", async () => {
    renderLayout();
    await waitFor(() => expect(screen.getByRole("navigation")).toBeInTheDocument());
    // The point here is that the shell actually renders and wires the action, not stranding the
    // user with tabs and no way to start a second analysis. "New analysis" appears twice by
    // design in the phone shell — the header's upload button and the bar's raised centre action.
    expect(screen.getAllByRole("button", { name: /New analysis/i }).length).toBeGreaterThan(0);
  });

  // End-to-end for the Fix 1 blocker: a real click, through a real router, actually lands the
  // user back on the studio — not just "the prop exists and got called" (covered above and in
  // components.LiffAppShell.test.tsx), but that AppLayout's own navigate("/app") fallback (used
  // here since this render passes no onNewAnalysis) fires for real off a non-studio page.
  it("clicking the header's New analysis action navigates into the studio from a non-studio page", async () => {
    render(
      <MemoryRouter initialEntries={["/history"]}>
        <LiffProvider>
          <AuthProvider>
            <I18nProvider>
              <Routes>
                <Route
                  path="/history"
                  element={
                    <AppLayout>
                      <p>history body</p>
                    </AppLayout>
                  }
                />
                <Route path="/app" element={<p>studio</p>} />
              </Routes>
            </I18nProvider>
          </AuthProvider>
        </LiffProvider>
      </MemoryRouter>
    );
    await waitFor(() => expect(screen.getByRole("navigation")).toBeInTheDocument());
    expect(screen.getByText("history body")).toBeInTheDocument();
    await userEvent.click(screen.getAllByRole("button", { name: /New analysis/i })[0]);
    await waitFor(() => expect(screen.getByText("studio")).toBeInTheDocument());
  });
});

describe("AppLayout — on the web (regression guard)", () => {
  it("keeps the existing navbar + sidebar shell", async () => {
    renderLayout();
    // AppLayout renders Sidebar twice (desktop rail hidden on mobile, mobile drawer hidden on
    // desktop), so its "New analysis" button appears twice in the DOM. The top row carries no
    // action pills — see components.Header.test.tsx.
    await waitFor(() => expect(screen.getAllByRole("button", { name: /New analysis/i })).toHaveLength(2));
    expect(screen.getByLabelText("X-Coach")).toBeInTheDocument();
    expect(screen.getByText("page body")).toBeInTheDocument();
  });

  // The rail's width is live layout state now, not a constant. Round-trip it through real clicks:
  // the toggle has to move the width AppLayout hands down, not just a flag inside Sidebar. Only
  // the desktop rail (the first <aside>) gets a toggle — the off-canvas drawer has none — so the
  // by-name query stays unambiguous even with both sidebars in the DOM.
  it("collapses and re-expands the desktop rail", async () => {
    const { container } = renderLayout();
    const railWidth = () => (container.querySelectorAll("aside")[0] as HTMLElement).style.width;
    await waitFor(() => expect(railWidth()).toBe("236px"));

    await userEvent.click(screen.getByRole("button", { name: /collapse navigation/i }));
    expect(railWidth()).toBe("76px");

    await userEvent.click(screen.getByRole("button", { name: /expand navigation/i }));
    expect(railWidth()).toBe("236px");
  });
});
