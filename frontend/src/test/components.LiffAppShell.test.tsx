import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import LiffAppShell from "../components/LiffAppShell";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";

// The shell no longer owns its chrome: it composes MobileTopBar + MobileTabBar, the same pair the
// phone web shell uses, so both phone surfaces are one design (motion_analysis_mobile.png). The
// bar has five slots, the middle one being the raised new-analysis action rather than a
// destination — so four tabs, two either side of it.
const renderAt = (path: string, title?: string) => {
  const onNewAnalysis = vi.fn();
  render(
    <MemoryRouter initialEntries={[path]}>
      <AuthProvider>
        <I18nProvider>
          <LiffAppShell onNewAnalysis={onNewAnalysis} title={title}>
            <p>page body</p>
          </LiffAppShell>
        </I18nProvider>
      </AuthProvider>
    </MemoryRouter>
  );
  return { onNewAnalysis };
};

describe("LiffAppShell — structure", () => {
  it("renders its children", () => {
    renderAt("/app");
    expect(screen.getByText("page body")).toBeInTheDocument();
  });

  it("shows four destination tabs, and nothing else links out of the bar", () => {
    renderAt("/app");
    const bar = screen.getByRole("navigation");
    const links = screen.getAllByRole("link");
    expect(links).toHaveLength(4);
    links.forEach((l) => expect(bar).toContainElement(l));
    ["Analyse", "My records", "Games", "Settings"].forEach((label) => {
      expect(screen.getByRole("link", { name: new RegExp(label, "i") })).toBeInTheDocument();
    });
  });

  // The brand LOCKUP (the linked mark that takes you home) is what the web shell has and this one
  // must not — checked as a link, since the header's fallback *title* is the product name and is
  // a heading, not navigation.
  it("omits the web chrome — no brand lockup, no sidebar toggle, no sign-in", () => {
    renderAt("/app", "Analyse");
    expect(screen.queryByRole("link", { name: "X-Coach" })).not.toBeInTheDocument();
    expect(screen.queryByText("X-Coach")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /navigation/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /sign in/i })).not.toBeInTheDocument();
  });

  it("insets the tab bar for the home indicator", () => {
    renderAt("/app");
    expect(screen.getByRole("navigation").className).toContain("safe-area-inset-bottom");
  });

  it("labels the tab bar as a navigation landmark", () => {
    renderAt("/app");
    expect(screen.getByRole("navigation", { name: /navigation/i })).toBeInTheDocument();
  });
});

describe("LiffAppShell — title", () => {
  // Titles came BACK with the phone design: the mock centres the page name between two round
  // buttons, and the active tab does not name every page on its own (Movements and Admin have no
  // tab at all). It is the shell's only heading.
  it("renders the supplied title as the header's heading", () => {
    renderAt("/history", "My records");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("My records");
  });

  it("falls back to the product name when a page supplies none", () => {
    renderAt("/history");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("X-Coach");
  });
});

describe("LiffAppShell — actions", () => {
  // "New analysis" is reachable twice by design: the header's upload button and the bar's raised
  // centre action, both from the mock. They call the same handler.
  it("offers the new-analysis action in both the header and the tab bar", async () => {
    const { onNewAnalysis } = renderAt("/app");
    const btns = screen.getAllByRole("button", { name: /New analysis/i });
    expect(btns).toHaveLength(2);
    await userEvent.click(btns[0]);
    await userEvent.click(btns[1]);
    expect(onNewAnalysis).toHaveBeenCalledTimes(2);
  });

  it("offers the new-analysis action on a non-studio page too", () => {
    renderAt("/history");
    expect(screen.getAllByRole("button", { name: /New analysis/i }).length).toBeGreaterThan(0);
  });

  // The sample library is gone, and with it the bar's Library tab. Pinned because the bar is the
  // whole of the in-LINE navigation: a leftover button here is an in-app route to a picker that
  // no longer exists.
  it("offers no Library action", () => {
    renderAt("/app");
    expect(screen.queryByRole("button", { name: /Library/i })).toBeNull();
  });
});

describe("LiffAppShell — active tab", () => {
  it.each([
    ["/app", "Analyse"],
    ["/history", "My records"],
    ["/games", "Games"],
    // The Games tab owns the two game routes as well as the hub, matching the desktop rail.
    ["/67", "Games"],
    ["/ninja", "Games"],
    ["/settings", "Settings"],
  ])("highlights exactly %s and marks it aria-current", (path, label) => {
    renderAt(path);
    const current = screen.getByRole("link", { current: "page" });
    expect(current).toHaveAccessibleName(new RegExp(label, "i"));
    expect(current.className).toContain("text-primary");
    screen.getAllByRole("link").forEach((link) => {
      if (link === current) return;
      expect(link).not.toHaveAttribute("aria-current");
      expect(link.className).not.toContain("text-primary");
    });
  });

  // Movements has no tab. Pinned so the route staying reachable is not mistaken for it being in
  // the bar.
  it.each(["/movements"])(
    "highlights nothing on the tab-less route %s",
    (path) => {
      renderAt(path);
      expect(screen.queryByRole("link", { current: "page" })).not.toBeInTheDocument();
      screen.getAllByRole("link").forEach((link) => {
        expect(link.className).not.toContain("text-primary");
      });
    }
  );
});
