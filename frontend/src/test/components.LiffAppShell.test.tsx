import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import LiffAppShell from "../components/LiffAppShell";
import { I18nProvider } from "../lib/i18n";

const renderAt = (path: string) => {
  const onOpenLibrary = vi.fn();
  const onNewAnalysis = vi.fn();
  render(
    <MemoryRouter initialEntries={[path]}>
      <I18nProvider>
        <LiffAppShell onOpenLibrary={onOpenLibrary} onNewAnalysis={onNewAnalysis}>
          <p>page body</p>
        </LiffAppShell>
      </I18nProvider>
    </MemoryRouter>
  );
  return { onOpenLibrary, onNewAnalysis };
};

describe("LiffAppShell — structure", () => {
  it("renders its children", () => {
    renderAt("/app");
    expect(screen.getByText("page body")).toBeInTheDocument();
  });

  it("shows the four tabs and nothing else in the tab bar", () => {
    renderAt("/app");
    const bar = screen.getByRole("navigation");
    const links = screen.getAllByRole("link");
    expect(links).toHaveLength(4);
    expect(bar).toContainElement(links[0]);
    ["Analyse", "My records", "Games", "Settings"].forEach((label) => {
      expect(screen.getByRole("link", { name: new RegExp(label, "i") })).toBeInTheDocument();
    });
  });

  it("omits the web chrome — no brand lockup, no sidebar toggle, no sign-in", () => {
    renderAt("/app");
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
  // The shell used to render a page title in its header ("My records", or "{movement} Analysis"
  // as the studio fallback). Titles were removed app-wide; the active bottom tab is what says
  // which page you're on. This pins the removal so a title can't creep back into the header.
  it("renders no page title", () => {
    renderAt("/history");
    expect(screen.queryByRole("heading")).not.toBeInTheDocument();
    expect(screen.queryByText(/Squat Analysis/i)).not.toBeInTheDocument();
  });
});

describe("LiffAppShell — header actions", () => {
  it("offers a New analysis action with an accessible name", async () => {
    const { onNewAnalysis } = renderAt("/app");
    const btn = screen.getByRole("button", { name: /New analysis/i });
    expect(btn).toHaveAttribute("title", "New analysis");
    await userEvent.click(btn);
    expect(onNewAnalysis).toHaveBeenCalledTimes(1);
  });

  it("offers a Library action with an accessible name", async () => {
    const { onOpenLibrary } = renderAt("/app");
    const btn = screen.getByRole("button", { name: /Library/i });
    expect(btn).toHaveAttribute("title", "Library");
    await userEvent.click(btn);
    expect(onOpenLibrary).toHaveBeenCalledTimes(1);
  });

  it("offers both header actions on a non-studio page too", () => {
    renderAt("/history");
    expect(screen.getByRole("button", { name: /New analysis/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Library/i })).toBeInTheDocument();
  });
});

describe("LiffAppShell — active tab", () => {
  it.each([
    ["/app", "Analyse"],
    ["/history", "My records"],
    ["/games", "Games"],
    ["/settings", "Settings"],
  ])("highlights exactly %s and marks it aria-current", (path, label) => {
    renderAt(path);
    const current = screen.getByRole("link", { current: "page" });
    expect(current).toHaveAccessibleName(new RegExp(label, "i"));
    expect(current.className).toContain("text-primary");
    // Every other tab is neither current nor coloured as active.
    screen.getAllByRole("link").forEach((link) => {
      if (link === current) return;
      expect(link).not.toHaveAttribute("aria-current");
      expect(link.className).not.toContain("text-primary");
    });
  });

  it.each(["/67", "/ninja"])("highlights the Games tab on the game route %s", (path) => {
    renderAt(path);
    const current = screen.getByRole("link", { current: "page" });
    expect(current).toHaveAccessibleName(/Games/i);
  });

  it("highlights nothing on a tab-less route", () => {
    renderAt("/movements");
    expect(screen.queryByRole("link", { current: "page" })).not.toBeInTheDocument();
    screen.getAllByRole("link").forEach((link) => {
      expect(link.className).not.toContain("text-primary");
    });
  });
});
