import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import LiffAppShell from "../components/LiffAppShell";
import { I18nProvider } from "../lib/i18n";

const renderAt = (path: string, title?: string, movement?: string) => {
  const onOpenLibrary = vi.fn();
  const onNewAnalysis = vi.fn();
  render(
    <MemoryRouter initialEntries={[path]}>
      <I18nProvider>
        <LiffAppShell title={title} movement={movement} onOpenLibrary={onOpenLibrary} onNewAnalysis={onNewAnalysis}>
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
  it("shows the page title when given one", () => {
    renderAt("/history", "My records");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("My records");
  });

  it("falls back to the studio title when untitled (the studio passes no title)", () => {
    renderAt("/app");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Squat Analysis");
  });

  // Finding 1 of the 2026-07-25 review: the LINE in-app shell has its own copy of this fallback
  // title (it renders instead of the web Header, see AppLayout), which must track the studio's
  // selection just like the web header does — not stay pinned to "Squat" for every movement.
  it("names the selected movement in the fallback title, not a hardcoded squat", () => {
    renderAt("/app", undefined, "Overhead Press");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Overhead Press Analysis");
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
    renderAt("/history", "My records");
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
