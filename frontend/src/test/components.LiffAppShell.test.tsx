import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import LiffAppShell from "../components/LiffAppShell";
import { I18nProvider } from "../lib/i18n";

const renderAt = (path: string, title?: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <I18nProvider>
        <LiffAppShell title={title}>
          <p>page body</p>
        </LiffAppShell>
      </I18nProvider>
    </MemoryRouter>
  );

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
});

describe("LiffAppShell — active tab", () => {
  it.each([
    ["/app", "Analyse"],
    ["/history", "My records"],
    ["/games", "Games"],
    ["/settings", "Settings"],
  ])("highlights %s", (path, label) => {
    renderAt(path);
    const link = screen.getByRole("link", { name: new RegExp(label, "i") });
    expect(link.className).toContain("text-primary");
  });

  it.each(["/67", "/ninja"])("highlights the Games tab on the game route %s", (path) => {
    renderAt(path);
    expect(screen.getByRole("link", { name: /Games/i }).className).toContain("text-primary");
  });

  it("highlights nothing on a tab-less route", () => {
    renderAt("/movements");
    screen.getAllByRole("link").forEach((link) => {
      expect(link.className).not.toContain("text-primary");
    });
  });
});
