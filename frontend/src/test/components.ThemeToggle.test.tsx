import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ThemeToggle from "../components/ThemeToggle";
import { renderWithProviders } from "./renderWithProviders";

describe("ThemeToggle — collapsed (expanded=false)", () => {
  beforeEach(() => localStorage.clear());

  it("renders a single cycling button", () => {
    renderWithProviders(<ThemeToggle expanded={false} />);
    expect(screen.getAllByRole("button")).toHaveLength(1);
  });

  it("cycles to 'system' when current is 'light'", async () => {
    const user = userEvent.setup();
    localStorage.setItem("theme", "light");
    renderWithProviders(<ThemeToggle expanded={false} />);
    const btn = screen.getByRole("button");
    expect(btn.getAttribute("aria-label")).toMatch(/Light/);
    await user.click(btn);
    // After clicking light → system; localStorage should reflect the new value
    expect(localStorage.getItem("theme")).toBe("system");
  });
});

describe("ThemeToggle — expanded (expanded=true)", () => {
  beforeEach(() => localStorage.clear());

  it("renders three option buttons", () => {
    renderWithProviders(<ThemeToggle expanded={true} />);
    expect(screen.getAllByRole("button")).toHaveLength(3);
  });

  it("marks the current theme button as pressed", () => {
    localStorage.setItem("theme", "dark");
    renderWithProviders(<ThemeToggle expanded={true} />);
    const darkBtn = screen.getByRole("button", { name: /dark/i });
    expect(darkBtn).toHaveAttribute("aria-pressed", "true");
  });

  it("un-presses other buttons", () => {
    localStorage.setItem("theme", "dark");
    renderWithProviders(<ThemeToggle expanded={true} />);
    const lightBtn = screen.getByRole("button", { name: /light/i });
    expect(lightBtn).toHaveAttribute("aria-pressed", "false");
  });

  it("switches theme when an option button is clicked", async () => {
    const user = userEvent.setup();
    localStorage.setItem("theme", "system");
    renderWithProviders(<ThemeToggle expanded={true} />);
    await user.click(screen.getByRole("button", { name: /light/i }));
    expect(localStorage.getItem("theme")).toBe("light");
  });
});
