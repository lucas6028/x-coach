import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import LanguageToggle from "../components/LanguageToggle";
import { renderWithProviders } from "./renderWithProviders";

describe("LanguageToggle — collapsed (expanded=false)", () => {
  beforeEach(() => localStorage.clear());

  it("renders a single cycling button", () => {
    renderWithProviders(<LanguageToggle expanded={false} />);
    expect(screen.getAllByRole("button")).toHaveLength(1);
  });

  it("cycles to zh-Hant when current is en", async () => {
    const user = userEvent.setup();
    localStorage.setItem("lang", "en");
    renderWithProviders(<LanguageToggle expanded={false} />);
    await user.click(screen.getByRole("button"));
    expect(localStorage.getItem("lang")).toBe("zh-Hant");
  });

  it("cycles back to en when current is zh-Hant", async () => {
    const user = userEvent.setup();
    localStorage.setItem("lang", "zh-Hant");
    renderWithProviders(<LanguageToggle expanded={false} />);
    await user.click(screen.getByRole("button"));
    expect(localStorage.getItem("lang")).toBe("en");
  });
});

describe("LanguageToggle — expanded (expanded=true)", () => {
  beforeEach(() => localStorage.clear());

  it("renders two language buttons", () => {
    renderWithProviders(<LanguageToggle expanded={true} />);
    expect(screen.getAllByRole("button")).toHaveLength(2);
  });

  it("marks the current language as pressed", () => {
    localStorage.setItem("lang", "zh-Hant");
    renderWithProviders(<LanguageToggle expanded={true} />);
    const zhBtn = screen.getByRole("button", { name: /繁體中文/i });
    expect(zhBtn).toHaveAttribute("aria-pressed", "true");
  });

  it("switches language when a button is clicked", async () => {
    const user = userEvent.setup();
    localStorage.setItem("lang", "en");
    renderWithProviders(<LanguageToggle expanded={true} />);
    await user.click(screen.getByRole("button", { name: /繁體中文/i }));
    expect(localStorage.getItem("lang")).toBe("zh-Hant");
  });
});
