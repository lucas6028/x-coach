import { describe, it, expect, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import LanguageToggle from "../components/LanguageToggle";
import { renderWithProviders } from "./renderWithProviders";

describe("LanguageToggle (dropdown)", () => {
  beforeEach(() => localStorage.clear());

  it("renders only the trigger button while closed", () => {
    renderWithProviders(<LanguageToggle />);
    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("opens a menu of the available languages when clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LanguageToggle />);
    await user.click(screen.getByRole("button"));
    expect(screen.getByRole("menu")).toBeInTheDocument();
    expect(screen.getAllByRole("menuitemradio")).toHaveLength(2);
  });

  it("marks the current language as checked", async () => {
    const user = userEvent.setup();
    localStorage.setItem("lang", "zh-Hant");
    renderWithProviders(<LanguageToggle />);
    await user.click(screen.getByRole("button"));
    expect(screen.getByRole("menuitemradio", { name: /繁體中文/i })).toHaveAttribute(
      "aria-checked",
      "true"
    );
  });

  it("switches language when an option is selected", async () => {
    const user = userEvent.setup();
    localStorage.setItem("lang", "en");
    renderWithProviders(<LanguageToggle />);
    await user.click(screen.getByRole("button"));
    await user.click(screen.getByRole("menuitemradio", { name: /繁體中文/i }));
    expect(localStorage.getItem("lang")).toBe("zh-Hant");
  });
});
