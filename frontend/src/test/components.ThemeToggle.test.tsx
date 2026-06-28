import { describe, it, expect, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ThemeToggle from "../components/ThemeToggle";
import { renderWithProviders } from "./renderWithProviders";

describe("ThemeToggle (dropdown)", () => {
  beforeEach(() => localStorage.clear());

  it("renders only the trigger button while closed", () => {
    renderWithProviders(<ThemeToggle />);
    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("opens a menu of three options when clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ThemeToggle />);
    await user.click(screen.getByRole("button"));
    expect(screen.getByRole("menu")).toBeInTheDocument();
    expect(screen.getAllByRole("menuitemradio")).toHaveLength(3);
  });

  it("marks the current theme as checked", async () => {
    const user = userEvent.setup();
    localStorage.setItem("theme", "dark");
    renderWithProviders(<ThemeToggle />);
    await user.click(screen.getByRole("button"));
    expect(screen.getByRole("menuitemradio", { name: /dark/i })).toHaveAttribute(
      "aria-checked",
      "true"
    );
  });

  it("switches theme when an option is selected", async () => {
    const user = userEvent.setup();
    localStorage.setItem("theme", "system");
    renderWithProviders(<ThemeToggle />);
    await user.click(screen.getByRole("button"));
    await user.click(screen.getByRole("menuitemradio", { name: /light/i }));
    expect(localStorage.getItem("theme")).toBe("light");
  });

  it("closes the menu after a selection", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ThemeToggle />);
    await user.click(screen.getByRole("button"));
    await user.click(screen.getByRole("menuitemradio", { name: /dark/i }));
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });
});
