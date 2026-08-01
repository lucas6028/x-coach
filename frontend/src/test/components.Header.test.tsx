import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Header from "../components/Header";
import { renderWithProviders } from "./renderWithProviders";

// The navbar used to carry a page title (`{movement} Analysis` / `Session: <id>` / the page name)
// and a status line (PROCESSING / ANALYSIS COMPLETE / AWAITING INPUT, plus movement, view and
// source). Both were removed: which page you're on is what the sidebar's active pill says, and the
// analysis metadata lives in the result panels. The tests that pinned that copy went with it.
describe("Header", () => {
  it("carries the brand and the nav controls, and nothing else", () => {
    renderWithProviders(<Header />);
    expect(screen.getByRole("link", { name: "X-Coach" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /show navigation/i })).toBeInTheDocument();
  });

  it("renders no page title or status line", () => {
    renderWithProviders(<Header />);
    // No heading at all in the navbar — a stray one would put the page's <h1> back.
    expect(screen.queryByRole("heading")).not.toBeInTheDocument();
    for (const gone of ["Squat Analysis", "AWAITING INPUT", "PROCESSING", "ANALYSIS COMPLETE"]) {
      expect(screen.queryByText(gone)).not.toBeInTheDocument();
    }
  });

  it("calls onMenu when the mobile menu button is clicked", async () => {
    const user = userEvent.setup();
    const onMenu = vi.fn();
    renderWithProviders(<Header onMenu={onMenu} />);
    await user.click(screen.getByRole("button", { name: /show navigation/i }));
    expect(onMenu).toHaveBeenCalledOnce();
  });
});
