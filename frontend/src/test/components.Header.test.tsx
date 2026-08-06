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
  it("keeps the mobile drawer opener — below lg it is the only way to reach the nav", () => {
    renderWithProviders(<Header />);
    expect(screen.getByRole("button", { name: /show navigation/i })).toBeInTheDocument();
  });

  // The brand lockup, the New-analysis / Library pills and the rail-collapse toggle all moved out
  // of the top row: the brand is the rail's mark (Sidebar), both actions are rail entries, and the
  // 84px rail was not worth a permanent control to collapse. Pinned so none of them drift back.
  it("carries no brand lockup, action pills or collapse toggle", () => {
    renderWithProviders(<Header />);
    expect(screen.queryByRole("link", { name: "X-Coach" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /New analysis/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Library/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /hide navigation/i })).not.toBeInTheDocument();
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
