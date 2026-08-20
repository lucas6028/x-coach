import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import StudioTitleBar from "../components/StudioTitleBar";
import { renderWithProviders } from "./renderWithProviders";

const base = {
  movement: "Squat",
  movements: [
    { name: "Squat", validated: true },
    { name: "Push-up", validated: false },
  ],
  onMovementChange: vi.fn(),
  tier: "heavy" as const,
  onTierChange: vi.fn(),
};

describe("StudioTitleBar", () => {
  it("names the selected movement in the title and breadcrumb", () => {
    renderWithProviders(<StudioTitleBar {...base} movement="Push-up" />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Push-up Motion Analysis"
    );
    expect(screen.getByText("Push-up Analysis")).toBeInTheDocument();
  });

  // The picker is a menu of `menuitemradio`s (the pattern the theme and language pickers use),
  // not a native <select> — a native one opens a browser-drawn list that ignores the palette.
  // The trigger announces both the label and the current value.
  it("exposes the movement picker as a labelled menu, and reports the choice", async () => {
    const onMovementChange = vi.fn();
    renderWithProviders(<StudioTitleBar {...base} onMovementChange={onMovementChange} />);
    const trigger = screen.getByLabelText("Movement: Squat");
    expect(trigger).toHaveAttribute("aria-expanded", "false");

    await userEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    // The open menu marks the current value, so the checked item is a real assertion, not decor.
    expect(screen.getByRole("menuitemradio", { name: /Squat/ })).toHaveAttribute(
      "aria-checked",
      "true"
    );

    await userEvent.click(screen.getByRole("menuitemradio", { name: /Push-up/ }));
    expect(onMovementChange).toHaveBeenCalledWith("Push-up");
    // Choosing closes it — the menu is not left hanging over the page.
    expect(screen.queryByRole("menu")).toBeNull();
  });

  // A URL-supplied movement the catalog does not list must stay visible rather than snapping the
  // control to something the user did not choose.
  it("keeps an unknown movement as an option of its own", () => {
    renderWithProviders(<StudioTitleBar {...base} movement="Lunge" />);
    expect(screen.getByLabelText("Movement: Lunge")).toBeInTheDocument();
  });

  it("tags an unvalidated movement as Beta, and a validated one not at all", () => {
    const { unmount } = renderWithProviders(<StudioTitleBar {...base} movement="Push-up" />);
    expect(screen.getByText("Beta")).toBeInTheDocument();
    unmount();
    renderWithProviders(<StudioTitleBar {...base} />);
    expect(screen.queryByText("Beta")).toBeNull();
  });

  it("persists a tier change through its callback", async () => {
    const onTierChange = vi.fn();
    renderWithProviders(<StudioTitleBar {...base} onTierChange={onTierChange} />);
    await userEvent.click(screen.getByLabelText("Precision: Heavy"));
    await userEvent.click(screen.getByRole("menuitemradio", { name: /Lite/ }));
    expect(onTierChange).toHaveBeenCalledWith("lite");
  });

  // The primary action only exists once a result is up: in the empty state the dropzone below is
  // already the call to action.
  it("shows the start/upload action only when given one", async () => {
    const onNewSession = vi.fn();
    const { unmount } = renderWithProviders(<StudioTitleBar {...base} />);
    expect(screen.queryByRole("button", { name: /start \/ upload/i })).toBeNull();
    unmount();
    renderWithProviders(<StudioTitleBar {...base} onNewSession={onNewSession} />);
    await userEvent.click(screen.getByRole("button", { name: /start \/ upload/i }));
    expect(onNewSession).toHaveBeenCalledOnce();
  });
});
