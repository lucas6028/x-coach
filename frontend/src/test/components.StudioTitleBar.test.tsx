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

  it("exposes the movement picker as a labelled select", async () => {
    const onMovementChange = vi.fn();
    renderWithProviders(<StudioTitleBar {...base} onMovementChange={onMovementChange} />);
    const select = screen.getByLabelText(/movement/i) as HTMLSelectElement;
    expect(select.value).toBe("Squat");
    await userEvent.selectOptions(select, "Push-up");
    expect(onMovementChange).toHaveBeenCalledWith("Push-up");
  });

  // A URL-supplied movement the catalog does not list must stay visible rather than snapping the
  // control to something the user did not choose.
  it("keeps an unknown movement as an option of its own", () => {
    renderWithProviders(<StudioTitleBar {...base} movement="Lunge" />);
    expect((screen.getByLabelText(/movement/i) as HTMLSelectElement).value).toBe("Lunge");
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
    await userEvent.selectOptions(screen.getByLabelText(/precision|tier|effort/i), "lite");
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
