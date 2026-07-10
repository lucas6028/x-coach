import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import DuelStartScreen from "../components/duel/DuelStartScreen";

describe("DuelStartScreen", () => {
  it("renders the pitch, how-to, pose deck, and empty board", () => {
    renderWithProviders(<DuelStartScreen results={[]} onStart={vi.fn()} />);
    expect(screen.getByText("Strike the pose. Beat your rival.")).toBeInTheDocument();
    expect(screen.getByText(/Both players stand side by side/)).toBeInTheDocument();
    expect(screen.getByText("T-Pose")).toBeInTheDocument();
    expect(screen.getByText("No duels yet — throw down!")).toBeInTheDocument();
  });

  it("fires onStart", () => {
    const onStart = vi.fn();
    renderWithProviders(<DuelStartScreen results={[]} onStart={onStart} />);
    fireEvent.click(screen.getByRole("button", { name: /Enable camera & duel/i }));
    expect(onStart).toHaveBeenCalledOnce();
  });

  it("disables the button while starting", () => {
    renderWithProviders(<DuelStartScreen results={[]} onStart={vi.fn()} starting />);
    expect(screen.getByRole("button", { name: /Starting camera/i })).toBeDisabled();
  });

  it("surfaces an error", () => {
    renderWithProviders(
      <DuelStartScreen results={[]} onStart={vi.fn()} error="Camera blocked" />
    );
    expect(screen.getByText("Camera blocked")).toBeInTheDocument();
  });
});
