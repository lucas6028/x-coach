import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import GameStartScreen from "../components/game/GameStartScreen";

describe("GameStartScreen", () => {
  it("renders the pitch, pose deck, and leaderboard", () => {
    renderWithProviders(<GameStartScreen leaderboard={[]} onStart={vi.fn()} />);
    expect(screen.getByText("Strike the pose. Beat the clock.")).toBeInTheDocument();
    expect(screen.getByText("T-Pose")).toBeInTheDocument();
    expect(screen.getByText("Squat hold")).toBeInTheDocument();
    expect(screen.getByText("No scores yet — be the first!")).toBeInTheDocument();
  });

  it("fires onStart when the button is clicked", () => {
    const onStart = vi.fn();
    renderWithProviders(<GameStartScreen leaderboard={[]} onStart={onStart} />);
    fireEvent.click(screen.getByRole("button", { name: /Enable camera/i }));
    expect(onStart).toHaveBeenCalledOnce();
  });

  it("shows a starting label and disables the button while warming up", () => {
    renderWithProviders(<GameStartScreen leaderboard={[]} onStart={vi.fn()} starting />);
    const btn = screen.getByRole("button", { name: /Starting camera/i });
    expect(btn).toBeDisabled();
  });

  it("surfaces an error message", () => {
    renderWithProviders(
      <GameStartScreen leaderboard={[]} onStart={vi.fn()} error="Camera blocked" />
    );
    expect(screen.getByText("Camera blocked")).toBeInTheDocument();
  });
});
