import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import NinjaStartScreen from "../components/ninja/NinjaStartScreen";

describe("NinjaStartScreen", () => {
  it("renders the pitch, how-to, and empty board", () => {
    renderWithProviders(<NinjaStartScreen leaderboard={[]} onStart={vi.fn()} />);
    expect(screen.getByText("Your hands are the blades.")).toBeInTheDocument();
    expect(screen.getByText(/Swipe a hand through the flying fruit/)).toBeInTheDocument();
    expect(screen.getByText("No scores yet — be the first!")).toBeInTheDocument();
  });

  it("fires onStart", () => {
    const onStart = vi.fn();
    renderWithProviders(<NinjaStartScreen leaderboard={[]} onStart={onStart} />);
    fireEvent.click(screen.getByRole("button", { name: /Enable camera & slice/i }));
    expect(onStart).toHaveBeenCalledOnce();
  });

  it("disables the button while starting", () => {
    renderWithProviders(<NinjaStartScreen leaderboard={[]} onStart={vi.fn()} starting />);
    expect(screen.getByRole("button", { name: /Starting camera/i })).toBeDisabled();
  });

  it("surfaces an error", () => {
    renderWithProviders(<NinjaStartScreen leaderboard={[]} onStart={vi.fn()} error="Camera blocked" />);
    expect(screen.getByText("Camera blocked")).toBeInTheDocument();
  });
});
