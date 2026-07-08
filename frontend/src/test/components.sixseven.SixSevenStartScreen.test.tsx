import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import SixSevenStartScreen from "../components/sixseven/SixSevenStartScreen";

describe("SixSevenStartScreen", () => {
  it("renders the pitch, how-to, and empty board", () => {
    renderWithProviders(<SixSevenStartScreen leaderboard={[]} onStart={vi.fn()} />);
    expect(screen.getByText("How many 67s can you hit?")).toBeInTheDocument();
    expect(screen.getByText(/every switch is one 67/)).toBeInTheDocument();
    expect(screen.getByText("No scores yet — be the first!")).toBeInTheDocument();
  });

  it("fires onStart", () => {
    const onStart = vi.fn();
    renderWithProviders(<SixSevenStartScreen leaderboard={[]} onStart={onStart} />);
    fireEvent.click(screen.getByRole("button", { name: /Enable camera & go/i }));
    expect(onStart).toHaveBeenCalledOnce();
  });

  it("disables the button while starting", () => {
    renderWithProviders(<SixSevenStartScreen leaderboard={[]} onStart={vi.fn()} starting />);
    expect(screen.getByRole("button", { name: /Starting camera/i })).toBeDisabled();
  });

  it("surfaces an error", () => {
    renderWithProviders(
      <SixSevenStartScreen leaderboard={[]} onStart={vi.fn()} error="Camera blocked" />
    );
    expect(screen.getByText("Camera blocked")).toBeInTheDocument();
  });
});
