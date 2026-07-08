import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import BlastStartScreen from "../components/blast/BlastStartScreen";

describe("BlastStartScreen", () => {
  it("renders the pitch, how-to, and leaderboard", () => {
    renderWithProviders(<BlastStartScreen leaderboard={[]} onStart={vi.fn()} />);
    expect(screen.getByText("Charge up. Blast the memes.")).toBeInTheDocument();
    expect(screen.getByText(/Bring both hands together to charge/)).toBeInTheDocument();
    expect(screen.getByText("No scores yet — be the first!")).toBeInTheDocument();
  });

  it("fires onStart", () => {
    const onStart = vi.fn();
    renderWithProviders(<BlastStartScreen leaderboard={[]} onStart={onStart} />);
    fireEvent.click(screen.getByRole("button", { name: /Enable camera/i }));
    expect(onStart).toHaveBeenCalledOnce();
  });

  it("disables the button and shows progress while starting", () => {
    renderWithProviders(<BlastStartScreen leaderboard={[]} onStart={vi.fn()} starting />);
    expect(screen.getByRole("button", { name: /Starting camera/i })).toBeDisabled();
  });

  it("surfaces an error", () => {
    renderWithProviders(
      <BlastStartScreen leaderboard={[]} onStart={vi.fn()} error="Camera blocked" />
    );
    expect(screen.getByText("Camera blocked")).toBeInTheDocument();
  });
});
