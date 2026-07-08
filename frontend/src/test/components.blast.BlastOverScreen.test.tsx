import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import BlastOverScreen from "../components/blast/BlastOverScreen";
import type { BlastEntry } from "../lib/blast/leaderboard";

const result = { score: 1800, hits: 15, bestCombo: 7 };
const board: BlastEntry[] = [{ name: "Me", score: 1800, hits: 15, bestCombo: 7, ts: 1 }];

function setup(props: Partial<React.ComponentProps<typeof BlastOverScreen>> = {}) {
  return renderWithProviders(
    <BlastOverScreen
      result={result}
      leaderboard={[]}
      rank={null}
      submitted={false}
      onSubmit={vi.fn()}
      onReplay={vi.fn()}
      {...props}
    />
  );
}

describe("BlastOverScreen", () => {
  it("shows the score and round stats", () => {
    setup();
    expect(screen.getByText("1,800")).toBeInTheDocument();
    expect(screen.getByText("15")).toBeInTheDocument();
    expect(screen.getByText("7×")).toBeInTheDocument();
  });

  it("submits a typed name", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    fireEvent.change(screen.getByPlaceholderText("Your name"), { target: { value: "Grace" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSubmit).toHaveBeenCalledWith("Grace");
  });

  it("falls back to Anonymous for a blank name", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSubmit).toHaveBeenCalledWith("Anonymous");
  });

  it("submits on Enter", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    const input = screen.getByPlaceholderText("Your name");
    fireEvent.change(input, { target: { value: "Alan" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSubmit).toHaveBeenCalledWith("Alan");
  });

  it("shows rank and leaderboard once submitted", () => {
    setup({ submitted: true, rank: 1, leaderboard: board });
    expect(screen.getByText("You're #1 on the local board!")).toBeInTheDocument();
    expect(screen.getByText("Me")).toBeInTheDocument();
  });

  it("shows an off-board message when not ranked", () => {
    setup({ submitted: true, rank: null, leaderboard: board });
    expect(
      screen.getByText("Saved — keep blasting to crack the top 10.")
    ).toBeInTheDocument();
  });

  it("fires onReplay", () => {
    const onReplay = vi.fn();
    setup({ onReplay });
    fireEvent.click(screen.getByRole("button", { name: /Play again/i }));
    expect(onReplay).toHaveBeenCalledOnce();
  });
});
