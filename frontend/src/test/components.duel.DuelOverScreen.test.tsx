import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import DuelOverScreen, { type DuelResult } from "../components/duel/DuelOverScreen";
import type { DuelEntry } from "../lib/duel/leaderboard";

const result: DuelResult = { winner: "a", aWins: 3, bWins: 1 };
const board: DuelEntry[] = [
  { winner: "Ann", loser: "Bo", winnerPoints: 3, loserPoints: 1, ts: 42 },
];

function setup(props: Partial<React.ComponentProps<typeof DuelOverScreen>> = {}) {
  return renderWithProviders(
    <DuelOverScreen
      result={result}
      results={[]}
      submitted={false}
      savedTs={null}
      onSubmit={vi.fn()}
      onReplay={vi.fn()}
      {...props}
    />
  );
}

describe("DuelOverScreen", () => {
  it("announces the winner and the final score", () => {
    setup();
    expect(screen.getByText("Player A wins!")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("submits the typed winner and loser names", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    fireEvent.change(screen.getByLabelText("Winner's name"), { target: { value: "Grace" } });
    fireEvent.change(screen.getByLabelText("Challenger's name"), { target: { value: "Hugo" } });
    fireEvent.click(screen.getByRole("button", { name: "Log this duel" }));
    expect(onSubmit).toHaveBeenCalledWith("Grace", "Hugo");
  });

  it("falls back to the player labels when names are blank", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    fireEvent.click(screen.getByRole("button", { name: "Log this duel" }));
    expect(onSubmit).toHaveBeenCalledWith("Player A", "Player B");
  });

  it("submits on Enter", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    const input = screen.getByLabelText("Winner's name");
    fireEvent.change(input, { target: { value: "Ivy" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSubmit).toHaveBeenCalledWith("Ivy", "Player B");
  });

  it("shows the saved message and board once submitted", () => {
    setup({ submitted: true, savedTs: 42, results: board });
    expect(screen.getByText("Logged to recent duels.")).toBeInTheDocument();
    expect(screen.getByText("Ann")).toBeInTheDocument();
  });

  it("fires onReplay for a rematch", () => {
    const onReplay = vi.fn();
    setup({ onReplay });
    fireEvent.click(screen.getByRole("button", { name: /Rematch/i }));
    expect(onReplay).toHaveBeenCalledOnce();
  });

  it("labels the winning side B when player B wins", () => {
    setup({ result: { winner: "b", aWins: 2, bWins: 3 } });
    expect(screen.getByText("Player B wins!")).toBeInTheDocument();
  });
});
