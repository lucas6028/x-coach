import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import DuelLeaderboard from "../components/duel/DuelLeaderboard";
import type { DuelEntry } from "../lib/duel/leaderboard";

const board: DuelEntry[] = [
  { winner: "Ann", loser: "Bo", winnerPoints: 3, loserPoints: 1, ts: 100 },
  { winner: "Cy", loser: "Dee", winnerPoints: 3, loserPoints: 2, ts: 200 },
];

describe("DuelLeaderboard", () => {
  it("shows an empty state with no entries", () => {
    renderWithProviders(<DuelLeaderboard entries={[]} />);
    expect(screen.getByText("No duels yet — throw down!")).toBeInTheDocument();
  });

  it("renders each duel with winner, loser, and score", () => {
    renderWithProviders(<DuelLeaderboard entries={board} />);
    expect(screen.getByText("Ann")).toBeInTheDocument();
    expect(screen.getByText("Bo")).toBeInTheDocument();
    expect(screen.getByText("3–1")).toBeInTheDocument();
    expect(screen.getByText("3–2")).toBeInTheDocument();
  });

  it("highlights the just-saved duel by timestamp", () => {
    const { container } = renderWithProviders(
      <DuelLeaderboard entries={board} highlightTs={200} />
    );
    expect(container.querySelector(".bg-primary\\/10")).not.toBeNull();
  });
});
