import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import Leaderboard from "../components/game/Leaderboard";
import type { ScoreEntry } from "../lib/game/leaderboard";

const entries: ScoreEntry[] = [
  { name: "Ada", score: 900, poses: 9, bestCombo: 5, ts: 1 },
  { name: "Bo", score: 500, poses: 5, bestCombo: 3, ts: 2 },
];

describe("Leaderboard", () => {
  it("renders an empty state with no entries", () => {
    renderWithProviders(<Leaderboard entries={[]} />);
    expect(screen.getByText("No scores yet — be the first!")).toBeInTheDocument();
  });

  it("lists entries with medals and scores", () => {
    renderWithProviders(<Leaderboard entries={entries} />);
    expect(screen.getByText("Ada")).toBeInTheDocument();
    expect(screen.getByText("900")).toBeInTheDocument();
    expect(screen.getByText("🥇")).toBeInTheDocument();
  });

  it("highlights the player's own row", () => {
    renderWithProviders(<Leaderboard entries={entries} highlightRank={2} />);
    expect(screen.getByText("You")).toBeInTheDocument();
  });
});
