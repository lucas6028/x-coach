import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import BlastLeaderboard from "../components/blast/BlastLeaderboard";
import type { BlastEntry } from "../lib/blast/leaderboard";

const entries: BlastEntry[] = [
  { name: "Ada", score: 1200, hits: 12, bestCombo: 5, ts: 1 },
  { name: "Bo", score: 700, hits: 7, bestCombo: 3, ts: 2 },
];

describe("BlastLeaderboard", () => {
  it("renders an empty state", () => {
    renderWithProviders(<BlastLeaderboard entries={[]} />);
    expect(screen.getByText("No scores yet — be the first!")).toBeInTheDocument();
  });

  it("lists entries with medals and scores", () => {
    renderWithProviders(<BlastLeaderboard entries={entries} />);
    expect(screen.getByText("Ada")).toBeInTheDocument();
    expect(screen.getByText("1,200")).toBeInTheDocument();
    expect(screen.getByText("🥇")).toBeInTheDocument();
  });

  it("highlights the player's own row", () => {
    renderWithProviders(<BlastLeaderboard entries={entries} highlightRank={2} />);
    expect(screen.getByText("You")).toBeInTheDocument();
  });
});
