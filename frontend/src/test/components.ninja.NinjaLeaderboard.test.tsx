import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import NinjaLeaderboard from "../components/ninja/NinjaLeaderboard";
import type { NinjaEntry } from "../lib/ninja/leaderboard";

const board: NinjaEntry[] = [
  { name: "Ann", score: 1200, bestCombo: 8, ts: 1 },
  { name: "Bo", score: 800, bestCombo: 5, ts: 2 },
];

describe("NinjaLeaderboard", () => {
  it("shows an empty state with no entries", () => {
    renderWithProviders(<NinjaLeaderboard entries={[]} />);
    expect(screen.getByText("No scores yet — be the first!")).toBeInTheDocument();
  });

  it("renders each entry with its score", () => {
    renderWithProviders(<NinjaLeaderboard entries={board} />);
    expect(screen.getByText("Ann")).toBeInTheDocument();
    expect(screen.getByText("1,200")).toBeInTheDocument();
    expect(screen.getByText("800")).toBeInTheDocument();
  });

  it("highlights the submitted rank", () => {
    renderWithProviders(<NinjaLeaderboard entries={board} highlightRank={1} />);
    expect(screen.getByText("You")).toBeInTheDocument();
  });
});
