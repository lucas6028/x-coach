import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import SixSevenLeaderboard from "../components/sixseven/SixSevenLeaderboard";
import type { SixSevenEntry } from "../lib/sixseven/leaderboard";

const board: SixSevenEntry[] = [
  { name: "Ann", count: 30, bestCombo: 6, ts: 1 },
  { name: "Bo", count: 20, bestCombo: 4, ts: 2 },
];

describe("SixSevenLeaderboard", () => {
  it("shows an empty state with no entries", () => {
    renderWithProviders(<SixSevenLeaderboard entries={[]} />);
    expect(screen.getByText("No scores yet — be the first!")).toBeInTheDocument();
  });

  it("renders each entry with its 67 count", () => {
    renderWithProviders(<SixSevenLeaderboard entries={board} />);
    expect(screen.getByText("Ann")).toBeInTheDocument();
    expect(screen.getByText("30 × 67")).toBeInTheDocument();
    expect(screen.getByText("20 × 67")).toBeInTheDocument();
  });

  it("highlights the submitted rank", () => {
    renderWithProviders(<SixSevenLeaderboard entries={board} highlightRank={1} />);
    expect(screen.getByText("You")).toBeInTheDocument();
  });
});
