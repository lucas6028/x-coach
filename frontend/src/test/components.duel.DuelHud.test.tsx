import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import DuelHud from "../components/duel/DuelHud";

const base = {
  poseEmoji: "🧍",
  poseNameKey: "t_pose",
  a: { wins: 1, hold: 0.5, present: true },
  b: { wins: 0, hold: 0.2, present: true },
  roundFlash: null,
};

describe("DuelHud", () => {
  it("shows both players, the target pose, and the prompts", () => {
    renderWithProviders(<DuelHud {...base} />);
    expect(screen.getByText("Player A")).toBeInTheDocument();
    expect(screen.getByText("Player B")).toBeInTheDocument();
    expect(screen.getByText("T-Pose")).toBeInTheDocument();
    expect(screen.getByText("Match this")).toBeInTheDocument();
    expect(screen.getByText("Hold it!")).toBeInTheDocument();
  });

  it("shows the round-win banner when a round is taken", () => {
    renderWithProviders(<DuelHud {...base} roundFlash="a" />);
    expect(screen.getByText("Player A takes the round!")).toBeInTheDocument();
  });

  it("shows the round-win banner for player B", () => {
    renderWithProviders(<DuelHud {...base} roundFlash="b" />);
    expect(screen.getByText("Player B takes the round!")).toBeInTheDocument();
  });

  it("has no banner mid-round", () => {
    renderWithProviders(<DuelHud {...base} roundFlash={null} />);
    expect(screen.queryByText(/takes the round/)).not.toBeInTheDocument();
  });
});
