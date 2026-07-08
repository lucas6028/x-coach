import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import GameHud from "../components/game/GameHud";
import { POSES } from "../lib/game/poses";

const target = POSES[0];

describe("GameHud", () => {
  it("shows score, time, and the target pose name", () => {
    renderWithProviders(
      <GameHud
        score={1250}
        combo={0}
        timeLeft={42}
        target={target}
        quality={0.3}
        holdProgress={0}
        lastGrade={null}
      />
    );
    expect(screen.getByText("1,250")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("T-Pose")).toBeInTheDocument();
    expect(screen.getByText("Match the pose")).toBeInTheDocument();
  });

  it("shows the combo tag once the streak reaches 2", () => {
    renderWithProviders(
      <GameHud
        score={0}
        combo={4}
        timeLeft={30}
        target={target}
        quality={0.5}
        holdProgress={0.2}
        lastGrade={null}
      />
    );
    expect(screen.getByText("4× combo")).toBeInTheDocument();
    expect(screen.getByText("Hold it…")).toBeInTheDocument();
  });

  it("flashes a positive grade", () => {
    renderWithProviders(
      <GameHud
        score={0}
        combo={0}
        timeLeft={10}
        target={target}
        quality={0.95}
        holdProgress={0}
        lastGrade="perfect"
      />
    );
    expect(screen.getByText("Perfect!")).toBeInTheDocument();
  });

  it("does not flash a grade banner on a miss", () => {
    renderWithProviders(
      <GameHud
        score={0}
        combo={0}
        timeLeft={10}
        target={target}
        quality={0.1}
        holdProgress={0}
        lastGrade="miss"
      />
    );
    expect(screen.queryByText("Miss")).not.toBeInTheDocument();
  });
});
