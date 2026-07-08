import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import BlastHud from "../components/blast/BlastHud";

describe("BlastHud", () => {
  it("shows score, time, and the charge prompt", () => {
    renderWithProviders(
      <BlastHud score={2400} combo={0} timeLeft={30} charge={0.4} armed={false} flash={null} />
    );
    expect(screen.getByText("2,400")).toBeInTheDocument();
    expect(screen.getByText("30")).toBeInTheDocument();
    expect(screen.getByText("Hands together — charge!")).toBeInTheDocument();
  });

  it("switches to the FIRE prompt when armed", () => {
    renderWithProviders(
      <BlastHud score={0} combo={0} timeLeft={30} charge={1} armed flash={null} />
    );
    expect(screen.getByText(/Fling arms apart to FIRE/)).toBeInTheDocument();
  });

  it("shows the combo tag from 2 and a hit flash with a multi-kill count", () => {
    renderWithProviders(
      <BlastHud
        score={0}
        combo={3}
        timeLeft={30}
        charge={0}
        armed={false}
        flash={{ hits: 2, points: 350 }}
      />
    );
    expect(screen.getByText("3× combo")).toBeInTheDocument();
    expect(screen.getByText("+350")).toBeInTheDocument();
    expect(screen.getByText("×2")).toBeInTheDocument();
  });

  it("shows a whiff flash when a shot hits nothing", () => {
    renderWithProviders(
      <BlastHud
        score={0}
        combo={0}
        timeLeft={30}
        charge={0}
        armed={false}
        flash={{ hits: 0, points: 0 }}
      />
    );
    expect(screen.getByText("Whiff!")).toBeInTheDocument();
  });
});
