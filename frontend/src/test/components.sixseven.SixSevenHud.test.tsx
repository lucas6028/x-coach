import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import SixSevenHud from "../components/sixseven/SixSevenHud";

describe("SixSevenHud", () => {
  it("shows the count, timer, and 6/7 hand indicators", () => {
    renderWithProviders(
      <SixSevenHud count={42} combo={0} timeLeft={30} lead="left" pop={0} />
    );
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("30")).toBeInTheDocument();
    expect(screen.getByText("sixty-sevens")).toBeInTheDocument();
    expect(screen.getByText("6")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
  });

  it("shows the rhythm combo tag from 2", () => {
    renderWithProviders(
      <SixSevenHud count={10} combo={4} timeLeft={20} lead="right" pop={2} />
    );
    expect(screen.getByText("4× rhythm")).toBeInTheDocument();
  });

  it("hides the combo tag below 2", () => {
    renderWithProviders(
      <SixSevenHud count={1} combo={1} timeLeft={20} lead="neutral" pop={0} />
    );
    expect(screen.queryByText(/rhythm/)).not.toBeInTheDocument();
  });
});
