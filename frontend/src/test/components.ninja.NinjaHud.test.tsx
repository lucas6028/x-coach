import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import NinjaHud from "../components/ninja/NinjaHud";

describe("NinjaHud", () => {
  it("shows the score and lives", () => {
    renderWithProviders(<NinjaHud score={1500} combo={0} lives={2} pop={0} bombFlash={false} />);
    expect(screen.getByText("1,500")).toBeInTheDocument();
    // 2 of 3 hearts filled.
    expect(screen.getByText("❤️❤️🖤")).toBeInTheDocument();
  });

  it("shows the combo pop from 3", () => {
    renderWithProviders(<NinjaHud score={0} combo={5} lives={3} pop={4} bombFlash={false} />);
    expect(screen.getByText("5 combo!")).toBeInTheDocument();
  });

  it("hides the combo pop below 3", () => {
    renderWithProviders(<NinjaHud score={0} combo={2} lives={3} pop={0} bombFlash={false} />);
    expect(screen.queryByText(/combo!/)).not.toBeInTheDocument();
  });

  it("shows the bomb banner over the combo", () => {
    renderWithProviders(<NinjaHud score={0} combo={5} lives={0} pop={4} bombFlash />);
    expect(screen.getByText("💥 BOOM")).toBeInTheDocument();
    expect(screen.queryByText("5 combo!")).not.toBeInTheDocument();
  });
});
