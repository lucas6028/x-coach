import { describe, it, expect, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import Games from "../pages/Games";
import { addCalories, clearCalories } from "../lib/calorieStore";
import { saveScore as saveNinja, clearLeaderboard as clearNinja } from "../lib/ninja/leaderboard";
import { saveScore as saveSix, clearLeaderboard as clearSix } from "../lib/sixseven/leaderboard";

describe("Games hub", () => {
  beforeEach(() => {
    clearCalories();
    clearNinja();
    clearSix();
  });

  it("lists both games with play links into their routes", () => {
    renderWithProviders(<Games />);
    expect(screen.getByText("Fruit Ninja")).toBeInTheDocument();
    expect(screen.getByText("67")).toBeInTheDocument();
    const links = screen.getAllByRole("link");
    const hrefs = links.map((l) => l.getAttribute("href"));
    expect(hrefs).toContain("/ninja");
    expect(hrefs).toContain("/67");
  });

  it("shows the empty calorie state before any round", () => {
    renderWithProviders(<Games />);
    expect(screen.getByText("Play a round to start burning.")).toBeInTheDocument();
  });

  it("uses the singular subtitle after exactly one round", () => {
    addCalories("ninja", 12);
    renderWithProviders(<Games />);
    expect(screen.getByText("Estimated across 1 round")).toBeInTheDocument();
  });

  it("shows the cumulative calorie total and per-game best once played", () => {
    addCalories("ninja", 20);
    addCalories("sixseven", 5);
    saveNinja({ name: "Me", score: 1500, bestCombo: 8, ts: 1 });
    saveSix({ name: "Me", count: 42, bestCombo: 6, ts: 1 });

    renderWithProviders(<Games />);
    // 20 + 5 across 2 rounds.
    expect(screen.getByText("25")).toBeInTheDocument();
    expect(screen.getByText("Estimated across 2 rounds")).toBeInTheDocument();
    // Per-game bests surface on the cards.
    expect(screen.getByText("1,500")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });
});
