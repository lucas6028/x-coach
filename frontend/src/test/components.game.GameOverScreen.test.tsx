import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import GameOverScreen from "../components/game/GameOverScreen";
import type { ScoreEntry } from "../lib/game/leaderboard";

const result = { score: 1500, poses: 12, bestCombo: 6 };
const board: ScoreEntry[] = [{ name: "Me", score: 1500, poses: 12, bestCombo: 6, ts: 1 }];

describe("GameOverScreen", () => {
  it("shows the round score and stats", () => {
    renderWithProviders(
      <GameOverScreen
        result={result}
        leaderboard={[]}
        rank={null}
        submitted={false}
        onSubmit={vi.fn()}
        onReplay={vi.fn()}
      />
    );
    expect(screen.getByText("1,500")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByText("6×")).toBeInTheDocument();
  });

  it("submits a typed name", () => {
    const onSubmit = vi.fn();
    renderWithProviders(
      <GameOverScreen
        result={result}
        leaderboard={[]}
        rank={null}
        submitted={false}
        onSubmit={onSubmit}
        onReplay={vi.fn()}
      />
    );
    fireEvent.change(screen.getByPlaceholderText("Your name"), {
      target: { value: "Grace" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSubmit).toHaveBeenCalledWith("Grace");
  });

  it("falls back to Anonymous when the name is blank", () => {
    const onSubmit = vi.fn();
    renderWithProviders(
      <GameOverScreen
        result={result}
        leaderboard={[]}
        rank={null}
        submitted={false}
        onSubmit={onSubmit}
        onReplay={vi.fn()}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSubmit).toHaveBeenCalledWith("Anonymous");
  });

  it("submits on Enter key", () => {
    const onSubmit = vi.fn();
    renderWithProviders(
      <GameOverScreen
        result={result}
        leaderboard={[]}
        rank={null}
        submitted={false}
        onSubmit={onSubmit}
        onReplay={vi.fn()}
      />
    );
    fireEvent.change(screen.getByPlaceholderText("Your name"), {
      target: { value: "Alan" },
    });
    fireEvent.keyDown(screen.getByPlaceholderText("Your name"), { key: "Enter" });
    expect(onSubmit).toHaveBeenCalledWith("Alan");
  });

  it("shows the rank and leaderboard once submitted", () => {
    renderWithProviders(
      <GameOverScreen
        result={result}
        leaderboard={board}
        rank={1}
        submitted
        onSubmit={vi.fn()}
        onReplay={vi.fn()}
      />
    );
    expect(screen.getByText("You're #1 on the local board!")).toBeInTheDocument();
    expect(screen.getByText("Me")).toBeInTheDocument();
  });

  it("shows an off-board message when not ranked", () => {
    renderWithProviders(
      <GameOverScreen
        result={result}
        leaderboard={board}
        rank={null}
        submitted
        onSubmit={vi.fn()}
        onReplay={vi.fn()}
      />
    );
    expect(
      screen.getByText("Saved — keep practising to crack the top 10.")
    ).toBeInTheDocument();
  });

  it("fires onReplay", () => {
    const onReplay = vi.fn();
    renderWithProviders(
      <GameOverScreen
        result={result}
        leaderboard={[]}
        rank={null}
        submitted={false}
        onSubmit={vi.fn()}
        onReplay={onReplay}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /Play again/i }));
    expect(onReplay).toHaveBeenCalledOnce();
  });
});
