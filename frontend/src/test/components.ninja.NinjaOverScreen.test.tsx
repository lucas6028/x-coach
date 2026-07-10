import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import NinjaOverScreen, { type NinjaResult } from "../components/ninja/NinjaOverScreen";
import type { NinjaEntry } from "../lib/ninja/leaderboard";

const result: NinjaResult = { score: 1340, bestCombo: 11, bombed: false, kcal: 24 };
const board: NinjaEntry[] = [{ name: "Me", score: 1340, bestCombo: 11, ts: 1 }];

function setup(props: Partial<React.ComponentProps<typeof NinjaOverScreen>> = {}) {
  return renderWithProviders(
    <NinjaOverScreen
      result={result}
      leaderboard={[]}
      rank={null}
      submitted={false}
      onSubmit={vi.fn()}
      onReplay={vi.fn()}
      {...props}
    />
  );
}

describe("NinjaOverScreen", () => {
  it("shows the score and best combo", () => {
    setup();
    expect(screen.getByText("1,340")).toBeInTheDocument();
    expect(screen.getByText("Best combo: 11")).toBeInTheDocument();
    expect(screen.getByText("Round over")).toBeInTheDocument();
  });

  it("shows the bomb message when bombed", () => {
    setup({ result: { ...result, bombed: true } });
    expect(screen.getByText("You hit a bomb!")).toBeInTheDocument();
  });

  it("submits a typed name", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    fireEvent.change(screen.getByPlaceholderText("Your name"), { target: { value: "Zed" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSubmit).toHaveBeenCalledWith("Zed");
  });

  it("falls back to Anonymous and guards against a double submit", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    const btn = screen.getByRole("button", { name: "Save" });
    fireEvent.click(btn);
    fireEvent.click(btn);
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith("Anonymous");
  });

  it("shows the rank and board once submitted", () => {
    setup({ submitted: true, rank: 1, leaderboard: board });
    expect(screen.getByText("You're #1 on the local board!")).toBeInTheDocument();
    expect(screen.getByText("Me")).toBeInTheDocument();
  });

  it("fires onReplay", () => {
    const onReplay = vi.fn();
    setup({ onReplay });
    fireEvent.click(screen.getByRole("button", { name: /Play again/i }));
    expect(onReplay).toHaveBeenCalledOnce();
  });
});
