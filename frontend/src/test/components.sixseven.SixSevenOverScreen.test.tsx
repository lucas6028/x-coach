import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import SixSevenOverScreen from "../components/sixseven/SixSevenOverScreen";
import type { SixSevenEntry } from "../lib/sixseven/leaderboard";

const result = { count: 37, bestCombo: 9, kcal: 6 };
const board: SixSevenEntry[] = [{ name: "Me", count: 37, bestCombo: 9, ts: 1 }];

function setup(props: Partial<React.ComponentProps<typeof SixSevenOverScreen>> = {}) {
  return renderWithProviders(
    <SixSevenOverScreen
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

describe("SixSevenOverScreen", () => {
  it("shows the final count and best combo", () => {
    setup();
    expect(screen.getByText("37")).toBeInTheDocument();
    expect(screen.getByText("Best rhythm streak: 9")).toBeInTheDocument();
  });

  it("submits a typed name", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    fireEvent.change(screen.getByPlaceholderText("Your name"), { target: { value: "Zoe" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSubmit).toHaveBeenCalledWith("Zoe");
  });

  it("falls back to Anonymous for a blank name", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSubmit).toHaveBeenCalledWith("Anonymous");
  });

  it("submits on Enter", () => {
    const onSubmit = vi.fn();
    setup({ onSubmit });
    const input = screen.getByPlaceholderText("Your name");
    fireEvent.change(input, { target: { value: "Ivy" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSubmit).toHaveBeenCalledWith("Ivy");
  });

  it("shows the rank and board once submitted", () => {
    setup({ submitted: true, rank: 1, leaderboard: board });
    expect(screen.getByText("You're #1 on the local board!")).toBeInTheDocument();
    expect(screen.getByText("Me")).toBeInTheDocument();
  });

  it("shows an off-board message when not ranked", () => {
    setup({ submitted: true, rank: null, leaderboard: board });
    expect(screen.getByText("Saved — keep bobbing to crack the top 10.")).toBeInTheDocument();
  });

  it("fires onReplay", () => {
    const onReplay = vi.fn();
    setup({ onReplay });
    fireEvent.click(screen.getByRole("button", { name: /Go again/i }));
    expect(onReplay).toHaveBeenCalledOnce();
  });
});
