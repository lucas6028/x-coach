import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";

// Stub the MediaPipe/WASM boundary so the test never touches WebGL / the CDN.
vi.mock("../components/duel/duelDetector", () => ({
  createPoseLandmarker: vi.fn().mockResolvedValue({ close: vi.fn() }),
  drawScene: vi.fn(),
}));

import PoseDuel from "../pages/PoseDuel";
import { clearResults, saveResult } from "../lib/duel/leaderboard";

describe("PoseDuel", () => {
  beforeEach(() => {
    clearResults();
    vi.restoreAllMocks();
  });

  it("renders the start screen with the recent-duels board", () => {
    saveResult({ winner: "Champ", loser: "Foe", winnerPoints: 3, loserPoints: 0, ts: 1 });
    renderWithProviders(<PoseDuel />);
    expect(screen.getByText("Strike the pose. Beat your rival.")).toBeInTheDocument();
    expect(screen.getByText("Champ")).toBeInTheDocument();
  });

  it("surfaces a camera error when access is denied", async () => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new Error("Permission denied")) },
    });
    renderWithProviders(<PoseDuel />);
    fireEvent.click(screen.getByRole("button", { name: /Enable camera & duel/i }));
    await waitFor(() =>
      expect(screen.getByText("Permission denied")).toBeInTheDocument()
    );
  });
});
