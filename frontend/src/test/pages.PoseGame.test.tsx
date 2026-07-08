import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";

// Stub the MediaPipe/WASM boundary so the test never touches real WebGL / the CDN.
vi.mock("../components/game/poseDetector", () => ({
  createPoseLandmarker: vi.fn().mockResolvedValue({ close: vi.fn() }),
  drawSkeleton: vi.fn(),
}));

import PoseGame from "../pages/PoseGame";
import { clearLeaderboard, saveScore } from "../lib/game/leaderboard";

describe("PoseGame", () => {
  beforeEach(() => {
    clearLeaderboard();
    vi.restoreAllMocks();
  });

  it("renders the start screen with the local leaderboard", () => {
    saveScore({ name: "Champ", score: 999, poses: 8, bestCombo: 4, ts: 1 });
    renderWithProviders(<PoseGame />);
    expect(screen.getByText("Strike the pose. Beat the clock.")).toBeInTheDocument();
    expect(screen.getByText("Champ")).toBeInTheDocument();
  });

  it("surfaces a camera error when access is denied", async () => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockRejectedValue(new Error("Permission denied")),
      },
    });
    renderWithProviders(<PoseGame />);
    fireEvent.click(screen.getByRole("button", { name: /Enable camera/i }));
    await waitFor(() =>
      expect(screen.getByText("Permission denied")).toBeInTheDocument()
    );
  });
});
