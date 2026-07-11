import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";

// Stub the MediaPipe/WASM boundary so the test never touches WebGL / the CDN.
vi.mock("../components/ninja/ninjaDetector", () => ({
  createPoseLandmarker: vi.fn().mockResolvedValue({ close: vi.fn() }),
  drawScene: vi.fn(),
}));

import FruitNinja from "../pages/FruitNinja";
import { clearLeaderboard, saveScore } from "../lib/ninja/leaderboard";

describe("FruitNinja", () => {
  beforeEach(() => {
    clearLeaderboard();
    vi.restoreAllMocks();
  });

  it("renders the start screen with the local leaderboard", () => {
    saveScore({ name: "Slicer", score: 999, bestCombo: 10, ts: 1 });
    renderWithProviders(<FruitNinja />);
    expect(screen.getByText("Your hands are the blades.")).toBeInTheDocument();
    expect(screen.getByText("Slicer")).toBeInTheDocument();
  });

  it("surfaces a camera error when access is denied", async () => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new Error("Permission denied")) },
    });
    renderWithProviders(<FruitNinja />);
    fireEvent.click(screen.getByRole("button", { name: /Enable camera & slice/i }));
    await waitFor(() => expect(screen.getByText("Permission denied")).toBeInTheDocument());
  });
});
