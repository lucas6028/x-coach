import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";

// Stub the MediaPipe/WASM boundary so the test never touches WebGL / the CDN.
vi.mock("../components/blast/blastDetector", () => ({
  createPoseLandmarker: vi.fn().mockResolvedValue({ close: vi.fn() }),
  drawScene: vi.fn(),
}));

import MemeBlast from "../pages/MemeBlast";
import { clearLeaderboard, saveScore } from "../lib/blast/leaderboard";

describe("MemeBlast", () => {
  beforeEach(() => {
    clearLeaderboard();
    vi.restoreAllMocks();
  });

  it("renders the start screen with the local leaderboard", () => {
    saveScore({ name: "Champ", score: 999, hits: 9, bestCombo: 4, ts: 1 });
    renderWithProviders(<MemeBlast />);
    expect(screen.getByText("Charge up. Blast the memes.")).toBeInTheDocument();
    expect(screen.getByText("Champ")).toBeInTheDocument();
  });

  it("surfaces a camera error when access is denied", async () => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new Error("Permission denied")) },
    });
    renderWithProviders(<MemeBlast />);
    fireEvent.click(screen.getByRole("button", { name: /Enable camera/i }));
    await waitFor(() =>
      expect(screen.getByText("Permission denied")).toBeInTheDocument()
    );
  });
});
