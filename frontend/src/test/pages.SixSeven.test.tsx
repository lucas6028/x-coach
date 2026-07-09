import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";

// Stub the MediaPipe/WASM boundary so the test never touches WebGL / the CDN.
vi.mock("../components/sixseven/sixSevenDetector", () => ({
  createPoseLandmarker: vi.fn().mockResolvedValue({ close: vi.fn() }),
  drawScene: vi.fn(),
}));

import SixSeven from "../pages/SixSeven";
import { clearLeaderboard, saveScore } from "../lib/sixseven/leaderboard";

describe("SixSeven", () => {
  beforeEach(() => {
    clearLeaderboard();
    vi.restoreAllMocks();
  });

  it("renders the start screen with the local leaderboard", () => {
    saveScore({ name: "Champ", count: 50, bestCombo: 12, ts: 1 });
    renderWithProviders(<SixSeven />);
    expect(screen.getByText("How many 67s can you hit?")).toBeInTheDocument();
    expect(screen.getByText("Champ")).toBeInTheDocument();
  });

  it("surfaces a localized camera error when access is denied", async () => {
    // Raw DOMExceptions are localized away — the user sees the translated string, not
    // the browser's English "Permission denied".
    vi.spyOn(console, "error").mockImplementation(() => {});
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new Error("Permission denied")) },
    });
    renderWithProviders(<SixSeven />);
    fireEvent.click(screen.getByRole("button", { name: /Enable camera & go/i }));
    await waitFor(() =>
      expect(screen.getByText("Couldn't start the camera.")).toBeInTheDocument()
    );
    expect(screen.queryByText("Permission denied")).not.toBeInTheDocument();
  });
});
