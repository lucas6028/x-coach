import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api } from "../api";
import App from "../App";
import { mockAnalysis } from "./fixtures";

const LIVE = [
  { name: "Squat", validated: true },
  { name: "Overhead Press", validated: false },
  { name: "Push-up", validated: false },
];

describe("studio movement selection", () => {
  beforeEach(() => {
    vi.spyOn(api, "getMovements").mockResolvedValue(LIVE);
  });
  afterEach(() => vi.restoreAllMocks());

  it("preselects the movement from the URL", async () => {
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });
    const select = (await screen.findByLabelText(/movement/i)) as HTMLSelectElement;
    expect(select.value).toBe("Push-up");
  });

  it("defaults to Squat when the URL says nothing", async () => {
    renderWithProviders(<App />, { route: "/app" });
    const select = (await screen.findByLabelText(/movement/i)) as HTMLSelectElement;
    expect(select.value).toBe("Squat");
  });

  it("sends the selected movement with the upload", async () => {
    // A full Analysis shape (not the brief's bare { video_id, movement } stub): App genuinely
    // transitions to the result view once the upload resolves, and VideoPanel/Header/CoachTray
    // destructure fields (metadata, view, retrievals) a partial object doesn't have — which
    // surfaced as an unrelated unhandled render exception, not this test's own failure.
    const upload = vi
      .spyOn(api, "analyzeUpload")
      .mockResolvedValue({ ...mockAnalysis, video_id: "v1", movement: "Push-up" });
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });
    await screen.findByLabelText(/movement/i);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await userEvent.upload(input, new File(["x"], "clip.mp4", { type: "video/mp4" }));
    expect(upload).toHaveBeenCalledWith(expect.any(File), "Push-up");
  });

  it("refuses an unanalyzable movement in the URL without spending an upload", async () => {
    const upload = vi.spyOn(api, "analyzeUpload");
    renderWithProviders(<App />, { route: "/app?movement=Lunge" });
    expect(await screen.findByText(/not.*analys|尚未/i)).toBeTruthy();
    expect(document.querySelector('input[type="file"]')).toBeNull();
    expect(upload).not.toHaveBeenCalled();
  });

  it("shows the Beta note for an unvalidated movement only", async () => {
    renderWithProviders(<App />, { route: "/app?movement=Overhead Press" });
    expect(await screen.findByText("Beta")).toBeTruthy();
  });

  it("does not claim a movement is unavailable while the list is still loading", async () => {
    let resolve!: (ms: { name: string; validated: boolean }[]) => void;
    vi.spyOn(api, "getMovements").mockReturnValue(
      new Promise((r) => {
        resolve = r;
      })
    );
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });
    // In flight: "we don't know yet" must not render as "no".
    expect(screen.queryByText(/not.*analys|尚未/i)).toBeNull();
    resolve(LIVE);
    expect(await screen.findByLabelText(/movement/i)).toBeTruthy();
    expect(screen.queryByText(/not.*analys|尚未/i)).toBeNull();
  });
});
