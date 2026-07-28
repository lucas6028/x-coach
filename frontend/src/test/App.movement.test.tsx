import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api } from "../api";
import App from "../App";
import { mockAnalysis } from "./fixtures";

// The upload path extracts pose client-side before hitting the API (CaptureStudio ->
// runPoseAnalysis -> extractPoseWithReps -> api.analyzePose). The real implementation needs a
// <video>/WASM pipeline jsdom cannot run, so stub it — these tests are about which MOVEMENT
// reaches the request, not about extraction. Mirrors App.test.tsx's stub.
vi.mock("../lib/poseExtract", () => ({
  extractPoseWithReps: vi.fn().mockResolvedValue({
    pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] },
    reps: { max_reps: 3, fallback: null, segments: [] },
  }),
}));

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
    const analyze = vi
      .spyOn(api, "analyzePose")
      .mockResolvedValue({ ...mockAnalysis, video_id: "v1", movement: "Push-up" });
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });
    await screen.findByLabelText(/movement/i);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await userEvent.upload(input, new File(["x"], "clip.mp4", { type: "video/mp4" }));
    // The movement is analyzePose's FIRST argument. It shipped hardcoded to "Squat" when the
    // client-capture path landed; this asserts the user's actual selection now reaches it.
    // Fourth arg is the rep plan extractPoseWithReps returns (Task 10) — the mock above supplies
    // one on every call now, so a bare 3-arg expectation would under-match the real call.
    await vi.waitFor(() =>
      expect(analyze).toHaveBeenCalledWith(
        "Push-up", expect.anything(), expect.anything(), expect.anything()
      )
    );
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

  // Finding 1 + 2 of the 2026-07-25 review: the header title, the demo heading, and the upload
  // prompt all stayed pinned to "squat" no matter which movement was selected — three surfaces
  // upstream of the verdict telling a Push-up user they were uploading a squat. This is the
  // end-to-end pin that all three now track the URL-driven selection together.
  it("names the URL-selected movement in the header, heading, and upload prompt", async () => {
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });
    await screen.findByLabelText(/movement/i);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Push-up Analysis");
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Analyze your Push-up in about 20 seconds."
    );
    expect(screen.getByText("Drop a Push-up video or tap to upload")).toBeInTheDocument();
    expect(screen.queryByText(/Squat Analysis/i)).not.toBeInTheDocument();
  });

  // The backend matches movements case-insensitively — `registry.get_detector` lowercases its
  // lookup key and `_validated_movement` returns the registry's own spelling — so `?movement=push-up`
  // is a request the API accepts and canonicalizes to "Push-up". The studio compared exactly, so a
  // differently-cased link (hand-typed, shared, or lowercased by another tool) locked the user out
  // of a movement the server would have analyzed fine.
  it.each(["push-up", "PUSH-UP", "Push-Up", " Push-up "])(
    "accepts %j as the URL movement, the way the backend does",
    async (param) => {
      const analyze = vi
        .spyOn(api, "analyzePose")
        .mockResolvedValue({ ...mockAnalysis, video_id: "v1", movement: "Push-up" });
      renderWithProviders(<App />, { route: `/app?movement=${encodeURIComponent(param)}` });

      const select = (await screen.findByLabelText(/movement/i)) as HTMLSelectElement;
      // Canonicalized, not merely accepted: the <select> value, the i18n label and the Beta badge
      // all key off the registry's spelling, so a case-insensitive gate alone would leave the
      // dropdown showing a phantom duplicate option and the Beta tag missing.
      expect(select.value).toBe("Push-up");
      expect(screen.queryByText(/not.*analys|尚未/i)).toBeNull();
      expect(screen.getByText("Beta")).toBeTruthy();
      expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Push-up Analysis");

      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      expect(input).not.toBeNull();
      await userEvent.upload(input, new File(["x"], "clip.mp4", { type: "video/mp4" }));
      // Fourth arg is the rep plan (Task 10) — see the note on the earlier assertion of this shape.
      await vi.waitFor(() =>
        expect(analyze).toHaveBeenCalledWith(
          "Push-up", expect.anything(), expect.anything(), expect.anything()
        )
      );
    }
  );

  it("still refuses a movement that differs by more than case", async () => {
    // Guards the fix from overreaching into fuzzy matching: only spelling/case may vary.
    renderWithProviders(<App />, { route: "/app?movement=pushup" });
    expect(await screen.findByText(/not.*analys|尚未/i)).toBeTruthy();
    expect(document.querySelector('input[type="file"]')).toBeNull();
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
    // AND the dropzone must not render either — a slow network must not let someone upload
    // against a movement we have not confirmed. This pins the App-level wiring of
    // `movementsLoaded` into DemoIntro, not just DemoIntro's own gate: without this assertion, a
    // mis-wired `movementsLoaded={true}` passed from App would still leave this test green.
    expect(document.querySelector('input[type="file"]')).toBeNull();
    resolve(LIVE);
    expect(await screen.findByLabelText(/movement/i)).toBeTruthy();
    expect(screen.queryByText(/not.*analys|尚未/i)).toBeNull();
  });
});
