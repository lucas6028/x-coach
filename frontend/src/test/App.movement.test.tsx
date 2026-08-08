import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api } from "../api";
import App from "../App";
import { mockAnalysis } from "./fixtures";

// The upload path extracts pose client-side before hitting the API (CaptureStudio ->
// runPoseAnalysis -> extractPoseFromBlob -> api.analyzePose). The real implementation needs a
// <video>/WASM pipeline jsdom cannot run, so stub it — these tests are about which MOVEMENT
// reaches the request, not about extraction. Mirrors App.test.tsx's stub.
vi.mock("../lib/poseExtract", () => ({
  extractPoseFromBlob: vi.fn().mockResolvedValue({
    metadata: { fps: 30, width: 1, height: 1, total_frames: 0 },
    frames: [],
  }),
}));

// runPoseAnalysis also calls captureThumbnail on the same blob. The real implementation drives a
// <video> decode jsdom cannot perform (it hung/failed these tests once App.tsx started calling
// it); these tests are about which MOVEMENT reaches the request, not about thumbnail capture.
vi.mock("../lib/thumbnail", () => ({ captureThumbnail: () => Promise.resolve(null) }));

const LIVE = [
  { name: "Squat", validated: true },
  { name: "Overhead Press", validated: false },
  { name: "Push-up", validated: false },
];

// The dropzone mounts only AFTER GET /api/movements settles — DemoIntro gates CaptureStudio on
// `movementsLoaded` — while the header's movement selector renders from the URL on the very first
// paint. So waiting on the selector does not imply the dropzone exists: it raced the fetch, and
// under full-suite load it lost, leaving `querySelector('input[type=file]')` null. Wait for the
// input itself; that also implies the catalog landed, so the Beta badge and the canonicalized
// label are safe to assert synchronously afterwards.
async function findFileInput(): Promise<HTMLInputElement> {
  await vi.waitFor(() => expect(document.querySelector('input[type="file"]')).not.toBeNull());
  return document.querySelector('input[type="file"]') as HTMLInputElement;
}

describe("studio movement selection", () => {
  beforeEach(() => {
    vi.spyOn(api, "getMovements").mockResolvedValue(LIVE);
  });
  afterEach(() => vi.restoreAllMocks());

  it("preselects the movement from the URL", async () => {
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });
    expect(await screen.findByLabelText("Movement: Push-up")).toBeInTheDocument();
  });

  it("defaults to Squat when the URL says nothing", async () => {
    renderWithProviders(<App />, { route: "/app" });
    expect(await screen.findByLabelText("Movement: Squat")).toBeInTheDocument();
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
    const input = await findFileInput();
    await userEvent.upload(input, new File(["x"], "clip.mp4", { type: "video/mp4" }));
    // The movement is analyzePose's FIRST argument. It shipped hardcoded to "Squat" when the
    // client-capture path landed; this asserts the user's actual selection now reaches it.
    // Trailing args: pose, video blob, and the (mocked-null) thumbnail — not this test's concern.
    await vi.waitFor(() =>
      expect(analyze).toHaveBeenCalledWith(
        "Push-up",
        expect.anything(),
        expect.anything(),
        // The mocked captureThumbnail above always resolves null; expect.anything() does not
        // match null/undefined, so this asserts the literal value it actually forwards.
        null
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
  // prompt all stayed pinned to "squat" no matter which movement was selected — surfaces upstream
  // of the verdict telling a Push-up user they were uploading a squat. The navbar title has since
  // been removed app-wide, so the two remaining surfaces carry the pin.
  it("names the URL-selected movement in the demo heading and upload prompt", async () => {
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });
    // The upload prompt lives INSIDE the dropzone, so this assertion needs the catalog to have
    // landed too — waiting on the selector alone raced it the same way.
    await findFileInput();
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Analyze your Push-up in about 20 seconds."
    );
    expect(screen.getByText("Drop a Push-up video or tap to upload")).toBeInTheDocument();
    expect(screen.queryByText(/your Squat in about/i)).not.toBeInTheDocument();
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

      // Gate every assertion below on the dropzone, not on the label: before the catalog lands,
      // an unresolved `" Push-up "` still renders `aria-label="Movement:  Push-up "`, which RTL's
      // default normalizer trims into a match — so `findByLabelText` could pass pre-load and drop
      // the following sync assertions (Beta especially) into the unloaded render.
      const input = await findFileInput();

      // Canonicalized, not merely accepted: the control's current value, the i18n label and the
      // Beta badge all key off the registry's spelling, so a case-insensitive gate alone would
      // leave the dropdown showing a phantom duplicate option and the Beta tag missing.
      expect(screen.getByLabelText("Movement: Push-up")).toBeInTheDocument();
      expect(screen.queryByText(/not.*analys|尚未/i)).toBeNull();
      expect(screen.getByText("Beta")).toBeTruthy();
      expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
        "Analyze your Push-up in about 20 seconds."
      );

      await userEvent.upload(input, new File(["x"], "clip.mp4", { type: "video/mp4" }));
      await vi.waitFor(() =>
        expect(analyze).toHaveBeenCalledWith(
          "Push-up",
          expect.anything(),
          expect.anything(),
          // The mocked captureThumbnail above always resolves null; expect.anything() does not
          // match null/undefined, so this asserts the literal value it actually forwards.
          null
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

  // The rest of the file mocks getMovements as an already-resolved promise, which settles within a
  // microtask of the first render — so close to it that whether the dropzone had mounted came down
  // to scheduling luck, and a loaded machine flipped it (the full suite failed here while the file
  // passed alone). This case pins the same wiring against a fetch that settles a macrotask later,
  // the way a real network does: the dropzone appears afterwards, and the upload still carries the
  // URL-selected movement.
  it("mounts the dropzone and sends the movement after a slower movements fetch", async () => {
    vi.spyOn(api, "getMovements").mockReturnValue(
      new Promise((r) => setTimeout(() => r(LIVE), 0))
    );
    const analyze = vi
      .spyOn(api, "analyzePose")
      .mockResolvedValue({ ...mockAnalysis, video_id: "v1", movement: "Push-up" });
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });

    const input = await findFileInput();
    await userEvent.upload(input, new File(["x"], "clip.mp4", { type: "video/mp4" }));
    await vi.waitFor(() =>
      expect(analyze).toHaveBeenCalledWith("Push-up", expect.anything(), expect.anything(), null)
    );
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
