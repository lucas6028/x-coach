import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";
import App from "../App";
import { api, UploadLimitError } from "../api";
import { mockAnalysis } from "./fixtures";

// The upload path now extracts pose client-side before hitting the API (CaptureStudio ->
// runPoseAnalysis -> extractPoseWithReps -> api.analyzePose). extractPoseWithReps's real
// implementation needs a real <video>/WASM pipeline that jsdom can't run — stub it so these tests
// exercise the new request path (api.analyzePose against the mocked fetch below) instead. The stub
// returns the same { pose, reps } shape the real two-pass extractor resolves to (Task 6).
vi.mock("../lib/poseExtract", () => ({
  extractPoseWithReps: vi.fn().mockResolvedValue({
    pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] },
    reps: { max_reps: 3, fallback: null, segments: [] },
  }),
}));

// runPoseAnalysis also calls captureThumbnail on the same blob, driving a <video> decode jsdom
// can't run either. Stub it the same way, for the same reason.
vi.mock("../lib/thumbnail", () => ({ captureThumbnail: () => Promise.resolve(null) }));

function renderApp() {
  return render(
    <MemoryRouter>
      <AuthProvider>
        <I18nProvider>
          <App />
        </I18nProvider>
      </AuthProvider>
    </MemoryRouter>
  );
}

// Exposes the current location.search so a test can assert the URL the app navigated to.
function LocationProbe() {
  const loc = useLocation();
  return <div data-testid="loc-search">{loc.search}</div>;
}

function renderAppWithLocation(entry = "/app") {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <AuthProvider>
        <I18nProvider>
          <App />
          <LocationProbe />
        </I18nProvider>
      </AuthProvider>
    </MemoryRouter>
  );
}

// The studio now gates the dropzone behind the GET /api/movements fetch settling (Task 11):
// it won't render the file input until movementsLoaded flips true, so callers must wait for it
// rather than assuming it's present the instant after render.
async function uploadAClip() {
  const input = await waitFor(() => {
    const el = document.querySelector("input[type=file]") as HTMLInputElement | null;
    expect(el).not.toBeNull();
    return el!;
  });
  fireEvent.change(input, {
    target: { files: [new File(["data"], "squat.mp4", { type: "video/mp4" })] },
  });
}

afterEach(() => vi.restoreAllMocks());

describe("App — initial state", () => {
  it("renders the demo intro (no analysis)", () => {
    renderApp();
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Analyze your Squat in about 20 seconds."
    );
  });

  it("renders the sidebar", () => {
    renderApp();
    expect(screen.getAllByText("X-Coach").length).toBeGreaterThan(0);
  });
});

// The sample library is gone: uploading (or recording) a clip is the only way into an analysis
// from the studio. Pinned through App because the picker had THREE entry points — the studio's
// own button, the sidebar and the mobile tab bar — and each one was its own way back to a modal
// that no longer exists.
describe("App — no sample library", () => {
  it("offers no route into a sample picker", () => {
    renderApp();
    expect(screen.queryByRole("button", { name: /sample/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /Library/i })).toBeNull();
    expect(screen.queryByText("Sample Library")).toBeNull();
  });
});

// The detector's key evidence line now renders TWICE on the result screen — once in the video
// card's floating "Detected errors" list and once on the coach panel's fault card — which is the
// reference design's own arrangement, so these look it up with getAllByText.
describe("App — analysis loaded", () => {
  it("shows analysis results when API returns data", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockAnalysis,
    } as Response);

    renderApp();
    await uploadAClip();
    await waitFor(() => expect(screen.getAllByText(/valgus angle 0\.35/)[0]).toBeInTheDocument());
  });

  it("shows an error message when an upload fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Error",
      json: async () => ({ detail: "Server error" }),
    } as Response);

    renderApp();

    // The hidden file input drives uploads — assert it exists so this test can't silently no-op
    // if the dropzone stops rendering it. It only appears once GET /api/movements settles
    // (Task 11's movementsLoaded gate), so wait for it rather than querying synchronously.
    const input = await waitFor(() => {
      const el = document.querySelector("input[type=file]") as HTMLInputElement | null;
      expect(el).not.toBeNull();
      return el!;
    });

    const file = new File(["data"], "squat.mp4", { type: "video/mp4" });
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() =>
      expect(screen.getByText("That clip did not go through")).toBeInTheDocument()
    );
    // The backend's detail message is surfaced to the user.
    expect(screen.getByText("Server error")).toBeInTheDocument();
  });

  it("shows a localised message when the upload exceeds the storage quota", async () => {
    vi.spyOn(api, "analyzePose").mockRejectedValue(
      new UploadLimitError("storage_quota_exceeded", 500, 480)
    );

    renderApp();
    await uploadAClip();

    expect(
      await screen.findByText(/Your storage is full \(480 MB of 500 MB\)/)
    ).toBeInTheDocument();
  });
});

// The movement detail page's "Start recording" card links here with ?capture=record. It is an
// instruction for the arrival, not a description of the session.
describe("App — ?capture=record", () => {
  it("opens the capture panel on the camera and then drops the param", async () => {
    renderAppWithLocation("/app?movement=Squat&capture=record");

    // Re-queried each poll rather than held: the capture panel is swapped for the loader while
    // GET /api/movements settles, so a node captured early can be a detached one.
    await waitFor(() =>
      expect(screen.getByRole("tab", { name: "Record live" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
    );

    // Consumed: left in the URL it would re-apply on every remount of the capture panel and
    // override whichever tab the user picked since. ?movement= survives — that one describes
    // the session.
    await waitFor(() =>
      expect(screen.getByTestId("loc-search").textContent).toBe("?movement=Squat")
    );
    // And the panel stays where the user puts it after the param is gone.
    await userEvent.click(screen.getByRole("tab", { name: "Upload video" }));
    expect(screen.getByRole("tab", { name: "Upload video" })).toHaveAttribute("aria-selected", "true");
  });

  it("opens on the dropzone without it", async () => {
    renderAppWithLocation("/app?movement=Squat");
    const uploadTab = await screen.findByRole("tab", { name: "Upload video" });
    expect(uploadTab).toHaveAttribute("aria-selected", "true");
  });
});

describe("App — upload reflects the analysis in the URL", () => {
  it("updates the URL to ?analysis=<id> after a persisted upload, without re-fetching", async () => {
    // A signed-in upload comes back with a persisted analysis_id.
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ...mockAnalysis, analysis_id: "bb718ecf" }),
    } as Response);

    renderAppWithLocation();
    await uploadAClip();

    await waitFor(() => expect(screen.getAllByText(/valgus angle 0\.35/)[0]).toBeInTheDocument());
    // The URL now carries the id (shareable + refresh-survivable). Poll it: the router's location
    // update can settle a tick after the analysis render.
    await waitFor(() =>
      expect(screen.getByTestId("loc-search").textContent).toBe("?analysis=bb718ecf")
    );
    // The replay effect is guarded, so the analysis is NOT re-fetched — if it were, GET
    // /api/analyses/<id> would return this same (result-less) shape and the fault card's evidence
    // line would disappear.
    expect(screen.getAllByText(/valgus angle 0\.35/)[0]).toBeInTheDocument();
  });

  it("leaves the URL untouched for an anonymous upload (no analysis_id)", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockAnalysis, // anonymous: nothing persisted, no analysis_id
    } as Response);

    renderAppWithLocation();
    await uploadAClip();

    await waitFor(() => expect(screen.getAllByText(/valgus angle 0\.35/)[0]).toBeInTheDocument());
    expect(screen.getByTestId("loc-search").textContent).toBe("");
  });
});

describe("App — new analysis reset", () => {
  it("clears a loaded analysis and the ?analysis= param when 'New analysis' is clicked", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ...mockAnalysis, analysis_id: "bb718ecf" }),
    } as Response);

    const user = userEvent.setup();
    renderAppWithLocation();
    await uploadAClip();

    await waitFor(() => expect(screen.getAllByText(/valgus angle 0\.35/)[0]).toBeInTheDocument());
    await waitFor(() =>
      expect(screen.getByTestId("loc-search").textContent).toBe("?analysis=bb718ecf")
    );

    // The sidebar CTA resets the studio in place (two sidebars render; the first is the desktop rail).
    await user.click(screen.getAllByRole("button", { name: /New analysis/i })[0]);

    // Back to the empty studio, and the shareable id is dropped from the URL.
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Analyze your Squat in about 20 seconds."
    );
    await waitFor(() => expect(screen.getByTestId("loc-search").textContent).toBe(""));
  });
});

describe("App — shell chrome", () => {
  // The navbar's rail-collapse toggle is gone: the rail is 84px, so collapsing it bought 20px in
  // exchange for a permanent control in the top row. Both rails (desktop + mobile drawer) now
  // always render their labels. The drawer's own ✕ keeps the "Hide navigation" name — the point
  // here is that the TOP ROW no longer carries one.
  it("renders both nav rails labelled, with no collapse toggle in the top row", () => {
    renderApp();
    expect(screen.getAllByText("Analyse").length).toBe(2);
    expect(screen.getAllByRole("button", { name: /Hide navigation/i })).toHaveLength(1);
  });
});
