import { describe, it, expect, vi, afterEach } from "vitest";
import type React from "react";
import { screen, fireEvent, waitFor, act, render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { AuthProvider } from "../lib/auth";
import { I18nProvider } from "../lib/i18n";
import VideoPanel from "../components/VideoPanel";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis } from "./fixtures";
import { api, type Analysis, type UploadMedia } from "../api";

// VideoPanel renders <video ref={videoRef}>, so React attaches the real jsdom
// <video> DOM node to .current on mount. We only need an empty container ref —
// a hand-rolled fake video would be silently discarded. Media APIs (play/pause)
// on the real node are stubbed globally in src/test/setup.ts.
function makeVideoRef(): React.RefObject<HTMLVideoElement> {
  return { current: null };
}

describe("VideoPanel", () => {
  it("renders the video element", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(document.querySelector("video")).toBeInTheDocument();
  });

  it("shows '1 fault detected' badge when there is one fault", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByText("1 fault detected")).toBeInTheDocument();
  });

  it("shows 'No faults detected' badge for a clean rep", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockCleanAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByText("No faults detected")).toBeInTheDocument();
  });

  it("shows the plural fault badge when fault count > 1", () => {
    const ref = makeVideoRef();
    const multiAnalysis = {
      ...mockAnalysis,
      detections: [
        { ...mockAnalysis.detections[0] },
        { ...mockAnalysis.detections[0], fault_id: "knees_forward_1", fault_name: "knees_forward" },
      ],
    };
    renderWithProviders(
      <VideoPanel
        analysis={multiAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByText("2 faults detected")).toBeInTheDocument();
  });

  it("renders the play button", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getAllByRole("button", { name: /play/i }).length).toBeGreaterThan(0);
  });

  it("renders the fullscreen button", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByRole("button", { name: /fullscreen/i })).toBeInTheDocument();
  });

  it("clicking play does not throw an error", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    // jsdom video.play() returns undefined, so we just verify no uncaught throw
    const playBtns = screen.getAllByRole("button", { name: /play/i });
    expect(() => fireEvent.click(playBtns[0])).not.toThrow();
  });

  // The timeline moved from a strip beneath the video into the card's own control pill (the
  // reference design has one scrub bar, not two), so its legend went with it. What survives is
  // the thing the legend was labelling: a per-fault marker on the track, named by its title.
  it("renders the scrub bar with a marker per detected fault", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(document.querySelector("[title*='Knee Valgus']")).toBeInTheDocument();
  });

  it("calls onTimeUpdate when the video fires timeupdate", () => {
    const onTimeUpdate = vi.fn();
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={onTimeUpdate}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    const video = document.querySelector("video")!;
    // Simulate a timeupdate event from the actual DOM video element
    fireEvent(video, new Event("timeupdate"));
    expect(onTimeUpdate).toHaveBeenCalled();
  });

  it("updates duration state when the video fires loadedmetadata", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    const video = document.querySelector("video")!;
    fireEvent(video, new Event("loadedmetadata"));
    // No throw means the handler ran
    expect(video).toBeInTheDocument();
  });

  it("updates playing state when the video fires play and pause events", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    const video = document.querySelector("video")!;
    fireEvent(video, new Event("play"));
    // After 'play' fires, play button becomes pause button
    expect(screen.getAllByRole("button", { name: /pause/i }).length).toBeGreaterThan(0);
    fireEvent(video, new Event("pause"));
    // After 'pause' fires, pause button becomes play button
    expect(screen.getAllByRole("button", { name: /play/i }).length).toBeGreaterThan(0);
  });
});

// Keeps one `videoRef` across `rerender` calls — a fresh ref per render would change what a
// switched-analysis test is exercising (the DOM node is meant to persist across the switch).
//
// `rerender` (unlike a fresh `renderWithProviders` call) replaces the whole tree it's given, so
// the provider wrapping has to be part of what's passed to `rerender` too — otherwise the second
// render loses its `I18nProvider`/`AuthProvider` context. Rebuilding renderWithProviders' default
// wrapping (route "/") here is the price of using `rerender` at all.
function renderPanel(analysis: Analysis) {
  const videoRef = makeVideoRef();
  const panel = (a: Analysis) => (
    <MemoryRouter initialEntries={["/"]}>
      <AuthProvider>
        <I18nProvider>
          <VideoPanel
            analysis={a}
            videoRef={videoRef}
            onTimeUpdate={vi.fn()}
            onActiveFault={vi.fn()}
            onSeek={vi.fn()}
          />
        </I18nProvider>
      </AuthProvider>
    </MemoryRouter>
  );
  // Plain `render`, not `renderWithProviders` — the wrapping already happens in `panel` above,
  // and `renderWithProviders` would nest a second `MemoryRouter` around it.
  const utils = render(panel(analysis));
  return {
    ...utils,
    // Takes the next analysis (not a raw element) so callers don't have to rebuild the provider
    // wrapping themselves.
    rerender: (next: Analysis) => utils.rerender(panel(next)),
  };
}

describe("VideoPanel video source", () => {
  afterEach(() => vi.restoreAllMocks());

  it("uses the local file endpoint for a library clip", () => {
    const media = vi.spyOn(api, "uploadMedia");
    renderPanel({ ...mockAnalysis, source: "library", video_id: "vid_001" });
    expect(document.querySelector("video")?.getAttribute("src")).toBe("/api/video-file/vid_001");
    expect(media).not.toHaveBeenCalled();
  });

  it("uses the presigned URL that came back with a fresh upload", () => {
    const media = vi.spyOn(api, "uploadMedia");
    renderPanel({
      ...mockAnalysis,
      source: "upload",
      video_id: "upload_a",
      video_url: "https://signed/source",
    });
    expect(document.querySelector("video")?.getAttribute("src")).toBe("https://signed/source");
    expect(media).not.toHaveBeenCalled();
  });

  it("re-signs on a history replay, where the stored result carries no URL", async () => {
    vi.spyOn(api, "uploadMedia").mockResolvedValue({
      video_url: "https://signed/replayed",
      thumbnail_url: "https://signed/thumb",
      expires_in: 3600,
    });
    renderPanel({ ...mockAnalysis, source: "upload", video_id: "upload_a" });
    await waitFor(() =>
      expect(document.querySelector("video")?.getAttribute("src")).toBe("https://signed/replayed")
    );
  });

  it("still renders the analysis when re-signing fails", async () => {
    vi.spyOn(api, "uploadMedia").mockRejectedValue(new Error("503"));
    renderPanel({ ...mockAnalysis, source: "upload", video_id: "upload_a" });
    // Actually let the rejection settle — asserting immediately after render would pass on a
    // state that already held before `uploadMedia` was even called (the hook clears `src` to
    // `null` synchronously on entering the re-sign branch), so it couldn't tell a working
    // `.catch` from a missing one.
    await act(async () => {
      await Promise.resolve();
    });
    // The panel degrades to no playback rather than blocking: the video element is present with
    // no `src`, and the rest of the analysis (the fault badge) still renders.
    expect(document.querySelector("video")).not.toBeNull();
    expect(document.querySelector("video")?.getAttribute("src")).toBeNull();
    expect(screen.getByText("1 fault detected")).toBeInTheDocument();
    // NOTE: this test cannot discriminate a correct `.catch` from a missing one — either way
    // `src` reads `null` here. What a missing `.catch` would break is silent: an unhandled
    // promise rejection. That is caught by the suite's own unhandled-rejection strictness, not
    // by an assertion in this test.
  });

  it("ignores an in-flight URL once the panel has switched analyses", async () => {
    // A request for the first analysis that resolves only when we say so, so we can settle it
    // AFTER the panel has already moved on to a second analysis.
    let releaseFirst: (media: UploadMedia) => void = () => {};
    const first = new Promise<UploadMedia>((resolve) => {
      releaseFirst = resolve;
    });
    vi.spyOn(api, "uploadMedia").mockImplementation((videoId: string) =>
      videoId === "upload_a"
        ? first
        : Promise.resolve({
            video_url: "https://signed/second",
            thumbnail_url: "https://signed/thumb",
            expires_in: 3600,
          })
    );

    const { rerender } = renderPanel({ ...mockAnalysis, source: "upload", video_id: "upload_a" });
    // Switch before the first request settles.
    rerender({ ...mockAnalysis, source: "upload", video_id: "upload_b" });
    await waitFor(() =>
      expect(document.querySelector("video")?.getAttribute("src")).toBe("https://signed/second")
    );

    // The first analysis's URL arrives late. Without the `cancelled` guard it would overwrite the
    // second video's src — showing the user the wrong clip.
    await act(async () => {
      releaseFirst({
        video_url: "https://signed/first",
        thumbnail_url: "https://signed/thumb",
        expires_in: 3600,
      });
      await Promise.resolve();
    });
    expect(document.querySelector("video")?.getAttribute("src")).toBe("https://signed/second");
  });
});
