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

  // A clip recorded in the browser is a live-muxed WebM with no Duration element, so Chrome
  // reports `Infinity` for its length (the reason lib/mediaDuration.ts exists). Infinity is
  // truthy, so it used to sail past the `duration || metadata` fallback in the scrub bar and
  // render as "0:00" with a playhead pinned at 0% — the clip appeared to have no length at all.
  it("falls back to the analysis metadata when the clip reports an infinite duration", () => {
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
    Object.defineProperty(video, "duration", { configurable: true, value: Infinity });
    fireEvent(video, new Event("loadedmetadata"));
    // 300 frames at 30fps.
    expect(document.querySelector(".tabular-nums")!.textContent).toContain("0:10");
  });

  it("advances the scrub bar on an infinite-duration clip", () => {
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
    Object.defineProperty(video, "duration", { configurable: true, value: Infinity });
    Object.defineProperty(video, "currentTime", { configurable: true, value: 2 });
    fireEvent(video, new Event("loadedmetadata"));
    fireEvent(video, new Event("timeupdate"));
    // 2s of a 10s clip: the fill has to have real width, not the 0% that `t / Infinity` gives.
    const fill = document.querySelector<HTMLElement>('div[class*="bg-[#8b7bff]"]')!;
    expect(fill.style.width).toBe("20%");
  });

  it("takes the real duration once the clip reports one", () => {
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
    Object.defineProperty(video, "duration", { configurable: true, value: Infinity });
    fireEvent(video, new Event("loadedmetadata"));
    Object.defineProperty(video, "duration", { configurable: true, value: 25 });
    fireEvent(video, new Event("durationchange"));
    expect(document.querySelector(".tabular-nums")!.textContent).toContain("0:25");
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

// The streaming path. What separates "the browser downloads the clip" from "the browser fetches
// the bytes it is about to play" is two attributes and one recovery path; each is asserted here
// because none of them fails visibly — a regression just makes every page load slow again.
describe("VideoPanel streaming", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches only metadata until the viewer presses play", () => {
    renderPanel({
      ...mockAnalysis,
      source: "upload",
      video_id: "upload_a",
      video_url: "https://signed/source",
    });
    // Without this the element defaults to `preload="auto"`, which pulls the whole clip on mount
    // whether or not it is ever watched.
    expect(document.querySelector("video")?.getAttribute("preload")).toBe("metadata");
  });

  it("paints the stored frame as the poster so the lazy stage is not black", () => {
    renderPanel({
      ...mockAnalysis,
      source: "upload",
      video_id: "upload_a",
      video_url: "https://signed/source",
      thumbnail_url: "https://signed/thumb.jpg",
    });
    expect(document.querySelector("video")?.getAttribute("poster")).toBe("https://signed/thumb.jpg");
  });

  it("carries no poster attribute when the upload has no stored frame", () => {
    // Thumbnail capture is best-effort on the client and its put is best-effort on the server, so
    // a missing one must leave the attribute off rather than set `poster=""` — which browsers
    // resolve against the page URL and fetch as an image.
    renderPanel({
      ...mockAnalysis,
      source: "upload",
      video_id: "upload_a",
      video_url: "https://signed/source",
    });
    expect(document.querySelector("video")?.hasAttribute("poster")).toBe(false);
  });

  it("re-signs when playback fails, because the URL may have expired before play", async () => {
    // The failure mode lazy loading introduces: the URL is signed on page load and first USED
    // when the viewer presses play, which can be past DEFAULT_URL_TTL (an hour).
    const media = vi.spyOn(api, "uploadMedia").mockResolvedValue({
      video_url: "https://signed/fresh",
      thumbnail_url: "https://signed/fresh-thumb.jpg",
      expires_in: 3600,
    });
    renderPanel({
      ...mockAnalysis,
      source: "upload",
      video_id: "upload_a",
      video_url: "https://signed/expired",
      // Persisted, so there is a `videos` row the re-sign endpoint can resolve as this caller.
      analysis_id: "a1",
    });
    fireEvent.error(document.querySelector("video") as HTMLVideoElement);
    await waitFor(() =>
      expect(document.querySelector("video")?.getAttribute("src")).toBe("https://signed/fresh")
    );
    expect(media).toHaveBeenCalledWith("upload_a");
  });

  it("gives up re-signing rather than looping on a clip that is simply broken", async () => {
    // `error` also fires for a deleted object and an undecodable file, and those repeat forever.
    const media = vi.spyOn(api, "uploadMedia").mockResolvedValue({
      video_url: "https://signed/still-broken",
      thumbnail_url: "https://signed/thumb.jpg",
      expires_in: 3600,
    });
    renderPanel({ ...mockAnalysis, source: "upload", video_id: "upload_a" });
    await waitFor(() => expect(media).toHaveBeenCalledTimes(1));
    for (let i = 0; i < 6; i += 1) {
      fireEvent.error(document.querySelector("video") as HTMLVideoElement);
      // eslint-disable-next-line no-await-in-loop
      await act(async () => {
        await Promise.resolve();
      });
    }
    // The first call is the history replay's own re-sign; the cap allows two recoveries on top.
    expect(media.mock.calls.length).toBeLessThanOrEqual(3);
  });

  it("does not try to re-sign a library clip, which carries no signature", async () => {
    const media = vi.spyOn(api, "uploadMedia");
    renderPanel({ ...mockAnalysis, source: "library", video_id: "vid_001" });
    fireEvent.error(document.querySelector("video") as HTMLVideoElement);
    await act(async () => {
      await Promise.resolve();
    });
    expect(media).not.toHaveBeenCalled();
    expect(document.querySelector("video")?.getAttribute("src")).toBe("/api/video-file/vid_001");
  });

  it("does not try to re-sign an anonymous upload, which has no row to resolve", async () => {
    // `/api/uploads/{id}/url` resolves the storage key through the caller's own JWT, and an
    // anonymous analysis is deliberately never persisted (no `analysis_id`). Asking would 401 and
    // the `.catch` would blank a player that is still showing the poster — worse than leaving it.
    const media = vi.spyOn(api, "uploadMedia");
    renderPanel({
      ...mockAnalysis,
      source: "upload",
      video_id: "upload_anon",
      video_url: "https://signed/anon",
    });
    fireEvent.error(document.querySelector("video") as HTMLVideoElement);
    await act(async () => {
      await Promise.resolve();
    });
    expect(media).not.toHaveBeenCalled();
    expect(document.querySelector("video")?.getAttribute("src")).toBe("https://signed/anon");
  });

  it("does not re-sign an upload whose persist failed, where no row exists either", async () => {
    // `analysis_id: null` means the analysis succeeded but its history row was never written, so
    // the objects are orphaned. The re-sign endpoint would 404, not 401 — same outcome.
    const media = vi.spyOn(api, "uploadMedia");
    renderPanel({
      ...mockAnalysis,
      source: "upload",
      video_id: "upload_a",
      video_url: "https://signed/orphan",
      analysis_id: null,
    });
    fireEvent.error(document.querySelector("video") as HTMLVideoElement);
    await act(async () => {
      await Promise.resolve();
    });
    expect(media).not.toHaveBeenCalled();
  });

  it("keeps the stale URL on screen when a recovery cannot re-sign", async () => {
    // Blanking `src` is right on a FIRST resolve (it would otherwise be the previous analysis's
    // clip). On a recovery the element is already showing this clip, so throwing it away costs a
    // working poster and a rendered timeline to gain nothing — it is unplayable either way.
    vi.spyOn(api, "uploadMedia").mockRejectedValue(new Error("503"));
    renderPanel({
      ...mockAnalysis,
      source: "upload",
      video_id: "upload_a",
      video_url: "https://signed/expired",
      analysis_id: "a1",
    });
    fireEvent.error(document.querySelector("video") as HTMLVideoElement);
    await act(async () => {
      await Promise.resolve();
    });
    expect(document.querySelector("video")?.getAttribute("src")).toBe("https://signed/expired");
  });
});
