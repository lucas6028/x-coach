import { StrictMode } from "react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { act, render, screen, fireEvent, waitFor } from "@testing-library/react";
import { I18nProvider } from "../lib/i18n";
import { mockAnalysis } from "./fixtures";
import type { Analysis, NlfJob } from "../api";

// A promise this test controls the settling of, for manufacturing races between two in-flight
// calls (used by the requestJob-invalidates-a-stale-fetch test below).
function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

// Isolate Nlf3dPanel's own state machine (request/poll/refresh/image-error) from the viewer's
// rendering — TurntableViewer has its own test file. The stub exposes just enough to drive the
// panel: how many key frames it was handed, and a button that fires the onImageError callback the
// panel passes down.
vi.mock("../components/nlf/TurntableViewer", () => ({
  default: ({
    keyFrames,
    onImageError,
  }: {
    keyFrames: unknown[];
    onImageError?: () => void;
  }) => (
    <div data-testid="turntable-viewer-stub">
      <span>{keyFrames.length} key frames</span>
      <button type="button" onClick={() => onImageError?.()}>
        trigger image error
      </button>
    </div>
  ),
}));

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
const mockUseAuth = vi.mocked(useAuth);

vi.mock("../api", async (importActual) => {
  const actual = await importActual<typeof import("../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getNlf: vi.fn(),
      requestNlf: vi.fn(),
    },
  };
});
import { api } from "../api";
const mockGetNlf = vi.mocked(api.getNlf);
const mockRequestNlf = vi.mocked(api.requestNlf);

import Nlf3dPanel from "../components/nlf/Nlf3dPanel";

const ANALYSIS_WITH_ID: Analysis = { ...mockAnalysis, analysis_id: "a1" };
const ANALYSIS_NO_ID: Analysis = { ...mockAnalysis, analysis_id: null };

function job(over: Partial<NlfJob>): NlfJob {
  return {
    status: "queued",
    worker_online: true,
    error: null,
    created_at: "2026-09-28T00:00:00Z",
    updated_at: "2026-09-28T00:00:00Z",
    key_frames: [],
    angles: 24,
    tile_px: 512,
    expires_in: 3600,
    ...over,
  };
}

function renderPanel(analysis: Analysis = ANALYSIS_WITH_ID, onSeek = vi.fn()) {
  const view = render(
    <I18nProvider>
      <Nlf3dPanel analysis={analysis} onSeek={onSeek} />
    </I18nProvider>
  );
  return { onSeek, unmount: view.unmount };
}

// Flushes a fake-timers render's initial-mount fetch: no macrotask/timer is involved yet (just the
// mocked promise's own microtasks), but it still needs `act` so React commits the resulting state
// update AND runs the passive effect that schedules the poll's own setTimeout before anything
// else advances the clock.
async function flushInitialFetch() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

function setVisibility(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", { configurable: true, value: state });
  document.dispatchEvent(new Event("visibilitychange"));
}

beforeEach(() => {
  mockGetNlf.mockReset();
  mockRequestNlf.mockReset();
  mockUseAuth.mockReturnValue({ canUse3d: true } as unknown as ReturnType<typeof useAuth>);
  setVisibility("visible");
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
  Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
});

describe("Nlf3dPanel — visibility gate", () => {
  it("renders nothing when the user is not NLF-enabled", () => {
    mockUseAuth.mockReturnValue({ canUse3d: false } as unknown as ReturnType<typeof useAuth>);
    const { container } = render(
      <I18nProvider>
        <Nlf3dPanel analysis={ANALYSIS_WITH_ID} />
      </I18nProvider>
    );
    expect(container).toBeEmptyDOMElement();
    expect(mockGetNlf).not.toHaveBeenCalled();
  });

  it("renders nothing when the analysis has no analysis_id", () => {
    const { container } = render(
      <I18nProvider>
        <Nlf3dPanel analysis={ANALYSIS_NO_ID} />
      </I18nProvider>
    );
    expect(container).toBeEmptyDOMElement();
    expect(mockGetNlf).not.toHaveBeenCalled();
  });
});

describe("Nlf3dPanel — no job yet", () => {
  it("shows a Get 3D view button that requests a job", async () => {
    mockGetNlf.mockResolvedValue(null);
    mockRequestNlf.mockResolvedValue(job({ status: "queued", worker_online: true }));
    renderPanel();

    const btn = await screen.findByRole("button", { name: /get 3d view/i });
    expect(screen.getByText(/research team's processing pc/i)).toBeInTheDocument();
    fireEvent.click(btn);

    await waitFor(() => expect(mockRequestNlf).toHaveBeenCalledWith("a1"));
    expect(await screen.findByText(/waiting for the processing pc/i)).toBeInTheDocument();
  });
});

describe("Nlf3dPanel — queued, worker offline", () => {
  it("shows the offline text and does not poll", async () => {
    // Fake timers installed BEFORE the first render: if the component ever scheduled a stray
    // timer here (it shouldn't — queued+offline isn't pollable), it would be a fake one that
    // advancing the clock below would actually catch, rather than a real one ticking unnoticed
    // in the background.
    vi.useFakeTimers();
    mockGetNlf.mockResolvedValue(job({ status: "queued", worker_online: false }));
    renderPanel();
    await flushInitialFetch();

    expect(screen.getByText(/waiting for the processing pc/i)).toBeInTheDocument();
    expect(screen.getByText(/processing pc is offline/i)).toBeInTheDocument();
    expect(mockGetNlf).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(mockGetNlf).toHaveBeenCalledTimes(1);
  });

  it("shows a Refresh button that fetches once when clicked", async () => {
    mockGetNlf.mockResolvedValue(job({ status: "queued", worker_online: false }));
    renderPanel();

    const btn = await screen.findByRole("button", { name: /refresh/i });
    expect(mockGetNlf).toHaveBeenCalledTimes(1);

    fireEvent.click(btn);
    await waitFor(() => expect(mockGetNlf).toHaveBeenCalledTimes(2));
  });
});

describe("Nlf3dPanel — claimed", () => {
  it("polls every 15s", async () => {
    // Fake timers from the start here: the poll's own setTimeout must be a FAKE one for
    // advanceTimersByTimeAsync to affect it — scheduling it under real timers (e.g. by waiting on
    // the initial fetch with findByText/waitFor first, then switching to fake timers) would leave
    // it pending against the real clock, unaffected by any later fake-time advance.
    vi.useFakeTimers();
    // A fresh object per call: React bails out of a state update (and the effects that depend on
    // it) when `setJob` is handed a reference-identical value, which a shared `mockResolvedValue`
    // object would be on every poll. (This test covers the "fresh object" path; the identical-
    // reference path is covered separately below — the fix must not depend on the objects differing.)
    mockGetNlf.mockImplementation(async () => job({ status: "claimed" }));
    renderPanel();
    await flushInitialFetch();
    expect(mockGetNlf).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/processing/i)).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    expect(mockGetNlf).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    expect(mockGetNlf).toHaveBeenCalledTimes(3);
  });

  // Regression test for the bug the coordinator flagged: scheduling used to be keyed on the `job`
  // state object's identity, so a poll that resolved to a reference-EQUAL value (same object,
  // every time, as a real `mockResolvedValue` behaves) never re-ran the scheduling effect and
  // silently stopped the poll loop after one tick.
  it("keeps polling even when every fetch resolves the SAME object reference", async () => {
    vi.useFakeTimers();
    const sameJob = job({ status: "claimed" });
    mockGetNlf.mockResolvedValue(sameJob);
    renderPanel();
    await flushInitialFetch();
    expect(mockGetNlf).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    expect(mockGetNlf).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    expect(mockGetNlf).toHaveBeenCalledTimes(3);
  });

  it("keeps polling (at a backed-off cadence) after a failed poll, and shows the error + Refresh", async () => {
    vi.useFakeTimers();
    let call = 0;
    mockGetNlf.mockImplementation(async () => {
      call += 1;
      if (call === 2) throw new Error("network blip");
      return job({ status: "claimed" });
    });
    renderPanel();
    await flushInitialFetch();
    expect(mockGetNlf).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/processing/i)).toBeInTheDocument();

    // The 2nd fetch (the first poll tick) fails.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    expect(mockGetNlf).toHaveBeenCalledTimes(2);
    expect(screen.getByText(/network blip/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /refresh/i })).toBeInTheDocument();
    // Still "claimed" underneath (the failure didn't change the job) — polling must continue.
    expect(screen.getByText(/processing/i)).toBeInTheDocument();

    // No poll yet at the ORIGINAL 15s cadence — the failure backs off to 60s.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    expect(mockGetNlf).toHaveBeenCalledTimes(2);

    // The backoff fires by 60s total after the failure and succeeds, clearing the error.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(45_000);
    });
    expect(mockGetNlf).toHaveBeenCalledTimes(3);
    expect(screen.queryByText(/network blip/i)).not.toBeInTheDocument();
  });
});

describe("Nlf3dPanel — hidden document", () => {
  it("stops polling and shows a Refresh button instead", async () => {
    // Fake timers before the first render, same reasoning as the queued+offline test above: a
    // stray real timer sitting in the background would make this test pass for the wrong reason.
    vi.useFakeTimers();
    mockGetNlf.mockImplementation(async () => job({ status: "claimed" }));
    renderPanel();
    await flushInitialFetch();
    expect(screen.getByText(/processing/i)).toBeInTheDocument();
    expect(mockGetNlf).toHaveBeenCalledTimes(1);

    act(() => setVisibility("hidden"));
    expect(screen.getByRole("button", { name: /refresh/i })).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    // Only the initial fetch — no poll fired while hidden, and no re-fetch-on-visible either
    // (that only fires on the hidden -> visible transition, not this one).
    expect(mockGetNlf).toHaveBeenCalledTimes(1);
  });

  it("re-fetches once on the hidden -> visible transition", async () => {
    mockGetNlf.mockResolvedValue(job({ status: "claimed" }));
    renderPanel();
    await vi.waitFor(() => expect(mockGetNlf).toHaveBeenCalledTimes(1));

    setVisibility("hidden");
    setVisibility("visible");
    await vi.waitFor(() => expect(mockGetNlf).toHaveBeenCalledTimes(2));
  });

  it("skips the visibility re-fetch for a job that already finished (done)", async () => {
    mockGetNlf.mockResolvedValue(job({ status: "done", key_frames: [] }));
    renderPanel();
    await screen.findByTestId("turntable-viewer-stub");
    expect(mockGetNlf).toHaveBeenCalledTimes(1);

    setVisibility("hidden");
    setVisibility("visible");
    // Give any (wrongly scheduled) re-fetch a chance to land before asserting it didn't happen.
    await new Promise((r) => setTimeout(r, 0));
    expect(mockGetNlf).toHaveBeenCalledTimes(1);
  });

  it("skips the visibility re-fetch for a job that already finished (failed)", async () => {
    mockGetNlf.mockResolvedValue(job({ status: "failed", error: "boom" }));
    renderPanel();
    await screen.findByText("boom");
    expect(mockGetNlf).toHaveBeenCalledTimes(1);

    setVisibility("hidden");
    setVisibility("visible");
    await new Promise((r) => setTimeout(r, 0));
    expect(mockGetNlf).toHaveBeenCalledTimes(1);
  });
});

describe("Nlf3dPanel — unmount", () => {
  it("clears the pending poll timer, so nothing fetches after unmount", async () => {
    vi.useFakeTimers();
    mockGetNlf.mockImplementation(async () => job({ status: "claimed" }));
    const { unmount } = renderPanel();
    await flushInitialFetch();
    expect(mockGetNlf).toHaveBeenCalledTimes(1);

    unmount();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(mockGetNlf).toHaveBeenCalledTimes(1);
  });
});

describe("Nlf3dPanel — done", () => {
  it("renders the turntable viewer", async () => {
    mockGetNlf.mockResolvedValue(
      job({
        status: "done",
        key_frames: [
          {
            t_s: 1,
            kind: "rep_bottom",
            fault_ids: [],
            rep_index: 1,
            strip_url: "https://example.com/s.webp",
          },
        ],
      })
    );
    renderPanel();
    expect(await screen.findByTestId("turntable-viewer-stub")).toBeInTheDocument();
    expect(screen.getByText("1 key frames")).toBeInTheDocument();
  });

  it("re-fetches at most twice on repeated image errors", async () => {
    mockGetNlf.mockResolvedValue(
      job({
        status: "done",
        key_frames: [
          {
            t_s: 1,
            kind: "rep_bottom",
            fault_ids: [],
            rep_index: 1,
            strip_url: "https://example.com/s.webp",
          },
        ],
      })
    );
    renderPanel();
    await screen.findByTestId("turntable-viewer-stub");
    expect(mockGetNlf).toHaveBeenCalledTimes(1);

    const btn = screen.getByRole("button", { name: /trigger image error/i });
    fireEvent.click(btn);
    await waitFor(() => expect(mockGetNlf).toHaveBeenCalledTimes(2));
    fireEvent.click(btn);
    await waitFor(() => expect(mockGetNlf).toHaveBeenCalledTimes(3));
    fireEvent.click(btn);
    // Third error: the cap (2 re-fetches per mount) is already spent, so no fourth call.
    await new Promise((r) => setTimeout(r, 0));
    expect(mockGetNlf).toHaveBeenCalledTimes(3);
  });
});

describe("Nlf3dPanel — failed", () => {
  it("shows the error text and re-requests on Try again", async () => {
    mockGetNlf.mockResolvedValue(job({ status: "failed", error: "worker crashed" }));
    mockRequestNlf.mockResolvedValue(job({ status: "queued", worker_online: true }));
    renderPanel();

    expect(await screen.findByText("worker crashed")).toBeInTheDocument();
    const btn = screen.getByRole("button", { name: /try again/i });
    fireEvent.click(btn);

    await waitFor(() => expect(mockRequestNlf).toHaveBeenCalledWith("a1"));
    expect(await screen.findByText(/waiting for the processing pc/i)).toBeInTheDocument();
  });

  it("shows the error when requestNlf itself rejects (Try again)", async () => {
    mockGetNlf.mockResolvedValue(job({ status: "failed", error: "worker crashed" }));
    mockRequestNlf.mockRejectedValue(new Error("quota exceeded"));
    renderPanel();

    const btn = await screen.findByRole("button", { name: /try again/i });
    fireEvent.click(btn);

    expect(await screen.findByText(/quota exceeded/i)).toBeInTheDocument();
    // The button must be re-enabled (not stuck disabled) so the person can try a third time.
    expect(screen.getByRole("button", { name: /try again/i })).not.toBeDisabled();
  });
});

describe("Nlf3dPanel — StrictMode", () => {
  // React 18 StrictMode (dev only) mounts every component, runs its effect cleanups, then
  // remounts it, to catch effects that aren't safe to run twice. Nlf3dPanel's cleanup used to
  // flip `mountedRef.current` to false and never back to true, so the REAL (second) mount's own
  // `if (!mountedRef.current) return;` guards silently discarded the fetch response forever —
  // the panel would be stuck on "Loading…" in `yarn dev`, though never in production or in a
  // plain (non-StrictMode) test render.
  it("still fetches and displays the job after StrictMode's mount/cleanup/remount cycle", async () => {
    mockGetNlf.mockResolvedValue(job({ status: "claimed" }));
    render(
      <StrictMode>
        <I18nProvider>
          <Nlf3dPanel analysis={ANALYSIS_WITH_ID} />
        </I18nProvider>
      </StrictMode>
    );
    expect(await screen.findByText(/processing/i)).toBeInTheDocument();
  });
});

describe("Nlf3dPanel — requestJob invalidates stale fetches", () => {
  it("a stale in-flight fetchJob response cannot overwrite a just-completed request", async () => {
    const staleFetch = deferred<NlfJob>();
    let getNlfCalls = 0;
    mockGetNlf.mockImplementation(() => {
      getNlfCalls += 1;
      // Call 1: the initial mount fetch, resolved immediately to "no job yet".
      if (getNlfCalls === 1) return Promise.resolve(null);
      // Call 2 (triggered below, while `job` is still null): left pending, to be resolved AFTER
      // requestJob has already completed — it must be ignored when it finally does resolve.
      return staleFetch.promise;
    });
    mockRequestNlf.mockResolvedValue(job({ status: "queued", worker_online: false }));

    renderPanel();
    const btn = await screen.findByRole("button", { name: /get 3d view/i });

    // Trigger a second fetchJob (call 2, the one left pending above) via a visibility round trip
    // — allowed while `job` is still null (non-terminal), same as any other "no job yet" state.
    setVisibility("hidden");
    setVisibility("visible");
    await waitFor(() => expect(getNlfCalls).toBe(2));

    // Request a fresh job WHILE that stale fetch is still pending.
    fireEvent.click(btn);
    await waitFor(() => expect(mockRequestNlf).toHaveBeenCalledWith("a1"));
    expect(await screen.findByText(/processing pc is offline/i)).toBeInTheDocument();

    // Now let the stale fetch resolve with a DIFFERENT, contradicting payload (worker online).
    // If requestJob hadn't invalidated it, this would flip the displayed state.
    await act(async () => {
      staleFetch.resolve(job({ status: "queued", worker_online: true }));
    });
    // Still the fresh (offline) job, not the stale (online) payload that just resolved.
    expect(screen.getByText(/processing pc is offline/i)).toBeInTheDocument();
  });
});

describe("Nlf3dPanel — initial fetch fails", () => {
  it("shows the error with a Retry button that re-fetches once (no auto backoff)", async () => {
    mockGetNlf.mockRejectedValueOnce(new Error("network down"));
    mockGetNlf.mockResolvedValueOnce(job({ status: "queued", worker_online: true }));
    renderPanel();

    expect(await screen.findByText(/network down/i)).toBeInTheDocument();
    // No auto-retry: still exactly one call a tick later, until the button is pressed.
    await new Promise((r) => setTimeout(r, 0));
    expect(mockGetNlf).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: /refresh/i }));
    await waitFor(() => expect(mockGetNlf).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/waiting for the processing pc/i)).toBeInTheDocument();
  });
});
