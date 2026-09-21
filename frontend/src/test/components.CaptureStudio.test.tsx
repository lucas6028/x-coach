import { fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import CaptureStudio from "../components/CaptureStudio";
import { DEFAULT_ANALYSIS_TIER } from "../lib/poseTier";
import { renderWithProviders } from "./renderWithProviders";

// RecordPanel is camera glue — stub it so the studio test stays in jsdom.
vi.mock("../components/RecordPanel", () => ({
  default: ({
    onRecorded,
    onError,
    fullscreen,
    onClose,
  }: {
    onRecorded: (b: Blob) => void;
    onError: (m: string) => void;
    fullscreen?: boolean;
    onClose?: () => void;
  }) => (
    <>
      <span>{fullscreen ? "layout-fullscreen" : "layout-inline"}</span>
      <button onClick={() => onClose?.()}>fake-close</button>
      <button onClick={() => onRecorded(new Blob(["v"], { type: "video/webm" }))}>fake-record</button>
      <button onClick={() => onError("cam fail")}>fake-error</button>
    </>
  ),
}));

// ComplexitySelector persists the chosen tier via saveAnalysisTier -> localStorage; clear it so a
// selection made in one test doesn't leak into the next.
afterEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("CaptureStudio", () => {
  it("defaults to upload mode and can switch to record", () => {
    renderWithProviders(<CaptureStudio onBlob={() => {}} busy={false} progress={0} movement="Squat" />);
    fireEvent.click(screen.getByRole("tab", { name: /record/i }));
    expect(screen.getByText("fake-record")).toBeInTheDocument();
  });

  it("hands a recorded blob + selected tier to onBlob", () => {
    const onBlob = vi.fn();
    renderWithProviders(<CaptureStudio onBlob={onBlob} busy={false} progress={0} movement="Squat" />);
    fireEvent.click(screen.getByRole("tab", { name: /record/i }));
    fireEvent.click(screen.getByText("fake-record"));
    expect(onBlob).toHaveBeenCalledWith(expect.any(Blob), DEFAULT_ANALYSIS_TIER);
  });

  it("resets to upload mode and reports the error when RecordPanel fails", () => {
    const onError = vi.fn();
    renderWithProviders(<CaptureStudio onBlob={() => {}} busy={false} progress={0} onError={onError} movement="Squat" />);
    fireEvent.click(screen.getByRole("tab", { name: /record/i }));
    fireEvent.click(screen.getByText("fake-error"));
    expect(onError).toHaveBeenCalledWith("cam fail");
    expect(screen.queryByText("fake-error")).not.toBeInTheDocument();
  });

  it("hands the selected non-default tier to onBlob", () => {
    const onBlob = vi.fn();
    renderWithProviders(<CaptureStudio onBlob={onBlob} busy={false} progress={0} onError={vi.fn()} movement="Squat" />);
    fireEvent.click(screen.getByRole("button", { name: /precision/i }));
    fireEvent.change(screen.getByRole("slider"), { target: { value: "0" } });
    fireEvent.mouseUp(screen.getByRole("slider"));
    fireEvent.click(screen.getByRole("tab", { name: /record/i }));
    fireEvent.click(screen.getByText("fake-record"));
    expect(onBlob).toHaveBeenCalledWith(expect.any(Blob), "lite");
  });
});

// On the phone the camera takes the whole viewport (the header + bottom nav otherwise push the
// preview off-screen). Driven by `useIsMobile`, so force its media query to match — the global
// setup stub answers `false` to everything, which keeps the describe above on the inline layout.
describe("CaptureStudio — recording on the phone", () => {
  const matchPhone = () =>
    vi.spyOn(window, "matchMedia").mockImplementation(
      (query: string) =>
        ({
          matches: query === "(max-width: 1023px)",
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn(),
        }) as unknown as MediaQueryList
    );

  it("keeps the inline camera on desktop", () => {
    renderWithProviders(
      <CaptureStudio onBlob={() => {}} busy={false} progress={0} movement="Squat" initialMode="record" />
    );
    expect(screen.getByText("layout-inline")).toBeInTheDocument();
  });

  it("opens the camera fullscreen, and closing it returns to the dropzone", () => {
    matchPhone();
    renderWithProviders(
      <CaptureStudio onBlob={() => {}} busy={false} progress={0} movement="Squat" initialMode="record" />
    );
    expect(screen.getByText("layout-fullscreen")).toBeInTheDocument();
    fireEvent.click(screen.getByText("fake-close"));
    expect(screen.queryByText("fake-close")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /upload/i })).toHaveAttribute("aria-selected", "true");
  });
});
