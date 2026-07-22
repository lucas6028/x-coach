import { fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import CaptureStudio from "../components/CaptureStudio";
import { DEFAULT_ANALYSIS_TIER } from "../lib/poseTier";
import { renderWithProviders } from "./renderWithProviders";

// RecordPanel is camera glue — stub it so the studio test stays in jsdom.
vi.mock("../components/RecordPanel", () => ({
  default: ({ onRecorded, onError }: { onRecorded: (b: Blob) => void; onError: (m: string) => void }) => (
    <>
      <button onClick={() => onRecorded(new Blob(["v"], { type: "video/webm" }))}>fake-record</button>
      <button onClick={() => onError("cam fail")}>fake-error</button>
    </>
  ),
}));

// ComplexitySelector persists the chosen tier via saveAnalysisTier -> localStorage; clear it so a
// selection made in one test doesn't leak into the next.
afterEach(() => {
  localStorage.clear();
});

describe("CaptureStudio", () => {
  it("defaults to upload mode and can switch to record", () => {
    renderWithProviders(<CaptureStudio onBlob={() => {}} busy={false} progress={0} />);
    fireEvent.click(screen.getByRole("tab", { name: /record/i }));
    expect(screen.getByText("fake-record")).toBeInTheDocument();
  });

  it("hands a recorded blob + selected tier to onBlob", () => {
    const onBlob = vi.fn();
    renderWithProviders(<CaptureStudio onBlob={onBlob} busy={false} progress={0} />);
    fireEvent.click(screen.getByRole("tab", { name: /record/i }));
    fireEvent.click(screen.getByText("fake-record"));
    expect(onBlob).toHaveBeenCalledWith(expect.any(Blob), DEFAULT_ANALYSIS_TIER);
  });

  it("resets to upload mode and reports the error when RecordPanel fails", () => {
    const onError = vi.fn();
    renderWithProviders(<CaptureStudio onBlob={() => {}} busy={false} progress={0} onError={onError} />);
    fireEvent.click(screen.getByRole("tab", { name: /record/i }));
    fireEvent.click(screen.getByText("fake-error"));
    expect(onError).toHaveBeenCalledWith("cam fail");
    expect(screen.queryByText("fake-error")).not.toBeInTheDocument();
  });

  it("hands the selected non-default tier to onBlob", () => {
    const onBlob = vi.fn();
    renderWithProviders(<CaptureStudio onBlob={onBlob} busy={false} progress={0} onError={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /precision/i }));
    fireEvent.change(screen.getByRole("slider"), { target: { value: "0" } });
    fireEvent.mouseUp(screen.getByRole("slider"));
    fireEvent.click(screen.getByRole("tab", { name: /record/i }));
    fireEvent.click(screen.getByText("fake-record"));
    expect(onBlob).toHaveBeenCalledWith(expect.any(Blob), "lite");
  });
});
