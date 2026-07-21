import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import CaptureStudio from "../components/CaptureStudio";
import { DEFAULT_ANALYSIS_TIER } from "../lib/poseTier";
import { renderWithProviders } from "./renderWithProviders";

// RecordPanel is camera glue — stub it so the studio test stays in jsdom.
vi.mock("../components/RecordPanel", () => ({
  default: ({ onRecorded }: { onRecorded: (b: Blob) => void }) => (
    <button onClick={() => onRecorded(new Blob(["v"], { type: "video/webm" }))}>fake-record</button>
  ),
}));

describe("CaptureStudio", () => {
  it("defaults to upload mode and can switch to record", () => {
    renderWithProviders(<CaptureStudio onBlob={() => {}} busy={false} progress={0} />);
    fireEvent.click(screen.getByRole("tab", { name: /錄影/ }));
    expect(screen.getByText("fake-record")).toBeInTheDocument();
  });

  it("hands a recorded blob + selected tier to onBlob", () => {
    const onBlob = vi.fn();
    renderWithProviders(<CaptureStudio onBlob={onBlob} busy={false} progress={0} />);
    fireEvent.click(screen.getByRole("tab", { name: /錄影/ }));
    fireEvent.click(screen.getByText("fake-record"));
    expect(onBlob).toHaveBeenCalledWith(expect.any(Blob), DEFAULT_ANALYSIS_TIER);
  });
});
