import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import UploadDropzone from "../components/UploadDropzone";
import { renderWithProviders } from "./renderWithProviders";

// UploadDropzone is the idle upload target only — the analysis *waiting* state is owned by Lumen
// (see components.DemoIntro.test.tsx "swaps the dropzone for the Lumen scan loader while loading").
describe("UploadDropzone", () => {
  it("shows the upload prompt", () => {
    renderWithProviders(<UploadDropzone onFile={vi.fn()} />);
    expect(screen.getByText("Drop a squat video or tap to upload")).toBeInTheDocument();
  });

  it("shows the hint text", () => {
    renderWithProviders(<UploadDropzone onFile={vi.fn()} />);
    expect(screen.getByText(/MP4 \/ MOV/)).toBeInTheDocument();
  });

  it("calls onFile when a file is dropped", () => {
    const onFile = vi.fn();
    renderWithProviders(<UploadDropzone onFile={onFile} />);
    const dropzone = screen.getByText("Drop a squat video or tap to upload").closest("div")!;
    const file = new File(["video"], "squat.mp4", { type: "video/mp4" });
    fireEvent.drop(dropzone, { dataTransfer: { files: [file] } });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("calls onFile when a file is selected via input", () => {
    const onFile = vi.fn();
    renderWithProviders(<UploadDropzone onFile={onFile} />);
    const input = document.querySelector("input[type=file]") as HTMLInputElement;
    const file = new File(["video"], "squat.mp4", { type: "video/mp4" });
    fireEvent.change(input, { target: { files: [file] } });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("does not call onFile when drop has no files", () => {
    const onFile = vi.fn();
    renderWithProviders(<UploadDropzone onFile={onFile} />);
    const dropzone = screen.getByText("Drop a squat video or tap to upload").closest("div")!;
    fireEvent.drop(dropzone, { dataTransfer: { files: [] } });
    expect(onFile).not.toHaveBeenCalled();
  });
});
