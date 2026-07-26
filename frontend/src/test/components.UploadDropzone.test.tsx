import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import UploadDropzone from "../components/UploadDropzone";
import { renderWithProviders } from "./renderWithProviders";

// UploadDropzone is the idle upload target only — the analysis *waiting* state is owned by Lumen
// (see components.DemoIntro.test.tsx "swaps the dropzone for the Lumen scan loader while loading").
describe("UploadDropzone", () => {
  it("shows the upload prompt", () => {
    renderWithProviders(<UploadDropzone onFile={vi.fn()} movement="Squat" />);
    expect(screen.getByText("Drop a Squat video or tap to upload")).toBeInTheDocument();
  });

  // The prompt sits directly beneath the movement selector — Finding 2 of the 2026-07-25 review
  // was that it stayed hardcoded to "squat" no matter what the user picked, telling a Push-up
  // user to upload the wrong thing. This pins that it now tracks the `movement` prop.
  it("names the selected movement in the prompt, not a hardcoded squat", () => {
    renderWithProviders(<UploadDropzone onFile={vi.fn()} movement="Push-up" />);
    expect(screen.getByText("Drop a Push-up video or tap to upload")).toBeInTheDocument();
    expect(screen.queryByText(/squat/i)).not.toBeInTheDocument();
  });

  it("shows the hint text", () => {
    renderWithProviders(<UploadDropzone onFile={vi.fn()} movement="Squat" />);
    expect(screen.getByText(/MP4 \/ MOV/)).toBeInTheDocument();
  });

  it("calls onFile when a file is dropped", () => {
    const onFile = vi.fn();
    renderWithProviders(<UploadDropzone onFile={onFile} movement="Squat" />);
    const dropzone = screen.getByText("Drop a Squat video or tap to upload").closest("div")!;
    const file = new File(["video"], "squat.mp4", { type: "video/mp4" });
    fireEvent.drop(dropzone, { dataTransfer: { files: [file] } });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("calls onFile when a file is selected via input", () => {
    const onFile = vi.fn();
    renderWithProviders(<UploadDropzone onFile={onFile} movement="Squat" />);
    const input = document.querySelector("input[type=file]") as HTMLInputElement;
    const file = new File(["video"], "squat.mp4", { type: "video/mp4" });
    fireEvent.change(input, { target: { files: [file] } });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("does not call onFile when drop has no files", () => {
    const onFile = vi.fn();
    renderWithProviders(<UploadDropzone onFile={onFile} movement="Squat" />);
    const dropzone = screen.getByText("Drop a Squat video or tap to upload").closest("div")!;
    fireEvent.drop(dropzone, { dataTransfer: { files: [] } });
    expect(onFile).not.toHaveBeenCalled();
  });
});
