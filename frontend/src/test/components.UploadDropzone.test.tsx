import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import UploadDropzone from "../components/UploadDropzone";
import { renderWithProviders } from "./renderWithProviders";

describe("UploadDropzone", () => {
  it("shows the upload prompt when idle", () => {
    renderWithProviders(<UploadDropzone onFile={vi.fn()} loading={false} statusMsg="" />);
    expect(screen.getByText("Drop a squat video or tap to upload")).toBeInTheDocument();
  });

  it("shows the hint text when idle", () => {
    renderWithProviders(<UploadDropzone onFile={vi.fn()} loading={false} statusMsg="" />);
    expect(screen.getByText(/MP4 \/ MOV/)).toBeInTheDocument();
  });

  it("shows 'Analysing…' when loading", () => {
    renderWithProviders(<UploadDropzone onFile={vi.fn()} loading={true} statusMsg="Processing..." />);
    expect(screen.getByText("Analysing…")).toBeInTheDocument();
  });

  it("shows the statusMsg when loading", () => {
    renderWithProviders(<UploadDropzone onFile={vi.fn()} loading={true} statusMsg="Please wait" />);
    expect(screen.getByText("Please wait")).toBeInTheDocument();
  });

  it("calls onFile when a file is dropped", () => {
    const onFile = vi.fn();
    renderWithProviders(<UploadDropzone onFile={onFile} loading={false} statusMsg="" />);
    const dropzone = screen.getByText("Drop a squat video or tap to upload").closest("div")!;
    const file = new File(["video"], "squat.mp4", { type: "video/mp4" });
    fireEvent.drop(dropzone, { dataTransfer: { files: [file] } });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("calls onFile when a file is selected via input", () => {
    const onFile = vi.fn();
    renderWithProviders(<UploadDropzone onFile={onFile} loading={false} statusMsg="" />);
    const input = document.querySelector("input[type=file]") as HTMLInputElement;
    const file = new File(["video"], "squat.mp4", { type: "video/mp4" });
    fireEvent.change(input, { target: { files: [file] } });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("does not call onFile when drop has no files", () => {
    const onFile = vi.fn();
    renderWithProviders(<UploadDropzone onFile={onFile} loading={false} statusMsg="" />);
    const dropzone = screen.getByText("Drop a squat video or tap to upload").closest("div")!;
    fireEvent.drop(dropzone, { dataTransfer: { files: [] } });
    expect(onFile).not.toHaveBeenCalled();
  });
});
