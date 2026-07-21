import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DemoIntro from "../components/DemoIntro";
import { renderWithProviders } from "./renderWithProviders";

describe("DemoIntro", () => {
  it("shows the heading", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" />
    );
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Analyze a squat in about 20 seconds."
    );
  });

  it("shows the sub-heading description", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" />
    );
    expect(screen.getByText(/Upload a clip or open a labeled sample/)).toBeInTheDocument();
  });

  it("shows the 'Open a sample clip' button", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" />
    );
    expect(screen.getByRole("button", { name: /Open a sample clip/i })).toBeInTheDocument();
  });

  it("calls onOpenLibrary when the sample button is clicked", async () => {
    const user = userEvent.setup();
    const onOpenLibrary = vi.fn();
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} onOpenLibrary={onOpenLibrary} loading={false} statusMsg="" error="" />
    );
    await user.click(screen.getByRole("button", { name: /Open a sample clip/i }));
    expect(onOpenLibrary).toHaveBeenCalledOnce();
  });

  it("disables the sample button while loading", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} onOpenLibrary={vi.fn()} loading={true} statusMsg="Processing…" error="" />
    );
    expect(screen.getByRole("button", { name: /Open a sample clip/i })).toBeDisabled();
  });

  it("swaps the dropzone for the Lumen scan loader while loading", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} onOpenLibrary={vi.fn()} loading={true} statusMsg="Extracting pose…" error="" />
    );
    // The Lumen waiting state takes over the upload target, carrying the status message…
    expect(screen.getByRole("status")).toHaveAttribute("aria-label", "Extracting pose…");
    // …and the idle dropzone prompt is gone.
    expect(screen.queryByText(/Drop a squat video/i)).not.toBeInTheDocument();
  });

  it("shows the error panel when error is non-empty", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="Video too short" />
    );
    expect(screen.getByText("That clip did not go through")).toBeInTheDocument();
    expect(screen.getByText("Video too short")).toBeInTheDocument();
  });

  it("does not show the error panel when error is empty", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" />
    );
    expect(screen.queryByText("That clip did not go through")).not.toBeInTheDocument();
  });

  it("renders the three 'what comes back' step cards", () => {
    renderWithProviders(
      <DemoIntro onBlob={vi.fn()} onError={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" />
    );
    expect(screen.getByText("Skeleton and faults")).toBeInTheDocument();
    expect(screen.getByText("Grounded feedback")).toBeInTheDocument();
    expect(screen.getByText("Knowledge graph")).toBeInTheDocument();
  });
});
