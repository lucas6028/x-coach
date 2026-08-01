import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import MetricsCards from "../components/MetricsCards";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis, mockUnmeasuredAnalysis } from "./fixtures";

describe("MetricsCards", () => {
  it("renders all four metric labels", () => {
    renderWithProviders(<MetricsCards analysis={mockAnalysis} />);
    expect(screen.getByText("Camera View")).toBeInTheDocument();
    expect(screen.getByText("Faults")).toBeInTheDocument();
    expect(screen.getByText("Lower-body Vis.")).toBeInTheDocument();
    expect(screen.getByText("Valid Frames")).toBeInTheDocument();
  });

  it("shows the translated view type", () => {
    renderWithProviders(<MetricsCards analysis={mockAnalysis} />);
    expect(screen.getByText("Side")).toBeInTheDocument();
  });

  it("shows the view confidence", () => {
    renderWithProviders(<MetricsCards analysis={mockAnalysis} />);
    expect(screen.getByText(/conf 0\.95/)).toBeInTheDocument();
  });

  it("shows the fault count", () => {
    renderWithProviders(<MetricsCards analysis={mockAnalysis} />);
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("shows 'clean rep' when there are no faults", () => {
    renderWithProviders(<MetricsCards analysis={mockCleanAnalysis} />);
    expect(screen.getByText("clean rep")).toBeInTheDocument();
  });

  it("keeps the clean-rep card green when the clip was actually measured", () => {
    renderWithProviders(<MetricsCards analysis={mockCleanAnalysis} />);
    // `text-secondary` is the "good" tone; the assertion is here so the unmeasured case below
    // is shown to differ in TONE and not only in wording.
    expect(screen.getByText("0")).toHaveClass("text-secondary");
  });

  // An empty `detections` list means BOTH "no faults found" and "never measured" — the detector
  // returns the same empty list either way. Before this gate, a clip whose pose extraction
  // produced zero valid frames rendered a green "Faults 0 / clean rep" card right beside
  // "Valid Frames 0%": a clean bill of health for a clip nothing was measured on.
  it("says 'not measured' instead of 'clean rep' when no frame was valid", () => {
    renderWithProviders(<MetricsCards analysis={mockUnmeasuredAnalysis} />);
    expect(screen.getByText("not measured")).toBeInTheDocument();
    expect(screen.queryByText("clean rep")).not.toBeInTheDocument();
    expect(screen.getByText("0%")).toBeInTheDocument();
  });

  it("drops the good tone when no frame was valid", () => {
    renderWithProviders(<MetricsCards analysis={mockUnmeasuredAnalysis} />);
    const faultCount = screen.getByText("0");
    expect(faultCount).not.toHaveClass("text-secondary");
    expect(faultCount).toHaveClass("text-white");
  });

  it("shows peak severity when faults exist", () => {
    renderWithProviders(<MetricsCards analysis={mockAnalysis} />);
    expect(screen.getByText(/peak severity 0\.80/)).toBeInTheDocument();
  });

  it("shows lower-body visibility as a percentage", () => {
    renderWithProviders(<MetricsCards analysis={mockAnalysis} />);
    expect(screen.getByText("88%")).toBeInTheDocument();
  });

  it("shows valid frame ratio as a percentage", () => {
    renderWithProviders(<MetricsCards analysis={mockAnalysis} />);
    expect(screen.getByText("92%")).toBeInTheDocument();
  });

  it("shows frames ratio sub-label", () => {
    renderWithProviders(<MetricsCards analysis={mockAnalysis} />);
    expect(screen.getByText("276/300 frames")).toBeInTheDocument();
  });
});
