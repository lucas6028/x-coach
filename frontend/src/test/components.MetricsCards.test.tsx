import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import MetricsCards from "../components/MetricsCards";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis } from "./fixtures";

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
