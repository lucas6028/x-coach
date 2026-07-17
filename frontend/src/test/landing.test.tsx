import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import Landing from "../landing/Landing";
import { renderWithProviders } from "./renderWithProviders";

describe("Landing page", () => {
  it("renders the x-coach brand in the nav", () => {
    renderWithProviders(<Landing />);
    expect(screen.getAllByText(/x-coach/i).length).toBeGreaterThan(0);
  });

  it("renders the hero heading with key phrase", () => {
    renderWithProviders(<Landing />);
    expect(screen.getByText(/trace to the joint/i)).toBeInTheDocument();
  });

  it("renders the hero sub-copy", () => {
    renderWithProviders(<Landing />);
    expect(screen.getByText(/reads a squat video/i)).toBeInTheDocument();
  });

  it("renders the 'Open the demo' CTA", () => {
    renderWithProviders(<Landing />);
    expect(screen.getAllByRole("link", { name: /Open the demo/i }).length).toBeGreaterThan(0);
  });

  it("renders the four pipeline stage titles", () => {
    renderWithProviders(<Landing />);
    expect(screen.getByText("Perceive")).toBeInTheDocument();
    expect(screen.getByText("Retrieve")).toBeInTheDocument();
    expect(screen.getByText("Reason")).toBeInTheDocument();
    expect(screen.getByText("Coach")).toBeInTheDocument();
  });

  it("renders the problem section heading", () => {
    renderWithProviders(<Landing />);
    expect(screen.getByText(/Scores don't coach/i)).toBeInTheDocument();
  });

  it("renders the evaluation section", () => {
    renderWithProviders(<Landing />);
    expect(screen.getByText(/Held to a measurable bar/i)).toBeInTheDocument();
  });

  it("renders the bento 'Four signals' section", () => {
    renderWithProviders(<Landing />);
    expect(screen.getByText(/Four signals/i)).toBeInTheDocument();
  });

  it("renders the diagnosis section", () => {
    renderWithProviders(<Landing />);
    expect(screen.getByText(/Every cue carries its reasoning/i)).toBeInTheDocument();
  });

  it("renders the closing CTA section", () => {
    renderWithProviders(<Landing />);
    expect(screen.getByText(/See it analyze a real squat/i)).toBeInTheDocument();
  });

  it("renders the movement showcase section", () => {
    renderWithProviders(<Landing />);
    expect(screen.getByText(/One pipeline, the whole movement library/i)).toBeInTheDocument();
  });

  it("renders the footer tagline", () => {
    renderWithProviders(<Landing />);
    expect(screen.getByText(/Explainable movement coaching/i)).toBeInTheDocument();
  });

  it("renders nav links for the three sections", () => {
    renderWithProviders(<Landing />);
    expect(screen.getAllByRole("link", { name: /How it works/i }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: /The pipeline/i }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: /Evaluation/i }).length).toBeGreaterThan(0);
  });

  it("renders language switch buttons", () => {
    renderWithProviders(<Landing />);
    // Should have EN and 中 buttons somewhere
    expect(screen.getAllByRole("button", { name: /EN|中/i }).length).toBeGreaterThan(0);
  });
});
