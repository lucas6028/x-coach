import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Header from "../components/Header";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis } from "./fixtures";

describe("Header", () => {
  it("shows the app title when there is no analysis", () => {
    renderWithProviders(<Header analysis={null} loading={false} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Squat Analysis");
  });

  it("shows the session id when an analysis is present", () => {
    renderWithProviders(<Header analysis={mockAnalysis} loading={false} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("vid_001");
  });

  it("shows PROCESSING status while loading", () => {
    renderWithProviders(<Header analysis={null} loading={true} />);
    expect(screen.getByText("PROCESSING")).toBeInTheDocument();
  });

  it("shows ANALYSIS COMPLETE when analysis is present and not loading", () => {
    renderWithProviders(<Header analysis={mockAnalysis} loading={false} />);
    expect(screen.getByText("ANALYSIS COMPLETE")).toBeInTheDocument();
  });

  it("shows AWAITING INPUT when no analysis and not loading", () => {
    renderWithProviders(<Header analysis={null} loading={false} />);
    expect(screen.getByText("AWAITING INPUT")).toBeInTheDocument();
  });

  it("shows the view type from the analysis", () => {
    renderWithProviders(<Header analysis={mockAnalysis} loading={false} />);
    expect(screen.getByText(/side view/i)).toBeInTheDocument();
  });

  it("shows the source label", () => {
    renderWithProviders(<Header analysis={mockAnalysis} loading={false} />);
    expect(screen.getByText("library")).toBeInTheDocument();
  });

  it("calls onMenu when the mobile menu button is clicked", async () => {
    const user = userEvent.setup();
    const onMenu = vi.fn();
    renderWithProviders(<Header analysis={null} loading={false} onMenu={onMenu} />);
    await user.click(screen.getByRole("button", { name: /show navigation/i }));
    expect(onMenu).toHaveBeenCalledOnce();
  });
});
