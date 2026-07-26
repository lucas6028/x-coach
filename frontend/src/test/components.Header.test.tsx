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

  // Finding 1 of the 2026-07-25 review: a user who picked Push-up landed on a header reading
  // "Squat Analysis" above a selector showing "Push-up". The pre-result title must track the
  // studio's actual selection via the `movement` prop, not a hardcoded literal.
  it("names the selected movement in the pre-result title, not a hardcoded squat", () => {
    renderWithProviders(<Header analysis={null} loading={false} movement="Push-up" />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Push-up Analysis");
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

  // Finding 1 of the 2026-07-25 review: on the fault-detected result path (the common case), no
  // UI surface named the movement whose rules produced the verdict — DemoIntro (the only surface
  // that did) unmounts once a result loads. The post-result status line must carry it, beside
  // view/source, and it must default to "Squat" for analyses predating the `movement` field.
  it("names the movement whose rules ran, beside view/source", () => {
    renderWithProviders(<Header analysis={mockAnalysis} loading={false} />);
    expect(screen.getByText("Squat")).toBeInTheDocument();
  });

  it("names a non-squat movement when the analysis carries one", () => {
    renderWithProviders(<Header analysis={{ ...mockAnalysis, movement: "Push-up" }} loading={false} />);
    expect(screen.getByText("Push-up")).toBeInTheDocument();
  });

  it("calls onMenu when the mobile menu button is clicked", async () => {
    const user = userEvent.setup();
    const onMenu = vi.fn();
    renderWithProviders(<Header analysis={null} loading={false} onMenu={onMenu} />);
    await user.click(screen.getByRole("button", { name: /show navigation/i }));
    expect(onMenu).toHaveBeenCalledOnce();
  });
});
