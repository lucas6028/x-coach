import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ComplexitySelector from "../components/ComplexitySelector";
import { renderWithProviders } from "./renderWithProviders";

const openPanel = (current: RegExp) => fireEvent.click(screen.getByRole("button", { name: current }));

describe("ComplexitySelector", () => {
  it("keeps the panel closed until the trigger is clicked", () => {
    renderWithProviders(<ComplexitySelector value="lite" onChange={() => {}} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    openPanel(/lite/i);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("puts the tiers on one ordered axis with the current one marked", () => {
    renderWithProviders(<ComplexitySelector value="full" onChange={() => {}} />);
    openPanel(/full/i);
    const slider = screen.getByRole("slider") as HTMLInputElement;
    // Lite=0 … Heavy=2: the tiers are a continuum, not parallel options.
    expect(slider.value).toBe("1");
    expect(slider).toHaveAttribute("aria-valuetext", "Full");
    expect(slider).toHaveAttribute("max", "2");
  });

  it("reports the tier the slider is dragged to", () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexitySelector value="lite" onChange={onChange} />);
    openPanel(/lite/i);
    fireEvent.change(screen.getByRole("slider"), { target: { value: "2" } });
    expect(onChange).toHaveBeenCalledWith("heavy");
  });

  it("reports the tier whose tick label is clicked", () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexitySelector value="heavy" onChange={onChange} />);
    openPanel(/heavy/i);
    fireEvent.click(screen.getByText("Lite"));
    expect(onChange).toHaveBeenCalledWith("lite");
  });

  it("describes only the current tier, the way an effort sublabel does", () => {
    renderWithProviders(<ComplexitySelector value="lite" onChange={() => {}} />);
    openPanel(/lite/i);
    const panel = screen.getByRole("dialog");
    // Lite's line must state the accuracy cost, not just speed — Task 1 measured only ~50%
    // Lite==Heavy squat verdict agreement, so that is the number the copy has to carry.
    expect(panel).toHaveTextContent(/half of squat verdicts/i);
    // The other tiers' descriptions are NOT on screen — that is what makes this an effort
    // control rather than a list of options to compare.
    expect(panel).not.toHaveTextContent(/thresholds were validated/i);
    expect(panel).not.toHaveTextContent(/default/i);
  });

  it("marks the sublabel Default only on the default tier", () => {
    renderWithProviders(<ComplexitySelector value="heavy" onChange={() => {}} />);
    openPanel(/heavy/i);
    const panel = screen.getByRole("dialog");
    expect(panel).toHaveTextContent(/thresholds were validated/i);
    // Tracks DEFAULT_ANALYSIS_TIER — if the default moves off Heavy this must move with it.
    expect(panel).toHaveTextContent(/default/i);
  });

  it("keeps the axis ends and the live-overlay carve-out visible", () => {
    renderWithProviders(<ComplexitySelector value="heavy" onChange={() => {}} />);
    openPanel(/heavy/i);
    const panel = screen.getByRole("dialog");
    expect(panel).toHaveTextContent(/faster/i);
    expect(panel).toHaveTextContent(/more accurate/i);
    // This control must not read as governing the live skeleton.
    expect(panel).toHaveTextContent(/live skeleton overlay always runs Lite/i);
  });

  it("focuses the slider on open so arrow keys adjust immediately", () => {
    renderWithProviders(<ComplexitySelector value="full" onChange={() => {}} />);
    openPanel(/full/i);
    expect(screen.getByRole("slider")).toHaveFocus();
  });

  it("closes on Escape and on an outside click", () => {
    renderWithProviders(<ComplexitySelector value="heavy" onChange={() => {}} />);

    openPanel(/heavy/i);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    openPanel(/heavy/i);
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
