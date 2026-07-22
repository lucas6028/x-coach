import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ComplexitySelector from "../components/ComplexitySelector";
import { renderWithProviders } from "./renderWithProviders";

describe("ComplexitySelector", () => {
  it("keeps the menu closed until the trigger is clicked", () => {
    renderWithProviders(<ComplexitySelector value="lite" onChange={() => {}} />);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /lite/i }));
    expect(screen.getByRole("menu")).toBeInTheDocument();
  });

  it("reports the picked tier and closes the menu", () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexitySelector value="lite" onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: /lite/i }));
    fireEvent.click(screen.getByRole("menuitemradio", { name: "Heavy" }));
    expect(onChange).toHaveBeenCalledWith("heavy");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("marks only the current tier as checked", () => {
    renderWithProviders(<ComplexitySelector value="full" onChange={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: /full/i }));
    expect(screen.getByRole("menuitemradio", { name: "Full" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("menuitemradio", { name: "Lite" })).toHaveAttribute("aria-checked", "false");
  });

  it("describes each tier's tradeoff and flags the default", () => {
    renderWithProviders(<ComplexitySelector value="lite" onChange={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: /lite/i }));
    // Lite's hint must state the accuracy cost, not just speed — Task 1 measured only ~50%
    // Lite==Heavy squat verdict agreement, so this is the number the copy has to carry.
    expect(screen.getByRole("menuitemradio", { name: "Lite" })).toHaveAccessibleDescription(/half/i);
    // The "Default" badge tracks DEFAULT_ANALYSIS_TIER, so it must sit on Heavy — not Lite.
    expect(screen.getByRole("menuitemradio", { name: "Heavy" })).toHaveTextContent(/default/i);
    expect(screen.getByRole("menuitemradio", { name: "Lite" })).not.toHaveTextContent(/default/i);
    // The overlay carve-out: this control must not read as governing the live skeleton.
    expect(screen.getByRole("menu")).toHaveTextContent(/live skeleton overlay always runs Lite/i);
  });

  it("shows the current tier on the trigger without opening the menu", () => {
    renderWithProviders(<ComplexitySelector value="heavy" onChange={() => {}} />);
    expect(screen.getByRole("button", { name: /heavy/i })).toHaveAttribute("aria-expanded", "false");
  });

  it("closes on Escape and on an outside click", () => {
    renderWithProviders(<ComplexitySelector value="heavy" onChange={() => {}} />);
    const trigger = screen.getByRole("button", { name: /heavy/i });

    fireEvent.click(trigger);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();

    fireEvent.click(trigger);
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });
});
