import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ComplexitySelector from "../components/ComplexitySelector";

describe("ComplexitySelector", () => {
  it("renders the three tiers and reports the picked one", () => {
    const onChange = vi.fn();
    render(<ComplexitySelector value="lite" onChange={onChange} />);
    fireEvent.click(screen.getByRole("radio", { name: /heavy/i }));
    expect(onChange).toHaveBeenCalledWith("heavy");
  });

  it("marks the current tier as checked", () => {
    render(<ComplexitySelector value="full" onChange={() => {}} />);
    expect(screen.getByRole("radio", { name: /full/i })).toBeChecked();
  });
});
