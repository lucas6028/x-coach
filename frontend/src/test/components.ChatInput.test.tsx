import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import ChatInput from "../components/ChatInput";
import { renderWithProviders } from "./renderWithProviders";

describe("ChatInput", () => {
  it("renders a disabled text input", () => {
    renderWithProviders(<ChatInput />);
    const input = screen.getByRole("textbox");
    expect(input).toBeDisabled();
  });

  it("shows the placeholder text", () => {
    renderWithProviders(<ChatInput />);
    expect(screen.getByPlaceholderText(/Ask the AI Coach/i)).toBeInTheDocument();
  });

  it("has a tooltip title explaining the feature is coming", () => {
    renderWithProviders(<ChatInput />);
    const wrapper = screen.getByRole("textbox").closest("[title]");
    expect(wrapper?.getAttribute("title")).toMatch(/LLM layer/i);
  });
});
