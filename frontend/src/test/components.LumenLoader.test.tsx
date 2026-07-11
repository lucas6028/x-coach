import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { LumenLoader, LumenAvatar } from "../components/LumenLoader";
import { renderWithProviders } from "./renderWithProviders";

describe("LumenLoader", () => {
  it("renders the scan stage as a status region labelled by its caption", () => {
    renderWithProviders(<LumenLoader variant="scan" caption="Extracting pose…" />);
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-label", "Extracting pose…");
    // The caption also renders as visible text below the stage.
    expect(screen.getByText("Extracting pose…")).toBeInTheDocument();
  });

  it("renders the scan stage without a caption (aria falls back to the generic label)", () => {
    renderWithProviders(<LumenLoader variant="scan" />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("renders the dots variant as a status region", () => {
    renderWithProviders(<LumenLoader variant="dots" />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("renders the Lumen avatar as a decorative head image", () => {
    const { container } = renderWithProviders(<LumenAvatar size={20} />);
    const img = container.querySelector("img");
    expect(img).toHaveAttribute("src", "/lumen/lumen-head.png");
    // Decorative — a "Lumen" text label always sits beside it, so the alt is empty.
    expect(img).toHaveAttribute("alt", "");
  });
});
