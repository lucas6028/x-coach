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

  // The waiting mascot RUNS rather than floating. The stride is faked on one still cutout, so the
  // body and its light sweep must share a single animated wrapper — animating them separately lets
  // the mask drift off the silhouette. Pinned structurally: that wrapper, the contact shadow and
  // the speed lines are what the CSS run cycle and the reduced-motion block both target by name.
  it("runs: body and light sweep share one animated wrapper, with a shadow and speed lines", () => {
    const { container } = renderWithProviders(<LumenLoader variant="scan" />);
    const runner = container.querySelector(".lm-runner");
    expect(runner).not.toBeNull();
    expect(runner!.querySelector(".lm-char")).not.toBeNull();
    expect(runner!.querySelector(".lm-scan")).not.toBeNull();
    expect(container.querySelector(".lm-shadow")).not.toBeNull();
    expect(container.querySelectorAll(".lm-track i")).toHaveLength(3);
    // Two dust puffs, one per footfall — the cue that replaced the scrolling ground line, which
    // read as a board under the feet rather than as running.
    expect(container.querySelectorAll(".lm-dust i")).toHaveLength(2);
    // Decorative — the status line already carries the announcement.
    expect(container.querySelector(".lm-track")).toHaveAttribute("aria-hidden", "true");
    expect(container.querySelector(".lm-shadow")).toHaveAttribute("aria-hidden", "true");
    expect(container.querySelector(".lm-dust")).toHaveAttribute("aria-hidden", "true");
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
