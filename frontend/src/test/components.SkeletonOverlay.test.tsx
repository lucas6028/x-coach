import { describe, it, expect, vi } from "vitest";
import type React from "react";
import SkeletonOverlay from "../components/SkeletonOverlay";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis } from "./fixtures";

function makeVideoRef(overrides: Partial<HTMLVideoElement> = {}): React.RefObject<HTMLVideoElement> {
  const video = {
    currentTime: 0,
    videoWidth: 1280,
    videoHeight: 720,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    ...overrides,
  } as unknown as HTMLVideoElement;
  return { current: video };
}

describe("SkeletonOverlay", () => {
  it("renders a canvas element", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <SkeletonOverlay analysis={mockAnalysis} videoRef={ref} onActiveFault={vi.fn()} />
    );
    expect(document.querySelector("canvas")).toBeInTheDocument();
  });

  it("canvas has pointer-events-none class", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <SkeletonOverlay analysis={mockAnalysis} videoRef={ref} onActiveFault={vi.fn()} />
    );
    const canvas = document.querySelector("canvas");
    expect(canvas?.className).toContain("pointer-events-none");
  });

  it("does not throw when video ref is null", () => {
    const nullRef: React.RefObject<HTMLVideoElement> = { current: null };
    // ref.current is null
    expect(() =>
      renderWithProviders(
        <SkeletonOverlay analysis={mockAnalysis} videoRef={nullRef} onActiveFault={vi.fn()} />
      )
    ).not.toThrow();
  });

  it("requests an animation frame on mount", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <SkeletonOverlay analysis={mockAnalysis} videoRef={ref} onActiveFault={vi.fn()} />
    );
    expect(requestAnimationFrame).toHaveBeenCalled();
  });
});
