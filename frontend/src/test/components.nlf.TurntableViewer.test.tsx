import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { I18nProvider } from "../lib/i18n";
import TurntableViewer from "../components/nlf/TurntableViewer";
import type { NlfKeyFrame } from "../api";

const ANGLES = 24;
const TILE_PX = 512;
const DISPLAY_PX = 288; // must match the component's own DISPLAY_PX

const KEY_FRAMES: NlfKeyFrame[] = [
  {
    t_s: 1.2,
    kind: "rep_bottom",
    fault_ids: [],
    rep_index: 2,
    strip_url: "https://example.com/strip1.webp",
  },
  {
    t_s: 3.4,
    kind: "fault_peak",
    fault_ids: ["knees_inward"],
    rep_index: null,
    strip_url: "https://example.com/strip2.webp",
  },
];

function renderViewer(over: Partial<React.ComponentProps<typeof TurntableViewer>> = {}) {
  const onSeek = vi.fn();
  const onImageError = vi.fn();
  render(
    <I18nProvider>
      <TurntableViewer
        keyFrames={KEY_FRAMES}
        angles={ANGLES}
        tilePx={TILE_PX}
        onSeek={onSeek}
        onImageError={onImageError}
        {...over}
      />
    </I18nProvider>
  );
  return { onSeek, onImageError };
}

function tile(): HTMLElement {
  return screen.getByTestId("turntable-tile");
}

function bgPosition(el: HTMLElement): number {
  const pos = el.style.backgroundPosition; // e.g. "-1728px 0"
  return -parseInt(pos.split(" ")[0], 10) || 0; // normalize -0 to 0
}

afterEach(() => vi.restoreAllMocks());

describe("TurntableViewer — background positioning", () => {
  it("starts on tile 0 (the camera's own view)", () => {
    renderViewer();
    expect(bgPosition(tile())).toBe(0);
    expect(tile().style.backgroundSize).toBe(`${ANGLES * DISPLAY_PX}px ${DISPLAY_PX}px`);
  });

  it("exposes itself as an accessible slider with a live angle value", () => {
    renderViewer();
    const t = tile();
    expect(t).toHaveAttribute("role", "slider");
    expect(t).toHaveAttribute("aria-valuemin", "0");
    expect(t).toHaveAttribute("aria-valuemax", String(ANGLES - 1));
    expect(t).toHaveAttribute("aria-valuenow", "0");
    expect(t).toHaveAttribute("aria-valuetext", "0°");
    expect(t.getAttribute("aria-label")).toMatch(/arrow keys/i);

    fireEvent.keyDown(t, { key: "ArrowRight" });
    expect(t).toHaveAttribute("aria-valuenow", "1");
    expect(t).toHaveAttribute("aria-valuetext", "15°");
  });

  it("dragging right steps the background position forward", () => {
    renderViewer();
    const t = tile();
    fireEvent.pointerDown(t, { clientX: 0, pointerId: 1 });
    fireEvent.pointerMove(t, { clientX: 28, pointerId: 1 }); // ~2 steps at 14px/step
    expect(bgPosition(t)).toBe(2 * DISPLAY_PX);
    fireEvent.pointerUp(t, { clientX: 28, pointerId: 1 });
  });

  it("dragging left steps the background position backward", () => {
    renderViewer();
    const t = tile();
    fireEvent.pointerDown(t, { clientX: 0, pointerId: 1 });
    fireEvent.pointerMove(t, { clientX: -14, pointerId: 1 });
    expect(bgPosition(t)).toBe((ANGLES - 1) * DISPLAY_PX);
  });

  it("wraps around past the last tile back to the first", () => {
    renderViewer();
    const t = tile();
    fireEvent.pointerDown(t, { clientX: 0, pointerId: 1 });
    // Drag far enough right to pass tile 23 and wrap to 0/1.
    fireEvent.pointerMove(t, { clientX: 24 * 14, pointerId: 1 });
    expect(bgPosition(t)).toBe(0);
  });
});

describe("TurntableViewer — side/front view shortcuts", () => {
  it("jumps to the 90-degree side view (tile 6 of 24)", () => {
    renderViewer();
    fireEvent.click(screen.getByRole("button", { name: /side view/i }));
    expect(bgPosition(tile())).toBe(6 * DISPLAY_PX);
  });

  it("front view returns to tile 0", () => {
    renderViewer();
    fireEvent.click(screen.getByRole("button", { name: /side view/i }));
    fireEvent.click(screen.getByRole("button", { name: /front view/i }));
    expect(bgPosition(tile())).toBe(0);
  });
});

describe("TurntableViewer — keyboard stepping", () => {
  it("ArrowRight steps forward and ArrowLeft steps backward", () => {
    renderViewer();
    const t = tile();
    fireEvent.keyDown(t, { key: "ArrowRight" });
    expect(bgPosition(t)).toBe(1 * DISPLAY_PX);
    fireEvent.keyDown(t, { key: "ArrowRight" });
    expect(bgPosition(t)).toBe(2 * DISPLAY_PX);
    fireEvent.keyDown(t, { key: "ArrowLeft" });
    expect(bgPosition(t)).toBe(1 * DISPLAY_PX);
  });

  it("ArrowLeft from tile 0 wraps to the last tile", () => {
    renderViewer();
    const t = tile();
    fireEvent.keyDown(t, { key: "ArrowLeft" });
    expect(bgPosition(t)).toBe((ANGLES - 1) * DISPLAY_PX);
  });
});

describe("TurntableViewer — key-frame list", () => {
  it("selecting a key frame calls onSeek with its time and resets to tile 0", () => {
    const { onSeek } = renderViewer();
    fireEvent.click(screen.getByRole("button", { name: /side view/i }));
    fireEvent.click(screen.getByText(/knee valgus/i));
    expect(onSeek).toHaveBeenCalledWith(3.4);
    expect(bgPosition(tile())).toBe(0);
  });

  it("labels a rep-bottom key frame with its rep number", () => {
    renderViewer();
    expect(screen.getByText(/Rep 2/)).toBeInTheDocument();
  });

  it("labels a fault-peak key frame with the fault name", () => {
    renderViewer();
    expect(screen.getByText(/Knee Valgus/i)).toBeInTheDocument();
  });
});

describe("TurntableViewer — image error", () => {
  it("calls onImageError when the preload image fails", () => {
    const { onImageError } = renderViewer();
    const img = document.querySelector("img")!;
    fireEvent.error(img);
    expect(onImageError).toHaveBeenCalledTimes(1);
  });
});

describe("TurntableViewer — caption", () => {
  it("explains the two faint lines as hip and knee joint-centre heights, never 'parallel'", () => {
    renderViewer();
    const caption = screen.getByText(/joint-centre heights/i);
    expect(caption.textContent?.toLowerCase()).not.toContain("parallel");
  });
});
