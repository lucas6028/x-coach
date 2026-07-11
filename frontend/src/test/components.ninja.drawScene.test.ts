import { describe, it, expect, vi } from "vitest";
import { drawScene, type Scene } from "../components/ninja/ninjaDetector";
import type { Piece } from "../lib/ninja/pieces";

// A mock 2D context: drawScene only ever calls methods / sets props, so a bag of spies suffices
// (no real canvas, which jsdom doesn't provide).
function mockCtx() {
  return {
    clearRect: vi.fn(),
    fillText: vi.fn(),
    beginPath: vi.fn(),
    rect: vi.fn(),
    clip: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    translate: vi.fn(),
    rotate: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    fillRect: vi.fn(),
  } as unknown as CanvasRenderingContext2D & Record<string, ReturnType<typeof vi.fn>>;
}

const piece = (over: Partial<Piece> = {}): Piece => ({
  id: 1,
  emoji: "🍉",
  x: 0.5,
  y: 0.5,
  vx: 0,
  vy: 0,
  rot: 0.3,
  spin: 2,
  radius: 0.075,
  half: "left",
  life: 0.8,
  ...over,
});

const emptyScene: Scene = { entities: [], pieces: [], trails: [], bombFlash: null };

describe("drawScene", () => {
  it("clears the canvas even with nothing to draw", () => {
    const ctx = mockCtx();
    drawScene(ctx, emptyScene, 640, 480);
    expect(ctx.clearRect).toHaveBeenCalledWith(0, 0, 640, 480);
  });

  it("draws each sliced-fruit half clipped in its own transformed frame", () => {
    const ctx = mockCtx();
    drawScene(
      ctx,
      { ...emptyScene, pieces: [piece({ half: "left" }), piece({ id: 2, half: "right" })] },
      640,
      480
    );
    // One save/translate/rotate/clip per piece, and the emoji painted for each.
    expect(ctx.save).toHaveBeenCalledTimes(2);
    expect(ctx.rotate).toHaveBeenCalledTimes(2);
    expect(ctx.clip).toHaveBeenCalledTimes(2);
    expect(ctx.restore).toHaveBeenCalledTimes(2);
    expect(ctx.fillText).toHaveBeenCalledWith("🍉", 0, 0);
  });

  it("draws fruit emoji and the bomb flash", () => {
    const ctx = mockCtx();
    drawScene(
      ctx,
      {
        entities: [{ id: 1, kind: "fruit", emoji: "🍎", x: 0.3, y: 0.4, vx: 0, vy: 0, radius: 0.075 }],
        pieces: [],
        trails: [],
        bombFlash: 0.5,
      },
      640,
      480
    );
    // Fruit position is mirrored (selfie space) so it lines up with the hand the player sees.
    expect(ctx.fillText).toHaveBeenCalledWith("🍎", (1 - 0.3) * 640, 0.4 * 480);
    expect(ctx.fillRect).toHaveBeenCalledWith(0, 0, 640, 480);
  });
});
