import "@testing-library/jest-dom";
import { vi } from "vitest";

// Stub matchMedia — jsdom doesn't implement it.
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Stub localStorage backed by a plain Map.
const store: Record<string, string> = {};
Object.defineProperty(window, "localStorage", {
  value: {
    getItem: (k: string) => store[k] ?? null,
    setItem: (k: string, v: string) => { store[k] = v; },
    removeItem: (k: string) => { delete store[k]; },
    clear: () => { Object.keys(store).forEach((k) => delete store[k]); },
  },
});

// Stub HTMLVideoElement.play/pause — jsdom doesn't implement media APIs.
Object.defineProperty(HTMLVideoElement.prototype, "play", {
  configurable: true,
  value: vi.fn().mockResolvedValue(undefined),
});
Object.defineProperty(HTMLVideoElement.prototype, "pause", {
  configurable: true,
  value: vi.fn(),
});

// Stub HTMLCanvasElement.getContext — jsdom has no GPU / 2-D rasterizer.
HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
  clearRect: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
  fill: vi.fn(),
  arc: vi.fn(),
  getBoundingClientRect: vi.fn().mockReturnValue({ width: 100, height: 100 }),
  getScreenCTM: vi.fn().mockReturnValue({ inverse: () => ({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }) }),
  strokeStyle: "",
  fillStyle: "",
  lineWidth: 1,
  lineCap: "butt",
  shadowColor: "",
  shadowBlur: 0,
} as unknown as CanvasRenderingContext2D);

// Stub requestAnimationFrame — tests don't need real animation frames.
vi.stubGlobal("requestAnimationFrame", vi.fn((_cb: FrameRequestCallback) => {
  // Don't call the callback — components that depend on rAF loops won't loop in tests.
  return 0;
}));
vi.stubGlobal("cancelAnimationFrame", vi.fn());

// Stub IntersectionObserver — jsdom doesn't implement it; used by framer-motion's
// scroll-triggered animations and the landing page's Reveal component.
vi.stubGlobal(
  "IntersectionObserver",
  class {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
    constructor(_cb: IntersectionObserverCallback, _opts?: IntersectionObserverInit) {}
  }
);

// Stub ResizeObserver — jsdom doesn't implement it.
vi.stubGlobal(
  "ResizeObserver",
  class {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
    constructor(_cb: ResizeObserverCallback) {}
  }
);
