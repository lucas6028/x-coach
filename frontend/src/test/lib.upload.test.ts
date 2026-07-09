import { describe, it, expect, vi, afterEach } from "vitest";
import {
  MAX_UPLOAD_BYTES,
  MAX_UPLOAD_DURATION_S,
  MAX_UPLOAD_MB,
  uploadLimitVars,
  probeDuration,
  validateUpload,
} from "../lib/upload";

// A fake <video> whose `src` setter fires the metadata / error callback asynchronously, so
// probeDuration's Promise resolves the way a real browser would (jsdom has no media pipeline).
function fakeVideo(duration: number, opts: { error?: boolean; silent?: boolean } = {}) {
  const v: Record<string, unknown> = {
    preload: "",
    onloadedmetadata: null as null | (() => void),
    onerror: null as null | (() => void),
    duration,
  };
  Object.defineProperty(v, "src", {
    set() {
      if (opts.silent) return; // never fires an event — exercises the timeout path
      Promise.resolve().then(() => {
        if (opts.error) (v.onerror as () => void)?.();
        else (v.onloadedmetadata as () => void)?.();
      });
    },
  });
  return v as unknown as HTMLVideoElement;
}

function stubVideo(video: HTMLVideoElement) {
  const realCreate = document.createElement.bind(document);
  vi.spyOn(document, "createElement").mockImplementation((tag: string) =>
    tag === "video" ? video : realCreate(tag)
  );
  // jsdom doesn't implement object URLs.
  vi.stubGlobal("URL", {
    ...URL,
    createObjectURL: vi.fn(() => "blob:fake"),
    revokeObjectURL: vi.fn(),
  });
}

function fileOfSize(bytes: number): File {
  const f = new File(["x"], "clip.mp4", { type: "video/mp4" });
  Object.defineProperty(f, "size", { value: bytes });
  return f;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("upload limits", () => {
  it("exposes MB and the shared i18n vars", () => {
    expect(MAX_UPLOAD_MB).toBe(100);
    expect(uploadLimitVars).toEqual({ maxMb: 100, maxS: MAX_UPLOAD_DURATION_S });
    expect(MAX_UPLOAD_BYTES).toBe(100 * 1024 * 1024);
  });
});

describe("probeDuration", () => {
  it("resolves the video's duration", async () => {
    stubVideo(fakeVideo(12.5));
    await expect(probeDuration(fileOfSize(1000))).resolves.toBe(12.5);
  });

  it("resolves NaN when the video errors", async () => {
    stubVideo(fakeVideo(0, { error: true }));
    await expect(probeDuration(fileOfSize(1000))).resolves.toBeNaN();
  });

  it("resolves NaN when metadata never loads (timeout)", async () => {
    vi.useFakeTimers();
    stubVideo(fakeVideo(0, { silent: true }));
    const p = probeDuration(fileOfSize(1000));
    await vi.advanceTimersByTimeAsync(15000);
    await expect(p).resolves.toBeNaN();
    vi.useRealTimers();
  });

  it("resolves NaN when object URLs are unavailable", async () => {
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn(() => {
        throw new Error("not implemented");
      }),
      revokeObjectURL: vi.fn(),
    });
    await expect(probeDuration(fileOfSize(1000))).resolves.toBeNaN();
  });
});

describe("validateUpload", () => {
  it("rejects a file over the size limit before probing duration", async () => {
    const spy = vi.spyOn(document, "createElement");
    const res = await validateUpload(fileOfSize(MAX_UPLOAD_BYTES + 1));
    expect(res).toEqual({ ok: false, errorKey: "upload.tooLarge" });
    expect(spy).not.toHaveBeenCalled();
  });

  it("rejects a clip over the duration limit", async () => {
    stubVideo(fakeVideo(MAX_UPLOAD_DURATION_S + 5));
    const res = await validateUpload(fileOfSize(1000));
    expect(res).toEqual({ ok: false, errorKey: "upload.tooLong" });
  });

  it("accepts a clip within both limits", async () => {
    stubVideo(fakeVideo(10));
    const res = await validateUpload(fileOfSize(1000));
    expect(res).toEqual({ ok: true });
  });

  it("accepts when duration can't be read (server backstops)", async () => {
    stubVideo(fakeVideo(0, { error: true }));
    const res = await validateUpload(fileOfSize(1000));
    expect(res).toEqual({ ok: true });
  });
});
