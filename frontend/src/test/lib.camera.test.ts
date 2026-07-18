import { describe, it, expect, vi, afterEach } from "vitest";
import { getCameraStream, probeCamera } from "../lib/camera";

// jsdom has no mediaDevices; each test installs exactly the shape it needs.
function installGetUserMedia(impl: (c: MediaStreamConstraints) => Promise<MediaStream>) {
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: impl },
  });
}

function removeMediaDevices() {
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: undefined });
}

function fakeStream() {
  const stop = vi.fn();
  return {
    stream: { getTracks: () => [{ stop }] } as unknown as MediaStream,
    stop,
  };
}

afterEach(() => {
  vi.useRealTimers();
  removeMediaDevices();
});

describe("getCameraStream", () => {
  it("rejects as unsupported when mediaDevices is missing", async () => {
    removeMediaDevices();
    await expect(getCameraStream({ video: true })).rejects.toMatchObject({
      name: "CameraError",
      reason: "unsupported",
    });
  });

  it("resolves with the stream when the browser answers", async () => {
    const { stream } = fakeStream();
    installGetUserMedia(vi.fn().mockResolvedValue(stream));
    await expect(getCameraStream({ video: true })).resolves.toBe(stream);
  });

  it("rethrows a real browser rejection (permission denied)", async () => {
    const denied = new DOMException("nope", "NotAllowedError");
    installGetUserMedia(vi.fn().mockRejectedValue(denied));
    await expect(getCameraStream({ video: true })).rejects.toBe(denied);
  });

  it("times out when getUserMedia never settles (the LIFF-on-iOS hang)", async () => {
    vi.useFakeTimers();
    installGetUserMedia(() => new Promise<MediaStream>(() => {}));
    const pending = getCameraStream({ video: true }, { timeoutMs: 5000 });
    const assertion = expect(pending).rejects.toMatchObject({ reason: "timeout" });
    await vi.advanceTimersByTimeAsync(5001);
    await assertion;
  });

  it("stops the tracks of a stream that arrives after the timeout", async () => {
    vi.useFakeTimers();
    const { stream, stop } = fakeStream();
    let deliver: (s: MediaStream) => void = () => {};
    installGetUserMedia(() => new Promise<MediaStream>((resolve) => (deliver = resolve)));
    const pending = getCameraStream({ video: true }, { timeoutMs: 1000 });
    const assertion = expect(pending).rejects.toMatchObject({ reason: "timeout" });
    await vi.advanceTimersByTimeAsync(1001);
    await assertion;
    deliver(stream); // the browser finally answers — nobody is listening
    await vi.advanceTimersByTimeAsync(0);
    expect(stop).toHaveBeenCalled();
  });
});

describe("probeCamera", () => {
  it("opens then releases the camera and reports ok", async () => {
    const { stream, stop } = fakeStream();
    installGetUserMedia(vi.fn().mockResolvedValue(stream));
    const result = await probeCamera();
    expect(result.ok).toBe(true);
    expect(result.reason).toBe("ok");
    expect(stop).toHaveBeenCalled();
  });

  it("reports unsupported without mediaDevices", async () => {
    removeMediaDevices();
    const result = await probeCamera();
    expect(result).toMatchObject({ ok: false, reason: "unsupported" });
  });

  it("reports a timeout", async () => {
    vi.useFakeTimers();
    installGetUserMedia(() => new Promise<MediaStream>(() => {}));
    const pending = probeCamera(2000);
    await vi.advanceTimersByTimeAsync(2001);
    expect(await pending).toMatchObject({ ok: false, reason: "timeout" });
  });

  it("reports denied on NotAllowedError", async () => {
    installGetUserMedia(vi.fn().mockRejectedValue(new DOMException("no", "NotAllowedError")));
    expect(await probeCamera()).toMatchObject({ ok: false, reason: "denied" });
  });

  it("reports a generic error otherwise", async () => {
    installGetUserMedia(vi.fn().mockRejectedValue(new Error("hardware fell off")));
    const result = await probeCamera();
    expect(result).toMatchObject({ ok: false, reason: "error" });
    expect(result.message).toContain("hardware fell off");
  });
});
