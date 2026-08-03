import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";

const pose = { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] };

afterEach(() => vi.restoreAllMocks());

describe("api.analyzePose", () => {
  it("posts movement + pose JSON + video and returns the analysis", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ video_id: "v1", source: "upload" }), { status: 200 })
    );
    const result = await api.analyzePose("Squat", pose as never, new Blob(["x"], { type: "video/webm" }));
    expect(result.video_id).toBe("v1");
    const [, init] = fetchMock.mock.calls[0];
    const form = init!.body as FormData;
    expect(form.get("movement")).toBe("Squat");
    expect(JSON.parse(await (form.get("pose") as Blob).text()).metadata.fps).toBe(30);
    expect(form.get("file")).toBeInstanceOf(Blob);
  });

  it("throws the backend detail on failure", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Pose JSON must have a 'frames' list." }), { status: 400 })
    );
    await expect(api.analyzePose("Squat", pose as never, new Blob(["x"]))).rejects.toThrow("frames");
  });

  it("surfaces the endpoint-specific pose payload limit", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: { code: "pose_too_large", limit_mb: 16 } }), { status: 413 })
    );
    await expect(api.analyzePose("Squat", pose as never, new Blob(["x"]))).rejects.toThrow("16 MB");
  });
});

describe("api.analyzePose thumbnail", () => {
  it("appends the thumbnail when one was captured", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ video_id: "v" }), { status: 200 }));
    await api.analyzePose("Squat", pose as never, new Blob(["v"], { type: "video/webm" }),
      new Blob(["jpeg"], { type: "image/jpeg" }));
    const form = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(form.get("thumbnail")).toBeInstanceOf(Blob);
  });
});
