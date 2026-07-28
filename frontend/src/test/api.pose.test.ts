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
    expect(JSON.parse(form.get("pose") as string).metadata.fps).toBe(30);
    expect(form.get("file")).toBeInstanceOf(Blob);
  });

  it("throws the backend detail on failure", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Pose JSON must have a 'frames' list." }), { status: 400 })
    );
    await expect(api.analyzePose("Squat", pose as never, new Blob(["x"]))).rejects.toThrow("frames");
  });

  it("posts the rep plan alongside the pose JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    const reps = { max_reps: 3, fallback: null, segments: [
      { index: 1, start_frame: 0, end_frame: 29, partial: false, analyzed: true, refined: true as const },
    ] };
    await api.analyzePose("Squat", { metadata: { fps: 30, width: 1, height: 1, total_frames: 1 }, frames: [] },
      new Blob([], { type: "video/webm" }), reps);
    const form = fetchMock.mock.calls[0][1].body as FormData;
    expect(JSON.parse(form.get("reps") as string)).toEqual(reps);
  });

  it("omits reps entirely when there is no plan, so old behaviour is unchanged", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await api.analyzePose("Squat", { metadata: { fps: 30, width: 1, height: 1, total_frames: 1 }, frames: [] },
      new Blob([], { type: "video/webm" }));
    expect((fetchMock.mock.calls[0][1].body as FormData).get("reps")).toBeNull();
  });
});
