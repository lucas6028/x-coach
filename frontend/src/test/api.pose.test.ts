import { afterEach, describe, expect, it, vi } from "vitest";
import { api, UploadLimitError } from "../api";
import { installFakeXhr } from "./fakeXhr";

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

  // `vi.spyOn`, NOT `vi.stubGlobal`: the file's afterEach only calls `restoreAllMocks`, which does
  // not unstub globals — a stubbed fetch would leak into the thumbnail case below and swallow its
  // request.
  it("posts the rep plan alongside the pose JSON", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ video_id: "v1" }), { status: 200 })
    );
    const reps = { max_reps: 3, fallback: null, segments: [
      { index: 1, start_frame: 0, end_frame: 29, partial: false, analyzed: true, refined: true as const },
    ] };
    await api.analyzePose("Squat", { metadata: { fps: 30, width: 1, height: 1, total_frames: 1 }, frames: [] },
      new Blob([], { type: "video/webm" }), null, reps);
    const form = fetchMock.mock.calls[0][1]!.body as FormData;
    expect(JSON.parse(form.get("reps") as string)).toEqual(reps);
  });

  it("omits reps entirely when there is no plan, so old behaviour is unchanged", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ video_id: "v1" }), { status: 200 })
    );
    await api.analyzePose("Squat", { metadata: { fps: 30, width: 1, height: 1, total_frames: 1 }, frames: [] },
      new Blob([], { type: "video/webm" }));
    expect((fetchMock.mock.calls[0][1]!.body as FormData).get("reps")).toBeNull();
  });
});

// With a progress callback the request moves to XMLHttpRequest — fetch cannot observe the request
// body going out. Everything after the transport (status handling, error mapping) is shared, so
// these pin the transport itself plus one failure to prove the sharing.
describe("api.analyzePose upload progress", () => {
  it("reports byte progress, then completion, and returns the analysis", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const { requests } = installFakeXhr({
      body: { video_id: "v1" },
      uploadEvents: [{ loaded: 25, total: 100 }, { loaded: 100, total: 100 }],
    });
    const seen: number[] = [];
    const result = await api.analyzePose(
      "Squat", pose as never, new Blob(["x"], { type: "video/webm" }), null, undefined,
      (p) => seen.push(p)
    );
    expect(result.video_id).toBe("v1");
    expect(seen).toEqual([0.25, 1, 1]);
    expect(requests).toHaveLength(1);
    expect(requests[0].method).toBe("POST");
    expect(requests[0].url).toBe("/api/analyze/pose");
    expect(requests[0].body!.get("movement")).toBe("Squat");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("skips progress events whose total is unknown", async () => {
    installFakeXhr({ uploadEvents: [{ loaded: 10, total: 0, lengthComputable: false }] });
    const seen: number[] = [];
    await api.analyzePose("Squat", pose as never, new Blob(["x"]), null, undefined, (p) => seen.push(p));
    expect(seen).toEqual([1]);
  });

  it("maps a failure status exactly as the fetch path does", async () => {
    installFakeXhr({ status: 413, body: { detail: { code: "upload_too_large", limit_mb: 200 } } });
    await expect(
      api.analyzePose("Squat", pose as never, new Blob(["x"]), null, undefined, () => {})
    ).rejects.toBeInstanceOf(UploadLimitError);
  });

  it("rejects when no response arrives", async () => {
    installFakeXhr({ networkError: true });
    await expect(
      api.analyzePose("Squat", pose as never, new Blob(["x"]), null, undefined, () => {})
    ).rejects.toThrow("Network request failed");
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
