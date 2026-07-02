import { describe, it, expect, vi, afterEach } from "vitest";
import { api } from "../api";

function mockFetch(body: unknown, ok = true, status = 200) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok,
    status,
    statusText: ok ? "OK" : "Not Found",
    json: async () => body,
  } as Response);
}

describe("api.health", () => {
  afterEach(() => vi.restoreAllMocks());

  it("calls /api/health and returns the parsed JSON", async () => {
    mockFetch({ status: "ok" });
    const result = await api.health();
    expect(result).toEqual({ status: "ok" });
    expect(fetch).toHaveBeenCalledWith("/api/health");
  });
});

describe("api.listVideos", () => {
  afterEach(() => vi.restoreAllMocks());

  it("builds the URL with default limit and offset", async () => {
    mockFetch({ total: 0, items: [] });
    await api.listVideos();
    expect(fetch).toHaveBeenCalledWith("/api/videos?limit=50&offset=0");
  });

  it("appends the fault filter when provided", async () => {
    mockFetch({ total: 1, items: [] });
    await api.listVideos(10, 0, "knees_inward");
    expect(fetch).toHaveBeenCalledWith("/api/videos?limit=10&offset=0&fault=knees_inward");
  });

  it("omits the fault param when not provided", async () => {
    mockFetch({ total: 0, items: [] });
    await api.listVideos(5, 0);
    const url = (fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(url).not.toContain("fault=");
  });
});

describe("api.getAnalysis", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches the correct endpoint", async () => {
    mockFetch({ video_id: "abc" });
    await api.getAnalysis("abc");
    expect(fetch).toHaveBeenCalledWith("/api/analysis/abc");
  });

  it("throws on non-ok responses", async () => {
    mockFetch({}, false, 404);
    await expect(api.getAnalysis("missing")).rejects.toThrow("404");
  });
});

describe("api.videoFileUrl", () => {
  it("returns the correct URL string", () => {
    expect(api.videoFileUrl("vid_001")).toBe("/api/video-file/vid_001");
  });
});

describe("api.graph", () => {
  afterEach(() => vi.restoreAllMocks());

  it("URL-encodes the query", async () => {
    mockFetch({ matched_nodes: [] });
    await api.graph("knee valgus");
    expect(fetch).toHaveBeenCalledWith("/api/knowledge/graph?query=knee%20valgus");
  });
});

describe("api.listAnalyses", () => {
  afterEach(() => vi.restoreAllMocks());

  it("builds the URL with default limit and offset", async () => {
    mockFetch({ total: 0, items: [] });
    await api.listAnalyses();
    expect(fetch).toHaveBeenCalledWith("/api/analyses?limit=50&offset=0");
  });

  it("passes a custom limit and offset", async () => {
    mockFetch({ total: 0, items: [] });
    await api.listAnalyses(10, 20);
    expect(fetch).toHaveBeenCalledWith("/api/analyses?limit=10&offset=20");
  });

  it("throws on non-ok responses", async () => {
    mockFetch({}, false, 401);
    await expect(api.listAnalyses()).rejects.toThrow("401");
  });
});

describe("api.getStoredAnalysis", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches the stored-analysis row endpoint", async () => {
    mockFetch({ id: "a1", result: { video_id: "v1" } });
    await api.getStoredAnalysis("a1");
    expect(fetch).toHaveBeenCalledWith("/api/analyses/a1");
  });
});

describe("api.deleteAnalyses", () => {
  afterEach(() => vi.restoreAllMocks());

  it("DELETEs the analyses endpoint and returns the count", async () => {
    const spy = mockFetch({ deleted: 4 });
    const result = await api.deleteAnalyses();
    expect(result).toEqual({ deleted: 4 });
    expect(spy.mock.calls[0][0]).toBe("/api/analyses");
    expect((spy.mock.calls[0][1] as RequestInit).method).toBe("DELETE");
  });

  it("throws on non-ok responses", async () => {
    mockFetch({}, false, 401);
    await expect(api.deleteAnalyses()).rejects.toThrow("401");
  });
});

describe("api.analyzeUpload", () => {
  afterEach(() => vi.restoreAllMocks());

  it("POSTs a FormData body and returns the analysis", async () => {
    const spy = mockFetch({ video_id: "upload_1" });
    const file = new File(["video"], "squat.mp4", { type: "video/mp4" });
    const result = await api.analyzeUpload(file);
    expect(result).toEqual({ video_id: "upload_1" });
    expect(spy.mock.calls[0][0]).toBe("/api/analyze");
    expect((spy.mock.calls[0][1] as RequestInit).method).toBe("POST");
  });

  it("throws with the backend detail message on failure", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 422,
      statusText: "Unprocessable Entity",
      json: async () => ({ detail: "Invalid video format" }),
    } as Response);
    const file = new File(["data"], "bad.mp4", { type: "video/mp4" });
    await expect(api.analyzeUpload(file)).rejects.toThrow("Invalid video format");
  });

  it("falls back to a generic message when backend detail is absent", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => ({}),
    } as Response);
    const file = new File(["data"], "bad.mp4", { type: "video/mp4" });
    await expect(api.analyzeUpload(file)).rejects.toThrow("Analyze failed (500)");
  });
});
