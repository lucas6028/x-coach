import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// With a signed-in Supabase session, requests must carry the bearer token.
const { mockGetSession, mockRefreshSession } = vi.hoisted(() => ({
  mockGetSession: vi.fn(),
  mockRefreshSession: vi.fn(),
}));

vi.mock("../lib/supabase", () => ({
  isSupabaseConfigured: true,
  supabase: { auth: { getSession: mockGetSession, refreshSession: mockRefreshSession } },
}));

import { api } from "../api";

function mockFetch(body: unknown) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => body,
  } as Response);
}

beforeEach(() => {
  mockGetSession.mockResolvedValue({ data: { session: { access_token: "tok123" } } });
});
afterEach(() => vi.restoreAllMocks());

describe("api auth header", () => {
  it("attaches the bearer token to JSON reads", async () => {
    const spy = mockFetch({ total: 0, items: [] });
    await api.listAnalyses();
    expect(spy).toHaveBeenCalledWith("/api/analyses?limit=50&offset=0", {
      headers: { Authorization: "Bearer tok123" },
    });
  });

  it("attaches the bearer token to uploads", async () => {
    const spy = mockFetch({ video_id: "u1" });
    await api.analyzeUpload(new File(["v"], "squat.mp4", { type: "video/mp4" }), "Squat");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ Authorization: "Bearer tok123" });
  });

  it("omits the header when there is no session", async () => {
    mockGetSession.mockResolvedValue({ data: { session: null } });
    const spy = mockFetch({ total: 0, items: [] });
    await api.listAnalyses();
    expect(spy).toHaveBeenCalledWith("/api/analyses?limit=50&offset=0");
  });
});

// A long upload can outlive the token that was read before it started: the body arrives intact
// and the server answers 401. Seen in production 2026-09-21.
describe("api.analyzePose token expiry", () => {
  const pose = { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] };
  const unauthorized = () =>
    new Response(JSON.stringify({ detail: "Invalid or expired token." }), { status: 401 });

  it("refreshes the session and resends once after a 401", async () => {
    mockRefreshSession.mockResolvedValue({ data: { session: { access_token: "fresh" } }, error: null });
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(unauthorized())
      .mockResolvedValueOnce(new Response(JSON.stringify({ video_id: "v1" }), { status: 200 }));
    const result = await api.analyzePose("Squat", pose as never, new Blob(["x"], { type: "video/webm" }));
    expect(result.video_id).toBe("v1");
    expect(spy).toHaveBeenCalledTimes(2);
    expect((spy.mock.calls[0][1] as RequestInit).headers).toEqual({ Authorization: "Bearer tok123" });
    expect((spy.mock.calls[1][1] as RequestInit).headers).toEqual({ Authorization: "Bearer fresh" });
    // Same body both times — the retry must not drop the video.
    expect(((spy.mock.calls[1][1] as RequestInit).body as FormData).get("file")).toBeInstanceOf(Blob);
  });

  it("does not loop: a 401 on the fresh token is reported", async () => {
    mockRefreshSession.mockResolvedValue({ data: { session: { access_token: "fresh" } }, error: null });
    const spy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => unauthorized());
    await expect(api.analyzePose("Squat", pose as never, new Blob(["x"]))).rejects.toThrow(
      "Invalid or expired token."
    );
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("keeps the original 401 when the refresh fails", async () => {
    mockRefreshSession.mockResolvedValue({ data: { session: null }, error: new Error("refresh_token_not_found") });
    const spy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => unauthorized());
    await expect(api.analyzePose("Squat", pose as never, new Blob(["x"]))).rejects.toThrow(
      "Invalid or expired token."
    );
    expect(spy).toHaveBeenCalledTimes(1);
  });
});
