import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// With a signed-in Supabase session, requests must carry the bearer token.
const { mockGetSession } = vi.hoisted(() => ({ mockGetSession: vi.fn() }));

vi.mock("../lib/supabase", () => ({
  isSupabaseConfigured: true,
  supabase: { auth: { getSession: mockGetSession } },
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
    await api.analyzeUpload(new File(["v"], "squat.mp4", { type: "video/mp4" }));
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
