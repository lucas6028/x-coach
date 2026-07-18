import { describe, it, expect, vi, afterEach } from "vitest";
import { api, ApiError } from "../api";

// api.lineLogin: the unauthenticated LIFF-token → Supabase-session exchange.

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "",
    json: () => Promise.resolve(body),
  } as Response;
}

afterEach(() => vi.unstubAllGlobals());

describe("api.lineLogin", () => {
  it("POSTs the id token and returns the minted session", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(200, { access_token: "acc", refresh_token: "ref" }));
    vi.stubGlobal("fetch", fetchMock);
    const session = await api.lineLogin("line-id-token");
    expect(session).toEqual({ access_token: "acc", refresh_token: "ref" });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/line",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token: "line-id-token" }),
      })
    );
  });

  it("throws ApiError carrying the status and server detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(401, { detail: "LINE ID token is invalid or expired." }))
    );
    const err = await api.lineLogin("stale").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(401);
    expect(err.message).toBe("LINE ID token is invalid or expired.");
  });

  it("falls back to a generic message when the error body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        json: () => Promise.reject(new Error("not json")),
      } as unknown as Response)
    );
    const err = await api.lineLogin("tok").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(503);
    expect(err.message).toContain("503");
  });
});
