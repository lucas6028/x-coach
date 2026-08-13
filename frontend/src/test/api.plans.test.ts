import { describe, it, expect, vi, afterEach } from "vitest";
import { api } from "../api";

function mockFetch(body: unknown, ok = true, status = 200, statusText = "OK") {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok,
    status,
    statusText,
    json: async () => body,
  } as Response);
}

/** The single fetch call this test made, as `[url, init]`. */
function lastCall() {
  return vi.mocked(globalThis.fetch).mock.calls[0] as [string, RequestInit | undefined];
}

afterEach(() => vi.restoreAllMocks());

describe("api.planTemplates", () => {
  it("GETs the public template catalog and unwraps it", async () => {
    mockFetch({ templates: [{ key: "quick_core", name: "Quick", description: "", items: [] }] });
    const templates = await api.planTemplates();
    expect(templates).toHaveLength(1);
    expect(lastCall()[0]).toBe("/api/plans/templates");
  });
});

describe("api.listPlans", () => {
  it("unwraps the plans array", async () => {
    mockFetch({ plans: [{ id: "p1" }] });
    expect(await api.listPlans()).toEqual([{ id: "p1" }]);
    expect(lastCall()[0]).toBe("/api/plans");
  });
});

describe("api.getPlan", () => {
  it("encodes the id into the path", async () => {
    mockFetch({ id: "p 1" });
    await api.getPlan("p 1");
    expect(lastCall()[0]).toBe("/api/plans/p%201");
  });
});

describe("api.createPlan", () => {
  it("POSTs the body as JSON", async () => {
    mockFetch({ id: "p1", items: [] });
    await api.createPlan({ name: "Week", template_key: "quick_core" });
    const [url, init] = lastCall();
    expect(url).toBe("/api/plans");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ name: "Week", template_key: "quick_core" });
  });

  it("surfaces the server's `detail` rather than the bare status line", async () => {
    // The plan endpoints answer 400 with something the user can act on ("Unknown movement
    // 'Burpee'."); swallowing it would leave the UI showing "400 Bad Request".
    mockFetch({ detail: "Unknown movement 'Burpee'." }, false, 400, "Bad Request");
    await expect(api.createPlan({ name: "x" })).rejects.toThrow("Unknown movement 'Burpee'.");
  });

  it("falls back to the status line when the body carries no detail", async () => {
    mockFetch({}, false, 500, "Internal Server Error");
    await expect(api.createPlan({ name: "x" })).rejects.toThrow(/500/);
  });
});

describe("api.updatePlan", () => {
  it("PATCHes only the given fields", async () => {
    mockFetch({ id: "p1" });
    await api.updatePlan("p1", { name: "Renamed" });
    const [url, init] = lastCall();
    expect(url).toBe("/api/plans/p1");
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(String(init?.body))).toEqual({ name: "Renamed" });
  });
});

describe("api.deletePlan", () => {
  it("DELETEs without a body", async () => {
    mockFetch({ deleted: 1 });
    expect(await api.deletePlan("p1")).toEqual({ deleted: 1 });
    const [url, init] = lastCall();
    expect(url).toBe("/api/plans/p1");
    expect(init?.method).toBe("DELETE");
    expect(init?.body).toBeUndefined();
    // No Content-Type on a bodyless request — sending one would describe a payload that isn't there.
    expect(init?.headers).not.toHaveProperty("Content-Type");
  });
});

describe("api.startPlan", () => {
  it("POSTs to the start path", async () => {
    mockFetch({ id: "p1", items: [] });
    await api.startPlan("p1");
    const [url, init] = lastCall();
    expect(url).toBe("/api/plans/p1/start");
    expect(init?.method).toBe("POST");
  });
});

describe("plan item endpoints", () => {
  it("addPlanItem POSTs under the plan", async () => {
    mockFetch({ id: "i1" });
    await api.addPlanItem("p1", { day_index: 2, movement: "Row", sets: 4, reps: 8 });
    const [url, init] = lastCall();
    expect(url).toBe("/api/plans/p1/items");
    expect(JSON.parse(String(init?.body))).toEqual({
      day_index: 2,
      movement: "Row",
      sets: 4,
      reps: 8,
    });
  });

  it("updatePlanItem PATCHes the completion and the analysis link together", async () => {
    mockFetch({ id: "i1" });
    await api.updatePlanItem("p1", "i1", { completed: true, analysis_id: "an-1" });
    const [url, init] = lastCall();
    expect(url).toBe("/api/plans/p1/items/i1");
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(String(init?.body))).toEqual({ completed: true, analysis_id: "an-1" });
  });

  it("deletePlanItem encodes both ids", async () => {
    mockFetch({ deleted: 1 });
    await api.deletePlanItem("p 1", "i/1");
    expect(lastCall()[0]).toBe("/api/plans/p%201/items/i%2F1");
  });
});
