import { describe, it, expect, vi, afterEach } from "vitest";
import { api, ChatError, UploadLimitError, type ChatContext, type ChatMessage } from "../api";

function mockFetch(body: unknown, ok = true, status = 200) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok,
    status,
    statusText: ok ? "OK" : "Not Found",
    json: async () => body,
  } as Response);
}

// Mock fetch with a real ReadableStream body carrying the given (already byte-splittable) chunks,
// so the client's frame-reassembly across chunk boundaries is exercised for real. Shared by every
// describe block that exercises the SSE stream (chatStream itself and the tool/reset frame tests).
function mockStream(chunks: string[]) {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const enc = new TextEncoder();
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
  return vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true,
    status: 200,
    statusText: "OK",
    body,
  } as unknown as Response);
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

describe("api.adminStatus", () => {
  afterEach(() => vi.restoreAllMocks());

  it("GETs the admin status endpoint and returns the flag", async () => {
    mockFetch({ is_admin: true });
    const result = await api.adminStatus();
    expect(result).toEqual({ is_admin: true });
    expect(fetch).toHaveBeenCalledWith("/api/admin/status");
  });

  it("throws on non-ok responses", async () => {
    mockFetch({}, false, 401);
    await expect(api.adminStatus()).rejects.toThrow("401");
  });
});

describe("api.getAdminSettings", () => {
  afterEach(() => vi.restoreAllMocks());

  it("GETs the admin settings endpoint and returns the parsed payload", async () => {
    const body = { effective: { rag_kg: { rag_top_k: 5 } }, defaults: {} };
    mockFetch(body);
    const result = await api.getAdminSettings();
    expect(result).toEqual(body);
    expect(fetch).toHaveBeenCalledWith("/api/admin/settings");
  });

  it("throws on non-ok responses", async () => {
    mockFetch({}, false, 403);
    await expect(api.getAdminSettings()).rejects.toThrow("403");
  });
});

describe("api.updateAdminSettings", () => {
  afterEach(() => vi.restoreAllMocks());

  it("PUTs the payload and returns the new settings", async () => {
    const body = { effective: { rag_kg: { rag_top_k: 9 } }, defaults: {} };
    const spy = mockFetch(body);
    const result = await api.updateAdminSettings({ rag_top_k: 9 });
    expect(result).toEqual(body);
    expect(spy.mock.calls[0][0]).toBe("/api/admin/settings");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body as string)).toEqual({ rag_top_k: 9 });
  });

  it("throws on non-ok responses (e.g. 422 validation)", async () => {
    mockFetch({}, false, 422);
    await expect(api.updateAdminSettings({ rag_top_k: 99 })).rejects.toThrow("422");
  });
});

describe("api.listAdminUsers", () => {
  afterEach(() => vi.restoreAllMocks());

  it("GETs the admin users endpoint and returns the parsed payload", async () => {
    const body = { users: [{ id: "u1", email: "a@x.com", is_admin: true }] };
    mockFetch(body);
    const result = await api.listAdminUsers();
    expect(result).toEqual(body);
    expect(fetch).toHaveBeenCalledWith("/api/admin/users");
  });

  it("throws on non-ok responses", async () => {
    mockFetch({}, false, 403);
    await expect(api.listAdminUsers()).rejects.toThrow("403");
  });
});

describe("api.getAdminOverview", () => {
  afterEach(() => vi.restoreAllMocks());

  it("GETs the admin overview endpoint and returns the parsed payload", async () => {
    const body = { auth_configured: true, total_users: 4, total_analyses: 12, stores: {} };
    mockFetch(body);
    const result = await api.getAdminOverview();
    expect(result).toEqual(body);
    expect(fetch).toHaveBeenCalledWith("/api/admin/overview");
  });

  it("throws on non-ok responses", async () => {
    mockFetch({}, false, 403);
    await expect(api.getAdminOverview()).rejects.toThrow("403");
  });
});

describe("api.setUserRole", () => {
  afterEach(() => vi.restoreAllMocks());

  it("PUTs the make_admin flag to the per-user role endpoint", async () => {
    const spy = mockFetch({ ok: true });
    const result = await api.setUserRole("u2", true);
    expect(result).toEqual({ ok: true });
    expect(spy.mock.calls[0][0]).toBe("/api/admin/users/u2/role");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body as string)).toEqual({ make_admin: true });
  });

  it("URL-encodes the user id and can revoke", async () => {
    const spy = mockFetch({ ok: true });
    await api.setUserRole("a b/c", false);
    expect(spy.mock.calls[0][0]).toBe("/api/admin/users/a%20b%2Fc/role");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ make_admin: false });
  });

  it("throws on non-ok responses (e.g. 400 self-demote)", async () => {
    mockFetch({}, false, 400);
    await expect(api.setUserRole("u1", false)).rejects.toThrow("400");
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

  it("omits the movement param when none is given", async () => {
    mockFetch({ matched_nodes: [] });
    await api.graph("x");
    expect(fetch).toHaveBeenCalledWith("/api/knowledge/graph?query=x");
  });

  it("appends the movement scope when provided", async () => {
    mockFetch({ matched_nodes: [] });
    await api.graph("knee valgus", "Squat");
    expect(fetch).toHaveBeenCalledWith(
      "/api/knowledge/graph?query=knee%20valgus&movement=Squat"
    );
  });
});

describe("api.movementFaults", () => {
  afterEach(() => vi.restoreAllMocks());

  it("URL-encodes the movement (spaces/hyphens) in the faults query", async () => {
    mockFetch({ movement: "Overhead Press", faults: [] });
    await api.movementFaults("Overhead Press");
    expect(fetch).toHaveBeenCalledWith("/api/knowledge/faults?movement=Overhead%20Press");
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

describe("api.deleteAnalysis", () => {
  afterEach(() => vi.restoreAllMocks());

  it("DELETEs the single analysis endpoint and returns the count", async () => {
    const spy = mockFetch({ deleted: 1 });
    const result = await api.deleteAnalysis("abc123");
    expect(result).toEqual({ deleted: 1 });
    expect(spy.mock.calls[0][0]).toBe("/api/analyses/abc123");
    expect((spy.mock.calls[0][1] as RequestInit).method).toBe("DELETE");
  });

  it("encodes the id into the URL", async () => {
    const spy = mockFetch({ deleted: 1 });
    await api.deleteAnalysis("urn:uuid:1/2 3");
    expect(spy.mock.calls[0][0]).toBe(
      `/api/analyses/${encodeURIComponent("urn:uuid:1/2 3")}`
    );
  });

  it("throws on non-ok responses", async () => {
    mockFetch({}, false, 404);
    await expect(api.deleteAnalysis("abc123")).rejects.toThrow("404");
  });
});

describe("api.analyzeUpload", () => {
  afterEach(() => vi.restoreAllMocks());

  it("POSTs a FormData body and returns the analysis", async () => {
    const spy = mockFetch({ video_id: "upload_1" });
    const file = new File(["video"], "squat.mp4", { type: "video/mp4" });
    const result = await api.analyzeUpload(file, "Squat");
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
    await expect(api.analyzeUpload(file, "Squat")).rejects.toThrow("Invalid video format");
  });

  it("falls back to a generic message when backend detail is absent", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => ({}),
    } as Response);
    const file = new File(["data"], "bad.mp4", { type: "video/mp4" });
    await expect(api.analyzeUpload(file, "Squat")).rejects.toThrow("Analyze failed (500)");
  });
});

describe("api.chatStream", () => {
  afterEach(() => vi.restoreAllMocks());

  const messages: ChatMessage[] = [{ role: "user", content: "why did my knees cave?" }];
  const context: ChatContext = { fault_count: 0, quality: {}, faults: [] };

  function collectHandlers() {
    const deltas: string[] = [];
    let model = "";
    let errorDetail = "";
    return {
      deltas,
      get model() {
        return model;
      },
      get errorDetail() {
        return errorDetail;
      },
      handlers: {
        onDelta: (t: string) => deltas.push(t),
        onDone: (m: string) => {
          model = m;
        },
        onError: (d: string) => {
          errorDetail = d;
        },
      },
    };
  }

  it("POSTs messages + context and streams delta frames then done to the handlers", async () => {
    // A frame deliberately split across two chunks to prove reassembly.
    const spy = mockStream([
      'event: delta\ndata: {"text":"Drive your knees ',
      'out."}\n\nevent: done\ndata: {"model":"m"}\n\n',
    ]);
    const c = collectHandlers();
    await api.chatStream(messages, context, c.handlers);

    expect(c.deltas.join("")).toBe("Drive your knees out.");
    expect(c.model).toBe("m");
    expect(c.errorDetail).toBe("");
    expect(spy.mock.calls[0][0]).toBe("/api/chat");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ messages, context });
  });

  it("includes the chosen model in the request body when provided", async () => {
    const spy = mockStream(['event: done\ndata: {"model":"minimax/minimax-m3"}\n\n']);
    const c = collectHandlers();
    await api.chatStream(messages, context, c.handlers, "minimax/minimax-m3");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({
      messages,
      context,
      model: "minimax/minimax-m3",
    });
  });

  it("omits model from the body when none is chosen (server default)", async () => {
    const spy = mockStream(['event: done\ndata: {"model":"x"}\n\n']);
    const c = collectHandlers();
    await api.chatStream(messages, context, c.handlers);
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ messages, context });
  });

  it("routes an in-band error frame to onError without throwing", async () => {
    mockStream(['event: error\ndata: {"detail":"LLM request failed: reset"}\n\n']);
    const c = collectHandlers();
    await api.chatStream(messages, context, c.handlers);
    expect(c.errorDetail).toContain("reset");
    expect(c.deltas).toHaveLength(0);
  });

  it("ignores malformed/eventless frames and defaults missing fields", async () => {
    mockStream([
      ": keep-alive comment\n\n", // no event line -> ignored
      "event: delta\ndata: {not valid json}\n\n", // unparseable -> ignored
      "event: delta\ndata: {}\n\n", // missing text -> ""
      "event: done\ndata: {}\n\n", // missing model -> ""
      "event: error\ndata: {}\n\n", // missing detail -> generic message
    ]);
    const c = collectHandlers();
    await api.chatStream(messages, context, c.handlers);
    expect(c.deltas).toEqual([""]); // only the `{}` delta reached a handler, defaulted to ""
    expect(c.model).toBe(""); // missing model -> ""
    expect(c.errorDetail).toBe("Chat failed"); // missing detail -> generic fallback
  });

  it("throws a ChatError carrying the HTTP status and backend detail on a pre-flight failure", async () => {
    mockFetch({ detail: "Missing bearer token." }, false, 401);
    const c = collectHandlers();
    await expect(api.chatStream(messages, context, c.handlers)).rejects.toMatchObject({
      name: "ChatError",
      status: 401,
      message: "Missing bearer token.",
    });
    await expect(api.chatStream(messages, context, c.handlers)).rejects.toBeInstanceOf(ChatError);
  });

  it("falls back to a generic message when the pre-flight error body isn't JSON", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      json: async () => {
        throw new Error("not json");
      },
    } as unknown as Response);
    const c = collectHandlers();
    await expect(api.chatStream(messages, context, c.handlers)).rejects.toMatchObject({
      status: 503,
      message: "Chat failed (503)",
    });
  });
});

describe("chat SSE tool frames", () => {
  afterEach(() => vi.restoreAllMocks());

  it("routes tool and reset frames to their handlers", async () => {
    const seen: string[] = [];
    const body = [
      'event: tool\ndata: {"id":0,"name":"kg_query","query":"knee valgus"}\n\n',
      'event: delta\ndata: {"text":"hm"}\n\n',
      "event: reset\ndata: {}\n\n",
      'event: delta\ndata: {"text":"answer"}\n\n',
      'event: done\ndata: {"model":"m"}\n\n',
    ].join("");
    mockStream([body]);
    await api.chatStream([{ role: "user", content: "hi" }], { fault_count: 0, quality: {}, faults: [] }, {
      onDelta: (t) => seen.push(`delta:${t}`),
      onDone: () => seen.push("done"),
      onError: () => seen.push("error"),
      onTool: (_id, n, q) => seen.push(`tool:${n}:${q}`),
      onReset: () => seen.push("reset"),
    });
    expect(seen).toEqual([
      "tool:kg_query:knee valgus",
      "delta:hm",
      "reset",
      "delta:answer",
      "done",
    ]);
  });

  it("falls back to id -1 for a tool frame that omits its id", async () => {
    const seen: number[] = [];
    mockStream(['event: tool\ndata: {"name":"kg_query","query":"x"}\n\n', 'event: done\ndata: {"model":"m"}\n\n']);
    await api.chatStream([{ role: "user", content: "hi" }], { fault_count: 0, quality: {}, faults: [] }, {
      onDelta: () => undefined,
      onDone: () => undefined,
      onError: () => undefined,
      onTool: (id) => seen.push(id),
    });
    expect(seen).toEqual([-1]);
  });

  it("ignores tool and reset frames when the handlers are absent", async () => {
    const seen: string[] = [];
    mockStream([
      'event: tool\ndata: {"name":"kg_query","query":"x"}\n\nevent: reset\ndata: {}\n\nevent: done\ndata: {"model":"m"}\n\n',
    ]);
    await api.chatStream([{ role: "user", content: "hi" }], { fault_count: 0, quality: {}, faults: [] }, {
      onDelta: () => seen.push("delta"),
      onDone: () => seen.push("done"),
      onError: () => seen.push("error"),
    });
    expect(seen).toEqual(["done"]);
  });

  it("strips detail from the followups request", async () => {
    const fetchMock = mockFetch({ questions: [] });
    await api.chatFollowups(
      [{ role: "user", content: "hi" }],
      { fault_count: 0, quality: {}, faults: [], detail: { detections: [] } }
    );
    const sent = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string);
    expect(sent.context.detail).toBeUndefined();
    expect(sent.context.fault_count).toBe(0);
  });

  it("routes tool_done sources to onToolDone, and an empty array when the frame omits them", async () => {
    const seen: Array<[number, unknown]> = [];
    mockStream([
      'event: tool\ndata: {"id":0,"name":"rag_search","query":"ankle"}\n\n',
      'event: tool_done\ndata: {"id":0,"sources":[{"label":"Wikipedia: Squat (exercise)","kind":"encyclopedia"}]}\n\n',
      'event: tool\ndata: {"id":1,"name":"get_analysis","query":"Depth"}\n\n',
      'event: tool_done\ndata: {"id":1}\n\n',
      'event: delta\ndata: {"text":"A"}\n\n',
      'event: done\ndata: {"model":"m"}\n\n',
    ]);
    await api.chatStream([{ role: "user", content: "hi" }], { fault_count: 0, quality: {}, faults: [] }, {
      onDelta: () => undefined,
      onDone: () => undefined,
      onError: () => undefined,
      onToolDone: (id, s) => seen.push([id, s]),
    });
    expect(seen).toEqual([
      [0, [{ label: "Wikipedia: Squat (exercise)", kind: "encyclopedia" }]],
      [1, []],
    ]);
  });

  it("drops a tool_done frame whose id is not a number", async () => {
    // A citation attached to the wrong tool is a worse failure than a citation lost: this feature
    // exists to make provenance trustworthy, so an uncorrelatable frame is discarded outright.
    const seen: unknown[] = [];
    mockStream([
      'event: tool_done\ndata: {"sources":[{"label":"L","kind":"paper"}]}\n\n',
      'event: done\ndata: {"model":"m"}\n\n',
    ]);
    await api.chatStream([{ role: "user", content: "hi" }], { fault_count: 0, quality: {}, faults: [] }, {
      onDelta: () => undefined,
      onDone: () => undefined,
      onError: () => undefined,
      onToolDone: (id, s) => seen.push([id, s]),
    });
    expect(seen).toEqual([]);
  });

  it("strips tools from messages on both chat endpoints", async () => {
    const thread = [
      { role: "user" as const, content: "hi" },
      {
        role: "assistant" as const,
        content: "answer",
        tools: [{ name: "rag_search", query: "ankle", sources: [{ label: "L", kind: "paper" }] }],
      },
    ];
    const ctx = { fault_count: 0, quality: {}, faults: [] };

    // Uses the file's `mockFetch` helper (a `vi.spyOn`, like every other test here) rather than
    // `vi.stubGlobal` directly: a raw global stub isn't undone by this describe block's
    // `afterEach(() => vi.restoreAllMocks())` and would leak a stubbed `fetch` into every test
    // that runs after this one in the file.
    const followupFetch = mockFetch({ questions: [] });
    await api.chatFollowups(thread, ctx);
    const followupBody = JSON.parse((followupFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(followupBody.messages[1].tools).toBeUndefined();
    expect(followupBody.messages[1].content).toBe("answer");

    const streamSpy = mockStream(['event: done\ndata: {"model":"m"}\n\n']);
    await api.chatStream(thread, ctx, {
      onDelta: () => undefined,
      onDone: () => undefined,
      onError: () => undefined,
    });
    const chatBody = JSON.parse((streamSpy.mock.calls[0][1] as RequestInit).body as string);
    expect(chatBody.messages[1].tools).toBeUndefined();
  });
});

describe("api.analyzeUpload thumbnail", () => {
  afterEach(() => vi.restoreAllMocks());

  it("appends the thumbnail when one was captured", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ video_id: "v" }), { status: 200 }));
    const file = new File(["v"], "squat.mp4", { type: "video/mp4" });
    await api.analyzeUpload(file, "Squat", new Blob(["jpeg"], { type: "image/jpeg" }));
    const form = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(form.get("thumbnail")).toBeInstanceOf(Blob);
  });

  it("omits the field when capture failed", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ video_id: "v" }), { status: 200 }));
    const file = new File(["v"], "squat.mp4", { type: "video/mp4" });
    await api.analyzeUpload(file, "Squat", null);
    const form = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(form.get("thumbnail")).toBeNull();
  });
});

describe("api.uploadMedia", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches the ownership-checked URL endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({ video_url: "u", thumbnail_url: "t", expires_in: 3600 }),
        { status: 200 }
      )
    );
    const media = await api.uploadMedia("upload_a");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/uploads/upload_a/url");
    expect(media.video_url).toBe("u");
  });
});

describe("api.uploadMediaBatch", () => {
  afterEach(() => vi.restoreAllMocks());

  it("posts every id and unwraps items", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({ items: { a: { video_url: "u", thumbnail_url: "t" } }, expires_in: 3600 }),
        { status: 200 }
      )
    );
    const items = await api.uploadMediaBatch(["a", "b"]);
    expect(JSON.parse(fetchMock.mock.calls[0][1]?.body as string)).toEqual({
      video_ids: ["a", "b"],
    });
    expect(items.a.video_url).toBe("u");
  });

  it("short-circuits an empty list without a request", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    expect(await api.uploadMediaBatch([])).toEqual({});
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("api.chatFollowups", () => {
  afterEach(() => vi.restoreAllMocks());

  const messages: ChatMessage[] = [
    { role: "user", content: "why did my knees cave?" },
    { role: "assistant", content: "Drive your knees out." },
  ];
  const context: ChatContext = { fault_count: 0, quality: {}, faults: [] };

  it("POSTs the thread + context and returns the questions", async () => {
    const spy = mockFetch({ questions: ["Widen my stance?", "Go lower?"] });
    const qs = await api.chatFollowups(messages, context);
    expect(qs).toEqual(["Widen my stance?", "Go lower?"]);
    expect(spy.mock.calls[0][0]).toBe("/api/chat/followups");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ messages, context });
  });

  it("includes the chosen model when provided", async () => {
    const spy = mockFetch({ questions: [] });
    await api.chatFollowups(messages, context, "minimax/minimax-m3");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({
      messages,
      context,
      model: "minimax/minimax-m3",
    });
  });

  it("returns [] on a non-ok response (best-effort, never throws)", async () => {
    mockFetch({}, false, 503);
    await expect(api.chatFollowups(messages, context)).resolves.toEqual([]);
  });

  it("defaults to [] when the body has no questions field", async () => {
    mockFetch({});
    await expect(api.chatFollowups(messages, context)).resolves.toEqual([]);
  });
});

describe("api.conversations", () => {
  afterEach(() => vi.restoreAllMocks());

  it("getConversation GETs the per-video thread and returns it parsed", async () => {
    const thread = { video_id: "vid", messages: [{ role: "user", content: "hi" }] };
    const spy = mockFetch(thread);
    const result = await api.getConversation("vid");
    expect(result).toEqual(thread);
    expect(spy.mock.calls[0][0]).toBe("/api/conversations/vid");
  });

  it("putConversation PUTs the thread body (chips default to empty when omitted)", async () => {
    const spy = mockFetch({ video_id: "vid", messages: [] });
    const msgs: ChatMessage[] = [{ role: "user", content: "why?" }];
    await api.putConversation("vid", msgs);
    expect(spy.mock.calls[0][0]).toBe("/api/conversations/vid");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body as string)).toEqual({ messages: msgs, followups: [] });
  });

  it("putConversation PUTs the followup chips when passed", async () => {
    const spy = mockFetch({ video_id: "vid", messages: [] });
    const msgs: ChatMessage[] = [{ role: "user", content: "why?" }];
    const fups = ["Should I widen my stance?", "How low should I go?"];
    await api.putConversation("vid", msgs, fups);
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ messages: msgs, followups: fups });
  });

  it("putConversation throws on a non-ok response", async () => {
    mockFetch({}, false, 500);
    await expect(api.putConversation("vid", [])).rejects.toThrow(/500/);
  });
});

describe("upload limit errors", () => {
  afterEach(() => vi.restoreAllMocks());

  it("turns a 413 quota body into a typed error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 413,
      json: async () => ({ detail: { code: "storage_quota_exceeded", used_mb: 480, limit_mb: 500 } }),
    } as Response);
    const err = await api
      .analyzePose("Squat", { metadata: {}, frames: [] } as never, new Blob(["v"]))
      .catch((e) => e);
    expect(err).toBeInstanceOf(UploadLimitError);
    expect(err.code).toBe("storage_quota_exceeded");
    expect(err.usedMb).toBe(480);
    expect(err.limitMb).toBe(500);
  });

  it("turns a 413 file-size body into a typed error with no used figure", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 413,
      json: async () => ({ detail: { code: "upload_too_large", limit_mb: 100 } }),
    } as Response);
    const err = await api
      .analyzePose("Squat", { metadata: {}, frames: [] } as never, new Blob(["v"]))
      .catch((e) => e);
    expect(err).toBeInstanceOf(UploadLimitError);
    expect(err.code).toBe("upload_too_large");
    expect(err.usedMb).toBeNull();
  });

  it("leaves a non-413 failure as a plain Error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ detail: "boom" }),
    } as Response);
    const err = await api
      .analyzePose("Squat", { metadata: {}, frames: [] } as never, new Blob(["v"]))
      .catch((e) => e);
    expect(err).not.toBeInstanceOf(UploadLimitError);
    expect(err.message).toBe("boom");
  });

  // Plan constraint: "A 413 the parser does not recognise (a reverse proxy's own error body,
  // say) must fall through to the generic error path rather than being mislabelled as a quota
  // problem." A body without one of the two known codes is exactly that case.
  it("leaves an unrecognised 413 as a plain Error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 413,
      json: async () => ({ detail: "Request Entity Too Large" }),
    } as Response);
    const err = await api
      .analyzePose("Squat", { metadata: {}, frames: [] } as never, new Blob(["v"]))
      .catch((e) => e);
    expect(err).not.toBeInstanceOf(UploadLimitError);
    expect(err.message).toBe("Request Entity Too Large");
  });
});
