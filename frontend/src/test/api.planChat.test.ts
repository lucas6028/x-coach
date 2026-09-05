import { describe, it, expect, vi, afterEach } from "vitest";
import { api, ChatError, type ChatMessage, type LiveToolRun, type Plan } from "../api";

// Same stream/fetch stubs as api.test.ts — the plan endpoint shares the transport, so the tests
// that matter here are the ones about the frames only IT emits: `tool_done.plan` and `done.plan_id`.
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

function mockFetch(body: unknown, ok = true, status = 200) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok,
    status,
    statusText: "",
    json: async () => body,
  } as Response);
}

const messages: ChatMessage[] = [{ role: "user", content: "three full-body days" }];

const plan: Plan = {
  id: "p9",
  name: "Full-body week",
  notes: null,
  template_key: null,
  started_at: null,
  created_at: "2026-09-01T00:00:00Z",
  updated_at: "2026-09-01T00:00:00Z",
  items: [],
};

function collect() {
  const deltas: string[] = [];
  const tools: LiveToolRun[] = [];
  const toolDone: Array<[number, { plan?: Plan }]> = [];
  const plans: Plan[] = [];
  const state = { done: null as { model: string; plan_id?: string } | null, error: "", resets: 0 };
  return {
    deltas,
    tools,
    toolDone,
    plans,
    state,
    handlers: {
      onDelta: (t: string) => deltas.push(t),
      onTool: (run: LiveToolRun) => tools.push(run),
      onToolDone: (id: number, extra: { plan?: Plan }) => toolDone.push([id, extra]),
      onPlan: (p: Plan) => plans.push(p),
      onReset: () => {
        state.resets++;
      },
      onDone: (info: { model: string; plan_id?: string }) => {
        state.done = info;
      },
      onError: (d: string) => {
        state.error = d;
      },
    },
  };
}

afterEach(() => vi.restoreAllMocks());

describe("api.planChatStream — request", () => {
  it("POSTs the lean thread, the plan id, the model and the language", async () => {
    const spy = mockStream(['event: done\ndata: {"model":"m"}\n\n']);
    const c = collect();
    // `tools` is a rendering/persistence record; it must not be re-uploaded every turn.
    const withTools: ChatMessage[] = [
      { role: "assistant", content: "done", tools: [{ name: "add_item", query: "Squat" }] },
      { role: "user", content: "one more" },
    ];
    await api.planChatStream(withTools, "p9", c.handlers, { model: "x/y", lang: "zh-Hant" });

    expect(spy.mock.calls[0][0]).toBe("/api/plans/chat");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      messages: [
        { role: "assistant", content: "done" },
        { role: "user", content: "one more" },
      ],
      plan_id: "p9",
      model: "x/y",
      lang: "zh-Hant",
    });
  });

  it("sends a null plan_id in builder mode and defaults model/lang", async () => {
    const spy = mockStream(['event: done\ndata: {"model":"m"}\n\n']);
    const c = collect();
    await api.planChatStream(messages, null, c.handlers);
    expect(JSON.parse((spy.mock.calls[0][1] as RequestInit).body as string)).toEqual({
      messages,
      plan_id: null,
      model: null,
      lang: "zh-Hant",
    });
  });
});

describe("api.planChatStream — frames", () => {
  it("streams deltas and reports the model on done", async () => {
    // Split mid-frame to prove the buffer reassembles across chunks.
    mockStream(['event: delta\ndata: {"text":"Day 1: ', 'Squat."}\n\nevent: done\ndata: {"model":"m"}\n\n']);
    const c = collect();
    await api.planChatStream(messages, null, c.handlers);
    expect(c.deltas.join("")).toBe("Day 1: Squat.");
    expect(c.state.done).toEqual({ model: "m" });
  });

  it("hands the fresh plan to BOTH onPlan and the tool_done extra, and marks the run pending", async () => {
    mockStream([
      'event: tool\ndata: {"id":0,"name":"create_plan","query":"Full-body week"}\n\n',
      `event: tool_done\ndata: ${JSON.stringify({ id: 0, plan })}\n\n`,
      'event: done\ndata: {"model":"m","plan_id":"p9"}\n\n',
    ]);
    const c = collect();
    await api.planChatStream(messages, null, c.handlers);

    expect(c.tools).toEqual([{ id: 0, name: "create_plan", query: "Full-body week", pending: true }]);
    expect(c.toolDone).toEqual([[0, { plan }]]);
    expect(c.plans).toEqual([plan]);
    expect(c.state.done).toEqual({ model: "m", plan_id: "p9" });
  });

  it("still delivers the plan when the tool_done frame carries no id", async () => {
    // The id guard exists to avoid mis-attributing a CITATION. A plan is not a citation: the write
    // already happened server-side, so dropping the frame would strand the user's real plan.
    mockStream([
      `event: tool_done\ndata: ${JSON.stringify({ plan })}\n\n`,
      'event: done\ndata: {"model":"m"}\n\n',
    ]);
    const c = collect();
    await api.planChatStream(messages, null, c.handlers);
    expect(c.plans).toEqual([plan]);
    expect(c.toolDone).toEqual([]); // uncorrelatable: no run to settle
  });

  it("omits plan and plan_id from the handler payloads for a read-only tool", async () => {
    mockStream([
      'event: tool\ndata: {"id":1,"name":"get_plan","query":""}\n\n',
      'event: tool_done\ndata: {"id":1}\n\n',
      'event: reset\ndata: {}\n\n',
      'event: done\ndata: {"model":"m"}\n\n',
    ]);
    const c = collect();
    await api.planChatStream(messages, "p9", c.handlers);
    expect(c.toolDone).toEqual([[1, {}]]);
    expect(c.plans).toEqual([]);
    expect(c.state.resets).toBe(1);
    expect(c.state.done).toEqual({ model: "m" });
  });

  it("routes an in-band error frame to onError without throwing", async () => {
    mockStream(['event: error\ndata: {"detail":"LLM request failed"}\n\n']);
    const c = collect();
    await api.planChatStream(messages, null, c.handlers);
    expect(c.state.error).toBe("LLM request failed");
    expect(c.state.done).toBeNull();
  });
});

describe("api.planChatStream — pre-flight failures", () => {
  it("throws a ChatError carrying the status and the backend detail", async () => {
    mockFetch({ detail: "Missing bearer token." }, false, 401);
    const c = collect();
    await expect(api.planChatStream(messages, null, c.handlers)).rejects.toMatchObject({
      name: "ChatError",
      status: 401,
      message: "Missing bearer token.",
    });
  });

  it("falls back to a generic message when the error body is not JSON", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      json: async () => {
        throw new Error("not json");
      },
    } as unknown as Response);
    const c = collect();
    const err = await api.planChatStream(messages, null, c.handlers).catch((e) => e);
    expect(err).toBeInstanceOf(ChatError);
    expect(err).toMatchObject({ status: 503, message: "Chat failed (503)" });
  });
});
