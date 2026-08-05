import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { ChatError, type Analysis } from "../api";

// Mutable auth state so tests can flip signed-in / signed-out (hoisted for the vi.mock factory).
const h = vi.hoisted(() => ({
  auth: { configured: true, user: { id: "u1" } as { id: string } | null },
  chatStream: vi.fn(),
  chatFollowups: vi.fn(),
  health: vi.fn(),
  getConversation: vi.fn(),
  putConversation: vi.fn(),
}));

vi.mock("../lib/auth", () => ({ useAuth: () => h.auth }));
// Preserve the real module (ChatError, types) and only stub the network methods, so the
// component's `instanceof ChatError` status check exercises the genuine error type.
vi.mock("../api", async (importActual) => {
  const actual = await importActual<typeof import("../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      chatStream: h.chatStream,
      chatFollowups: h.chatFollowups,
      health: h.health,
      getConversation: h.getConversation,
      putConversation: h.putConversation,
    },
  };
});

// Drive the component's stream handlers with a fixed reply, then complete — the streaming analogue
// of a resolved `{ reply }`. Deltas are chunked to exercise incremental accumulation.
function streamReply(reply: string, model = "m") {
  return async (
    _messages: unknown,
    _context: unknown,
    handlers: { onDelta: (t: string) => void; onDone: (m: string) => void }
  ) => {
    const mid = Math.ceil(reply.length / 2);
    handlers.onDelta(reply.slice(0, mid));
    handlers.onDelta(reply.slice(mid));
    handlers.onDone(model);
  };
}

import CoachTray from "../components/CoachTray";

const analysis = {
  video_id: "v1",
  view: { view_type: "front", view_confidence: 0.9 },
  quality: { lower_body_visibility_mean: 0.9 },
  detections: [
    { fault_id: "f1", fault_name: "knees_inward", phase: "descent", severity: 0.8, start_time: 1, end_time: 2, evidence: { primary_label: "knee valgus ratio", primary_value: 0.82 } },
  ],
  retrievals: [
    {
      fault_id: "f1",
      retrieval_mode: "kg",
      context: { results: [{ summary: { corrections: [{ node_id: "knees out", label: "z" }] } }] },
    },
  ],
} as unknown as Analysis;

function renderTray() {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <CoachTray analysis={analysis} currentTime={0} onSeek={vi.fn()} />
      </I18nProvider>
    </MemoryRouter>
  );
}

// Types a follow-up into the composer and sends it — the common "user acts" prelude shared by tests
// that need to observe in-flight stream state (deliberately not awaited by some callers; see below).
async function sendMessage(text: string) {
  await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), text);
  await userEvent.click(screen.getByLabelText(/Send message/i));
}

describe("CoachTray — follow-up chat", () => {
  beforeEach(() => {
    h.auth = { configured: true, user: { id: "u1" } };
    h.chatStream.mockReset();
    h.chatFollowups.mockReset();
    h.chatFollowups.mockResolvedValue([]); // no follow-up chips by default
    h.health.mockReset();
    h.health.mockResolvedValue({ status: "ok", chat_configured: true });
    h.getConversation.mockReset();
    h.getConversation.mockResolvedValue({ video_id: "v1", messages: [] }); // no saved thread by default
    h.putConversation.mockReset();
    h.putConversation.mockResolvedValue(undefined);
  });

  it("sends a grounded user turn and renders the streamed coach reply in the same thread", async () => {
    h.chatStream.mockImplementation(streamReply("Drive your knees out over your toes."));
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "why did my knees cave?");
    await userEvent.click(screen.getByLabelText(/Send message/i));

    expect(h.chatStream).toHaveBeenCalledTimes(1);
    const [messages, context] = h.chatStream.mock.calls[0];
    expect(messages.at(-1)).toEqual({ role: "user", content: "why did my knees cave?" });
    // Grounding blob carries the detected fault + its retrieved corrective cue.
    expect(context.faults[0].fault_name).toBe("knees_inward");
    expect(context.faults[0].corrections).toContain("knees out");

    expect(await screen.findByText("Drive your knees out over your toes.")).toBeInTheDocument();

    // The completed turn (user + assistant) is persisted for replay.
    const putCall = h.putConversation.mock.calls[0];
    expect(putCall[0]).toBe("v1");
    expect(putCall[1]).toEqual([
      { role: "user", content: "why did my knees cave?" },
      { role: "assistant", content: "Drive your knees out over your toes." },
    ]);
  });

  it("fetches follow-up chips after an answer (fire-and-forget) and sends one when clicked", async () => {
    h.chatStream
      .mockImplementationOnce(streamReply("Drive your knees out."))
      .mockImplementationOnce(streamReply("Aim for hip crease below the knee."));
    // The follow-up chips come from a SEPARATE request fired after the answer commits.
    h.chatFollowups.mockResolvedValueOnce(["Should I widen my stance?", "How low should I go?"]);
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "why did my knees cave?");
    await userEvent.click(screen.getByLabelText(/Send message/i));

    // The answer renders immediately; the chips arrive a beat later from chatFollowups.
    expect(await screen.findByText("Drive your knees out.")).toBeInTheDocument();
    const chip = await screen.findByRole("button", { name: /Should I widen my stance/i });
    expect(screen.getByRole("button", { name: /How low should I go/i })).toBeInTheDocument();
    // The follow-up request received the committed thread (ending on the assistant answer).
    const [fuMessages] = h.chatFollowups.mock.calls[0];
    expect(fuMessages.at(-1)).toEqual({ role: "assistant", content: "Drive your knees out." });

    // Once the chips land they're re-persisted with the thread, so a reload restores them too. The
    // first PUT (on `done`) clears the previous answer's chips; the second carries this answer's.
    await vi.waitFor(() => expect(h.putConversation).toHaveBeenCalledTimes(2));
    expect(h.putConversation.mock.calls[1][2]).toEqual([
      "Should I widen my stance?",
      "How low should I go?",
    ]);

    // Clicking a suggestion sends it as the next user turn (same behaviour as a starter chip).
    await userEvent.click(chip);
    expect(h.chatStream).toHaveBeenCalledTimes(2);
    const [messages] = h.chatStream.mock.calls[1];
    expect(messages.at(-1)).toEqual({ role: "user", content: "Should I widen my stance?" });
    expect(await screen.findByText("Aim for hip crease below the knee.")).toBeInTheDocument();
    // The previous answer's chips are cleared once the new turn is sent.
    expect(screen.queryByRole("button", { name: /How low should I go/i })).not.toBeInTheDocument();
  });

  it("stays usable when restoring a saved thread fails", async () => {
    h.getConversation.mockRejectedValue(new Error("network"));
    renderTray();
    // The restore rejection is swallowed — the composer still renders, no crash, empty thread.
    expect(await screen.findByPlaceholderText(/Ask a follow-up/i)).toBeInTheDocument();
  });

  it("restores a saved thread on load (history replay)", async () => {
    h.getConversation.mockResolvedValue({
      video_id: "v1",
      messages: [
        { role: "user", content: "earlier question" },
        { role: "assistant", content: "earlier answer" },
      ],
    });
    renderTray();

    expect(await screen.findByText("earlier answer")).toBeInTheDocument();
    expect(screen.getByText("earlier question")).toBeInTheDocument();
    expect(h.getConversation).toHaveBeenCalledWith("v1");
  });

  it("restores the saved follow-up chips on load (not just the answer)", async () => {
    h.getConversation.mockResolvedValue({
      video_id: "v1",
      messages: [
        { role: "user", content: "why did my knees cave?" },
        { role: "assistant", content: "Drive your knees out." },
      ],
      followups: ["Should I widen my stance?", "How low should I go?"],
    });
    renderTray();

    // The persisted chips come back as clickable suggestions under the restored answer — the whole
    // point of persisting them (a reload used to leave the response with no chips).
    expect(await screen.findByText("Drive your knees out.")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Should I widen my stance/i })
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /How low should I go/i })).toBeInTheDocument();
  });

  it("does not persist when a turn fails mid-stream", async () => {
    h.chatStream.mockImplementation(
      async (
        _m: unknown,
        _c: unknown,
        handlers: { onError: (d: string) => void }
      ) => {
        handlers.onError("LLM request failed: reset");
      }
    );
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "why?");
    await userEvent.keyboard("{Enter}");

    await screen.findByRole("alert");
    expect(h.putConversation).not.toHaveBeenCalled();
  });

  it("rolls back and does not persist when the stream ends without done or error", async () => {
    // FIX 1 (whole-branch review): api.chatStream resolves normally the instant the reader drains,
    // regardless of which frame (if any) was last seen -- a dropped connection / proxy idle-timeout
    // looks exactly like success here unless the component checks that `done` actually arrived.
    // Without that check this would commit `{ role: "assistant", content: acc }` (here, a truncated
    // partial answer) and persist it via putConversation.
    h.chatStream.mockImplementation(
      async (_m: unknown, _c: unknown, handlers: { onDelta: (t: string) => void }) => {
        handlers.onDelta("Drive your kne");
        // Resolves WITHOUT ever calling onDone or onError.
      }
    );
    renderTray();

    const box = screen.getByPlaceholderText(/Ask a follow-up/i) as HTMLInputElement;
    await userEvent.type(box, "why?");
    await userEvent.keyboard("{Enter}");

    await screen.findByRole("alert");
    // No assistant turn committed, and the (rolled-back) user turn isn't left orphaned either.
    expect(screen.queryByText("Drive your kne")).not.toBeInTheDocument();
    expect(screen.queryByText("why?")).not.toBeInTheDocument();
    expect(box.value).toBe("why?"); // restored for retry
    expect(h.putConversation).not.toHaveBeenCalled();
  });

  it("sends a starter suggestion directly when its chip is clicked", async () => {
    h.chatStream.mockImplementation(streamReply("Fix your knees first."));
    renderTray();

    await userEvent.click(screen.getByRole("button", { name: /What should I fix first/i }));

    expect(h.chatStream).toHaveBeenCalledTimes(1);
    const [messages] = h.chatStream.mock.calls[0];
    expect(messages.at(-1)).toEqual({ role: "user", content: "What should I fix first?" });
    expect(await screen.findByText("Fix your knees first.")).toBeInTheDocument();
  });

  it("shows an error and rolls back the optimistic turn when the coach is unreachable", async () => {
    h.chatStream.mockRejectedValue(new Error("network"));
    renderTray();

    const box = screen.getByPlaceholderText(/Ask a follow-up/i) as HTMLInputElement;
    await userEvent.type(box, "hi");
    await userEvent.keyboard("{Enter}"); // exercises Enter-to-send

    expect(await screen.findByRole("alert")).toHaveTextContent(/try again/i);
    // Failed user turn rolled back (no orphaned "You" bubble) and text restored so retry can't dup.
    expect(screen.queryByText(/^You$/)).not.toBeInTheDocument();
    expect(box.value).toBe("hi");
  });

  it("tells the user to sign in again on a 401 (expired session)", async () => {
    h.chatStream.mockRejectedValue(new ChatError("Missing bearer token.", 401));
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "hi");
    await userEvent.keyboard("{Enter}");

    expect(await screen.findByRole("alert")).toHaveTextContent(/sign in again/i);
  });

  it("rolls back the optimistic turn and shows an error on an in-band stream error", async () => {
    // The stream opens (200) but the LLM provider fails mid-flight: the client delivers an `error` frame
    // via onError rather than throwing. The partial turn must be discarded, not left orphaned.
    h.chatStream.mockImplementation(
      async (
        _m: unknown,
        _c: unknown,
        handlers: { onDelta: (t: string) => void; onError: (d: string) => void }
      ) => {
        handlers.onDelta("Drive your kne");
        handlers.onError("LLM request failed: reset");
      }
    );
    renderTray();

    const box = screen.getByPlaceholderText(/Ask a follow-up/i) as HTMLInputElement;
    await userEvent.type(box, "why?");
    await userEvent.keyboard("{Enter}");

    expect(await screen.findByRole("alert")).toHaveTextContent(/try again/i);
    // The partial assistant text is gone and the user turn rolled back; input restored for retry.
    expect(screen.queryByText("Drive your kne")).not.toBeInTheDocument();
    expect(screen.queryByText("why?")).not.toBeInTheDocument();
    expect(box.value).toBe("why?");
  });

  it("keeps a working composer when the one-shot health check fails transiently", async () => {
    h.health.mockRejectedValue(new Error("blip"));
    renderTray();

    const box = await screen.findByPlaceholderText(/Ask a follow-up/i);
    expect(box).not.toBeDisabled();
  });

  it("shows the feedback but a sign-in composer when signed out", () => {
    h.auth = { configured: true, user: null };
    renderTray();

    // The grounded feedback is still visible…
    expect(screen.getByText("Knee Valgus")).toBeInTheDocument();
    // …but the composer invites sign-in rather than faking an input.
    expect(screen.getByText(/Sign in to chat/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute("href", "/login");
    expect(screen.queryByPlaceholderText(/Ask a follow-up/i)).not.toBeInTheDocument();
  });

  it("falls back to the disabled composer when the server has no LLM key", async () => {
    h.health.mockResolvedValue({ status: "ok", chat_configured: false });
    renderTray();

    const disabled = await screen.findByPlaceholderText(/Ask Lumen/i);
    expect(disabled).toBeDisabled();
    expect(h.chatStream).not.toHaveBeenCalled();
  });

  it("does nothing when Enter is pressed with an empty composer", async () => {
    renderTray();
    const box = await screen.findByPlaceholderText(/Ask a follow-up/i);
    box.focus();
    await userEvent.keyboard("{Enter}");
    expect(h.chatStream).not.toHaveBeenCalled();
  });

  it("scrolls the thread into view once a turn is sent, when scrollTo is available", async () => {
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      writable: true,
      value: scrollTo,
    });

    h.chatStream.mockImplementation(streamReply("Ok."));
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "hi");
    await userEvent.click(screen.getByLabelText(/Send message/i));
    await screen.findByText("Ok.");

    expect(scrollTo).toHaveBeenCalledWith({ top: expect.any(Number) });

    Reflect.deleteProperty(HTMLElement.prototype, "scrollTo");
  });

  it("scrolls again when the follow-up chips land (they arrive after the answer commits)", async () => {
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      writable: true,
      value: scrollTo,
    });

    // Defer the follow-up result so we can measure the scroll count *before* the chips render, then
    // resolve it and assert the chips landing triggered a further scroll (not folded below the view).
    let resolveFollowups!: (qs: string[]) => void;
    h.chatFollowups.mockReturnValueOnce(
      new Promise<string[]>((res) => {
        resolveFollowups = res;
      })
    );
    h.chatStream.mockImplementation(streamReply("Ok."));
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "hi");
    await userEvent.click(screen.getByLabelText(/Send message/i));
    await screen.findByText("Ok.");
    const beforeChips = scrollTo.mock.calls.length;

    resolveFollowups(["Chip one?", "Chip two?"]);
    await screen.findByRole("button", { name: /Chip one/i });
    expect(scrollTo.mock.calls.length).toBeGreaterThan(beforeChips);

    Reflect.deleteProperty(HTMLElement.prototype, "scrollTo");
  });

  it("swallows a follow-up fetch failure without disrupting the committed answer", async () => {
    h.chatStream.mockImplementation(streamReply("Drive your knees out."));
    // The background suggestion fetch rejects — fire-and-forget, so it must not surface an error.
    h.chatFollowups.mockRejectedValueOnce(new Error("network"));
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "why?");
    await userEvent.click(screen.getByLabelText(/Send message/i));

    // The answer is on screen and the failed suggestion fetch is silently ignored — no chips, no alert.
    expect(await screen.findByText("Drive your knees out.")).toBeInTheDocument();
    expect(h.chatFollowups).toHaveBeenCalledTimes(1);
    await Promise.resolve(); // let the rejection settle through the .catch
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("drops stale follow-ups once a newer turn has superseded them", async () => {
    h.chatStream
      .mockImplementationOnce(streamReply("First answer."))
      .mockImplementationOnce(streamReply("Second answer."));
    // Defer the FIRST turn's suggestions so they resolve only after a second turn has been sent.
    let resolveStale!: (qs: string[]) => void;
    h.chatFollowups
      .mockReturnValueOnce(
        new Promise<string[]>((res) => {
          resolveStale = res;
        })
      )
      .mockResolvedValueOnce([]); // the second turn yields no chips
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "first?");
    await userEvent.click(screen.getByLabelText(/Send message/i));
    await screen.findByText("First answer.");

    // A second send bumps the follow-up sequence token before the first fetch resolves.
    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "second?");
    await userEvent.click(screen.getByLabelText(/Send message/i));
    await screen.findByText("Second answer.");

    // The first turn's suggestions finally arrive — but they belong to a superseded turn, so the
    // sequence guard drops them rather than flashing chips under the newer answer.
    resolveStale(["Stale chip?"]);
    await new Promise((r) => setTimeout(r, 0)); // flush the resolved .then through the guard
    expect(screen.queryByRole("button", { name: /Stale chip/i })).not.toBeInTheDocument();
  });

  it("does not auto-scroll when the user has scrolled up to read earlier messages", async () => {
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      writable: true,
      value: scrollTo,
    });

    let resolveFollowups!: (qs: string[]) => void;
    h.chatFollowups.mockReturnValueOnce(
      new Promise<string[]>((res) => {
        resolveFollowups = res;
      })
    );
    h.chatStream.mockImplementation(streamReply("Ok."));
    const { container } = renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "hi");
    await userEvent.click(screen.getByLabelText(/Send message/i));
    await screen.findByText("Ok.");

    // Simulate the user scrolling up, away from the foot (distance-from-bottom well past threshold).
    const scroller = container.querySelector(".overflow-y-auto") as HTMLElement;
    Object.defineProperty(scroller, "scrollHeight", { configurable: true, value: 1000 });
    Object.defineProperty(scroller, "clientHeight", { configurable: true, value: 300 });
    scroller.scrollTop = 0;
    fireEvent.scroll(scroller);

    const beforeChips = scrollTo.mock.calls.length;
    resolveFollowups(["Chip one?", "Chip two?"]);
    await screen.findByRole("button", { name: /Chip one/i });
    // The chips rendered, but the view stayed where the user was reading — no forced scroll.
    expect(scrollTo.mock.calls.length).toBe(beforeChips);

    Reflect.deleteProperty(HTMLElement.prototype, "scrollTo");
  });

  it("shows a named tool record with its label and subject, and keeps it once the answer streams", async () => {
    // Subject is deliberately NOT "knee valgus": the fixture's own FaultCard evidence already
    // renders "knee valgus ratio 0.82" on first paint (lib/retrieval.ts keyEvidence), so that text
    // would satisfy findByText immediately — before the tool ever ran — and this test would pass
    // vacuously (releaseTool never actually needed). A collision-free subject forces the assertion
    // to wait on the real tool line. Uses h.chatStream (like every other test in this file), not
    // vi.spyOn(api, ...) — this file's vi.mock("../api") already makes api.chatStream === h.chatStream,
    // so spying on api.chatStream would replace that reference for every test that runs after this
    // one (nothing in setup.ts restores mocks between tests), leaking this implementation forward.
    let releaseTool!: () => void;
    let releaseDelta!: () => void;
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "kg_query", "zzqury-kg-subject");
      handlers.onToolDone?.(0, []);
      await new Promise<void>((r) => (releaseTool = r)); // assert while the tool is running
      handlers.onDelta("Answer");
      await new Promise<void>((r) => (releaseDelta = r)); // assert while the answer streams
      handlers.onDone("m");
    });
    const { container } = renderTray();
    void sendMessage("why?"); // NOT awaited — the stream is deliberately still open

    // The tool line is up, naming BOTH the tool's label and its subject in one node — asserting on
    // the subject alone would also pass if the label lookup were broken (e.g. hard-coded to the
    // generic fallback), since the subject would still render.
    // Separator now goes through t("chat.tool.sep") (FIX 5, whole-branch review) -- English renders
    // ": " (halfwidth), not the fullwidth "：" that used to be hard-coded regardless of language.
    expect(
      await screen.findByText(/Searching the knowledge graph: zzqury-kg-subject/)
    ).toBeTruthy();

    // onToolDone already fired (with an empty `sources`, as get_analysis always does) and must have
    // cleared `pending` on this run. CoachTray's own thinking dots are suppressed once a tool row
    // exists (toolRuns.length === 0 gate), so the only ".lm-dots" left in the tree would be this
    // row's own pending marker — its absence proves the empty-sources tool_done actually resolved
    // the row instead of leaving it spinning forever (the exact failure this feature prevents).
    expect(container.querySelector(".lm-dots")).toBeNull();

    // Unlike v3, the record is NOT cleared once the answer starts streaming — it is the answer's
    // provenance and belongs beside it (v3.1).
    releaseTool();
    expect(await screen.findByText("Answer")).toBeTruthy();
    expect(screen.getByText(/zzqury-kg-subject/)).toBeTruthy();

    releaseDelta();
  });

  it("falls back to a generic label for a tool it has no i18n string for", async () => {
    // `t()` returns the key itself on a miss (i18n.tsx:1421), so an unguarded lookup would render
    // "chat.tool.something_else" straight into the tray. The subject is deliberately unique gibberish
    // — a bare "x" is a substring of the KnowledgeGraphWidget caption ("Fault → cause → fix"), which
    // let this assertion pass even before the tool line existed.
    let release!: () => void;
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "something_else", "zzqux-subject");
      handlers.onToolDone?.(0, []);
      await new Promise<void>((r) => (release = r));
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    void sendMessage("why?");
    expect(await screen.findByText(/zzqux-subject/)).toBeTruthy();
    expect(screen.queryByText(/chat\.tool\./)).toBeNull();
    release();
  });

  it("discards streamed narration when the server sends reset, but keeps tool records made before it", async () => {
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "kg_query", "zzq-early-subject");
      handlers.onToolDone?.(0, []);
      handlers.onDelta("Let me check.");
      handlers.onReset?.();
      handlers.onTool?.(1, "kg_query", "valgus");
      handlers.onToolDone?.(1, []);
      handlers.onDelta("Real answer");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    expect(await screen.findByText("Real answer")).toBeTruthy();
    expect(screen.queryByText(/Let me check/)).toBeNull();
    // The tool call made BEFORE the reset really happened and really fed the answer — reset only
    // retracts the narration text, not the record of what was looked up.
    expect(screen.getByText(/zzq-early-subject/)).toBeTruthy();
  });

  it("keeps tool records visible after the answer starts streaming", async () => {
    // Live-state assertion (parked, NOT awaited): sendMessage is deliberately left in flight so this
    // is the sole guard against a reintroduced `setToolRuns([])` in `onDelta` — every other new test
    // in this block awaits `sendMessage` and therefore only ever sees committed state. (This test
    // owns the name that best describes it; the sibling below the record surviving into the
    // committed message is a related but distinct guarantee.)
    let release!: () => void;
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "rag_search", "zzq-live-subject");
      handlers.onToolDone?.(0, [{ label: "zzq-Live-Source", kind: "encyclopedia" }]);
      handlers.onDelta("Real ans");
      await new Promise<void>((r) => (release = r)); // assert while still streaming, pre-commit
      handlers.onDone("m");
    });
    const { container } = renderTray();
    void sendMessage("why?"); // NOT awaited — the stream is deliberately still open
    expect(await screen.findByText("Real ans")).toBeTruthy();
    expect(screen.getByText(/zzq-live-subject/)).toBeTruthy();
    // Collapsed by default (v3.2) — the count is what shows; the label is one click away.
    await userEvent.click(screen.getByRole("button", { name: /Sources · 1/ }));
    expect(screen.getByText("zzq-Live-Source")).toBeTruthy();
    // The record sits ABOVE the streamed answer text, matching where it lands on the committed
    // message once the turn commits (nothing should shift at commit time) — the layout decision the
    // whole feature was specified around ("答案上方,預設展開").
    const text = container.textContent ?? "";
    expect(text.indexOf("zzq-live-subject")).toBeGreaterThan(-1);
    expect(text.indexOf("zzq-live-subject")).toBeLessThan(text.indexOf("Real ans"));
    release();
  });

  it("keeps tool records on the committed assistant message once the turn completes", async () => {
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "rag_search", "zzq-ankle-subject");
      handlers.onToolDone?.(0, [{ label: "zzq-Wiki-Source", kind: "encyclopedia" }]);
      handlers.onDelta("Real answer");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    // Committed, so this survives the finally — the whole point of v3.1.
    expect(await screen.findByText("Real answer")).toBeTruthy();
    expect(screen.getByText(/zzq-ankle-subject/)).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: /Sources · 1/ }));
    expect(screen.getByText("zzq-Wiki-Source")).toBeTruthy();
  });

  it("appends successive tool calls instead of replacing them", async () => {
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "kg_query", "zzq-first-subject");
      handlers.onToolDone?.(0, []);
      handlers.onTool?.(1, "rag_search", "zzq-second-subject");
      handlers.onToolDone?.(1, []);
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    expect(await screen.findByText("A")).toBeTruthy();
    expect(screen.getByText(/zzq-first-subject/)).toBeTruthy();
    expect(screen.getByText(/zzq-second-subject/)).toBeTruthy();
  });

  it("counts a kg_query run as concepts and a rag_search run as sources, keyed by kind not tool name", async () => {
    // v3.1's red line, carried through the v3.2 collapse: a knowledge-graph node carries no citation
    // anywhere in the graph, so it must never be counted under the same word as a retrieved paper.
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "kg_query", "zzq-concept-subject");
      handlers.onToolDone?.(0, [{ label: "zzq-Concept-Label", kind: "concept" }]);
      handlers.onTool?.(1, "rag_search", "zzq-paper-subject");
      handlers.onToolDone?.(1, [{ label: "zzq-Paper-Label", kind: "paper" }]);
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    expect(await screen.findByText("A")).toBeTruthy();

    const conceptToggle = screen.getByRole("button", { name: /Knowledge-graph concepts · 1/ });
    const sourceToggle = screen.getByRole("button", { name: /Sources · 1/ });
    await userEvent.click(conceptToggle);
    await userEvent.click(sourceToggle);

    // Each row lists only its own tool's label, under its own count — not the other's.
    const conceptBlock = conceptToggle.parentElement as HTMLElement;
    const sourceBlock = sourceToggle.parentElement as HTMLElement;
    expect(conceptBlock.textContent).toContain("zzq-Concept-Label");
    expect(conceptBlock.textContent).not.toContain("zzq-Paper-Label");
    expect(sourceBlock.textContent).toContain("zzq-Paper-Label");
    expect(sourceBlock.textContent).not.toContain("zzq-Concept-Label");
  });

  it("keeps tool records when the server retracts narration with reset", async () => {
    // reset retracts the model's narration, but the tool calls really happened and really fed the
    // answer — erasing them would misreport the reasoning chain. The tool call fires BEFORE the
    // reset (a realistic round-1-tool, round-2-narrate-then-reset sequence) so that a handler which
    // clears `runs` inside `onReset` is actually caught: clearing an empty list would be unobservable.
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "kg_query", "zzq-early-subject");
      handlers.onToolDone?.(0, []);
      handlers.onDelta("zzq-narration");
      handlers.onReset?.();
      handlers.onTool?.(1, "rag_search", "zzq-kept-subject");
      handlers.onToolDone?.(1, []);
      handlers.onDelta("Real answer");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    expect(await screen.findByText("Real answer")).toBeTruthy();
    expect(screen.queryByText(/zzq-narration/)).toBeNull();
    expect(screen.getByText(/zzq-early-subject/)).toBeTruthy();
    expect(screen.getByText(/zzq-kept-subject/)).toBeTruthy();
  });

  it("persists tool records with the committed turn", async () => {
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "rag_search", "ankle");
      handlers.onToolDone?.(0, [{ label: "zzq-Persisted-Source", kind: "paper" }]);
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    await screen.findByText("A");
    const thread = h.putConversation.mock.calls[0][1] as Array<{ role: string; tools?: unknown[] }>;
    expect(thread[thread.length - 1].tools).toEqual([
      { name: "rag_search", query: "ankle", sources: [{ label: "zzq-Persisted-Source", kind: "paper" }] },
    ]);
  });

  it("restores tool records from a stored conversation", async () => {
    h.getConversation.mockResolvedValue({
      video_id: "v1",
      messages: [
        { role: "user", content: "why?" },
        {
          role: "assistant",
          content: "stored answer",
          tools: [{ name: "kg_query", query: "zzq-restored-subject", sources: [] }],
        },
      ],
      followups: [],
    });
    renderTray();
    expect(await screen.findByText("stored answer")).toBeTruthy();
    expect(screen.getByText(/zzq-restored-subject/)).toBeTruthy();
  });

  it("lands each tool call's sources on its own row when the same tool is called twice", async () => {
    // The case that actually exercises correlation, and it is reachable today: one round can call
    // rag_search twice, and two rounds routinely do. tool_done arrives OUT of start order here, so
    // a "last pending run" rule would attach both source sets to the wrong rows.
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "rag_search", "zzq-first-query");
      handlers.onTool?.(1, "rag_search", "zzq-second-query");
      handlers.onToolDone?.(1, [{ label: "zzq-Second-Source", kind: "paper" }]);
      handlers.onToolDone?.(0, [{ label: "zzq-First-Source", kind: "paper" }]);
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    await screen.findByText("A");
    const thread = h.putConversation.mock.calls[0][1] as Array<{ tools?: unknown[] }>;
    expect(thread[thread.length - 1].tools).toEqual([
      { name: "rag_search", query: "zzq-first-query", sources: [{ label: "zzq-First-Source", kind: "paper" }] },
      { name: "rag_search", query: "zzq-second-query", sources: [{ label: "zzq-Second-Source", kind: "paper" }] },
    ]);
  });

  it("drops a tool_done whose id matches no run", async () => {
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "rag_search", "zzq-orphan-query");
      handlers.onToolDone?.(99, [{ label: "zzq-Orphan-Source", kind: "paper" }]);
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    await screen.findByText("A");
    const thread = h.putConversation.mock.calls[0][1] as Array<{ tools?: unknown[] }>;
    expect(thread[thread.length - 1].tools).toEqual([
      { name: "rag_search", query: "zzq-orphan-query" },
    ]);
  });

  it("commits tool records without the in-memory id and pending fields", async () => {
    // These are transport/UI state. The backend's ToolRun model would ignore them, but that
    // backstop is coincidental — the strip is the mechanism, exactly as with `tools` on /api/chat.
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "kg_query", "zzq-strip-query");
      handlers.onToolDone?.(0, []);
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    await screen.findByText("A");
    const thread = h.putConversation.mock.calls[0][1] as Array<{ tools?: unknown[] }>;
    const blob = JSON.stringify(thread[thread.length - 1].tools);
    expect(blob).not.toContain("pending");
    expect(blob).not.toContain('"id"');
  });
});
