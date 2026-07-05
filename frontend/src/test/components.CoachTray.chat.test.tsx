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

  it("does not persist when a turn fails mid-stream", async () => {
    h.chatStream.mockImplementation(
      async (
        _m: unknown,
        _c: unknown,
        handlers: { onError: (d: string) => void }
      ) => {
        handlers.onError("OpenRouter request failed: reset");
      }
    );
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "why?");
    await userEvent.keyboard("{Enter}");

    await screen.findByRole("alert");
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
    // The stream opens (200) but OpenRouter fails mid-flight: the client delivers an `error` frame
    // via onError rather than throwing. The partial turn must be discarded, not left orphaned.
    h.chatStream.mockImplementation(
      async (
        _m: unknown,
        _c: unknown,
        handlers: { onDelta: (t: string) => void; onError: (d: string) => void }
      ) => {
        handlers.onDelta("Drive your kne");
        handlers.onError("OpenRouter request failed: reset");
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

    const disabled = await screen.findByPlaceholderText(/Ask the AI Coach/i);
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
});
