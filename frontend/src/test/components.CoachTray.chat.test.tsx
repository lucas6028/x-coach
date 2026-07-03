import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { ChatError, type Analysis } from "../api";

// Mutable auth state so tests can flip signed-in / signed-out (hoisted for the vi.mock factory).
const h = vi.hoisted(() => ({
  auth: { configured: true, user: { id: "u1" } as { id: string } | null },
  chat: vi.fn(),
  health: vi.fn(),
}));

vi.mock("../lib/auth", () => ({ useAuth: () => h.auth }));
// Preserve the real module (ChatError, types) and only stub the network methods, so the
// component's `instanceof ChatError` status check exercises the genuine error type.
vi.mock("../api", async (importActual) => {
  const actual = await importActual<typeof import("../api")>();
  return { ...actual, api: { ...actual.api, chat: h.chat, health: h.health } };
});

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
    h.chat.mockReset();
    h.health.mockReset();
    h.health.mockResolvedValue({ status: "ok", chat_configured: true });
  });

  it("sends a grounded user turn and renders the coach reply in the same thread", async () => {
    h.chat.mockResolvedValue({ reply: "Drive your knees out over your toes.", model: "m" });
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "why did my knees cave?");
    await userEvent.click(screen.getByLabelText(/Send message/i));

    expect(h.chat).toHaveBeenCalledTimes(1);
    const [messages, context] = h.chat.mock.calls[0];
    expect(messages.at(-1)).toEqual({ role: "user", content: "why did my knees cave?" });
    // Grounding blob carries the detected fault + its retrieved corrective cue.
    expect(context.faults[0].fault_name).toBe("knees_inward");
    expect(context.faults[0].corrections).toContain("knees out");

    expect(await screen.findByText("Drive your knees out over your toes.")).toBeInTheDocument();
  });

  it("shows an error and rolls back the optimistic turn when the coach is unreachable", async () => {
    h.chat.mockRejectedValue(new Error("network"));
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
    h.chat.mockRejectedValue(new ChatError("Missing bearer token.", 401));
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "hi");
    await userEvent.keyboard("{Enter}");

    expect(await screen.findByRole("alert")).toHaveTextContent(/sign in again/i);
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
    expect(h.chat).not.toHaveBeenCalled();
  });

  it("does nothing when Enter is pressed with an empty composer", async () => {
    renderTray();
    const box = await screen.findByPlaceholderText(/Ask a follow-up/i);
    box.focus();
    await userEvent.keyboard("{Enter}");
    expect(h.chat).not.toHaveBeenCalled();
  });

  it("scrolls the thread into view once a turn is sent, when scrollTo is available", async () => {
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      writable: true,
      value: scrollTo,
    });

    h.chat.mockResolvedValue({ reply: "Ok.", model: "m" });
    renderTray();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "hi");
    await userEvent.click(screen.getByLabelText(/Send message/i));
    await screen.findByText("Ok.");

    expect(scrollTo).toHaveBeenCalledWith({ top: expect.any(Number) });

    Reflect.deleteProperty(HTMLElement.prototype, "scrollTo");
  });
});
