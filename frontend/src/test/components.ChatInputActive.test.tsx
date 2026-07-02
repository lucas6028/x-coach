import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import type { Analysis } from "../api";

// Mutable auth state so individual tests can flip signed-in vs signed-out (hoisted so the
// vi.mock factory can close over it).
const h = vi.hoisted(() => ({
  auth: { configured: true, user: { id: "u1" } as { id: string } | null },
  chat: vi.fn(),
  health: vi.fn(),
}));

vi.mock("../lib/auth", () => ({ useAuth: () => h.auth }));
vi.mock("../api", () => ({ api: { chat: h.chat, health: h.health } }));

import ChatInput from "../components/ChatInput";

const analysis = {
  video_id: "v1",
  view: { view_type: "front", view_confidence: 0.9 },
  quality: { lower_body_visibility_mean: 0.9 },
  detections: [
    { fault_id: "f1", fault_name: "knees_inward", phase: "descent", severity: 0.8, start_time: 1, end_time: 2, evidence: { knee_valgus_ratio: 0.82 } },
  ],
  retrievals: [
    {
      fault_id: "f1",
      retrieval_mode: "kg",
      context: { results: [{ summary: { corrections: [{ node_id: "knees out", label: "z" }] } }] },
    },
  ],
} as unknown as Analysis;

function renderPanel() {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <ChatInput analysis={analysis} />
      </I18nProvider>
    </MemoryRouter>
  );
}

describe("ChatInput (active)", () => {
  beforeEach(() => {
    h.auth = { configured: true, user: { id: "u1" } };
    h.chat.mockReset();
    h.health.mockReset();
    h.health.mockResolvedValue({ status: "ok", chat_configured: true });
  });

  it("sends a grounded user turn and renders the coach reply", async () => {
    h.chat.mockResolvedValue({ reply: "Drive your knees out over your toes.", model: "m" });
    renderPanel();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "why did my knees cave?");
    await userEvent.click(screen.getByLabelText(/Send message/i));

    expect(h.chat).toHaveBeenCalledTimes(1);
    const [messages, context] = h.chat.mock.calls[0];
    expect(messages.at(-1)).toEqual({ role: "user", content: "why did my knees cave?" });
    // The grounding blob carries the detected fault + its retrieved corrective cue.
    expect(context.faults[0].fault_name).toBe("knees_inward");
    expect(context.faults[0].corrections).toContain("knees out");

    expect(await screen.findByText("Drive your knees out over your toes.")).toBeInTheDocument();
  });

  it("shows an error when the coach is unreachable", async () => {
    h.chat.mockRejectedValue(new Error("network"));
    renderPanel();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "hi");
    await userEvent.keyboard("{Enter}"); // exercises Enter-to-send

    expect(await screen.findByRole("alert")).toHaveTextContent(/try again/i);
  });

  it("invites sign-in when configured but signed out", () => {
    h.auth = { configured: true, user: null };
    renderPanel();

    expect(screen.getByText(/Sign in to chat/i)).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /sign in/i });
    expect(link).toHaveAttribute("href", "/login");
  });

  it("falls back to the disabled affordance when the server has no LLM key", async () => {
    h.health.mockResolvedValue({ status: "ok", chat_configured: false });
    renderPanel();

    // Once /api/health resolves, the working input is replaced by the honest coming-soon input.
    const disabled = await screen.findByPlaceholderText(/Ask the AI Coach/i);
    expect(disabled).toBeDisabled();
    expect(h.chat).not.toHaveBeenCalled();
  });
});
