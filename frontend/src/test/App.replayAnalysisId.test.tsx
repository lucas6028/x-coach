import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { mockAnalysis } from "./fixtures";

// WP4's chat context needs `analysis_id` on every replayed analysis, not just a fresh upload's
// immediate response — see App.tsx's `loadStored`. The backend persists `analyses.result` (the
// JSONB blob GET /api/analyses/{id} returns as `row.result`) BEFORE it knows the row's own id
// (backend/app/routers/analyze.py sets `result["analysis_id"]` only on the in-memory response
// handed back right after upload, never on what gets written to the row), so a real replay's
// `row.result.analysis_id` is always absent — only `row.id` carries it. This suite renders App at
// `/app?analysis=<id>` against exactly that shape and proves the id still reaches the coach's chat
// context, i.e. that App.tsx stitches `row.id` back in rather than trusting `row.result` alone.
const h = vi.hoisted(() => ({
  auth: { configured: true, user: { id: "u1" } as { id: string } | null },
  chatStream: vi.fn(),
}));

vi.mock("../lib/auth", () => ({ useAuth: () => h.auth }));

vi.mock("../api", async (importActual) => {
  const actual = await importActual<typeof import("../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      chatStream: h.chatStream,
      chatFollowups: vi.fn().mockResolvedValue([]),
      health: vi.fn().mockResolvedValue({ status: "ok", chat_configured: true }),
      getConversation: vi.fn().mockResolvedValue({ video_id: "vid_history", messages: [] }),
      putConversation: vi.fn().mockResolvedValue(undefined),
      getMovements: vi.fn().mockResolvedValue([{ name: "Squat", validated: true }]),
      // The realistic replay shape: `result` carries NO `analysis_id` — only the row's own `id`
      // does. A fix that merely read `row.result.analysis_id` would leave this undefined.
      getStoredAnalysis: vi.fn().mockResolvedValue({
        id: "a1",
        video_id: "vid_history",
        source: "upload",
        view_type: "side",
        fault_count: 1,
        created_at: "2026-09-14T00:00:00Z",
        result: { ...mockAnalysis, source: "upload", video_id: "vid_history" },
      }),
    },
  };
});

import App from "../App";

function renderAppAt(entry: string) {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <I18nProvider>
        <App />
      </I18nProvider>
    </MemoryRouter>
  );
}

beforeEach(() => {
  h.chatStream.mockReset();
  h.chatStream.mockImplementation(
    async (
      _messages: unknown,
      _context: unknown,
      handlers: { onDone: (m: string) => void }
    ) => {
      handlers.onDone("m");
    }
  );
});

describe("App replay — analysis_id reaches the chat context", () => {
  it("stitches the row's own id into the replayed analysis's chat grounding", async () => {
    renderAppAt("/app?analysis=a1");

    await userEvent.type(await screen.findByPlaceholderText(/Ask a follow-up/i), "why?");
    await userEvent.click(screen.getByLabelText(/Send message/i));

    expect(h.chatStream).toHaveBeenCalledTimes(1);
    const [, context] = h.chatStream.mock.calls[0];
    expect(context.analysis_id).toBe("a1");
  });
});
