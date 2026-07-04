import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import type { Analysis } from "../api";

// Isolated from components.CoachTray.chat.test.tsx: framer-motion's `useReducedMotion` lazily
// reads window.matchMedia("(prefers-reduced-motion)") ONCE per module instance, so this needs to
// be the first render in a fresh module graph (Vitest isolates modules per file) for the mocked
// `matches: true` to actually take effect.
const h = vi.hoisted(() => ({
  auth: { configured: true, user: { id: "u1" } as { id: string } | null },
  chatStream: vi.fn(),
  health: vi.fn(),
}));

vi.mock("../lib/auth", () => ({ useAuth: () => h.auth }));
vi.mock("../api", async (importActual) => {
  const actual = await importActual<typeof import("../api")>();
  return { ...actual, api: { ...actual.api, chatStream: h.chatStream, health: h.health } };
});

import CoachTray from "../components/CoachTray";

const analysis = {
  video_id: "v1",
  view: { view_type: "front", view_confidence: 0.9 },
  quality: { lower_body_visibility_mean: 0.9 },
  detections: [
    { fault_id: "f1", fault_name: "knees_inward", phase: "descent", severity: 0.8, start_time: 1, end_time: 2, evidence: { primary_label: "knee valgus ratio", primary_value: 0.82 } },
  ],
  retrievals: [],
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

describe("CoachTray — prefers-reduced-motion", () => {
  beforeEach(() => {
    h.auth = { configured: true, user: { id: "u1" } };
    h.chatStream.mockReset();
    h.health.mockReset();
    h.health.mockResolvedValue({ status: "ok", chat_configured: true });

    vi.spyOn(window, "matchMedia").mockImplementation((query: string) => ({
      matches: true,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
  });

  it("skips the entrance animation for fault cards and chat turns", async () => {
    h.chatStream.mockImplementation(
      async (
        _m: unknown,
        _c: unknown,
        handlers: { onDelta: (t: string) => void; onDone: (m: string) => void }
      ) => {
        handlers.onDelta("Keep your chest up.");
        handlers.onDone("m");
      }
    );
    renderTray();

    expect(screen.getByText("Knee Valgus")).toBeInTheDocument();

    await userEvent.type(screen.getByPlaceholderText(/Ask a follow-up/i), "hi");
    await userEvent.click(screen.getByLabelText(/Send message/i));

    expect(await screen.findByText("Keep your chest up.")).toBeInTheDocument();
  });
});
