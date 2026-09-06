import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import type { LiveToolRun, Plan } from "../api";

const h = vi.hoisted(() => ({
  auth: { configured: true, user: { id: "u1" } as { id: string } | null },
  planChatStream: vi.fn(),
  health: vi.fn(),
  getConversation: vi.fn(),
  putConversation: vi.fn(),
}));

vi.mock("../lib/auth", () => ({
  useAuth: () => h.auth,
  // AppLayout's chrome (sidebar/account menu) reads the same module.
  AuthProvider: ({ children }: { children: React.ReactNode }) => children,
}));
vi.mock("../api", async (importActual) => {
  const actual = await importActual<typeof import("../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      planChatStream: h.planChatStream,
      health: h.health,
      getConversation: h.getConversation,
      putConversation: h.putConversation,
    },
  };
});

import PlanBuilder from "../pages/PlanBuilder";

const plan: Plan = {
  id: "p9",
  name: "zzq-Full-body week",
  notes: null,
  template_key: null,
  started_at: null,
  created_at: "2026-09-01T00:00:00Z",
  updated_at: "2026-09-01T00:00:00Z",
  items: [
    {
      id: "i1",
      plan_id: "p9",
      day_index: 1,
      position: 0,
      movement: "Squat",
      sets: 3,
      reps: 8,
      notes: null,
      completed_at: null,
      analysis_id: null,
      created_at: "2026-09-01T00:00:00Z",
    },
  ],
};

type Handlers = {
  onDelta: (t: string) => void;
  onTool?: (run: LiveToolRun) => void;
  onToolDone?: (id: number, extra: { plan?: Plan }) => void;
  onPlan?: (p: Plan) => void;
  onDone: (info: { model: string; plan_id?: string }) => void;
};

beforeEach(() => {
  h.auth = { configured: true, user: { id: "u1" } };
  h.planChatStream.mockReset().mockImplementation(async (_m, _p, handlers: Handlers) => {
    handlers.onPlan?.(plan);
    handlers.onDelta("zzq-here is your week");
    handlers.onDone({ model: "m", plan_id: "p9" });
  });
  h.health.mockReset().mockResolvedValue({ status: "ok", chat_configured: true });
  h.getConversation.mockReset().mockResolvedValue({ video_id: "", messages: [] });
  h.putConversation.mockReset().mockResolvedValue(undefined);
});
afterEach(() => vi.restoreAllMocks());

describe("PlanBuilder", () => {
  it("renders the page header, Lumen's greeting and a way back to the plans list", async () => {
    renderWithProviders(<PlanBuilder />, { route: "/plans/new" });
    expect(await screen.findByRole("heading", { name: /Plan with Lumen/i })).toBeInTheDocument();
    expect(screen.getByText(/Hi, I'm Lumen/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /All plans/i })).toHaveAttribute("href", "/plans");
  });

  it("shows the composed placeholder before any plan exists", async () => {
    renderWithProviders(<PlanBuilder />, { route: "/plans/new" });
    expect(await screen.findByText(/the plan will appear here/i)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Open the plan/i })).toBeNull();
  });

  it("fills the preview and offers the CTA once the stream reports a plan", async () => {
    renderWithProviders(<PlanBuilder />, { route: "/plans/new" });
    await userEvent.click(
      await screen.findByRole("button", { name: /Three full-body days a week/i })
    );

    // The preview is driven purely by the streamed plan — no refetch.
    expect(await screen.findByText("zzq-Full-body week")).toBeInTheDocument();
    expect(screen.getByText("Squat")).toBeInTheDocument();
    expect(screen.getByText(/3 × 8/)).toBeInTheDocument();
    expect(screen.queryByText(/the plan will appear here/i)).toBeNull();

    await waitFor(() =>
      expect(screen.getByRole("link", { name: /Open the plan/i })).toHaveAttribute(
        "href",
        "/plans/p9"
      )
    );
  });
});

describe("PlanBuilder — what the drafted plan trains", () => {
  it("has no coverage summary before a plan exists, and one after", async () => {
    renderWithProviders(<PlanBuilder />, { route: "/plans/new" });
    await screen.findByText(/the plan will appear here/i);
    // Absent, not an empty box: there is nothing to summarise yet.
    expect(screen.queryByRole("heading", { name: /what this plan trains/i })).toBeNull();

    await userEvent.click(
      await screen.findByRole("button", { name: /Three full-body days a week/i })
    );

    const card = (await screen.findByRole("heading", {
      name: /what this plan trains/i,
    })).closest("section") as HTMLElement;
    expect(within(card).getByText("Quadriceps")).toBeInTheDocument();
    expect(within(card).getByText(/not trained this week/i)).toBeInTheDocument();
  });
});

// On the phone the preview lives in a bottom sheet. Driven by `useIsMobile`, so this forces its
// media query to match — the global setup stub answers `false` to everything else.
describe("PlanBuilder — on the phone", () => {
  beforeEach(() => {
    vi.spyOn(window, "matchMedia").mockImplementation(
      (query: string) =>
        ({
          matches: query === "(max-width: 1023px)",
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn(),
        }) as unknown as MediaQueryList
    );
  });

  it("opens the preview as a dialog and closes it on Escape", async () => {
    renderWithProviders(<PlanBuilder />, { route: "/plans/new" });
    await userEvent.click(await screen.findByRole("button", { name: /Preview plan/i }));

    const sheet = screen.getByRole("dialog", { name: /Plan preview/i });
    expect(sheet).toHaveAttribute("aria-modal", "true");
    expect(screen.getByText(/the plan will appear here/i)).toBeInTheDocument();

    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes the sheet from its close button", async () => {
    renderWithProviders(<PlanBuilder />, { route: "/plans/new" });
    await userEvent.click(await screen.findByRole("button", { name: /Preview plan/i }));
    const closers = screen.getAllByRole("button", { name: /Close preview/i });
    await userEvent.click(closers[closers.length - 1]);
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
