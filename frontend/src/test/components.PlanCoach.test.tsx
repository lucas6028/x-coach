import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import type { LiveToolRun, Plan } from "../api";

// Mutable auth state so a test can flip signed-in / signed-out (hoisted for the vi.mock factory),
// exactly as components.CoachTray.chat.test.tsx does.
const h = vi.hoisted(() => ({
  auth: { configured: true, user: { id: "u1" } as { id: string } | null },
  planChatStream: vi.fn(),
  health: vi.fn(),
  getConversation: vi.fn(),
  putConversation: vi.fn(),
}));

vi.mock("../lib/auth", () => ({ useAuth: () => h.auth }));
// Keep the real module (ChatError, types) and stub only the network methods.
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

import PlanCoach from "../components/plans/PlanCoach";

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

type Handlers = {
  onDelta: (t: string) => void;
  onTool?: (run: LiveToolRun) => void;
  onToolDone?: (id: number, extra: { plan?: Plan }) => void;
  onPlan?: (p: Plan) => void;
  onDone: (info: { model: string; plan_id?: string }) => void;
};

/** Drive the component's handlers with a fixed reply, optionally creating a plan first. */
function streamReply(reply: string, opts: { plan?: Plan; planId?: string } = {}) {
  return async (_m: unknown, _planId: unknown, handlers: Handlers) => {
    if (opts.plan) {
      handlers.onTool?.({ id: 0, name: "create_plan", query: opts.plan.name, pending: true });
      handlers.onPlan?.(opts.plan);
      handlers.onToolDone?.(0, { plan: opts.plan });
    }
    const mid = Math.ceil(reply.length / 2);
    handlers.onDelta(reply.slice(0, mid));
    handlers.onDelta(reply.slice(mid));
    handlers.onDone({ model: "m", ...(opts.planId ? { plan_id: opts.planId } : {}) });
  };
}

function renderCoach(ui: React.ReactElement) {
  return render(
    <MemoryRouter>
      <I18nProvider>{ui}</I18nProvider>
    </MemoryRouter>
  );
}

beforeEach(() => {
  h.auth = { configured: true, user: { id: "u1" } };
  h.planChatStream.mockReset();
  h.health.mockReset().mockResolvedValue({ status: "ok", chat_configured: true });
  h.getConversation.mockReset().mockResolvedValue({ video_id: "plan:p9", messages: [] });
  h.putConversation.mockReset().mockResolvedValue(undefined);
});
afterEach(() => {
  localStorage.removeItem("lang");
  vi.restoreAllMocks();
});

describe("PlanCoach — opening state", () => {
  it("renders the greeting and the quick prompts while the thread is empty", async () => {
    renderCoach(
      <PlanCoach planId={null} greeting="zzq-greeting" suggestions={["zzq-chip-a", "zzq-chip-b"]} />
    );
    expect(await screen.findByText("zzq-greeting")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /zzq-chip-a/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /zzq-chip-b/ })).toBeInTheDocument();
  });

  it("drops the greeting and the chips once the thread has a turn", async () => {
    // The greeting is a rendered opening line, not a message — a restored thread replaces it.
    renderCoach(
      <PlanCoach
        planId="p9"
        greeting="zzq-greeting"
        suggestions={["zzq-chip-a"]}
        initialMessages={[{ role: "user", content: "zzq-earlier" }]}
      />
    );
    expect(await screen.findByText("zzq-earlier")).toBeInTheDocument();
    expect(screen.queryByText("zzq-greeting")).toBeNull();
    expect(screen.queryByRole("button", { name: /zzq-chip-a/ })).toBeNull();
    // A caller-supplied thread wins over the persisted one, so no restore is attempted.
    expect(h.getConversation).not.toHaveBeenCalled();
  });

  it("restores the persisted thread under the plan: key", async () => {
    h.getConversation.mockResolvedValue({
      video_id: "plan:p9",
      messages: [{ role: "assistant", content: "zzq-restored" }],
    });
    renderCoach(<PlanCoach planId="p9" greeting="zzq-greeting" />);
    expect(await screen.findByText("zzq-restored")).toBeInTheDocument();
    expect(h.getConversation).toHaveBeenCalledWith("plan:p9");
  });
});

describe("PlanCoach — sending", () => {
  it("sends the chip's text with the plan id and the current language, and commits the answer", async () => {
    localStorage.setItem("lang", "zh-Hant");
    h.planChatStream.mockImplementation(streamReply("zzq-answer"));
    renderCoach(<PlanCoach planId="p9" suggestions={["zzq-chip-a"]} />);

    await userEvent.click(await screen.findByRole("button", { name: /zzq-chip-a/ }));

    await waitFor(() => expect(h.planChatStream).toHaveBeenCalled());
    const [messages, planId, , opts] = h.planChatStream.mock.calls[0];
    expect(messages).toEqual([{ role: "user", content: "zzq-chip-a" }]);
    expect(planId).toBe("p9");
    expect(opts.lang).toBe("zh-Hant");
    // The streamed deltas are committed as one assistant turn.
    expect(await screen.findByText("zzq-answer")).toBeInTheDocument();
    await waitFor(() =>
      expect(h.putConversation).toHaveBeenCalledWith("plan:p9", [
        { role: "user", content: "zzq-chip-a" },
        { role: "assistant", content: "zzq-answer", tools: undefined },
      ])
    );
  });

  it("shows the tool label for a plan tool while the turn streams", async () => {
    h.planChatStream.mockImplementation(streamReply("zzq-answer", { plan, planId: "p9" }));
    renderCoach(<PlanCoach planId={null} suggestions={["zzq-chip-a"]} />);
    await userEvent.click(await screen.findByRole("button", { name: /zzq-chip-a/ }));
    // Committed with the answer: the localized label, not the raw tool name.
    expect(await screen.findByText(/Building the plan/)).toBeInTheDocument();
  });

  it("reports a plan created mid-conversation through onPlan and onCreated", async () => {
    const onPlan = vi.fn();
    const onCreated = vi.fn();
    h.planChatStream.mockImplementation(streamReply("zzq-answer", { plan, planId: "p9" }));
    renderCoach(
      <PlanCoach planId={null} onPlan={onPlan} onCreated={onCreated} suggestions={["zzq-chip-a"]} />
    );

    await userEvent.click(await screen.findByRole("button", { name: /zzq-chip-a/ }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("p9"));
    expect(onPlan).toHaveBeenCalledWith(plan);
    expect(onCreated).toHaveBeenCalledTimes(1);
    // The id is adopted for persistence too, even though the prop was null when the turn started.
    await waitFor(() => expect(h.putConversation).toHaveBeenCalledWith("plan:p9", expect.anything()));
  });

  it("announces a plan id that only reached the tool frame, not the done frame", async () => {
    // A stream that dies after create_plan still created a plan; losing its id would strand it.
    const onCreated = vi.fn();
    h.planChatStream.mockImplementation(streamReply("zzq-answer", { plan }));
    renderCoach(<PlanCoach planId={null} onCreated={onCreated} suggestions={["zzq-chip-a"]} />);
    await userEvent.click(await screen.findByRole("button", { name: /zzq-chip-a/ }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("p9"));
  });

  it("rolls the user turn back and restores the text when the stream fails", async () => {
    const onPlan = vi.fn();
    h.planChatStream.mockImplementation(async (_m: unknown, _p: unknown, handlers: Handlers) => {
      handlers.onPlan?.(plan); // the tool really wrote the plan before the failure
      throw new Error("boom");
    });
    renderCoach(<PlanCoach planId="p9" onPlan={onPlan} suggestions={["zzq-chip-a"]} />);
    await userEvent.click(await screen.findByRole("button", { name: /zzq-chip-a/ }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/Lumen/i);
    // The chat turn is rolled back — the thread is empty again (so the chips are back) and the
    // text returns to the composer so a retry doesn't duplicate it. The PLAN is NOT rolled back:
    // a tool that reported a write really wrote.
    await waitFor(() =>
      expect(screen.getByLabelText(/Message Lumen/i)).toHaveValue("zzq-chip-a")
    );
    expect(screen.getByRole("button", { name: /zzq-chip-a/ })).toBeInTheDocument();
    expect(onPlan).toHaveBeenCalledWith(plan);
    expect(h.putConversation).not.toHaveBeenCalled();
  });
});

describe("PlanCoach — composer states", () => {
  it("offers a sign-in instead of a composer when signed out", async () => {
    h.auth = { configured: true, user: null };
    renderCoach(<PlanCoach planId={null} suggestions={["zzq-chip-a"]} />);
    expect(await screen.findByText(/Sign in to plan with Lumen/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Message Lumen/i)).toBeNull();
    // Chips would send a request that can only 401, so they stay away too.
    expect(screen.queryByRole("button", { name: /zzq-chip-a/ })).toBeNull();
  });

  it("disables the composer when the server has no LLM key", async () => {
    h.health.mockResolvedValue({ status: "ok", chat_configured: false });
    renderCoach(<PlanCoach planId={null} />);
    await waitFor(() => expect(screen.getByPlaceholderText(/coming soon/i)).toBeDisabled());
  });
});
