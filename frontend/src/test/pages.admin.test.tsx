import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import {
  api,
  type AdminOverview,
  type AdminSettingsResponse,
  type AdminUserRow,
} from "../api";
import AdminLayout from "../pages/admin/AdminLayout";
import AdminOverviewPage from "../pages/admin/AdminOverview";
import AdminUsers from "../pages/admin/AdminUsers";
import AdminSettingsLlm from "../pages/admin/AdminSettingsLlm";
import AdminSettingsRag from "../pages/admin/AdminSettingsRag";
import AdminSettingsAnalyze from "../pages/admin/AdminSettingsAnalyze";

const SAMPLE_SETTINGS: AdminSettingsResponse = {
  effective: {
    llm: {
      llm_models: ["a/model", "b/model"],
      llm_followup_model: "fast/model",
      llm_base_url: "https://openrouter.ai/api/v1",
      chat_temperature: null,
      chat_timeout: 60,
      followup_timeout: 15,
    },
    rag_kg: { rag_top_k: 5, kg_hops: 1, kg_seeds: 5 },
    analyze: { allowed_upload_suffixes: [".mp4", ".mov"], max_concurrent_analyses: 2 },
  },
  defaults: {
    llm: {
      llm_models: ["a/model", "b/model"],
      llm_followup_model: "fast/model",
      llm_base_url: "https://openrouter.ai/api/v1",
      chat_temperature: null,
      chat_timeout: 60,
      followup_timeout: 15,
    },
    rag_kg: { rag_top_k: 5, kg_hops: 1, kg_seeds: 5 },
    analyze: { allowed_upload_suffixes: [".mp4", ".mov"], max_concurrent_analyses: 2 },
  },
};

const SAMPLE_OVERVIEW: AdminOverview = {
  auth_configured: true,
  chat_configured: true,
  chat_models: ["a/model"],
  chat_default: "a/model",
  stores: { labeled_videos: true, detections: false },
  total_users: 2,
  total_analyses: 7,
};

// Row "u1" is the signed-in admin (self, cannot self-demote); "u2" is a non-admin togglable user.
const SAMPLE_USERS: AdminUserRow[] = [
  {
    id: "u1",
    email: "ada@x.com",
    created_at: "2026-01-01T00:00:00Z",
    last_sign_in_at: "2026-07-01T00:00:00Z",
    analyses_count: 4,
    conversations_count: 2,
    is_admin: true,
  },
  {
    id: "u2",
    email: "bob@x.com",
    created_at: "2026-02-01T00:00:00Z",
    last_sign_in_at: null,
    analyses_count: 3,
    conversations_count: 1,
    is_admin: false,
  },
];

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
const mockUseAuth = vi.mocked(useAuth);

// Render the real nested admin route tree at `path` so <Outlet/> (and its context) resolve exactly as
// in production. A stub /app route stands in for the "back to app" destination.
function renderAdmin(path: string) {
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/admin" element={<AdminLayout />}>
            <Route index element={<AdminOverviewPage />} />
            <Route path="users" element={<AdminUsers />} />
            <Route path="settings/llm" element={<AdminSettingsLlm />} />
            <Route path="settings/rag" element={<AdminSettingsRag />} />
            <Route path="settings/analyze" element={<AdminSettingsAnalyze />} />
          </Route>
          <Route path="/app" element={<div>app studio</div>} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

// The admin role now comes from AuthProvider (probed once per session), so the gate reads
// `isAdmin`/`adminState` off useAuth rather than calling api.adminStatus per mount.
function authValue(overrides: Record<string, unknown> = {}) {
  return {
    user: { id: "u1", email: "ada@x.com" },
    signOut: vi.fn(),
    isAdmin: true,
    adminState: "ready",
    refreshAdmin: vi.fn(),
    ...overrides,
  } as unknown as ReturnType<typeof useAuth>;
}

beforeEach(() => {
  mockUseAuth.mockReturnValue(authValue());
  vi.spyOn(api, "getAdminOverview").mockResolvedValue(SAMPLE_OVERVIEW);
  vi.spyOn(api, "listAdminUsers").mockResolvedValue({ users: SAMPLE_USERS });
  vi.spyOn(api, "getAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
});
afterEach(() => vi.restoreAllMocks());

describe("AdminLayout gate", () => {
  it("renders the admin nav for an admin", async () => {
    renderAdmin("/admin");
    // The nav lists all five admin destinations (plus the back-to-app link). The rail is rendered in
    // both the desktop shell and the (off-canvas) mobile drawer, so each link appears more than once.
    expect((await screen.findAllByRole("link", { name: "Users" })).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "LLM chat" }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "Back to app" }).length).toBeGreaterThan(0);
    expect(
      screen.queryByText("You don't have access to the admin panel.")
    ).not.toBeInTheDocument();
  });

  it("shows an access-denied card for a non-admin (and does not mount the child page)", async () => {
    mockUseAuth.mockReturnValue(authValue({ isAdmin: false }));
    const overview = vi.spyOn(api, "getAdminOverview").mockResolvedValue(SAMPLE_OVERVIEW);
    renderAdmin("/admin");
    expect(
      await screen.findByText("You don't have access to the admin panel.")
    ).toBeInTheDocument();
    // The gated child never mounts, so it never fetches.
    expect(overview).not.toHaveBeenCalled();
  });

  it("surfaces an error when the admin probe fails", async () => {
    mockUseAuth.mockReturnValue(authValue({ adminState: "error" }));
    renderAdmin("/admin");
    expect(await screen.findByText(/Couldn't verify your admin access/i)).toBeInTheDocument();
  });

  it("offers a Retry that re-runs the admin probe (no reload needed)", async () => {
    const refreshAdmin = vi.fn();
    mockUseAuth.mockReturnValue(authValue({ adminState: "error", refreshAdmin }));
    renderAdmin("/admin");
    const retry = await screen.findByRole("button", { name: "Retry" });
    fireEvent.click(retry);
    expect(refreshAdmin).toHaveBeenCalledTimes(1);
  });
});

describe("AdminOverview", () => {
  it("renders the system overview cards", async () => {
    renderAdmin("/admin");
    expect(await screen.findByText("Total users")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument(); // total analyses value
    expect(screen.getByText("1/2 ready")).toBeInTheDocument();
  });
});

describe("AdminUsers", () => {
  it("renders the users table rows with the self tag", async () => {
    renderAdmin("/admin/users");
    expect(await screen.findByText("bob@x.com")).toBeInTheDocument();
    expect(screen.getByText("ada@x.com")).toBeInTheDocument();
    expect(screen.getByText("You")).toBeInTheDocument();
  });

  it("toggles a non-self user's role and refreshes the list", async () => {
    const setRole = vi.spyOn(api, "setUserRole").mockResolvedValue({ ok: true });
    const list = vi.spyOn(api, "listAdminUsers").mockResolvedValue({ users: SAMPLE_USERS });
    renderAdmin("/admin/users");

    const makeAdmin = await screen.findByRole("button", { name: "Make admin" });
    fireEvent.click(makeAdmin);
    await waitFor(() => expect(setRole).toHaveBeenCalledWith("u2", true));
    await waitFor(() => expect(list.mock.calls.length).toBeGreaterThanOrEqual(2));
  });

  it("disables the toggle on the signed-in admin's own row (no self-demote)", async () => {
    renderAdmin("/admin/users");
    const revoke = await screen.findByRole("button", { name: "Revoke admin" });
    expect(revoke).toBeDisabled();
  });
});

describe("Admin settings pages", () => {
  it("loads the LLM settings and shows current values", async () => {
    renderAdmin("/admin/settings/llm");
    const baseUrl = await screen.findByLabelText("Provider base URL");
    expect(baseUrl).toHaveValue("https://openrouter.ai/api/v1");
  });

  it("saves only the RAG group's keys from the RAG settings page", async () => {
    const update = vi.spyOn(api, "updateAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin("/admin/settings/rag");

    const topK = await screen.findByLabelText("RAG top-k");
    expect(topK).toHaveValue("5");
    fireEvent.change(topK, { target: { value: "9" } });
    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));

    await waitFor(() => expect(update).toHaveBeenCalled());
    // Only the RAG group's keys are sent; no LLM/analyze keys leak in.
    const payload = update.mock.calls[0][0];
    expect(payload).toEqual({ rag_top_k: 9, kg_hops: 1, kg_seeds: 5 });
    expect(await screen.findByText("Settings saved.")).toBeInTheDocument();
  });

  it("shows a load-error card when the settings fetch fails", async () => {
    vi.spyOn(api, "getAdminSettings").mockRejectedValue(new Error("500 boom"));
    renderAdmin("/admin/settings/analyze");
    expect(await screen.findByText("Couldn't load the current settings.")).toBeInTheDocument();
  });

  it("blocks the save and shows an invalid-number error when a required field is cleared", async () => {
    const update = vi.spyOn(api, "updateAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin("/admin/settings/rag");

    const topK = await screen.findByLabelText("RAG top-k");
    fireEvent.change(topK, { target: { value: "" } }); // cleared → not a valid number
    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));

    expect(
      await screen.findByText("Please enter a valid number for every field.")
    ).toBeInTheDocument();
    expect(update).not.toHaveBeenCalled();
  });

  it("blocks the save on non-numeric input and never submits it", async () => {
    const update = vi.spyOn(api, "updateAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin("/admin/settings/rag");

    const hops = await screen.findByLabelText("KG hops");
    fireEvent.change(hops, { target: { value: "abc" } });
    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));

    expect(
      await screen.findByText("Please enter a valid number for every field.")
    ).toBeInTheDocument();
    expect(update).not.toHaveBeenCalled();
  });

  it("submits parsed numbers when every required field is valid", async () => {
    const update = vi.spyOn(api, "updateAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin("/admin/settings/rag");

    const topK = await screen.findByLabelText("RAG top-k");
    fireEvent.change(topK, { target: { value: "8" } });
    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));

    await waitFor(() => expect(update).toHaveBeenCalled());
    // Sent as real numbers, not strings.
    expect(update.mock.calls[0][0]).toEqual({ rag_top_k: 8, kg_hops: 1, kg_seeds: 5 });
  });

  it("shows max concurrent analyses read-only: no editable input, never sent in an update", async () => {
    const update = vi.spyOn(api, "updateAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin("/admin/settings/analyze");

    // The read-only env-var note is shown, and there is no editable control for the value.
    expect(
      await screen.findByText(/XCOACH_MAX_CONCURRENT_ANALYSES/)
    ).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: /max concurrent/i })).toBeNull();

    // Saving only sends the editable upload-formats field — never max_concurrent_analyses.
    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));
    await waitFor(() => expect(update).toHaveBeenCalled());
    expect(update.mock.calls[0][0]).toEqual({ allowed_upload_suffixes: [".mp4", ".mov"] });
  });
});

// Deep-clone a settings response so a per-test tweak (e.g. a non-null temperature) never leaks.
const cloneSettings = (): AdminSettingsResponse =>
  JSON.parse(JSON.stringify(SAMPLE_SETTINGS)) as AdminSettingsResponse;

describe("AdminSettingsLlm", () => {
  it("renders the current LLM values across every field", async () => {
    renderAdmin("/admin/settings/llm");
    // Models are shown newline-joined in the textarea; temperature=null renders as an empty field.
    expect(await screen.findByLabelText("Selectable models")).toHaveValue("a/model\nb/model");
    expect(screen.getByLabelText("Follow-up model")).toHaveValue("fast/model");
    expect(screen.getByLabelText("Provider base URL")).toHaveValue("https://openrouter.ai/api/v1");
    expect(screen.getByLabelText("Temperature")).toHaveValue("");
    expect(screen.getByLabelText("Answer timeout (s)")).toHaveValue("60");
    expect(screen.getByLabelText("Follow-up timeout (s)")).toHaveValue("15");
  });

  it("renders a stored (non-null) temperature as a populated field", async () => {
    const withTemp = cloneSettings();
    withTemp.effective.llm.chat_temperature = 0.5;
    vi.spyOn(api, "getAdminSettings").mockResolvedValue(withTemp);
    renderAdmin("/admin/settings/llm");
    expect(await screen.findByLabelText("Temperature")).toHaveValue("0.5");
  });

  it("edits every field and saves the parsed LLM payload, then reflects the reload", async () => {
    // The save response carries a non-null temperature so the reload exercises toForm's non-null branch.
    const saved = cloneSettings();
    saved.effective.llm.chat_temperature = 0.7;
    const update = vi.spyOn(api, "updateAdminSettings").mockResolvedValue(saved);
    renderAdmin("/admin/settings/llm");

    fireEvent.change(await screen.findByLabelText("Selectable models"), {
      target: { value: "x/m, y/m\nz/m" }, // splitList handles both comma and newline separators
    });
    fireEvent.change(screen.getByLabelText("Follow-up model"), { target: { value: "  quick/model  " } });
    fireEvent.change(screen.getByLabelText("Provider base URL"), { target: { value: "  https://alt/v1  " } });
    fireEvent.change(screen.getByLabelText("Temperature"), { target: { value: "0.7" } });
    fireEvent.change(screen.getByLabelText("Answer timeout (s)"), { target: { value: "45" } });
    fireEvent.change(screen.getByLabelText("Follow-up timeout (s)"), { target: { value: "20" } });

    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));

    await waitFor(() => expect(update).toHaveBeenCalled());
    expect(update.mock.calls[0][0]).toEqual({
      llm_models: ["x/m", "y/m", "z/m"],
      llm_followup_model: "quick/model", // trimmed
      llm_base_url: "https://alt/v1", // trimmed
      chat_temperature: 0.7, // parseNumber of a non-blank value
      chat_timeout: 45,
      followup_timeout: 20,
    });
    expect(await screen.findByText("Settings saved.")).toBeInTheDocument();
    // The reload re-derives the form from the (non-null temperature) response.
    expect(screen.getByLabelText("Temperature")).toHaveValue("0.7");
  });

  it("keeps chat_temperature null when the field is left blank", async () => {
    const update = vi.spyOn(api, "updateAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin("/admin/settings/llm");
    // Leave temperature blank; only bump a required timeout so the submit still fires.
    fireEvent.change(await screen.findByLabelText("Answer timeout (s)"), { target: { value: "30" } });
    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));
    await waitFor(() => expect(update).toHaveBeenCalled());
    expect(update.mock.calls[0][0].chat_temperature).toBeNull();
  });

  it("blocks the save and shows the invalid-number error when a required timeout is cleared", async () => {
    const update = vi.spyOn(api, "updateAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin("/admin/settings/llm");
    fireEvent.change(await screen.findByLabelText("Answer timeout (s)"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));
    expect(
      await screen.findByText("Please enter a valid number for every field.")
    ).toBeInTheDocument();
    expect(update).not.toHaveBeenCalled();
  });

  it("shows the save-error state when the update request rejects", async () => {
    vi.spyOn(api, "updateAdminSettings").mockRejectedValue(new Error("500 boom"));
    renderAdmin("/admin/settings/llm");
    fireEvent.click(await screen.findByRole("button", { name: /Save changes/i }));
    expect(await screen.findByText("Couldn't save the settings.")).toBeInTheDocument();
  });

  it("renders the load-error card when the settings fetch fails", async () => {
    vi.spyOn(api, "getAdminSettings").mockRejectedValue(new Error("500 boom"));
    renderAdmin("/admin/settings/llm");
    expect(await screen.findByText("Couldn't load the current settings.")).toBeInTheDocument();
  });
});

describe("AdminSettingsRag save-error + full edit", () => {
  it("edits kg_seeds and surfaces the save-error state when the update rejects", async () => {
    vi.spyOn(api, "updateAdminSettings").mockRejectedValue(new Error("boom"));
    renderAdmin("/admin/settings/rag");
    fireEvent.change(await screen.findByLabelText("KG seed nodes"), { target: { value: "6" } });
    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));
    expect(await screen.findByText("Couldn't save the settings.")).toBeInTheDocument();
  });
});

describe("AdminSettingsAnalyze edit + save-error", () => {
  it("sends only the edited upload suffixes on save", async () => {
    const update = vi.spyOn(api, "updateAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin("/admin/settings/analyze");
    fireEvent.change(await screen.findByLabelText("Allowed upload formats"), {
      target: { value: ".mp4, .avi" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));
    await waitFor(() => expect(update).toHaveBeenCalled());
    expect(update.mock.calls[0][0]).toEqual({ allowed_upload_suffixes: [".mp4", ".avi"] });
  });

  it("shows the save-error state when the update rejects", async () => {
    vi.spyOn(api, "updateAdminSettings").mockRejectedValue(new Error("boom"));
    renderAdmin("/admin/settings/analyze");
    fireEvent.click(await screen.findByRole("button", { name: /Save changes/i }));
    expect(await screen.findByText("Couldn't save the settings.")).toBeInTheDocument();
  });
});

describe("AdminOverview error + not-configured", () => {
  it("shows the load-error card when the overview fetch fails", async () => {
    vi.spyOn(api, "getAdminOverview").mockRejectedValue(new Error("500 boom"));
    renderAdmin("/admin");
    expect(await screen.findByText("Couldn't load the system overview.")).toBeInTheDocument();
  });

  it("renders 'Not configured' badges when auth/chat are down", async () => {
    const down: AdminOverview = { ...SAMPLE_OVERVIEW, auth_configured: false, chat_configured: false };
    vi.spyOn(api, "getAdminOverview").mockResolvedValue(down);
    renderAdmin("/admin");
    // Both the auth and chat cards resolve to the not-configured value.
    expect((await screen.findAllByText("Not configured")).length).toBe(2);
  });
});

describe("AdminUsers error / empty / toggle-failure", () => {
  it("shows the load-error card when the users fetch fails", async () => {
    vi.spyOn(api, "listAdminUsers").mockRejectedValue(new Error("500 boom"));
    renderAdmin("/admin/users");
    expect(await screen.findByText("Couldn't load the users list.")).toBeInTheDocument();
  });

  it("shows the empty state when there are no users", async () => {
    vi.spyOn(api, "listAdminUsers").mockResolvedValue({ users: [] });
    renderAdmin("/admin/users");
    expect(await screen.findByText("No users yet.")).toBeInTheDocument();
  });

  it("surfaces an inline row error when the role toggle fails and leaves the list intact", async () => {
    const setRole = vi.spyOn(api, "setUserRole").mockRejectedValue(new Error("nope"));
    const list = vi.spyOn(api, "listAdminUsers").mockResolvedValue({ users: SAMPLE_USERS });
    renderAdmin("/admin/users");

    fireEvent.click(await screen.findByRole("button", { name: "Make admin" }));
    await waitFor(() => expect(setRole).toHaveBeenCalledWith("u2", true));
    // Inline per-row error is shown; the list was NOT re-fetched (only the initial load ran).
    expect(await screen.findByText("Couldn't update this user's role.")).toBeInTheDocument();
    expect(list).toHaveBeenCalledTimes(1);
    // Both rows survive — the failed toggle didn't corrupt the table.
    expect(screen.getByText("ada@x.com")).toBeInTheDocument();
    expect(screen.getByText("bob@x.com")).toBeInTheDocument();
  });

  it("falls back to the id for a null email and shows 'Never' for an unparseable date", async () => {
    const edge: AdminUserRow[] = [
      {
        id: "u9",
        email: null,
        created_at: "not-a-date",
        last_sign_in_at: null,
        analyses_count: 0,
        conversations_count: 0,
        is_admin: false,
      },
    ];
    vi.spyOn(api, "listAdminUsers").mockResolvedValue({ users: edge });
    renderAdmin("/admin/users");
    // Null email → the row id is displayed instead.
    expect(await screen.findByText("u9")).toBeInTheDocument();
    // Both the invalid created_at and the null last_sign_in_at render as the "Never" fallback.
    expect(screen.getAllByText("Never").length).toBe(2);
  });
});

describe("AdminLayout loading + mobile nav", () => {
  it("shows the spinner while the admin probe is still loading", async () => {
    mockUseAuth.mockReturnValue(authValue({ adminState: "loading" }));
    renderAdmin("/admin");
    expect(await screen.findByText("Checking your access…")).toBeInTheDocument();
  });

  it("opens the mobile drawer (backdrop) and closes it via the drawer's close button", async () => {
    const { container } = renderAdmin("/admin");
    await screen.findByLabelText("Show navigation");
    // Drawer starts closed: the off-canvas backdrop is not mounted yet.
    expect(container.querySelector(".bg-black\\/50")).toBeNull();
    // Open: the header's menu button mounts the backdrop.
    fireEvent.click(screen.getByLabelText("Show navigation"));
    expect(container.querySelector(".bg-black\\/50")).not.toBeNull();
    // Close: the drawer's onNavigate close button collapses it again.
    fireEvent.click(screen.getByLabelText("Hide navigation"));
    await waitFor(() => expect(container.querySelector(".bg-black\\/50")).toBeNull());
  });

  it("closes the mobile drawer when the backdrop is clicked", async () => {
    const { container } = renderAdmin("/admin");
    fireEvent.click(await screen.findByLabelText("Show navigation"));
    const backdrop = container.querySelector(".bg-black\\/50");
    expect(backdrop).not.toBeNull();
    fireEvent.click(backdrop as Element);
    await waitFor(() => expect(container.querySelector(".bg-black\\/50")).toBeNull());
  });
});
