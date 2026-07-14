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

  it("keeps the restart-required note on the analyze page", async () => {
    renderAdmin("/admin/settings/analyze");
    expect(await screen.findByText("Restart required to take effect")).toBeInTheDocument();
  });
});
