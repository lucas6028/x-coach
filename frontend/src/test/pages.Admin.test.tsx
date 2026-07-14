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
import Admin from "../pages/Admin";

const mockUseAuth = vi.mocked(useAuth);

function renderAdmin() {
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={["/admin"]}>
        <Routes>
          <Route path="/admin" element={<Admin />} />
          <Route path="/app" element={<div>app studio</div>} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

beforeEach(() => {
  mockUseAuth.mockReturnValue({
    user: { id: "u1", email: "ada@x.com" },
    signOut: vi.fn(),
  } as unknown as ReturnType<typeof useAuth>);
  // Default happy-path stubs for the P3 overview + users sections so the existing admin-ready tests
  // don't hit real fetch when those sections mount. Individual tests override as needed.
  vi.spyOn(api, "getAdminOverview").mockResolvedValue(SAMPLE_OVERVIEW);
  vi.spyOn(api, "listAdminUsers").mockResolvedValue({ users: SAMPLE_USERS });
});
afterEach(() => vi.restoreAllMocks());

describe("Admin", () => {
  it("renders the admin panel for an admin", async () => {
    vi.spyOn(api, "adminStatus").mockResolvedValue({ is_admin: true });
    vi.spyOn(api, "getAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin();
    // The panel subtitle is unique to the admin ready-state (the "Admin" title also appears in the
    // shared Header, so we key off the subtitle rather than the heading).
    expect(
      await screen.findByText(/Manage users, LLM settings, and pipeline parameters/i)
    ).toBeInTheDocument();
    // The denied card must NOT show for an admin.
    expect(
      screen.queryByText("You don't have access to the admin panel.")
    ).not.toBeInTheDocument();
  });

  it("loads the settings form with current values and saves the edited payload", async () => {
    vi.spyOn(api, "adminStatus").mockResolvedValue({ is_admin: true });
    vi.spyOn(api, "getAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    const update = vi.spyOn(api, "updateAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin();

    // The current effective values populate the controlled inputs.
    const baseUrl = await screen.findByLabelText("Provider base URL");
    expect(baseUrl).toHaveValue("https://openrouter.ai/api/v1");
    expect(screen.getByLabelText("RAG top-k")).toHaveValue("5");

    // Edit a knob and save.
    fireEvent.change(screen.getByLabelText("RAG top-k"), { target: { value: "9" } });
    fireEvent.click(screen.getByRole("button", { name: /Save changes/i }));

    await waitFor(() => expect(update).toHaveBeenCalled());
    expect(update).toHaveBeenCalledWith(expect.objectContaining({ rag_top_k: 9 }));
    // The edited value is surfaced back and a success state renders.
    expect(await screen.findByText("Settings saved.")).toBeInTheDocument();
  });

  it("shows a load-error card when the settings fetch fails", async () => {
    vi.spyOn(api, "adminStatus").mockResolvedValue({ is_admin: true });
    vi.spyOn(api, "getAdminSettings").mockRejectedValue(new Error("500 boom"));
    renderAdmin();
    expect(await screen.findByText("Couldn't load the current settings.")).toBeInTheDocument();
  });

  it("shows an access-denied card for a non-admin", async () => {
    vi.spyOn(api, "adminStatus").mockResolvedValue({ is_admin: false });
    renderAdmin();
    expect(
      await screen.findByText("You don't have access to the admin panel.")
    ).toBeInTheDocument();
    // The admin panel subtitle must NOT render for a non-admin.
    expect(
      screen.queryByText(/Manage users, LLM settings, and pipeline parameters/i)
    ).not.toBeInTheDocument();
  });

  it("surfaces an error when the status check fails", async () => {
    vi.spyOn(api, "adminStatus").mockRejectedValue(new Error("500 boom"));
    renderAdmin();
    expect(
      await screen.findByText(/Couldn't verify your admin access/i)
    ).toBeInTheDocument();
  });

  it("renders the system overview cards from getAdminOverview", async () => {
    vi.spyOn(api, "adminStatus").mockResolvedValue({ is_admin: true });
    vi.spyOn(api, "getAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin();
    // Totals surface as their own cards; the stores card shows "1/2 ready" (one of two present).
    expect(await screen.findByText("Total users")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument(); // total analyses value
    expect(screen.getByText("1/2 ready")).toBeInTheDocument();
  });

  it("renders the users table rows from listAdminUsers", async () => {
    vi.spyOn(api, "adminStatus").mockResolvedValue({ is_admin: true });
    vi.spyOn(api, "getAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin();
    expect(await screen.findByText("bob@x.com")).toBeInTheDocument();
    expect(screen.getByText("ada@x.com")).toBeInTheDocument();
    // The signed-in admin's own row is tagged "You".
    expect(screen.getByText("You")).toBeInTheDocument();
  });

  it("toggles a non-self user's role and refreshes the list", async () => {
    vi.spyOn(api, "adminStatus").mockResolvedValue({ is_admin: true });
    vi.spyOn(api, "getAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    const setRole = vi.spyOn(api, "setUserRole").mockResolvedValue({ ok: true });
    const list = vi.spyOn(api, "listAdminUsers").mockResolvedValue({ users: SAMPLE_USERS });
    renderAdmin();

    // bob (u2) is not admin -> the button offers "Make admin".
    const makeAdmin = await screen.findByRole("button", { name: "Make admin" });
    fireEvent.click(makeAdmin);
    await waitFor(() => expect(setRole).toHaveBeenCalledWith("u2", true));
    // The list is re-fetched after a successful toggle (initial mount + post-toggle refresh).
    await waitFor(() => expect(list.mock.calls.length).toBeGreaterThanOrEqual(2));
  });

  it("disables the toggle on the signed-in admin's own row (no self-demote)", async () => {
    vi.spyOn(api, "adminStatus").mockResolvedValue({ is_admin: true });
    vi.spyOn(api, "getAdminSettings").mockResolvedValue(SAMPLE_SETTINGS);
    renderAdmin();
    // ada (u1) is the current user and an admin -> her row's "Revoke admin" button is disabled.
    const revoke = await screen.findByRole("button", { name: "Revoke admin" });
    expect(revoke).toBeDisabled();
  });
});
