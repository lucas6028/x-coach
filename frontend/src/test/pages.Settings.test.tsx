import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { api } from "../api";

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
import Settings from "../pages/Settings";

const mockUseAuth = vi.mocked(useAuth);

function renderSettings() {
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={["/settings"]}>
        <Routes>
          <Route path="/settings" element={<Settings />} />
          <Route path="/app" element={<div>app studio</div>} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

beforeEach(() => {
  localStorage.clear();
  mockUseAuth.mockReturnValue({
    user: {
      email: "ada@x.com",
      app_metadata: { provider: "google" },
      user_metadata: { full_name: "Ada Lovelace", avatar_url: "https://x/me.png" },
    },
    signOut: vi.fn(),
  } as unknown as ReturnType<typeof useAuth>);
});
afterEach(() => vi.restoreAllMocks());

describe("Settings", () => {
  it("shows the profile name, email, and avatar", () => {
    const { container } = renderSettings();
    expect(screen.getByText("Ada Lovelace")).toBeInTheDocument();
    expect(screen.getByText("ada@x.com")).toBeInTheDocument();
    expect(container.querySelector("img[src='https://x/me.png']")).toBeInTheDocument();
  });

  it("offers server-default + the four models, defaulting to server default", () => {
    renderSettings();
    expect(screen.getByRole("heading", { name: "Coach model" })).toBeInTheDocument();
    // Fresh user: "Server default" (= OPENROUTER_MODEL) is selected, not a specific model.
    expect(screen.getByRole("radio", { name: /Server default/i })).toBeChecked();
    expect(screen.getByRole("radio", { name: /DeepSeek V4 Flash/i })).not.toBeChecked();
    // Server default + the four curated models.
    expect(screen.getAllByRole("radio")).toHaveLength(5);
  });

  it("persists the chosen coach model to localStorage", async () => {
    renderSettings();
    await userEvent.click(screen.getByRole("radio", { name: /MiniMax M3/i }));
    expect(localStorage.getItem("chat_model")).toBe("minimax/minimax-m3");
    expect(screen.getByRole("radio", { name: /MiniMax M3/i })).toBeChecked();
  });

  it("can switch back to the server default", async () => {
    localStorage.setItem("chat_model", "minimax/minimax-m3");
    renderSettings();
    expect(screen.getByRole("radio", { name: /MiniMax M3/i })).toBeChecked();
    await userEvent.click(screen.getByRole("radio", { name: /Server default/i }));
    expect(localStorage.getItem("chat_model")).toBe("");
    expect(screen.getByRole("radio", { name: /Server default/i })).toBeChecked();
  });

  it("requires confirmation before clearing analyses", async () => {
    const spy = vi.spyOn(api, "deleteAnalyses").mockResolvedValue({ deleted: 3 });
    renderSettings();

    // First click reveals the confirm step; it does not call the API yet.
    await userEvent.click(screen.getByRole("button", { name: /clear all/i }));
    expect(spy).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole("button", { name: /yes, delete everything/i }));
    await waitFor(() => expect(spy).toHaveBeenCalledOnce());
    expect(await screen.findByText("Deleted 3 saved analyses.")).toBeInTheDocument();
  });

  it("can cancel the clear confirmation", async () => {
    const spy = vi.spyOn(api, "deleteAnalyses").mockResolvedValue({ deleted: 0 });
    renderSettings();
    await userEvent.click(screen.getByRole("button", { name: /clear all/i }));
    await userEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(screen.getByRole("button", { name: /clear all/i })).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();
  });

  it("surfaces an error when clearing fails", async () => {
    vi.spyOn(api, "deleteAnalyses").mockRejectedValue(new Error("500 boom"));
    renderSettings();
    await userEvent.click(screen.getByRole("button", { name: /clear all/i }));
    await userEvent.click(screen.getByRole("button", { name: /yes, delete everything/i }));
    expect(await screen.findByText(/Couldn't clear your analyses/i)).toBeInTheDocument();
  });
});
