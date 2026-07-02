import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

// Auth drives which honest state ChatInput shows; mock it (hoisted) so the test is deterministic
// regardless of whether a local frontend/.env happens to configure Supabase. The working
// signed-in flow lives in components.ChatInputActive.test.tsx.
const h = vi.hoisted(() => ({ auth: { configured: false, user: null as { id: string } | null } }));
vi.mock("../lib/auth", () => ({ useAuth: () => h.auth }));

import ChatInput from "../components/ChatInput";

function renderPanel() {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <ChatInput />
      </I18nProvider>
    </MemoryRouter>
  );
}

describe("ChatInput", () => {
  beforeEach(() => {
    h.auth = { configured: false, user: null };
  });

  describe("when the auth/LLM layer is not configured", () => {
    it("renders a disabled text input (honest 'coming soon' fallback)", () => {
      renderPanel();
      expect(screen.getByRole("textbox")).toBeDisabled();
    });

    it("shows the coming-soon placeholder", () => {
      renderPanel();
      expect(screen.getByPlaceholderText(/Ask the AI Coach/i)).toBeInTheDocument();
    });

    it("has a tooltip explaining the feature arrives with the LLM layer", () => {
      renderPanel();
      const wrapper = screen.getByRole("textbox").closest("[title]");
      expect(wrapper?.getAttribute("title")).toMatch(/LLM layer/i);
    });
  });

  describe("when configured but signed out", () => {
    it("invites the user to sign in and does not fake an input", () => {
      h.auth = { configured: true, user: null };
      renderPanel();
      expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
      expect(screen.getByText(/Sign in to chat/i)).toBeInTheDocument();
      expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute("href", "/login");
    });
  });
});
