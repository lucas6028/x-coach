import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import Sidebar from "../components/Sidebar";

// The Admin link is UX gating on `isAdmin` from useAuth; mock the hook so both branches are exercised.
// (The sibling components.Sidebar.test.tsx uses the REAL AuthProvider for its games-route cases, so the
// mock lives in this separate file to avoid the file-hoisted vi.mock clobbering those.)
vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
const mockUseAuth = vi.mocked(useAuth);

function renderSidebar(isAdmin: boolean, path = "/app") {
  mockUseAuth.mockReturnValue({ isAdmin } as unknown as ReturnType<typeof useAuth>);
  return render(
    <MemoryRouter initialEntries={[path]}>
      <I18nProvider>
        <Sidebar
          open
          width={240}
          animate={false}
          onOpenLibrary={vi.fn()}
          onNewAnalysis={vi.fn()}
        />
      </I18nProvider>
    </MemoryRouter>
  );
}

beforeEach(() => mockUseAuth.mockReset());

describe("Sidebar — admin link gating", () => {
  it("shows the Admin link when the signed-in user is an admin", () => {
    renderSidebar(true);
    expect(screen.getByRole("link", { name: "Admin" })).toBeInTheDocument();
  });

  it("hides the Admin link for a non-admin", () => {
    renderSidebar(false);
    expect(screen.queryByRole("link", { name: "Admin" })).not.toBeInTheDocument();
  });

  it("highlights the Admin link when on the /admin route", () => {
    renderSidebar(true, "/admin");
    expect(screen.getByRole("link", { name: "Admin" }).className).toContain("text-primary");
  });

  it("does not highlight the Admin link on an unrelated route", () => {
    renderSidebar(true, "/history");
    expect(screen.getByRole("link", { name: "Admin" }).className).not.toContain("text-primary");
  });
});
