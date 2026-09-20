import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import Sidebar from "../components/Sidebar";

// The 個案管理 link is UX gating on `isClinician` from useAuth; mirrors components.Sidebar.admin.test.tsx
// exactly. Mocking useAuth here (rather than using the real AuthProvider) lives in its own file for
// the same reason the admin version does: components.Sidebar.test.tsx's file-hoisted vi.mock would
// otherwise clobber those tests.
vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
const mockUseAuth = vi.mocked(useAuth);

function renderSidebar(isClinician: boolean, path = "/app") {
  mockUseAuth.mockReturnValue({ isClinician } as unknown as ReturnType<typeof useAuth>);
  return render(
    <MemoryRouter initialEntries={[path]}>
      <I18nProvider>
        <Sidebar open width={240} animate={false} onNewAnalysis={vi.fn()} />
      </I18nProvider>
    </MemoryRouter>
  );
}

beforeEach(() => mockUseAuth.mockReset());

describe("Sidebar — clinician link gating", () => {
  it("shows the 個案管理 link when the signed-in user is a clinician", () => {
    renderSidebar(true);
    expect(screen.getByRole("link", { name: "Patients" })).toBeInTheDocument();
  });

  it("hides the link for a non-clinician", () => {
    renderSidebar(false);
    expect(screen.queryByRole("link", { name: "Patients" })).not.toBeInTheDocument();
  });

  it("highlights the link on the /clinic route", () => {
    renderSidebar(true, "/clinic");
    expect(screen.getByRole("link", { name: "Patients" }).className).toContain("text-primary");
  });

  it("highlights the link on a nested /clinic/* route", () => {
    renderSidebar(true, "/clinic/patients/p1");
    expect(screen.getByRole("link", { name: "Patients" }).className).toContain("text-primary");
  });

  it("does not highlight the link on an unrelated route", () => {
    renderSidebar(true, "/history");
    expect(screen.getByRole("link", { name: "Patients" }).className).not.toContain("text-primary");
  });
});
