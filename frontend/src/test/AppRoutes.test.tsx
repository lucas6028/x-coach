import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

// Which routes require a session is configuration, and configuration that nothing asserts drifts.
// /movements spent its whole life gated only because it inherited RequireAuth from the /explore
// page it replaced (commit 9058abe9 swapped the path and component and left the wrapper), which
// nobody noticed because no test looked. This file pins the public/gated split so re-adding a
// guard — or dropping one from /history or /settings — fails loudly.
vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";

// Every page is stubbed to a marker: this test is about the ROUTE TABLE, not what the pages
// render. Stubbing also keeps it fast and free of each page's own fetches and providers.
vi.mock("../App", () => ({ default: () => <div>studio page</div> }));
vi.mock("../landing/Landing", () => ({ default: () => <div>landing page</div> }));
vi.mock("../pages/Login", () => ({ default: () => <div>login page</div> }));
vi.mock("../pages/History", () => ({ default: () => <div>history page</div> }));
vi.mock("../pages/Movements", () => ({ default: () => <div>movements page</div> }));
vi.mock("../pages/Settings", () => ({ default: () => <div>settings page</div> }));
vi.mock("../pages/Games", () => ({ default: () => <div>games page</div> }));
vi.mock("../pages/WebSlinger", () => ({ default: () => <div>web slinger page</div> }));
vi.mock("../pages/LiffDiag", () => ({ default: () => <div>liff diag page</div> }));
vi.mock("../pages/admin/AdminLogin", () => ({ default: () => <div>admin login page</div> }));
vi.mock("../pages/admin/AdminLayout", () => ({ default: () => <div>admin page</div> }));
vi.mock("../pages/admin/AdminOverview", () => ({ default: () => <div>admin overview</div> }));
vi.mock("../pages/admin/AdminLine", () => ({ default: () => <div>admin line</div> }));
vi.mock("../pages/admin/AdminUsers", () => ({ default: () => <div>admin users</div> }));
vi.mock("../pages/admin/AdminSettingsLlm", () => ({ default: () => <div>admin llm</div> }));
vi.mock("../pages/admin/AdminSettingsRag", () => ({ default: () => <div>admin rag</div> }));
vi.mock("../pages/admin/AdminSettingsAnalyze", () => ({ default: () => <div>admin analyze</div> }));

import AppRoutes from "../AppRoutes";

const mockUseAuth = vi.mocked(useAuth);

/** Render the real route table at `path` with no signed-in session. */
function renderAnonymousAt(path: string) {
  mockUseAuth.mockReturnValue({
    user: null,
    loading: false,
    lineAuthenticating: false,
  } as ReturnType<typeof useAuth>);
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </I18nProvider>
  );
}

beforeEach(() => mockUseAuth.mockReset());

describe("AppRoutes — public routes", () => {
  // The regression this file exists for.
  it("serves /movements to a signed-out visitor", () => {
    renderAnonymousAt("/movements");
    expect(screen.getByText("movements page")).toBeInTheDocument();
    expect(screen.queryByText("login page")).not.toBeInTheDocument();
  });

  it.each([
    ["/", "landing page"],
    ["/app", "studio page"],
    ["/games", "games page"],
    ["/login", "login page"],
    ["/liff/diag", "liff diag page"],
    ["/admin/login", "admin login page"],
  ])("serves %s to a signed-out visitor", (path, marker) => {
    renderAnonymousAt(path);
    expect(screen.getByText(marker)).toBeInTheDocument();
  });

  it("serves the lazy web-slinger route to a signed-out visitor", async () => {
    renderAnonymousAt("/web-slinger");
    expect(await screen.findByText("web slinger page")).toBeInTheDocument();
  });
});

describe("AppRoutes — gated routes", () => {
  // These show one specific user's own data, so they must stay behind RequireAuth. Asserting the
  // gated side too means making /movements public cannot quietly take these with it.
  it.each([
    ["/history", "history page"],
    ["/settings", "settings page"],
  ])("redirects %s to /login when signed out", (path, marker) => {
    renderAnonymousAt(path);
    expect(screen.getByText("login page")).toBeInTheDocument();
    expect(screen.queryByText(marker)).not.toBeInTheDocument();
  });

  it("redirects /admin to the admin login, not the ordinary one", () => {
    renderAnonymousAt("/admin");
    expect(screen.getByText("admin login page")).toBeInTheDocument();
    expect(screen.queryByText("admin page")).not.toBeInTheDocument();
    expect(screen.queryByText("login page")).not.toBeInTheDocument();
  });
});
