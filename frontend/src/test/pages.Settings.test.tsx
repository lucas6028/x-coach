import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, MemoryRouter, Routes, Route } from "react-router-dom";
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

/**
 * Settings is a popup with a category rail: only the selected pane is mounted, so every test that
 * touches a non-default pane has to walk there first. "General" is what opens.
 */
async function openPane(name: RegExp) {
  await userEvent.click(screen.getByRole("button", { name }));
}

beforeEach(() => {
  localStorage.clear();
  // Server-driven model catalog (from /api/health).
  vi.spyOn(api, "health").mockResolvedValue({
    status: "ok",
    chat_models: ["deepseek/deepseek-v4-flash", "minimax/minimax-m3"],
    chat_default: "deepseek/deepseek-v4-flash",
  });
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

describe("Settings — popup shell", () => {
  it("opens on General, which stacks the profile and preferences sections", () => {
    renderSettings();
    expect(screen.getByRole("dialog")).toHaveAttribute("aria-modal", "true");
    expect(screen.getByRole("heading", { name: "Profile" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Preferences" })).toBeInTheDocument();
    // Other panes are not mounted until their category is picked.
    expect(screen.queryByRole("heading", { name: "Coach model" })).not.toBeInTheDocument();
  });

  // The overlay is `fixed inset-0`, and one of the things that opens it is the account menu at
  // the foot of the nav rail — whose `.glass-rail` backdrop-filter makes that 236px rail the
  // containing block for fixed descendants, so an in-tree overlay sizes itself to the rail
  // (measured 234×697 against a 1495×726 viewport) instead of the screen. jsdom has no layout
  // and cannot see that, but it can see the invariant that prevents it: the dialog is rendered
  // outside the tree that mounted it.
  it("portals out of the tree that mounted it", () => {
    const { container } = renderSettings();
    expect(container).not.toContainElement(screen.getByRole("dialog"));
  });

  it("switches panes from the category rail", async () => {
    renderSettings();
    await openPane(/^Account$/);
    expect(screen.getByRole("heading", { name: "Account" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Profile" })).not.toBeInTheDocument();
  });

  it("filters the rail by search, and says so when nothing matches", async () => {
    renderSettings();
    const search = screen.getByRole("searchbox", { name: /search/i });

    await userEvent.type(search, "coach");
    expect(screen.getByRole("button", { name: /^Coach model$/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Account$/ })).not.toBeInTheDocument();
    // Narrowing the rail must not yank the reader out of the pane they were on.
    expect(screen.getByRole("heading", { name: "Profile" })).toBeInTheDocument();

    await userEvent.clear(search);
    await userEvent.type(search, "zzz");
    expect(screen.getByText("No matching settings.")).toBeInTheDocument();
  });

  it("leaves the route when closed, landing in the studio from a deep link", async () => {
    renderSettings();
    await userEvent.click(screen.getByRole("button", { name: /close settings/i }));
    expect(await screen.findByText("app studio")).toBeInTheDocument();
  });

  it("closes on Escape", async () => {
    renderSettings();
    await userEvent.keyboard("{Escape}");
    expect(await screen.findByText("app studio")).toBeInTheDocument();
  });

  it("goes back to the page you opened it from", async () => {
    render(
      <I18nProvider>
        <MemoryRouter initialEntries={["/history"]}>
          <Routes>
            <Route path="/history" element={<Link to="/settings">my records</Link>} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/app" element={<div>app studio</div>} />
          </Routes>
        </MemoryRouter>
      </I18nProvider>
    );

    await userEvent.click(screen.getByRole("link", { name: "my records" }));
    await userEvent.click(await screen.findByRole("button", { name: /close settings/i }));

    // Back to /history — not the studio, which is only the fallback for a first-entry deep link.
    expect(await screen.findByRole("link", { name: "my records" })).toBeInTheDocument();
    expect(screen.queryByText("app studio")).not.toBeInTheDocument();
  });
});

describe("Settings — profile", () => {
  // Scoped to the dialog: the shell's rail carries the same display name and avatar in its
  // account footer, so an unscoped query matches the chrome as well as the pane under test.
  it("shows the profile name, email, and avatar", () => {
    renderSettings();
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Ada Lovelace")).toBeInTheDocument();
    expect(within(dialog).getByText("ada@x.com")).toBeInTheDocument();
    // querySelector rather than a role query: the avatar is decorative (alt=""), so it exposes
    // no `img` role at all.
    expect(dialog.querySelector("img[src='https://x/me.png']")).toBeInTheDocument();
  });
});

describe("Settings — preferences", () => {
  // Language is the only preference left: the theme picker is gone app-wide (light-only), and
  // this row is now the single place language can be changed anywhere in the app.
  it("carries language and nothing else", () => {
    renderSettings();
    const prefs = screen.getByRole("heading", { name: "Preferences" }).closest("section")!;
    expect(within(prefs).getByText("Language")).toBeInTheDocument();
    expect(within(prefs).queryByText("Appearance")).not.toBeInTheDocument();
    expect(within(prefs).queryByRole("radio")).not.toBeInTheDocument();
  });

  it("Escape closes an open language menu without closing the dialog", async () => {
    const user = userEvent.setup();
    renderSettings();
    const languageRow = screen.getByText("Language").closest("div")!;
    await user.click(within(languageRow).getByRole("button"));
    expect(screen.getByRole("menuitemradio", { name: /繁體中文/i })).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menuitemradio", { name: /繁體中文/i })).not.toBeInTheDocument();
    // The popup itself survives — one Escape must not tear down both layers.
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.queryByText("app studio")).not.toBeInTheDocument();
  });

  it("switches language from within Settings", async () => {
    const user = userEvent.setup();
    renderSettings();
    // AppLayout also renders the web Header, which carries its own LanguageToggle — so scope
    // to the Settings row's copy rather than matching on the icon-only trigger button alone.
    const languageRow = screen.getByText("Language").closest("div")!;
    await user.click(within(languageRow).getByRole("button"));
    await user.click(await screen.findByRole("menuitemradio", { name: /繁體中文/i }));
    expect(screen.getByRole("heading", { name: "偏好設定" })).toBeInTheDocument();
  });
});

describe("Settings — coach model", () => {
  it("renders the server-driven catalog and pre-selects the server default", async () => {
    renderSettings();
    await openPane(/^Coach model$/);
    expect(screen.getByRole("heading", { name: "Coach model" })).toBeInTheDocument();
    // Radios appear once the catalog loads from /api/health.
    const deepseek = await screen.findByRole("radio", { name: /DeepSeek V4 Flash/i });
    expect(deepseek).toBeChecked(); // fresh user -> server default (chat_default) pre-selected
    expect(screen.getByRole("radio", { name: /MiniMax M3/i })).not.toBeChecked();
    expect(screen.getAllByRole("radio")).toHaveLength(2); // exactly what the server offered
  });

  it("persists the chosen coach model to localStorage", async () => {
    renderSettings();
    await openPane(/^Coach model$/);
    await userEvent.click(await screen.findByRole("radio", { name: /MiniMax M3/i }));
    expect(localStorage.getItem("chat_model")).toBe("minimax/minimax-m3");
    expect(screen.getByRole("radio", { name: /MiniMax M3/i })).toBeChecked();
  });
});

describe("Settings — account", () => {
  it("requires confirmation before clearing analyses", async () => {
    const spy = vi.spyOn(api, "deleteAnalyses").mockResolvedValue({ deleted: 3 });
    renderSettings();
    await openPane(/^Account$/);

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
    await openPane(/^Account$/);
    await userEvent.click(screen.getByRole("button", { name: /clear all/i }));
    await userEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(screen.getByRole("button", { name: /clear all/i })).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();
  });

  it("surfaces an error when clearing fails", async () => {
    vi.spyOn(api, "deleteAnalyses").mockRejectedValue(new Error("500 boom"));
    renderSettings();
    await openPane(/^Account$/);
    await userEvent.click(screen.getByRole("button", { name: /clear all/i }));
    await userEvent.click(screen.getByRole("button", { name: /yes, delete everything/i }));
    expect(await screen.findByText(/Couldn't clear your analyses/i)).toBeInTheDocument();
  });
});
