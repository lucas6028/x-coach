import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { api, ApiError, type CareClinician } from "../api";

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

async function openTherapistPane() {
  await userEvent.click(screen.getByRole("button", { name: /^My therapist$/ }));
}

function clinician(overrides: Partial<CareClinician> = {}): CareClinician {
  return {
    link_id: "l1",
    clinician_id: "clin-1",
    email: "therapist@clinic.com",
    display_name: "Dr. Wu",
    accepted_at: "2026-09-15T00:00:00Z",
    ...overrides,
  };
}

beforeEach(() => {
  localStorage.clear();
  vi.spyOn(api, "health").mockResolvedValue({ status: "ok" });
  vi.spyOn(api, "careClinicians").mockResolvedValue([]);
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

describe("Settings — my therapist", () => {
  it("accepting a code lists the linked therapist", async () => {
    const accept = vi.spyOn(api, "careAccept").mockResolvedValue({ link: { id: "l1" } });
    vi.mocked(api.careClinicians).mockResolvedValueOnce([]).mockResolvedValueOnce([clinician()]);

    renderSettings();
    await openTherapistPane();
    expect(await screen.findByText(/no linked therapists yet/i)).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText(/invite code/i), "ABC123");
    await userEvent.click(screen.getByRole("button", { name: /^link$/i }));

    await waitFor(() => expect(accept).toHaveBeenCalledWith("ABC123"));
    expect(await screen.findByText("Dr. Wu")).toBeInTheDocument();
  });

  it("shows the invalid/expired copy on a 404", async () => {
    vi.spyOn(api, "careAccept").mockRejectedValue(
      new ApiError("Invite code is invalid or expired.", 404)
    );
    renderSettings();
    await openTherapistPane();
    await userEvent.type(screen.getByLabelText(/invite code/i), "BAD");
    await userEvent.click(screen.getByRole("button", { name: /^link$/i }));
    expect(await screen.findByText(/invite code is invalid or expired/i)).toBeInTheDocument();
  });

  it("shows the own-code copy on a 400", async () => {
    vi.spyOn(api, "careAccept").mockRejectedValue(new ApiError("own code", 400));
    renderSettings();
    await openTherapistPane();
    await userEvent.type(screen.getByLabelText(/invite code/i), "MINE");
    await userEvent.click(screen.getByRole("button", { name: /^link$/i }));
    expect(await screen.findByText(/can't use your own invite code/i)).toBeInTheDocument();
  });

  it("unlink goes through confirm and calls careUnlink", async () => {
    vi.mocked(api.careClinicians).mockResolvedValueOnce([clinician()]).mockResolvedValueOnce([]);
    const unlink = vi.spyOn(api, "careUnlink").mockResolvedValue({ link: { id: "l1" } });

    renderSettings();
    await openTherapistPane();
    expect(await screen.findByText("Dr. Wu")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /^unlink$/i }));
    // The confirm dialog is up; the API must not fire until it is confirmed. The settings popup
    // is ITSELF a `role="dialog"`, so scope by the confirm dialog's own accessible name rather
    // than querying `dialog` unscoped (which would match both).
    expect(unlink).not.toHaveBeenCalled();
    const dialog = screen.getByRole("dialog", { name: /unlink this therapist/i });
    await userEvent.click(within(dialog).getByRole("button", { name: /^unlink$/i }));

    await waitFor(() => expect(unlink).toHaveBeenCalledWith("l1"));
  });
});
