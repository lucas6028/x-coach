import { describe, it, expect, vi, afterEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type ClinicInvite as ClinicInviteRow } from "../api";
import ClinicInvite from "../pages/clinic/ClinicInvite";

function invite(overrides: Partial<ClinicInviteRow> = {}): ClinicInviteRow {
  return {
    id: "i1",
    invite_code: "AB12CD",
    status: "pending",
    created_at: "2026-09-15T00:00:00Z",
    expires_at: "2026-09-22T00:00:00Z",
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("ClinicInvite", () => {
  it("generates a code and shows it large, with its expiry date", async () => {
    vi.spyOn(api, "clinicInvites").mockResolvedValue([]);
    const create = vi.spyOn(api, "clinicCreateInvite").mockResolvedValue(invite());
    renderWithProviders(<ClinicInvite />);

    await userEvent.click(await screen.findByRole("button", { name: "Generate an invite code" }));
    expect(create).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("AB12CD")).toBeInTheDocument();
    expect(screen.getByText(/Expires/)).toBeInTheDocument();
  });

  it("lists pending invites and revokes one via api.clinicRevokeLink", async () => {
    const revoke = vi.spyOn(api, "clinicRevokeLink").mockResolvedValue({ link: {} });
    const list = vi
      .spyOn(api, "clinicInvites")
      .mockResolvedValueOnce([invite({ id: "i1", invite_code: "AAAAAA" })])
      .mockResolvedValueOnce([]);
    renderWithProviders(<ClinicInvite />);

    expect(await screen.findByText("AAAAAA")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Revoke" }));

    await waitFor(() => expect(revoke).toHaveBeenCalledWith("i1"));
    await waitFor(() => expect(list.mock.calls.length).toBeGreaterThanOrEqual(2));
  });

  it("shows the no-pending-invites empty state", async () => {
    vi.spyOn(api, "clinicInvites").mockResolvedValue([]);
    renderWithProviders(<ClinicInvite />);
    expect(await screen.findByText("No pending invites.")).toBeInTheDocument();
  });

  it("copies the generated code to the clipboard", async () => {
    vi.spyOn(api, "clinicInvites").mockResolvedValue([]);
    vi.spyOn(api, "clinicCreateInvite").mockResolvedValue(invite());
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    renderWithProviders(<ClinicInvite />);

    await userEvent.click(await screen.findByRole("button", { name: "Generate an invite code" }));
    await screen.findByText("AB12CD");
    await userEvent.click(screen.getByRole("button", { name: "Copy code" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("AB12CD"));
    expect(await screen.findByText("Copied")).toBeInTheDocument();
  });

  it("names the instructions for where a patient enters the code", async () => {
    vi.spyOn(api, "clinicInvites").mockResolvedValue([]);
    renderWithProviders(<ClinicInvite />);
    expect(
      await screen.findByText(/Ask the patient to enter this code under Settings → My therapist\./)
    ).toBeInTheDocument();
  });
});
