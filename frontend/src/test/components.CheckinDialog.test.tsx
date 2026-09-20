import { describe, it, expect, vi, afterEach } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Checkin } from "../api";
import CheckinDialog from "../components/checkin/CheckinDialog";

function baseCheckin(overrides: Partial<Checkin> = {}): Checkin {
  return {
    id: "c1",
    user_id: "u1",
    plan_id: "p1",
    plan_item_id: "i1",
    analysis_id: "an-1",
    pain_nrs: 2,
    rpe: null,
    note: null,
    form_score: null,
    flagged: false,
    flag_reasons: [],
    acknowledged_at: null,
    created_at: "2026-09-15T00:00:00Z",
    ...overrides,
  };
}

afterEach(() => vi.restoreAllMocks());

describe("CheckinDialog", () => {
  it("submits pain (required), rpe and note in the payload", async () => {
    const spy = vi.spyOn(api, "createCheckin").mockResolvedValue(baseCheckin());
    const onSubmitted = vi.fn();
    renderWithProviders(
      <CheckinDialog
        open
        planId="p1"
        planItemId="i1"
        analysisId="an-1"
        movementLabel="Squat"
        onClose={vi.fn()}
        onSubmitted={onSubmitted}
      />
    );
    const dialog = screen.getByRole("dialog");
    await userEvent.click(within(dialog).getByRole("button", { name: "4" }));
    await userEvent.selectOptions(within(dialog).getByLabelText(/perceived effort/i), "7");
    await userEvent.type(within(dialog).getByLabelText(/^note/i), "Felt stiff today");
    await userEvent.click(within(dialog).getByRole("button", { name: /^submit$/i }));

    await vi.waitFor(() =>
      expect(spy).toHaveBeenCalledWith({
        plan_id: "p1",
        plan_item_id: "i1",
        analysis_id: "an-1",
        pain_nrs: 4,
        rpe: 7,
        note: "Felt stiff today",
      })
    );
    expect(onSubmitted).toHaveBeenCalledWith(baseCheckin());
  });

  it("shows the fixed copy and localized reasons when the response is flagged, and stays open", async () => {
    vi.spyOn(api, "createCheckin").mockResolvedValue(
      baseCheckin({ flagged: true, flag_reasons: ["pain_high", "form_drop"] })
    );
    const onClose = vi.fn();
    const onSubmitted = vi.fn();
    renderWithProviders(
      <CheckinDialog open planId="p1" onClose={onClose} onSubmitted={onSubmitted} />
    );
    const dialog = screen.getByRole("dialog");
    await userEvent.click(within(dialog).getByRole("button", { name: "9" }));
    await userEvent.click(within(dialog).getByRole("button", { name: /^submit$/i }));

    expect(await screen.findByText(/this check-in needs attention/i)).toBeInTheDocument();
    expect(screen.getByText("Pain score is high")).toBeInTheDocument();
    expect(screen.getByText("Form score dropped sharply")).toBeInTheDocument();
    expect(onSubmitted).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();

    // Two "Close" buttons exist while flagged (the header X and the flagged panel's own button);
    // clicking either one must close the dialog.
    const closeButtons = screen.getAllByRole("button", { name: /close/i });
    await userEvent.click(closeButtons[closeButtons.length - 1]);
    expect(onClose).toHaveBeenCalled();
  });

  it("skip closes without calling the API", async () => {
    const spy = vi.spyOn(api, "createCheckin");
    const onClose = vi.fn();
    renderWithProviders(<CheckinDialog open planId="p1" onClose={onClose} />);
    await userEvent.click(screen.getByRole("button", { name: /^skip$/i }));
    expect(onClose).toHaveBeenCalled();
    expect(spy).not.toHaveBeenCalled();
  });

  it("requires a pain rating before submitting", async () => {
    const spy = vi.spyOn(api, "createCheckin");
    renderWithProviders(<CheckinDialog open planId="p1" onClose={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /^submit$/i }));
    expect(spy).not.toHaveBeenCalled();
    expect(screen.getByText(/please rate your pain/i)).toBeInTheDocument();
  });
});
