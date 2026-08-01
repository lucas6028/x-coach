import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ConfirmDialog from "../components/ConfirmDialog";

function setup(over: Partial<Parameters<typeof ConfirmDialog>[0]> = {}) {
  const onConfirm = vi.fn();
  const onCancel = vi.fn();
  const view = render(
    <ConfirmDialog
      open
      title="Delete this record?"
      description="This can't be undone."
      detail="Side Squat · 10:24"
      confirmLabel="Delete"
      cancelLabel="Cancel"
      onConfirm={onConfirm}
      onCancel={onCancel}
      {...over}
    />
  );
  return { onConfirm, onCancel, view };
}

afterEach(() => vi.restoreAllMocks());

describe("ConfirmDialog", () => {
  it("renders nothing when closed", () => {
    setup({ open: false });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("labels the dialog with its own title", () => {
    setup();
    // aria-labelledby must resolve to the heading, otherwise a screen reader announces the dialog
    // as unnamed.
    expect(screen.getByRole("dialog", { name: "Delete this record?" })).toBeInTheDocument();
    expect(screen.getByText("Side Squat · 10:24")).toBeInTheDocument();
  });

  it("focuses Cancel on open, not the destructive button", async () => {
    setup();
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  });

  it("confirms and cancels through the buttons", async () => {
    const { onConfirm, onCancel } = setup();
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("cancels on Escape and on a backdrop click", async () => {
    const { onCancel } = setup();
    await userEvent.keyboard("{Escape}");
    expect(onCancel).toHaveBeenCalledTimes(1);

    // The backdrop is the dialog's parent; a mousedown on the card itself must not close it.
    const backdrop = screen.getByRole("dialog").parentElement as HTMLElement;
    await userEvent.click(screen.getByRole("dialog"));
    expect(onCancel).toHaveBeenCalledTimes(1);
    await userEvent.click(backdrop);
    expect(onCancel).toHaveBeenCalledTimes(2);
  });

  it("refuses to close while busy", async () => {
    const { onCancel, onConfirm } = setup({ busy: true });
    await userEvent.keyboard("{Escape}");
    await userEvent.click(screen.getByRole("dialog").parentElement as HTMLElement);
    expect(onCancel).not.toHaveBeenCalled();
    // Both buttons are locked, so a double-confirm can't fire a second delete.
    expect(screen.getByRole("button", { name: "Delete" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("returns focus to the element that opened it", async () => {
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();

    const { view } = setup();
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();

    view.unmount();
    expect(opener).toHaveFocus();
    opener.remove();
  });
});
