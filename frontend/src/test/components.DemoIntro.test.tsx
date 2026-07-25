import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DemoIntro from "../components/DemoIntro";
import { renderWithProviders } from "./renderWithProviders";

// Task 11 gave the movement selector its own dedicated suite (App.movement.test.tsx, exercised
// through App so the URL-driven wiring is covered end to end). These props are fixed to an
// always-available, already-loaded Squat so the pre-existing assertions below — none of which are
// about movement selection — keep observing the same dropzone/loader behavior as before.
const movementProps = {
  movements: [{ name: "Squat", validated: true }],
  movement: "Squat",
  onMovementChange: vi.fn(),
  movementError: "",
  movementsLoaded: true,
};

describe("DemoIntro", () => {
  it("shows the heading", () => {
    renderWithProviders(
      <DemoIntro onFile={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Analyze your Squat in about 20 seconds."
    );
  });

  // Finding 2 of the 2026-07-25 review: this heading sits directly above the movement selector and
  // stayed hardcoded to "squat" regardless of what was selected. Pin that it now names the actual
  // selection, and that the dropzone below it (see UploadDropzone.test.tsx) agrees.
  it("names the selected movement in the heading, not a hardcoded squat", () => {
    renderWithProviders(
      <DemoIntro
        onFile={vi.fn()}
        onOpenLibrary={vi.fn()}
        loading={false}
        statusMsg=""
        error=""
        {...movementProps}
        movement="Push-up"
        movements={[{ name: "Push-up", validated: true }]}
      />
    );
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Analyze your Push-up in about 20 seconds."
    );
    expect(screen.getByText(/Drop a Push-up video/i)).toBeInTheDocument();
  });

  it("shows the sub-heading description", () => {
    renderWithProviders(
      <DemoIntro onFile={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.getByText(/Upload a clip or open a labeled sample/)).toBeInTheDocument();
  });

  it("shows the 'Open a sample clip' button", () => {
    renderWithProviders(
      <DemoIntro onFile={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.getByRole("button", { name: /Open a sample clip/i })).toBeInTheDocument();
  });

  it("calls onOpenLibrary when the sample button is clicked", async () => {
    const user = userEvent.setup();
    const onOpenLibrary = vi.fn();
    renderWithProviders(
      <DemoIntro onFile={vi.fn()} onOpenLibrary={onOpenLibrary} loading={false} statusMsg="" error="" {...movementProps} />
    );
    await user.click(screen.getByRole("button", { name: /Open a sample clip/i }));
    expect(onOpenLibrary).toHaveBeenCalledOnce();
  });

  it("disables the sample button while loading", () => {
    renderWithProviders(
      <DemoIntro onFile={vi.fn()} onOpenLibrary={vi.fn()} loading={true} statusMsg="Processing…" error="" {...movementProps} />
    );
    expect(screen.getByRole("button", { name: /Open a sample clip/i })).toBeDisabled();
  });

  it("swaps the dropzone for the Lumen scan loader while loading", () => {
    renderWithProviders(
      <DemoIntro onFile={vi.fn()} onOpenLibrary={vi.fn()} loading={true} statusMsg="Extracting pose…" error="" {...movementProps} />
    );
    // The Lumen waiting state takes over the upload target, carrying the status message…
    expect(screen.getByRole("status")).toHaveAttribute("aria-label", "Extracting pose…");
    // …and the idle dropzone prompt is gone.
    expect(screen.queryByText(/Drop a squat video/i)).not.toBeInTheDocument();
  });

  it("shows the Lumen scan loader while the movement catalog is still loading", () => {
    renderWithProviders(
      <DemoIntro
        onFile={vi.fn()}
        onOpenLibrary={vi.fn()}
        loading={false}
        statusMsg=""
        error=""
        {...movementProps}
        movementsLoaded={false}
      />
    );
    // Same "we don't know yet" guard as the loading case: the dropzone must not appear before
    // the catalog fetch settles, so a slow network can't let an unconfirmed movement upload.
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.queryByText(/Drop a squat video/i)).not.toBeInTheDocument();
  });

  it("shows the error panel when error is non-empty", () => {
    renderWithProviders(
      <DemoIntro onFile={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="Video too short" {...movementProps} />
    );
    expect(screen.getByText("That clip did not go through")).toBeInTheDocument();
    expect(screen.getByText("Video too short")).toBeInTheDocument();
  });

  it("does not show the error panel when error is empty", () => {
    renderWithProviders(
      <DemoIntro onFile={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.queryByText("That clip did not go through")).not.toBeInTheDocument();
  });

  it("renders the three 'what comes back' step cards", () => {
    renderWithProviders(
      <DemoIntro onFile={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect(screen.getByText("Skeleton and faults")).toBeInTheDocument();
    expect(screen.getByText("Grounded feedback")).toBeInTheDocument();
    expect(screen.getByText("Knowledge graph")).toBeInTheDocument();
  });

  it("shows the movement select with the current value", () => {
    renderWithProviders(
      <DemoIntro onFile={vi.fn()} onOpenLibrary={vi.fn()} loading={false} statusMsg="" error="" {...movementProps} />
    );
    expect((screen.getByLabelText(/movement/i) as HTMLSelectElement).value).toBe("Squat");
  });

  it("shows the movementError panel instead of the dropzone, and hides no other content", () => {
    renderWithProviders(
      <DemoIntro
        onFile={vi.fn()}
        onOpenLibrary={vi.fn()}
        loading={false}
        statusMsg=""
        error=""
        {...movementProps}
        movement="Lunge"
        movementError='"Lunge" cannot be analysed yet. Pick one of the available movements.'
      />
    );
    expect(
      screen.getByText('"Lunge" cannot be analysed yet. Pick one of the available movements.')
    ).toBeInTheDocument();
    expect(screen.queryByText(/Drop a squat video/i)).not.toBeInTheDocument();
  });
});
